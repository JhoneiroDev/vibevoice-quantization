#!/usr/bin/env python3
"""Build and quality-gate selective TorchAO W4A16 VibeVoice artifacts."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
import whisper
from torchao.quantization import Int4WeightOnlyConfig
from torchao.quantization.quantize_.workflows import (
    Int4ChooseQParamsAlgorithm,
    Int4PackingFormat,
)
from transformers import TorchAoConfig


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: E402
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: E402

try:  # noqa: E402
    from quantization_common import (
        atomic_json,
        expected_qwen_weights,
        sha256,
        source_hashes,
        validate_canonical_source,
        validate_manifest_source,
    )
    from validate_vibevoice_es_quality import TEXTS, normalize
except ModuleNotFoundError:  # noqa: E402
    from scripts.quantization_common import (
        atomic_json,
        expected_qwen_weights,
        sha256,
        source_hashes,
        validate_canonical_source,
        validate_manifest_source,
    )
    from scripts.validate_vibevoice_es_quality import TEXTS, normalize


SOURCE = ROOT / "weights" / "vibevoice-1.5b-es"
FINAL = ROOT / "weights" / "vibevoice-1.5b-es-torchao-int4"
PARTIAL = FINAL.with_name(FINAL.name + ".partial")
OUTPUT = ROOT / "outputs" / "sprint2_torchao_int4"
VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"
TARGET_SR = 24000
MAX_WER = 0.30
TORCHAO_VERSION = "0.18.0"
GROUP_SIZE = 128
PACKING_FORMAT = Int4PackingFormat.TILE_PACKED_TO_4D
QPARAMS_ALGORITHM = Int4ChooseQParamsAlgorithm.HQQ
QUANTIZED_WEIGHT_TYPE = "Int4TilePackedTo4dTensor"
SENSITIVE_MODULES = [
    "model.language_model.embed_tokens",
    "lm_head",
    "model.prediction_head",
    "model.acoustic_tokenizer",
    "model.semantic_tokenizer",
    "model.acoustic_connector",
    "model.semantic_connector",
]


def _configuration() -> TorchAoConfig:
    return TorchAoConfig(
        Int4WeightOnlyConfig(
            group_size=GROUP_SIZE,
            int4_packing_format=PACKING_FORMAT,
            int4_choose_qparams_algorithm=QPARAMS_ALGORITHM,
        ),
        modules_to_not_convert=SENSITIVE_MODULES,
    )


def _cast_unquantized_to_bfloat16(model) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            if type(parameter).__name__ == QUANTIZED_WEIGHT_TYPE:
                continue
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)


def _validate_modules(model) -> list[str]:
    expected = {
        name.removesuffix(".weight"): shape
        for name, shape in expected_qwen_weights().items()
    }
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and type(module.weight).__name__ == QUANTIZED_WEIGHT_TYPE
    }
    if set(modules) != set(expected):
        raise AssertionError(
            "Invalid TorchAO INT4 targets: "
            f"missing={sorted(set(expected) - set(modules))[:5]}, "
            f"unexpected={sorted(set(modules) - set(expected))[:5]}"
        )
    invalid_shapes = [
        (name, tuple(module.weight.shape))
        for name, module in modules.items()
        if tuple(module.weight.shape) != expected[name]
    ]
    if invalid_shapes:
        raise AssertionError(f"Invalid TorchAO INT4 shapes: {invalid_shapes[:5]}")
    probed_shapes = set()
    for name, module in modules.items():
        weight = module.weight
        if weight.qdata.dtype != torch.int32 or not weight.qdata.is_contiguous():
            raise AssertionError(f"Invalid packed TorchAO INT4 data: {name}")
        if (
            weight.scale_and_zero.dtype != torch.bfloat16
            or not weight.scale_and_zero.is_contiguous()
            or not torch.isfinite(weight.scale_and_zero).all()
        ):
            raise AssertionError(f"Invalid TorchAO INT4 scales or zeros: {name}")
        if list(weight.block_size) != [1, GROUP_SIZE]:
            raise AssertionError(f"Invalid TorchAO INT4 block size: {name}")
        if weight.act_pre_scale is not None and not torch.isfinite(weight.act_pre_scale).all():
            raise AssertionError(f"Invalid TorchAO INT4 activation scale: {name}")

        shape = (module.in_features, module.out_features)
        if shape not in probed_shapes:
            probe = torch.zeros(
                1, module.in_features, dtype=torch.bfloat16, device=weight.device
            )
            with torch.inference_mode():
                output = module(probe)
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise AssertionError(f"Invalid TorchAO INT4 kernel output: {name}")
            probed_shapes.add(shape)

    embedding = model.model.language_model.embed_tokens.weight
    if embedding.dtype != torch.bfloat16:
        raise AssertionError(f"embed_tokens has unexpected dtype: {embedding.dtype}")
    if model.lm_head.weight.data_ptr() != embedding.data_ptr():
        raise AssertionError("lm_head and embed_tokens are not tied")
    invalid_dense_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if type(parameter).__name__ != QUANTIZED_WEIGHT_TYPE
        and parameter.is_floating_point()
        and parameter.dtype != torch.bfloat16
    ]
    if invalid_dense_dtypes:
        raise AssertionError(
            f"Non-quantized parameters are not BF16: {invalid_dense_dtypes[:5]}"
        )
    return sorted(modules)


def _artifact_inventory(path: Path) -> dict[str, dict[str, int | str]]:
    files = [file for file in path.rglob("*") if file.is_file() and file.name != "manifest.json"]
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(files)
    }


def _validate_existing(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "vibevoice-selective-torchao-int4-v1":
        raise ValueError(f"Unsupported TorchAO artifact: {path}")
    validate_manifest_source(manifest)
    if manifest.get("status") != "validated":
        raise ValueError(f"Artifact is not validated: {manifest.get('status')}")
    if manifest.get("artifacts") != _artifact_inventory(path):
        raise ValueError(f"Artifact hashes do not match manifest: {path}")
    if manifest.get("validation", {}).get("quantized_modules") != 196:
        raise ValueError("TorchAO artifact does not contain 196 validated modules")
    return manifest


def run() -> None:
    if importlib.metadata.version("torchao") != TORCHAO_VERSION:
        raise RuntimeError(f"Sprint 2 requires torchao=={TORCHAO_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("TorchAO W4A16 requires a CUDA GPU")
    gate = validate_canonical_source(SOURCE)
    if not VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {VOICE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if FINAL.is_dir():
        manifest = _validate_existing(FINAL)
        atomic_json(OUTPUT / "metrics.json", manifest["validation"]["quality"])
        print(f"Validated existing TorchAO INT4 artifact: {FINAL}")
        return
    names = None
    if PARTIAL.exists():
        if (PARTIAL / "manifest.json").exists():
            raise FileExistsError(
                f"Existing evaluated TorchAO candidate requires manual review: {PARTIAL}"
            )
        print(f"Resuming saved TorchAO candidate without requantizing: {PARTIAL}")
    else:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            SOURCE,
            quantization_config=_configuration(),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.eval()
        model.set_ddpm_inference_steps(num_steps=20)
        _cast_unquantized_to_bfloat16(model)
        names = _validate_modules(model)
        model.save_pretrained(PARTIAL, safe_serialization=False, max_shard_size="4GB")
        processor = VibeVoiceProcessor.from_pretrained(SOURCE, local_files_only=True)
        processor.save_pretrained(PARTIAL)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    reloaded = VibeVoiceForConditionalGenerationInference.from_pretrained(
        PARTIAL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
        weights_only=False,
    )
    reloaded.eval()
    reloaded.set_ddpm_inference_steps(num_steps=20)
    _cast_unquantized_to_bfloat16(reloaded)
    reloaded_names = _validate_modules(reloaded)
    if names is not None and reloaded_names != names:
        raise AssertionError("TorchAO target set changed after reload")
    names = reloaded_names
    shutil.copy2(
        SOURCE / "preprocessor_config.json", PARTIAL / "preprocessor_config.json"
    )
    processor = VibeVoiceProcessor.from_pretrained(PARTIAL, local_files_only=True)
    quantized_params = sum(np.prod(shape) for shape in expected_qwen_weights().values())
    total_params = sum(parameter.numel() for parameter in reloaded.parameters())

    idle_vram = torch.cuda.memory_allocated() / 1024**3
    peak_vram = idle_vram
    rtfs = []
    wav_paths = []
    for index, text in enumerate(TEXTS, start=1):
        torch.manual_seed(42 + index - 1)
        inputs = processor(
            text=[f"Speaker 1: {text}"],
            voice_samples=[[str(VOICE)]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in inputs.items()
            if value is not None
        }
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = reloaded.generate(
                **inputs,
                max_new_tokens=600,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=True,
            )
        elapsed = time.perf_counter() - started
        if not generated.speech_outputs or generated.speech_outputs[0] is None:
            raise RuntimeError(f"No TorchAO INT4 audio produced for sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite TorchAO INT4 audio for sample {index}")
        duration = len(audio) / TARGET_SR
        if duration <= 0:
            raise RuntimeError(f"Empty TorchAO INT4 audio for sample {index}")
        rtfs.append(elapsed / duration)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)
        wav = OUTPUT / f"sample_{index:02d}.wav"
        temporary = wav.with_suffix(".wav.partial")
        sf.write(temporary, audio, TARGET_SR, subtype="PCM_16", format="WAV")
        temporary.replace(wav)
        wav_paths.append(wav)

    del reloaded, generated, inputs, processor
    gc.collect()
    torch.cuda.empty_cache()

    asr = whisper.load_model(
        "large-v3", device="cpu", download_root=str(Path.home() / ".cache" / "whisper")
    )
    samples = []
    for text, wav in zip(TEXTS, wav_paths):
        audio, _ = librosa.load(wav, sr=16000, mono=True)
        transcript = asr.transcribe(
            audio,
            language="es",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=1,
            sample_len=64,
        )["text"].strip()
        samples.append(
            {
                "reference": text,
                "transcript": transcript,
                "wer": float(jiwer.wer(normalize(text), normalize(transcript))),
                "cer": float(jiwer.cer(normalize(text), normalize(transcript))),
            }
        )
    del asr

    wer = float(np.mean([sample["wer"] for sample in samples]))
    metrics = {
        "model": str(FINAL.resolve()),
        "source": str(SOURCE.resolve()),
        "source_hashes": source_hashes(SOURCE),
        "source_gate_wer": gate["wer"],
        "quantization": "TorchAO W4A16 HQQ g128",
        "quantized_modules": len(names),
        "quantized_parameter_fraction": quantized_params / total_params,
        "disk_gib": sum(
            file.stat().st_size for file in PARTIAL.rglob("*") if file.is_file()
        )
        / 1024**3,
        "vram_idle_gib": idle_vram,
        "vram_peak_gib": peak_vram,
        "rtf": float(np.mean(rtfs)),
        "wer": wer,
        "cer": float(np.mean([sample["cer"] for sample in samples])),
        "max_wer": MAX_WER,
        "passed": wer <= MAX_WER,
        "samples": samples,
    }
    atomic_json(OUTPUT / "metrics.json", metrics)
    manifest = {
        "schema": "vibevoice-selective-torchao-int4-v1",
        "status": "validated" if metrics["passed"] else "quality_failed",
        "source": str(SOURCE.resolve()),
        "source_hashes": source_hashes(SOURCE),
        "quantization": {
            "type": "W4A16",
            "algorithm": QPARAMS_ALGORITHM.value,
            "group_size": GROUP_SIZE,
            "packing_format": PACKING_FORMAT.value,
            "torchao_version": TORCHAO_VERSION,
            "expected_modules": 196,
            "compute_dtype": "torch.bfloat16",
        },
        "artifacts": _artifact_inventory(PARTIAL),
        "validation": {
            "quantized_modules": len(names),
            "reload_verified": True,
            "quality": metrics,
        },
    }
    atomic_json(PARTIAL / "manifest.json", manifest)
    if not metrics["passed"]:
        raise RuntimeError(
            f"TorchAO candidate not promoted: WER {wer:.4f} > {MAX_WER:.4f}"
        )
    PARTIAL.replace(FINAL)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
