#!/usr/bin/env python3
"""Build and quality-gate selective bitsandbytes VibeVoice artifacts."""

from __future__ import annotations

import gc
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
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from transformers import BitsAndBytesConfig


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
VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"
TARGET_SR = 24000
MAX_WER = 0.30
SENSITIVE_MODULES = [
    "embed_tokens",
    "lm_head",
    "prediction_head",
    "acoustic_tokenizer",
    "semantic_tokenizer",
    "acoustic_connector",
    "semantic_connector",
]


def _configuration(mode: str) -> tuple[BitsAndBytesConfig, type, torch.dtype]:
    if mode == "int8":
        return (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_skip_modules=SENSITIVE_MODULES,
            ),
            Linear8bitLt,
            torch.bfloat16,
        )
    if mode == "nf4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8,
                llm_int8_skip_modules=SENSITIVE_MODULES,
            ),
            Linear4bit,
            torch.bfloat16,
        )
    raise ValueError(f"Unsupported bitsandbytes mode: {mode}")


def _validate_modules(model, module_type: type, mode: str) -> list[str]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, module_type)
    }
    expected_shapes = {
        name.removesuffix(".weight"): shape
        for name, shape in expected_qwen_weights().items()
    }
    if set(modules) != set(expected_shapes):
        raise AssertionError(
            f"Invalid {mode.upper()} targets: "
            f"missing={sorted(set(expected_shapes) - set(modules))[:5]}, "
            f"unexpected={sorted(set(modules) - set(expected_shapes))[:5]}"
        )
    bad_shapes = [
        (name, module.out_features, module.in_features)
        for name, module in modules.items()
        if (module.out_features, module.in_features) != expected_shapes[name]
    ]
    if bad_shapes:
        raise AssertionError(f"Invalid {mode.upper()} module shapes: {bad_shapes[:5]}")

    for name, module in modules.items():
        if mode == "nf4":
            state = module.weight.quant_state
            if state is None or not state.nested or module.compute_dtype != torch.bfloat16:
                raise AssertionError(f"Invalid NF4/Double Quant state: {name}")
            if not torch.isfinite(state.absmax.float()).all():
                raise AssertionError(f"Non-finite NF4 scales: {name}")
        else:
            scale = getattr(module.weight, "SCB", None)
            if scale is not None and not torch.isfinite(scale.float()).all():
                raise AssertionError(f"Non-finite INT8 scales: {name}")

    embedding = model.model.language_model.embed_tokens.weight
    if embedding.dtype not in {torch.float16, torch.bfloat16}:
        raise AssertionError(f"embed_tokens has unexpected dtype: {embedding.dtype}")
    if model.lm_head.weight.data_ptr() != embedding.data_ptr():
        raise AssertionError("lm_head and embed_tokens are not tied")
    return sorted(modules)


def _artifact_inventory(path: Path) -> dict[str, dict[str, int | str]]:
    files = [file for file in path.rglob("*") if file.is_file() and file.name != "manifest.json"]
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(files)
    }


def _validate_existing(path: Path, mode: str) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "vibevoice-selective-bnb-v1":
        raise ValueError(f"Unsupported bitsandbytes artifact: {path}")
    validate_manifest_source(manifest)
    if manifest.get("status") != "validated":
        raise ValueError(f"Artifact is not validated: {manifest.get('status')}")
    if manifest.get("quantization", {}).get("type") != mode:
        raise ValueError(f"Artifact mode does not match {mode}: {path}")
    if manifest.get("artifacts") != _artifact_inventory(path):
        raise ValueError(f"Artifact hashes do not match manifest: {path}")
    if manifest.get("validation", {}).get("quantized_modules") != 196:
        raise ValueError(f"Artifact does not contain 196 validated modules: {path}")
    return manifest


def run(mode: str) -> None:
    assert torch.cuda.is_available(), "INT8/NF4 requires a CUDA GPU"
    gate = validate_canonical_source(SOURCE)
    if not VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {VOICE}")

    bnb_config, module_type, model_dtype = _configuration(mode)
    final = ROOT / "weights" / f"vibevoice-1.5b-es-{mode}"
    partial = final.with_name(final.name + ".partial")
    output = ROOT / "outputs" / f"sprint{5 if mode == 'int8' else 6}_{mode}"
    output.mkdir(parents=True, exist_ok=True)
    if final.is_dir():
        manifest = _validate_existing(final, mode)
        atomic_json(output / "metrics.json", manifest["validation"]["quality"])
        print(f"Validated existing {mode.upper()} artifact: {final}")
        return
    if partial.exists():
        shutil.rmtree(partial)

    print(f"Loading canonical VibeVoice-ES with selective {mode.upper()}...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        SOURCE,
        quantization_config=bnb_config,
        torch_dtype=model_dtype,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    names = _validate_modules(model, module_type, mode)
    quantized_modules = [
        module for module in model.modules() if isinstance(module, module_type)
    ]
    quantized_weight_ids = {id(module.weight) for module in quantized_modules}
    quantized_params = sum(
        module.in_features * module.out_features for module in quantized_modules
    )
    protected_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in quantized_weight_ids
    )
    total_params = protected_params + quantized_params

    model.save_pretrained(partial, safe_serialization=True, max_shard_size="4GB")
    processor = VibeVoiceProcessor.from_pretrained(SOURCE, local_files_only=True)
    processor.save_pretrained(partial)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = VibeVoiceForConditionalGenerationInference.from_pretrained(
        partial,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    reloaded.eval()
    reloaded.set_ddpm_inference_steps(num_steps=20)
    reloaded_names = _validate_modules(reloaded, module_type, mode)
    if reloaded_names != names:
        raise AssertionError(f"{mode.upper()} target set changed after reload")

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
            raise RuntimeError(f"No {mode.upper()} audio produced for sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite {mode.upper()} audio for sample {index}")
        duration = len(audio) / TARGET_SR
        if duration <= 0:
            raise RuntimeError(f"Empty {mode.upper()} audio for sample {index}")
        rtfs.append(elapsed / duration)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)
        wav = output / f"sample_{index:02d}.wav"
        temporary_wav = wav.with_suffix(".wav.partial")
        sf.write(temporary_wav, audio, TARGET_SR, subtype="PCM_16", format="WAV")
        temporary_wav.replace(wav)
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
        "model": str(final.resolve()),
        "source": str(SOURCE.resolve()),
        "source_hashes": source_hashes(SOURCE),
        "source_gate_wer": gate["wer"],
        "quantization": mode,
        "quantized_modules": len(names),
        "quantized_parameter_fraction": quantized_params / total_params,
        "vram_idle_gib": idle_vram,
        "vram_peak_gib": peak_vram,
        "rtf": float(np.mean(rtfs)),
        "wer": wer,
        "cer": float(np.mean([sample["cer"] for sample in samples])),
        "max_wer": MAX_WER,
        "passed": wer <= MAX_WER,
        "samples": samples,
    }
    atomic_json(output / "metrics.json", metrics)
    manifest = {
        "schema": "vibevoice-selective-bnb-v1",
        "status": "validated" if metrics["passed"] else "quality_failed",
        "source": str(SOURCE.resolve()),
        "source_hashes": source_hashes(SOURCE),
        "quantization": {
            "type": mode,
            "expected_modules": 196,
            "double_quant": mode == "nf4",
            "compute_dtype": str(model_dtype),
        },
        "artifacts": _artifact_inventory(partial),
        "validation": {
            "quantized_modules": len(names),
            "reload_verified": True,
            "quality": metrics,
        },
    }
    atomic_json(partial / "manifest.json", manifest)
    if not metrics["passed"]:
        print(
            f"{mode.upper()} candidate was not promoted: "
            f"WER {wer:.4f} > {MAX_WER:.4f}"
        )
        return
    if final.exists():
        shutil.rmtree(final)
    partial.replace(final)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
