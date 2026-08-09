#!/usr/bin/env python3
# ruff: noqa: E402
"""Sprint 7: selective SmoothQuant W8A8 on the Qwen decoder."""

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
import whisper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "VibeVoice_repo") not in sys.path:
    sys.path.insert(0, str(ROOT / "VibeVoice_repo"))

from scripts.quantization_common import (  # noqa: E402
    atomic_json,
    expected_qwen_weights,
    sha256,
    source_hashes,
    validate_canonical_source,
)
from scripts.smoothquant_vibevoice import (  # noqa: E402
    ALPHA,
    SCHEMA,
    TARGETS,
    SmoothQuantLinear,
    collect_activation_scales,
    load_smoothquant_vibevoice,
    replace_qwen_linears_with_smoothquant,
)
try:  # noqa: E402
    from validate_vibevoice_es_quality import TEXTS, normalize
except ModuleNotFoundError:  # noqa: E402
    from scripts.validate_vibevoice_es_quality import TEXTS, normalize
from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: E402
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: E402


MODEL = ROOT / "weights" / "vibevoice-1.5b-es"
FINAL = ROOT / "weights" / "vibevoice-1.5b-es-smoothquant"
PARTIAL = FINAL.with_name(FINAL.name + ".partial")
OUTPUT = ROOT / "outputs" / "sprint7_smoothquant"
VOICE = ROOT / "VibeVoice_repo" / "demo" / "voices" / "en-Alice_woman.wav"
CALIBRATION_METADATA = ROOT / "data" / "calibration_metadata.json"
TARGET_SR = 24000
MAX_WER = 0.30
CALIBRATION_MODE = "tts_prefill_inputs_embeds"


def capture_tts_prefills(model, processor, records: list[dict], device: str):
    if not VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {VOICE}")
    selected = records[::2]
    if len(selected) != 256:
        raise ValueError(f"SmoothQuant requires 256 TTS-prefill samples, found {len(selected)}")
    voice = processor.audio_processor._load_audio_from_path(str(VOICE))
    cached_speech_embeds = None
    expected_speech_positions = None
    prefills = []
    with torch.inference_mode():
        for index, record in enumerate(selected):
            inputs = processor(
                text=[f"Speaker 1: {record['sentence'].strip()}"],
                voice_samples=[[voice]], padding=True, return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = inputs["input_ids"].to(device)
            speech_mask = inputs["speech_input_mask"].to(device)
            embeds = model.get_input_embeddings()(input_ids)
            speech_positions = int(speech_mask.sum().item())
            if cached_speech_embeds is None:
                torch.manual_seed(42)
                _, cached_speech_embeds = model._process_speech_inputs(
                    inputs["speech_tensors"].to(device, dtype=torch.bfloat16),
                    inputs["speech_masks"],
                )
                cached_speech_embeds = cached_speech_embeds.detach()
                expected_speech_positions = speech_positions
            if speech_positions != expected_speech_positions:
                raise ValueError(f"Voice prompt length changed at sample {index}")
            embeds[speech_mask] = cached_speech_embeds
            mask = inputs["attention_mask"].to(device)
            if embeds.ndim != 3 or embeds.shape[-1] != 1536:
                raise ValueError(f"Invalid TTS prefill shape at sample {index}: {embeds.shape}")
            prefills.append((embeds.cpu().contiguous(), mask.cpu().contiguous()))
    return prefills


def artifact_inventory(path: Path) -> dict:
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(file for file in path.rglob("*") if file.is_file() and file.name != "manifest.json")
    }


def validate_model(model) -> list[str]:
    modules = {name: module for name, module in model.named_modules() if isinstance(module, SmoothQuantLinear)}
    if set(modules) != TARGETS:
        raise AssertionError("SmoothQuant reload did not preserve exactly 196 target layers")
    expected_shapes = {
        name.removesuffix(".weight"): shape for name, shape in expected_qwen_weights().items()
    }
    if any(
        module.qweight.dtype != torch.int8
        or (module.out_features, module.in_features) != expected_shapes[name]
        or module.qweight.shape != (module.out_features, module.in_features)
        or module.input_scale.shape != (module.in_features,)
        or module.weight_scale.shape != (module.out_features,)
        or (module.input_scale <= 0).any()
        or (module.weight_scale <= 0).any()
        or not torch.isfinite(module.input_scale).all()
        or not torch.isfinite(module.weight_scale).all()
        for name, module in modules.items()
    ):
        raise AssertionError("SmoothQuant artifact contains invalid scales or weights")
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise AssertionError("lm_head and embed_tokens are not tied")
    protected = (
        model.model.prediction_head, model.model.acoustic_tokenizer,
        model.model.semantic_tokenizer, model.model.acoustic_connector,
        model.model.semantic_connector, model.model.language_model.embed_tokens,
    )
    for component in protected:
        for parameter in component.parameters():
            if parameter.is_floating_point() and parameter.dtype not in {torch.float16, torch.bfloat16}:
                raise AssertionError(f"Unsupported protected dtype: {parameter.dtype}")
    return sorted(modules)


def main() -> None:
    validate_canonical_source(MODEL)
    if not torch.cuda.is_available():
        raise RuntimeError("SmoothQuant W8A8 requires CUDA")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("This PyTorch does not expose torch._int_mm")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | kernel: torch._int_mm")
    if FINAL.exists():
        raise FileExistsError(f"Existing SmoothQuant artifact will not be overwritten: {FINAL}")
    if PARTIAL.exists():
        raise FileExistsError(
            f"Existing SmoothQuant candidate requires manual review or removal: {PARTIAL}"
        )

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    processor = VibeVoiceProcessor.from_pretrained(MODEL, local_files_only=True)
    calibration = json.loads(CALIBRATION_METADATA.read_text(encoding="utf-8"))
    records = calibration.get("records", [])
    if calibration.get("split_origin") != "validation":
        raise ValueError("SmoothQuant calibration must use the reserved validation split")
    calibration_texts = [record.get("sentence", "").strip() for record in records]
    if calibration.get("n_samples") != 512 or len(records) != 512:
        raise ValueError(f"SmoothQuant requires exactly 512 calibration records, found {len(records)}")
    if any(not text for text in calibration_texts) or len(set(calibration_texts)) != 512:
        raise ValueError("SmoothQuant calibration contains empty or duplicate transcriptions")
    calibration_info = {
        "mode": CALIBRATION_MODE,
        "source_records": len(records),
        "tts_samples": len(records[::2]),
        "source_indices": [record["local_idx"] for record in records[::2]],
        "metadata_sha256": sha256(CALIBRATION_METADATA),
        "voice": str(VOICE.resolve()),
        "voice_sha256": sha256(VOICE),
        "alpha": ALPHA,
    }
    PARTIAL.mkdir(parents=True, exist_ok=False)
    atomic_json(PARTIAL / "calibration.json", calibration_info)
    prefills = capture_tts_prefills(model, processor, records, "cuda:0")
    activation_scales = collect_activation_scales(
        model, prefills, "cuda:0"
    )
    source_modules = {
        name: module for name, module in model.named_modules() if name in TARGETS
    }
    quantized_params = sum(module.weight.numel() for module in source_modules.values())
    protected_params = sum(parameter.numel() for parameter in model.parameters()) - quantized_params
    replace_qwen_linears_with_smoothquant(model, activation_scales)
    modules = validate_model(model)
    model.config.vibevoice_smoothquant_config = {
        "format": "W8A8",
        "method": "SmoothQuant",
        "alpha": ALPHA,
        "activation_quantization": "dynamic_per_token",
        "weight_quantization": "per_output_channel",
        "compute_kernel": "torch._int_mm",
        "output_dtype": "bfloat16",
        "protected_modules": [
            "embed_tokens", "lm_head", "prediction_head", "acoustic_tokenizer",
            "semantic_tokenizer", "acoustic_connector", "semantic_connector",
        ],
    }
    model.save_pretrained(PARTIAL, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(PARTIAL)
    # Keep the Qwen identifier so VibeVoiceProcessor recognizes the local tokenizer.
    shutil.copy2(MODEL / "preprocessor_config.json", PARTIAL / "preprocessor_config.json")
    del model, source_modules, activation_scales, prefills, processor
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = load_smoothquant_vibevoice(PARTIAL, device_map={"": "cuda:0"})
    reloaded.set_ddpm_inference_steps(num_steps=20)
    if validate_model(reloaded) != modules:
        raise AssertionError("SmoothQuant target order changed after reload")
    processor = VibeVoiceProcessor.from_pretrained(PARTIAL, local_files_only=True)
    idle_vram = torch.cuda.memory_allocated() / 1024**3
    peak_vram = idle_vram
    rtfs = []
    wav_paths = []
    for index, text in enumerate(TEXTS, start=1):
        torch.manual_seed(42 + index - 1)
        inputs = processor(
            text=[f"Speaker 1: {text}"], voice_samples=[[str(VOICE)]],
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        inputs = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                  for key, value in inputs.items() if value is not None}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = reloaded.generate(
                **inputs, max_new_tokens=600, tokenizer=processor.tokenizer,
                generation_config={"do_sample": False}, verbose=False, is_prefill=True,
            )
        elapsed = time.perf_counter() - started
        if not generated.speech_outputs or generated.speech_outputs[0] is None:
            raise RuntimeError(f"No SmoothQuant audio for sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite SmoothQuant audio for sample {index}")
        duration = len(audio) / TARGET_SR
        rtfs.append(elapsed / duration)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)
        wav = OUTPUT / f"sample_{index:02d}.wav"
        sf.write(
            wav.with_suffix(".wav.partial"), audio, TARGET_SR,
            subtype="PCM_16", format="WAV",
        )
        wav.with_suffix(".wav.partial").replace(wav)
        wav_paths.append(wav)

    del reloaded, generated, inputs, processor
    gc.collect()
    torch.cuda.empty_cache()
    asr = whisper.load_model("large-v3", device="cpu", download_root=str(Path.home() / ".cache" / "whisper"))
    samples = []
    for text, wav in zip(TEXTS, wav_paths):
        audio, _ = librosa.load(wav, sr=16000, mono=True)
        transcript = asr.transcribe(
            audio, language="es", fp16=False, verbose=False,
            condition_on_previous_text=False, temperature=0.0, beam_size=1, sample_len=64,
        )["text"].strip()
        samples.append({
            "reference": text, "transcript": transcript,
            "wer": float(jiwer.wer(normalize(text), normalize(transcript))),
            "cer": float(jiwer.cer(normalize(text), normalize(transcript))),
        })
    wer = float(np.mean([sample["wer"] for sample in samples]))
    disk_gib = sum(path.stat().st_size for path in PARTIAL.rglob("*.safetensors")) / 1024**3
    metrics = {
        "model": str(FINAL.resolve()), "source": str(MODEL.resolve()),
        "source_hashes": source_hashes(MODEL), "method": "SmoothQuant W8A8",
        "voice": str(VOICE.resolve()),
        "protocol": "six-texts-en-Alice-large-v3-WER-0.30-v1",
        "alpha": ALPHA, "quantized_modules": len(modules),
        "quantized_parameter_fraction": quantized_params / (quantized_params + protected_params),
        "disk_gib": disk_gib, "vram_idle_gib": idle_vram,
        "vram_peak_gib": peak_vram, "rtf": float(np.mean(rtfs)), "wer": wer,
        "cer": float(np.mean([sample["cer"] for sample in samples])),
        "max_wer": MAX_WER, "passed": wer <= MAX_WER, "samples": samples,
    }
    atomic_json(OUTPUT / "metrics.json", metrics)
    manifest = {
        "schema": SCHEMA, "status": "validated" if metrics["passed"] else "quality_failed",
        "source": str(MODEL.resolve()), "source_hashes": source_hashes(MODEL),
        "smoothquant": {"alpha": ALPHA, "expected_modules": 196, "format": "W8A8"},
        "calibration": {
            "dataset_id": calibration["dataset_id"],
            "dataset_config": calibration["dataset_config"],
            "split_origin": calibration["split_origin"],
            "records": len(calibration_texts),
            **calibration_info,
        },
        "artifacts": artifact_inventory(PARTIAL), "validation": metrics,
    }
    atomic_json(PARTIAL / "manifest.json", manifest)
    if not metrics["passed"]:
        print(f"SmoothQuant no fue promovido: WER={wer:.4f} > {MAX_WER:.4f}")
        return
    PARTIAL.replace(FINAL)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
