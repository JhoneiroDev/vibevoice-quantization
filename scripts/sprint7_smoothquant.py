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
OUTPUT.mkdir(parents=True, exist_ok=True)
TARGET_SR = 24000
MAX_WER = 0.30


def artifact_inventory(path: Path) -> dict:
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(file for file in path.rglob("*") if file.is_file() and file.name != "manifest.json")
    }


def validate_model(model) -> list[str]:
    modules = {name: module for name, module in model.named_modules() if isinstance(module, SmoothQuantLinear)}
    if set(modules) != TARGETS:
        raise AssertionError("SmoothQuant reload did not preserve exactly 196 target layers")
    if any(
        module.qweight.dtype != torch.int8
        or not torch.isfinite(module.input_scale).all()
        or not torch.isfinite(module.weight_scale).all()
        for module in modules.values()
    ):
        raise AssertionError("SmoothQuant artifact contains invalid scales or weights")
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise AssertionError("lm_head and embed_tokens are not tied")
    return sorted(modules)


def main() -> None:
    validate_canonical_source(MODEL)
    if not torch.cuda.is_available():
        raise RuntimeError("SmoothQuant W8A8 requires CUDA")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("This PyTorch does not expose torch._int_mm")
    print(f"GPU: {torch.cuda.get_device_name(0)} | kernel: torch._int_mm")
    if PARTIAL.exists():
        shutil.rmtree(PARTIAL)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    processor = VibeVoiceProcessor.from_pretrained(MODEL, local_files_only=True)
    calibration = json.loads(CALIBRATION_METADATA.read_text(encoding="utf-8"))
    if calibration.get("split_origin") != "validation":
        raise ValueError("SmoothQuant calibration must use the reserved validation split")
    calibration_texts = [record["sentence"].strip() for record in calibration["records"]]
    if len(calibration_texts) != 512:
        raise ValueError(f"SmoothQuant requires 512 calibration records, found {len(calibration_texts)}")
    activation_scales = collect_activation_scales(
        model, processor.tokenizer, calibration_texts, "cuda:0"
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
    del model, source_modules, activation_scales, processor
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
        },
        "artifacts": artifact_inventory(PARTIAL), "validation": metrics,
    }
    atomic_json(PARTIAL / "manifest.json", manifest)
    if not metrics["passed"]:
        print(f"SmoothQuant no fue promovido: WER={wer:.4f} > {MAX_WER:.4f}")
        return
    if FINAL.exists():
        shutil.rmtree(FINAL)
    PARTIAL.replace(FINAL)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
