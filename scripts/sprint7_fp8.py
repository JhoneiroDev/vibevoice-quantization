#!/usr/bin/env python3
# ruff: noqa: E402

# === Sprint 7.1: Conversion selectiva, validacion y persistencia FP8 ===
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

for _v in ["model", "model_fp8", "m_fp8", "wm", "out"]:
    if _v in globals():
        del globals()[_v]
gc.collect()
torch.cuda.empty_cache()

S7_ROOT = Path.cwd()
if not (S7_ROOT / "weights").exists():
    S7_ROOT = S7_ROOT.parent
for path in [S7_ROOT, S7_ROOT / "VibeVoice_repo"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import transformers
if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from scripts.fp8_vibevoice import (
    DynamicScaledFP8Linear,
    load_fp8_vibevoice,
    replace_qwen_linears_with_fp8,
)
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_ES_PATH = S7_ROOT / "weights" / "vibevoice-1.5b-es-corrected"
FP8_FINAL = S7_ROOT / "weights" / "vibevoice-1.5b-es-fp8"
FP8_OUT = FP8_FINAL.with_name(FP8_FINAL.name + ".partial")
SPRINT7_OUTPUT = S7_ROOT / "outputs" / "sprint7_fp8"
SPRINT7_OUTPUT.mkdir(parents=True, exist_ok=True)
TARGET_SR = 24000
FP8_RANGE_LIMIT = 240.0

assert torch.cuda.is_available(), "Sprint 7 requiere una GPU CUDA."
assert hasattr(torch, "_scaled_mm"), "Este PyTorch no expone torch._scaled_mm."
fp8_max = torch.finfo(torch.float8_e4m3fn).max
assert FP8_RANGE_LIMIT < fp8_max
print(f"GPU: {torch.cuda.get_device_name(0)} | FP8 max={fp8_max} | limite={FP8_RANGE_LIMIT}")

print("Cargando VibeVoice-ES en BF16 para conversion selectiva...")
model_fp8 = VibeVoiceForConditionalGenerationInference.from_pretrained(
    str(MODEL_ES_PATH), torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, attn_implementation="sdpa",
)
model_fp8.eval()
model_fp8.set_ddpm_inference_steps(num_steps=20)
total_params_fp8 = sum(param.numel() for param in model_fp8.parameters())
fp8_params = sum(
    module.weight.numel()
    for name, module in model_fp8.named_modules()
    if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)
)
converted_names_fp8 = replace_qwen_linears_with_fp8(
    model_fp8, range_limit=FP8_RANGE_LIMIT
)
fp8_modules = [
    module for module in model_fp8.modules()
    if isinstance(module, DynamicScaledFP8Linear)
]
assert len(converted_names_fp8) == len(fp8_modules) == 196
assert all(name.startswith("model.language_model.") for name in converted_names_fp8)
assert all(module.weight.dtype == torch.float8_e4m3fn for module in fp8_modules)
assert all(torch.isfinite(module.weight.float()).all() for module in fp8_modules)
assert all(module.weight.float().abs().max() <= FP8_RANGE_LIMIT for module in fp8_modules)
assert model_fp8.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
assert model_fp8.lm_head.weight is model_fp8.model.language_model.embed_tokens.weight
for protected_name in [
    "prediction_head", "acoustic_tokenizer", "semantic_tokenizer",
    "acoustic_connector", "semantic_connector",
]:
    protected = getattr(model_fp8.model, protected_name)
    assert not any(isinstance(module, DynamicScaledFP8Linear) for module in protected.modules())

fp8_fraction = fp8_params / total_params_fp8
vram_fp8_idle = torch.cuda.memory_allocated() / 1024**3
print(f"Capas DynamicScaledFP8Linear: {len(fp8_modules)}")
print(f"Parametros FP8             : {fp8_params:,} ({fp8_fraction:.1%})")
print(f"VRAM reposo FP8 hibrido   : {vram_fp8_idle:.2f} GB")

model_fp8.config.vibevoice_fp8_config = {
    "format": "e4m3fn", "range_limit": FP8_RANGE_LIMIT,
    "activation_scaling": "dynamic_per_tensor",
    "weight_scaling": "static_per_layer",
    "compute_kernel": "torch._scaled_mm",
    "output_dtype": "bfloat16",
    "loader": "scripts.fp8_vibevoice.load_fp8_vibevoice",
}
if FP8_OUT.exists():
    shutil.rmtree(FP8_OUT)
model_fp8.save_pretrained(str(FP8_OUT), safe_serialization=True, max_shard_size="4GB")
processor_fp8 = VibeVoiceProcessor.from_pretrained(str(MODEL_ES_PATH))
processor_fp8.save_pretrained(str(FP8_OUT))
disk_fp8_gb = sum(path.stat().st_size for path in FP8_OUT.rglob("*.safetensors")) / 1024**3
print(f"Modelo FP8 guardado: {FP8_OUT} ({disk_fp8_gb:.2f} GB)")

del fp8_modules, model_fp8
gc.collect()
torch.cuda.empty_cache()


# === Sprint 7.2: Recarga del checkpoint y benchmark TTS ===
m_fp8 = load_fp8_vibevoice(FP8_OUT, device_map={"": "cuda:0"})
m_fp8.set_ddpm_inference_steps(num_steps=20)
reloaded_fp8_names = [
    name for name, module in m_fp8.named_modules()
    if isinstance(module, DynamicScaledFP8Linear)
]
assert reloaded_fp8_names == converted_names_fp8
assert m_fp8.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
assert m_fp8.lm_head.weight is m_fp8.model.language_model.embed_tokens.weight
vram_fp8_idle = torch.cuda.memory_allocated() / 1024**3
print(f"Recarga FP8 verificada: {len(reloaded_fp8_names)} capas, {vram_fp8_idle:.2f} GB")
if FP8_FINAL.exists():
    shutil.rmtree(FP8_FINAL)
FP8_OUT.replace(FP8_FINAL)
FP8_OUT = FP8_FINAL
print(f"Artefacto FP8 promovido atomicamente: {FP8_FINAL}")

# En GPUs menores puede usarse device_map="auto" y max_memory. Esto ofrece
# offload por modulos de Accelerate, no descarga circular custom del KV cache.
test_texts_fp8 = [
    "Hola, buenos dias. Este es un modelo de sintesis de voz en espanol.",
    "La inteligencia artificial permite crear sistemas de voz cada vez mas naturales.",
    "El aprendizaje profundo ha revolucionado el procesamiento del lenguaje.",
    "Los modelos de lenguaje pueden generar texto y voz con gran precision.",
]
audios_fp8 = []
rtfs_fp8 = []
peak_vram_fp8 = vram_fp8_idle
for text in test_texts_fp8:
    inputs_fp8 = processor_fp8(
        text=[f"Speaker 1: {text}"], voice_samples=None, padding=True,
        return_tensors="pt", return_attention_mask=True,
    )
    inputs_fp8 = {
        key: (value.to(m_fp8.device) if torch.is_tensor(value) else value)
        for key, value in inputs_fp8.items() if value is not None
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = m_fp8.generate(
            **inputs_fp8, max_new_tokens=600, tokenizer=processor_fp8.tokenizer,
            generation_config={"do_sample": False}, verbose=False,
        )
    elapsed = time.perf_counter() - start
    peak_vram_fp8 = max(peak_vram_fp8, torch.cuda.max_memory_allocated() / 1024**3)
    if out.speech_outputs and out.speech_outputs[0] is not None:
        audio = out.speech_outputs[0].squeeze().cpu().float().numpy()
        duration = len(audio) / TARGET_SR
        rtfs_fp8.append(elapsed / duration)
        audios_fp8.append(audio)
        sf.write(
            SPRINT7_OUTPUT / f"sample_{len(audios_fp8):02d}.wav",
            audio, TARGET_SR, subtype="PCM_16",
        )
        print(f"Audio: {duration:.1f}s | RTF: {rtfs_fp8[-1]:.2f}")
    else:
        audios_fp8.append(None)
        print("ERROR: generacion sin audio")

del m_fp8, out, inputs_fp8, reloaded_fp8_names
gc.collect()
torch.cuda.empty_cache()
vram_before_asr_fp8 = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM tras liberar VibeVoice: {vram_before_asr_fp8:.2f} GB")
if vram_before_asr_fp8 > 0.5:
    print("ADVERTENCIA: quedan referencias CUDA activas antes de Whisper.")


# === Sprint 7.3: WER/CER con Whisper y reporte ===
whisper_cache = Path.home() / ".cache" / "whisper"
whisper_checkpoint = whisper_cache / "large-v3.pt"
if not whisper_checkpoint.exists():
    raise FileNotFoundError(
        f"Falta {whisper_checkpoint}. Descargue large-v3 por separado."
    )
print(f"Cargando Whisper large-v3 ({whisper_checkpoint.stat().st_size / 1024**3:.2f} GB)...")
asr_start = time.perf_counter()
wm = whisper.load_model("large-v3", device="cuda", download_root=str(whisper_cache))
print(f"Whisper cargado en {time.perf_counter() - asr_start:.1f}s")
wers_fp8 = []
cers_fp8 = []
for index, (text, audio) in enumerate(zip(test_texts_fp8, audios_fp8), start=1):
    if audio is None:
        wers_fp8.append(1.0)
        cers_fp8.append(1.0)
        continue
    print(f"Transcribiendo muestra {index}/{len(audios_fp8)}...", flush=True)
    audio_16k = librosa.resample(
        audio.astype(np.float32), orig_sr=TARGET_SR, target_sr=16000
    )
    transcript = wm.transcribe(
        audio_16k, language="es", fp16=True, verbose=False,
        condition_on_previous_text=False,
    )["text"].strip()
    wers_fp8.append(jiwer.wer(text.lower(), transcript.lower()))
    cers_fp8.append(jiwer.cer(text.lower(), transcript.lower()))
    print(f"[{index}] WER={wers_fp8[-1]:.3f} CER={cers_fp8[-1]:.3f} | {transcript[:80]}")

del wm
gc.collect()
torch.cuda.empty_cache()
rtf_fp8 = float(np.mean(rtfs_fp8)) if rtfs_fp8 else float("inf")
wer_fp8 = float(np.mean(wers_fp8)) if wers_fp8 else 1.0
cer_fp8 = float(np.mean(cers_fp8)) if cers_fp8 else 1.0
metrics_fp8 = {
    "model": "vibevoice-1.5b-es-fp8-dynamic",
    "format": "float8_e4m3fn",
    "range_limit": FP8_RANGE_LIMIT,
    "fp8_layers": len(converted_names_fp8),
    "fp8_parameter_fraction": fp8_fraction,
    "disk_gb": disk_fp8_gb,
    "vram_idle_gb": vram_fp8_idle,
    "vram_peak_gb": peak_vram_fp8,
    "rtf": rtf_fp8, "wer": wer_fp8, "cer": cer_fp8,
}
metrics_path = SPRINT7_OUTPUT / "metrics.json"
metrics_temporary = metrics_path.with_suffix(".json.tmp")
with open(metrics_temporary, "w", encoding="utf-8") as file:
    json.dump(metrics_fp8, file, indent=2)
metrics_temporary.replace(metrics_path)
print(
    f"\n=== FP8 DINAMICO === VRAM idle: {vram_fp8_idle:.2f} GB | "
    f"VRAM peak: {peak_vram_fp8:.2f} GB | RTF: {rtf_fp8:.4f} | "
    f"WER: {wer_fp8:.4f} | CER: {cer_fp8:.4f}"
)
