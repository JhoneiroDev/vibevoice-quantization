#!/usr/bin/env python3
# ruff: noqa: E402

# === Sprint 5: INT8 hibrido selectivo (Fabio Sarracino) ===
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import librosa
import jiwer
import numpy as np
import soundfile as sf
import torch
import whisper

for _v in ["model", "model_int8", "m_int8", "wm", "out"]:
    if _v in globals():
        del globals()[_v]
gc.collect()
torch.cuda.empty_cache()

repo_path = os.path.abspath("../VibeVoice_repo")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Parche obligatorio antes de importar las clases VibeVoice.
import transformers
if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from bitsandbytes.nn import Linear8bitLt
from transformers import BitsAndBytesConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_ES_PATH = Path("../weights/vibevoice-1.5b-es-corrected")
INT8_FINAL = Path("../weights/vibevoice-1.5b-es-int8")
INT8_OUT = INT8_FINAL.with_name(INT8_FINAL.name + ".partial")
SPRINT5_OUTPUT = Path("../outputs/sprint5_int8")
SPRINT5_OUTPUT.mkdir(parents=True, exist_ok=True)
TARGET_SR = 24000

# En el fork actual diffusion_head se instancia como model.prediction_head.
SENSITIVE_MODULES = [
    "prediction_head",
    "acoustic_tokenizer",
    "semantic_tokenizer",
    "acoustic_connector",
    "semantic_connector",
    "lm_head",
]
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_skip_modules=SENSITIVE_MODULES,
)

print("Cargando VibeVoice-ES con INT8 selectivo sobre Qwen2...")
model_int8 = VibeVoiceForConditionalGenerationInference.from_pretrained(
    str(MODEL_ES_PATH),
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
)
model_int8.eval()
model_int8.set_ddpm_inference_steps(num_steps=20)

# Verificacion estructural: solo el language_model puede contener Linear8bitLt.
quantized_linears = [
    name for name, module in model_int8.named_modules()
    if isinstance(module, Linear8bitLt)
]
sensitive_quantized = [
    name for name in quantized_linears
    if any(part in name for part in SENSITIVE_MODULES)
]
unexpected_quantized = [
    name for name in quantized_linears
    if not name.startswith("model.language_model.")
]
assert quantized_linears, "No se encontro ninguna capa Linear8bitLt; revisar bitsandbytes/CUDA."
assert not sensitive_quantized, f"Modulos sensibles cuantizados: {sensitive_quantized[:5]}"
assert not unexpected_quantized, f"Capas fuera de Qwen2 cuantizadas: {unexpected_quantized[:5]}"
assert model_int8.lm_head.weight is model_int8.model.language_model.embed_tokens.weight, (
    "lm_head y embed_tokens dejaron de compartir pesos."
)

quantized_weight_ids = {
    id(module.weight) for module in model_int8.modules()
    if isinstance(module, Linear8bitLt)
}
total_params = sum(param.numel() for param in model_int8.parameters())
int8_params = sum(
    param.numel() for param in model_int8.parameters()
    if id(param) in quantized_weight_ids
)
int8_fraction = int8_params / total_params
vram_int8_idle = torch.cuda.memory_allocated() / 1024**3
print(f"Capas Linear8bitLt       : {len(quantized_linears)}")
print(f"Parametros INT8          : {int8_params:,} ({int8_fraction:.1%})")
print(f"VRAM reposo INT8 hibrido: {vram_int8_idle:.2f} GB")
print("Audio modules         : FP16 (verificacion OK)")

# Guardar el modelo bnb directamente. No convertir el state_dict a FP16.
if INT8_OUT.exists():
    shutil.rmtree(INT8_OUT)
model_int8.save_pretrained(str(INT8_OUT), safe_serialization=True)
processor = VibeVoiceProcessor.from_pretrained(str(MODEL_ES_PATH))
processor.save_pretrained(str(INT8_OUT))
disk_gb = sum(path.stat().st_size for path in INT8_OUT.rglob("*.safetensors")) / 1024**3
print(f"Modelo INT8 guardado    : {INT8_OUT} ({disk_gb:.2f} GB en safetensors)")

# Recargar desde disco para validar que el artefacto conserva la cuantizacion.
del model_int8
gc.collect()
torch.cuda.empty_cache()
m_int8 = VibeVoiceForConditionalGenerationInference.from_pretrained(
    str(INT8_OUT),
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
)
m_int8.eval()
m_int8.set_ddpm_inference_steps(num_steps=20)
reloaded_quantized_names = [
    name for name, module in m_int8.named_modules()
    if isinstance(module, Linear8bitLt)
]
assert len(reloaded_quantized_names) == len(quantized_linears), (
    f"La recarga cambio las capas INT8: {len(quantized_linears)} -> {len(reloaded_quantized_names)}"
)
vram_int8_idle = torch.cuda.memory_allocated() / 1024**3
print(f"Recarga INT8 verificada : {len(reloaded_quantized_names)} capas, {vram_int8_idle:.2f} GB")
if INT8_FINAL.exists():
    shutil.rmtree(INT8_FINAL)
INT8_OUT.replace(INT8_FINAL)
INT8_OUT = INT8_FINAL
print(f"Artefacto INT8 promovido atomicamente: {INT8_FINAL}")

# Smoke benchmark de 4 frases. El benchmark estadistico de 200 muestras es Sprint 8.
test_texts = [
    "Hola, buenos dias. Este es un modelo de sintesis de voz en espanol.",
    "La inteligencia artificial permite crear sistemas de voz cada vez mas naturales.",
    "El aprendizaje profundo ha revolucionado el procesamiento del lenguaje.",
    "Los modelos de lenguaje pueden generar texto y voz con gran precision.",
]
audios = []
rtfs = []
peak_vram = vram_int8_idle
for text in test_texts:
    inputs = processor(
        text=[f"Speaker 1: {text}"],
        voice_samples=None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        key: (value.to(m_int8.device) if torch.is_tensor(value) else value)
        for key, value in inputs.items() if value is not None
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = m_int8.generate(
            **inputs,
            max_new_tokens=600,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )
    elapsed = time.perf_counter() - start
    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)
    if out.speech_outputs and out.speech_outputs[0] is not None:
        audio = out.speech_outputs[0].squeeze().cpu().float().numpy()
        duration = len(audio) / TARGET_SR
        rtfs.append(elapsed / duration)
        audios.append(audio)
        audio_path = SPRINT5_OUTPUT / f"sample_{len(audios):02d}.wav"
        sf.write(audio_path, audio, TARGET_SR, subtype="PCM_16")
        print(f"Audio: {duration:.1f}s | RTF: {rtfs[-1]:.2f}")
    else:
        audios.append(None)
        print("ERROR: generacion sin audio")

# Liberar todas las referencias CUDA antes de cargar Whisper. No conservar listas
# de modulos: cada objeto Linear8bitLt mantiene vivo su peso INT8.
del m_int8, out, inputs, reloaded_quantized_names
gc.collect()
torch.cuda.empty_cache()
vram_before_asr = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM tras liberar VibeVoice: {vram_before_asr:.2f} GB")
if vram_before_asr > 0.5:
    print("ADVERTENCIA: quedan referencias CUDA activas antes de Whisper.")

whisper_cache = Path.home() / ".cache" / "whisper"
whisper_checkpoint = whisper_cache / "large-v3.pt"
if not whisper_checkpoint.exists():
    raise FileNotFoundError(
        f"Falta {whisper_checkpoint}. Descargue large-v3 por separado para evitar "
        "que el benchmark quede esperando una descarga sin progreso visible."
    )
print(f"Cargando Whisper large-v3 desde cache ({whisper_checkpoint.stat().st_size / 1024**3:.2f} GB)...")
asr_load_start = time.perf_counter()
wm = whisper.load_model("large-v3", device="cuda", download_root=str(whisper_cache))
print(f"Whisper cargado en {time.perf_counter() - asr_load_start:.1f}s")
wers = []
cers = []
for index, (text, audio) in enumerate(zip(test_texts, audios), start=1):
    if audio is None:
        wers.append(1.0)
        cers.append(1.0)
        continue
    print(f"Transcribiendo muestra {index}/{len(audios)}...", flush=True)
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=TARGET_SR, target_sr=16000)
    transcript = wm.transcribe(
        audio_16k, language="es", fp16=True, verbose=False,
        condition_on_previous_text=False,
    )["text"].strip()
    wers.append(jiwer.wer(text.lower(), transcript.lower()))
    cers.append(jiwer.cer(text.lower(), transcript.lower()))
    print(f"[{index}] WER={wers[-1]:.3f} CER={cers[-1]:.3f} | {transcript[:80]}")

del wm
gc.collect()
torch.cuda.empty_cache()
rtf_int8 = float(np.mean(rtfs)) if rtfs else float("inf")
wer_int8 = float(np.mean(wers)) if wers else 1.0
cer_int8 = float(np.mean(cers)) if cers else 1.0
metrics_int8 = {
    "model": "vibevoice-1.5b-es-int8-selective",
    "int8_layers": len(quantized_linears),
    "int8_parameter_fraction": int8_fraction,
    "disk_gb": disk_gb,
    "vram_idle_gb": vram_int8_idle,
    "vram_peak_gb": peak_vram,
    "rtf": rtf_int8,
    "wer": wer_int8,
    "cer": cer_int8,
}
metrics_path = SPRINT5_OUTPUT / "metrics.json"
metrics_temporary = metrics_path.with_suffix(".json.tmp")
with open(metrics_temporary, "w", encoding="utf-8") as file:
    json.dump(metrics_int8, file, indent=2)
metrics_temporary.replace(metrics_path)
print(
    f"\n=== INT8 HIBRIDO === VRAM idle: {vram_int8_idle:.2f} GB | "
    f"VRAM peak: {peak_vram:.2f} GB | RTF: {rtf_int8:.4f} | "
    f"WER: {wer_int8:.4f} | CER: {cer_int8:.4f}"
)
