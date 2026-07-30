#!/usr/bin/env python3
# ruff: noqa: E402

# === Sprint 6: NF4 hibrido + Double Quant ===
import gc
import json
import os
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

for _v in ["model", "model_nf4", "m_nf4", "wm", "out"]:
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

from bitsandbytes.nn import Linear4bit
from transformers import BitsAndBytesConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_ES_PATH = Path("../weights/vibevoice-1.5b-es-corrected")
NF4_FINAL = Path("../weights/vibevoice-1.5b-es-nf4")
NF4_OUT = NF4_FINAL.with_name(NF4_FINAL.name + ".partial")
SPRINT6_OUTPUT = Path("../outputs/sprint6_nf4")
SPRINT6_OUTPUT.mkdir(parents=True, exist_ok=True)
TARGET_SR = 24000

# embed_tokens es nn.Embedding (bnb no lo cuantiza), pero se declara para dejar
# explicita la proteccion. lm_head comparte ese mismo peso.
SENSITIVE_MODULES_NF4 = [
    "embed_tokens",
    "lm_head",
    "prediction_head",
    "acoustic_tokenizer",
    "semantic_tokenizer",
    "acoustic_connector",
    "semantic_connector",
]
bnb_nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_skip_modules=SENSITIVE_MODULES_NF4,
)

print("Cargando VibeVoice-ES con NF4 selectivo sobre Qwen2...")
model_nf4 = VibeVoiceForConditionalGenerationInference.from_pretrained(
    str(MODEL_ES_PATH),
    quantization_config=bnb_nf4_config,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
)
model_nf4.eval()
model_nf4.set_ddpm_inference_steps(num_steps=20)

# Verificar ubicacion, formato, doble cuantizacion y compute dtype.
quantized_linears_nf4 = [
    name for name, module in model_nf4.named_modules()
    if isinstance(module, Linear4bit)
]
sensitive_quantized_nf4 = [
    name for name in quantized_linears_nf4
    if any(part in name for part in SENSITIVE_MODULES_NF4)
]
unexpected_quantized_nf4 = [
    name for name in quantized_linears_nf4
    if not name.startswith("model.language_model.")
]
nf4_modules = [
    module for module in model_nf4.modules()
    if isinstance(module, Linear4bit)
]
assert quantized_linears_nf4, "No se encontro ninguna capa Linear4bit."
assert not sensitive_quantized_nf4, f"Modulos sensibles cuantizados: {sensitive_quantized_nf4[:5]}"
assert not unexpected_quantized_nf4, f"Capas fuera de Qwen2 cuantizadas: {unexpected_quantized_nf4[:5]}"
assert all(module.compute_dtype == torch.bfloat16 for module in nf4_modules), (
    "Hay capas NF4 cuyo compute dtype no es BF16."
)
assert all(module.weight.quant_state is not None for module in nf4_modules), (
    "Hay capas Linear4bit sin estado de cuantizacion."
)
double_quant_layers = sum(
    bool(getattr(module.weight.quant_state, "nested", False))
    for module in nf4_modules
)
assert double_quant_layers == len(nf4_modules), (
    f"Double Quant solo esta activo en {double_quant_layers}/{len(nf4_modules)} capas."
)
assert isinstance(model_nf4.model.language_model.embed_tokens, torch.nn.Embedding)
assert model_nf4.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
assert isinstance(model_nf4.lm_head, torch.nn.Linear)
assert model_nf4.lm_head.weight is model_nf4.model.language_model.embed_tokens.weight, (
    "lm_head y embed_tokens dejaron de compartir pesos."
)

quantized_weight_ids_nf4 = {id(module.weight) for module in nf4_modules}
total_params_nf4 = sum(param.numel() for param in model_nf4.parameters())
nf4_params = sum(
    param.numel() for param in model_nf4.parameters()
    if id(param) in quantized_weight_ids_nf4
)
nf4_fraction = nf4_params / total_params_nf4
vram_nf4_idle = torch.cuda.memory_allocated() / 1024**3
print(f"Capas Linear4bit NF4    : {len(quantized_linears_nf4)}")
print(f"Capas con Double Quant : {double_quant_layers}")
print(f"Parametros NF4         : {nf4_params:,} ({nf4_fraction:.1%})")
print("embed_tokens / lm_head : BF16 tied (verificacion OK)")
print("Audio modules          : BF16 (verificacion OK)")
print(f"VRAM reposo NF4 hibrido: {vram_nf4_idle:.2f} GB")

# Guardado nativo: conservar Params4bit y quant_state; nunca castear a FP16.
if NF4_OUT.exists():
    shutil.rmtree(NF4_OUT)
model_nf4.save_pretrained(str(NF4_OUT), safe_serialization=True)
processor_nf4 = VibeVoiceProcessor.from_pretrained(str(MODEL_ES_PATH))
processor_nf4.save_pretrained(str(NF4_OUT))
disk_nf4_gb = sum(path.stat().st_size for path in NF4_OUT.rglob("*.safetensors")) / 1024**3
print(f"Modelo NF4 guardado     : {NF4_OUT} ({disk_nf4_gb:.2f} GB en safetensors)")

# nf4_modules contiene objetos CUDA; eliminarlo antes de liberar el modelo.
del nf4_modules, model_nf4
gc.collect()
torch.cuda.empty_cache()
m_nf4 = VibeVoiceForConditionalGenerationInference.from_pretrained(
    str(NF4_OUT),
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
)
m_nf4.eval()
m_nf4.set_ddpm_inference_steps(num_steps=20)
reloaded_nf4_names = [
    name for name, module in m_nf4.named_modules()
    if isinstance(module, Linear4bit)
]
reloaded_double_quant = sum(
    bool(getattr(module.weight.quant_state, "nested", False))
    for module in m_nf4.modules() if isinstance(module, Linear4bit)
)
assert len(reloaded_nf4_names) == len(quantized_linears_nf4)
assert reloaded_double_quant == len(reloaded_nf4_names)
assert m_nf4.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
assert m_nf4.lm_head.weight is m_nf4.model.language_model.embed_tokens.weight
vram_nf4_idle = torch.cuda.memory_allocated() / 1024**3
print(
    f"Recarga NF4 verificada  : {len(reloaded_nf4_names)} capas, "
    f"Double Quant={reloaded_double_quant}, {vram_nf4_idle:.2f} GB"
)
if NF4_FINAL.exists():
    shutil.rmtree(NF4_FINAL)
NF4_OUT.replace(NF4_FINAL)
NF4_OUT = NF4_FINAL
print(f"Artefacto NF4 promovido atomicamente: {NF4_FINAL}")

# Smoke benchmark de 4 frases. Sprint 8 realizara la evaluacion estadistica.
test_texts_nf4 = [
    "Hola, buenos dias. Este es un modelo de sintesis de voz en espanol.",
    "La inteligencia artificial permite crear sistemas de voz cada vez mas naturales.",
    "El aprendizaje profundo ha revolucionado el procesamiento del lenguaje.",
    "Los modelos de lenguaje pueden generar texto y voz con gran precision.",
]
audios_nf4 = []
rtfs_nf4 = []
peak_vram_nf4 = vram_nf4_idle
for text in test_texts_nf4:
    inputs_nf4 = processor_nf4(
        text=[f"Speaker 1: {text}"],
        voice_samples=None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs_nf4 = {
        key: (value.to(m_nf4.device) if torch.is_tensor(value) else value)
        for key, value in inputs_nf4.items() if value is not None
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = m_nf4.generate(
            **inputs_nf4,
            max_new_tokens=600,
            tokenizer=processor_nf4.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )
    elapsed = time.perf_counter() - start
    peak_vram_nf4 = max(peak_vram_nf4, torch.cuda.max_memory_allocated() / 1024**3)
    if out.speech_outputs and out.speech_outputs[0] is not None:
        audio = out.speech_outputs[0].squeeze().cpu().float().numpy()
        duration = len(audio) / TARGET_SR
        rtfs_nf4.append(elapsed / duration)
        audios_nf4.append(audio)
        audio_path = SPRINT6_OUTPUT / f"sample_{len(audios_nf4):02d}.wav"
        sf.write(audio_path, audio, TARGET_SR, subtype="PCM_16")
        print(f"Audio: {duration:.1f}s | RTF: {rtfs_nf4[-1]:.2f}")
    else:
        audios_nf4.append(None)
        print("ERROR: generacion sin audio")

# Liberar VibeVoice y cualquier referencia CUDA antes de cargar Whisper large-v3.
del m_nf4, out, inputs_nf4, reloaded_nf4_names
gc.collect()
torch.cuda.empty_cache()
vram_before_asr_nf4 = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM tras liberar VibeVoice: {vram_before_asr_nf4:.2f} GB")
if vram_before_asr_nf4 > 0.5:
    print("ADVERTENCIA: quedan referencias CUDA activas antes de Whisper.")

whisper_cache = Path.home() / ".cache" / "whisper"
whisper_checkpoint = whisper_cache / "large-v3.pt"
if not whisper_checkpoint.exists():
    raise FileNotFoundError(
        f"Falta {whisper_checkpoint}. Descargue large-v3 por separado para evitar "
        "una descarga sin progreso visible durante el benchmark."
    )
print(f"Cargando Whisper large-v3 desde cache ({whisper_checkpoint.stat().st_size / 1024**3:.2f} GB)...")
asr_load_start = time.perf_counter()
wm = whisper.load_model("large-v3", device="cuda", download_root=str(whisper_cache))
print(f"Whisper cargado en {time.perf_counter() - asr_load_start:.1f}s")
wers_nf4 = []
cers_nf4 = []
for index, (text, audio) in enumerate(zip(test_texts_nf4, audios_nf4), start=1):
    if audio is None:
        wers_nf4.append(1.0)
        cers_nf4.append(1.0)
        continue
    print(f"Transcribiendo muestra {index}/{len(audios_nf4)}...", flush=True)
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=TARGET_SR, target_sr=16000)
    transcript = wm.transcribe(
        audio_16k, language="es", fp16=True, verbose=False,
        condition_on_previous_text=False,
    )["text"].strip()
    wers_nf4.append(jiwer.wer(text.lower(), transcript.lower()))
    cers_nf4.append(jiwer.cer(text.lower(), transcript.lower()))
    print(f"[{index}] WER={wers_nf4[-1]:.3f} CER={cers_nf4[-1]:.3f} | {transcript[:80]}")

del wm
gc.collect()
torch.cuda.empty_cache()
rtf_nf4 = float(np.mean(rtfs_nf4)) if rtfs_nf4 else float("inf")
wer_nf4 = float(np.mean(wers_nf4)) if wers_nf4 else 1.0
cer_nf4 = float(np.mean(cers_nf4)) if cers_nf4 else 1.0
metrics_nf4 = {
    "model": "vibevoice-1.5b-es-nf4-selective",
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "bfloat16",
    "nf4_layers": len(quantized_linears_nf4),
    "nf4_parameter_fraction": nf4_fraction,
    "disk_gb": disk_nf4_gb,
    "vram_idle_gb": vram_nf4_idle,
    "vram_peak_gb": peak_vram_nf4,
    "rtf": rtf_nf4,
    "wer": wer_nf4,
    "cer": cer_nf4,
}
metrics_path = SPRINT6_OUTPUT / "metrics.json"
metrics_temporary = metrics_path.with_suffix(".json.tmp")
with open(metrics_temporary, "w", encoding="utf-8") as file:
    json.dump(metrics_nf4, file, indent=2)
metrics_temporary.replace(metrics_path)
print(
    f"\n=== NF4 HIBRIDO === VRAM idle: {vram_nf4_idle:.2f} GB | "
    f"VRAM peak: {peak_vram_nf4:.2f} GB | RTF: {rtf_nf4:.4f} | "
    f"WER: {wer_nf4:.4f} | CER: {cer_nf4:.4f}"
)
