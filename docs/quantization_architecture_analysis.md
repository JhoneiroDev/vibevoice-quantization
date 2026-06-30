# VibeVoice: Analisis Completo de Cuantizabilidad y Diagnostico de Fallos

Documento para el agente de programacion (OpenCode). Viabilidad, arquitectura detallada, fallos previos y plan de accion para cuantizacion PTQ de VibeVoice 1.5B.

---

## 0. Resumen Ejecutivo

**Veredicto: La cuantizacion GPTQ/AWQ del modelo VibeVoice completo es TOTALMENTE VIABLE.**

El modelo usa `nn.Linear` estandar de PyTorch en todas sus capas proyectivas. No hay kernels fusionados, subclases de Linear ni operaciones custom que bloqueen GPTQ o AWQ. Los fallos previos (Sprint 2 y 3) se debieron a:

1. Uso de `bitsandbytes` (RTN) que no esta disenado para modelos custom (+38% overhead VRAM)
2. Cuantizacion de solo el LLM standalone, no VibeVoice completo, con datos insuficientes
3. Enfoque incorrecto: cuantizar componentes aislados en lugar del modelo integrado

Estrategia correcta: cuantizar VibeVoice completo con GPTQ/AWQ, excluyendo capas convolucionales de tokenizers.

---

## 1. Arquitectura Completa del Modelo

### 1.1 Arbol de Componentes

```
VibeVoiceForConditionalGenerationInference  (2,704,021,985 params totales)

+- model: VibeVoiceModel
   +- language_model: Qwen2Model (from Qwen2Config)
   |  +- embed_tokens: Embedding(151643, 1536)
   |  +- layers: ModuleList[Qwen2DecoderLayer] x 28
   |     +- Cada capa:
   |        +- self_attn: Qwen2Attention
   |        |  +- q_proj: nn.Linear(1536, 1536, bias=True)     CUANTIZABLE
   |        |  +- k_proj: nn.Linear(1536, 256, bias=True)      CUANTIZABLE
   |        |  +- v_proj: nn.Linear(1536, 256, bias=True)      CUANTIZABLE
   |        |  +- o_proj: nn.Linear(1536, 1536, bias=False)    CUANTIZABLE
   |        +- mlp: Qwen2MLP
   |        |  +- gate_proj: nn.Linear(1536, 8960)             CUANTIZABLE
   |        |  +- up_proj:   nn.Linear(1536, 8960)             CUANTIZABLE
   |        |  +- down_proj: nn.Linear(8960, 1536)             CUANTIZABLE
   |        +- input_layernorm, post_attention_layernorm
   |  +- norm: Qwen2RMSNorm(1536)
   |
   +- acoustic_tokenizer: VibeVoiceAcousticTokenizerModel
   |  +- encoder: TokenizerEncoder (SConv1d + FFN lineal)
   |  |  Conv1d NO cuantizable, FFN SI pero bajo impacto
   |  +- decoder: TokenizerDecoder (SConvTranspose1d + FFN)
   |
   +- semantic_tokenizer: VibeVoiceSemanticTokenizerModel
   |  +- encoder: TokenizerEncoder (sin decoder)
   |
   +- acoustic_connector: SpeechConnector
   |  +- fc1: nn.Linear(64, 1536)    CUANTIZABLE
   |  +- fc2: nn.Linear(1536, 1536)  CUANTIZABLE
   |
   +- semantic_connector: SpeechConnector
   |  +- fc1: nn.Linear(128, 1536)   CUANTIZABLE
   |  +- fc2: nn.Linear(1536, 1536)  CUANTIZABLE
   |
   +- prediction_head: VibeVoiceDiffusionHead
   |  +- noisy_images_proj: nn.Linear(64, 768)       CUANTIZABLE
   |  +- cond_proj: nn.Linear(768, 768)              CUANTIZABLE
   |  +- t_embedder.mlp: Sequential(Linear(256,768), SiLU, Linear(768,768))
   |  +- layers x 4: HeadLayer
   |  |  +- ffn.gate_proj:  nn.Linear(768, 2304)     CUANTIZABLE
   |  |  +- ffn.up_proj:    nn.Linear(768, 2304)     CUANTIZABLE
   |  |  +- ffn.down_proj:  nn.Linear(2304, 768)     CUANTIZABLE
   |  |  +- adaLN_modulation[-1]: nn.Linear(768, 2304) CUANTIZABLE
   |  +- final_layer.linear: nn.Linear(768, 64)      CUANTIZABLE
   |  +- final_layer.adaLN[-1]: nn.Linear(768, 1536) CUANTIZABLE
   |
   +- noise_scheduler: DPMSolverMultistepScheduler (sin parametros)

+- lm_head: nn.Linear(1536, 151643, bias=False)       CUANTIZABLE
   (tied to language_model.embed_tokens.weight)
```

### 1.2 Distribucion de Parametros

| Componente | Tipo de capas | Parametros | % del total | Cuantizable |
|------------|--------------|------------|-------------|-------------|
| language_model | nn.Linear (Q/K/V/O/gate/up/down) | ~1,310M | 48.5% | SI |
| language_model | embed_tokens, RMSNorm | ~233M | 8.6% | NO |
| acoustic_tokenizer | SConv1d, SConvTranspose1d | ~350M | 12.9% | NO (conv) |
| semantic_tokenizer | SConv1d | ~600M | 22.2% | NO (conv) |
| prediction_head | nn.Linear (todas) | ~123M | 4.6% | SI |
| connectors (2x) | nn.Linear x 4 total | ~0.8M | 0.03% | SI |
| lm_head | nn.Linear | ~233M | 8.6% | SI (tied) |
| noise_scheduler | Sin parametros | 0 | 0% | N/A |

**Total cuantizable (solo nn.Linear): ~1,660M parametros (61.4%)**

Ahorro esperado en disco: de ~5.04 GB (FP16) a ~2.5-3.0 GB.
Ahorro esperado en VRAM: de 5.04 GB a ~3.0-3.5 GB.

---

## 2. Analisis de Compatibilidad con GPTQ / AWQ

### 2.1 Verdicto por Componente

| Componente | GPTQ | AWQ | Notas |
|------------|------|-----|-------|
| language_model (Qwen2.5) | Totalmente | Totalmente | Modelo HF estandar, soportado nativamente |
| prediction_head | Totalmente | Totalmente | Todas nn.Linear, DiT-style sin atencion |
| acoustic_connector | Totalmente | Totalmente | MLP simple 2 capas lineales |
| semantic_connector | Totalmente | Totalmente | Identico al acustico |
| lm_head | Totalmente | Totalmente | nn.Linear estandar, tied con embed_tokens |
| acoustic_tokenizer | Parcial | Parcial | Solo FFN lineales en Block1D. Conv1d NO |
| semantic_tokenizer | Parcial | Parcial | Igual que acustico |

### 2.2 Evidencia Tecnica

1. **Todas las capas usan nn.Linear de PyTorch estandar.** Sin subclases custom. Verificado en modeling_vibevoice.py, modular_vibevoice_diffusion_head.py, modular_vibevoice_tokenizer.py.

2. **LLM backbone es Qwen2.5** (modeling_vibevoice.py:121). Mismo modelo que GPTQModel cuantiza en scripts/run_gptq_quantize.py.

3. **_supports_quantized_cache = True** declarado en VibeVoicePreTrainedModel (modeling_vibevoice.py:81).

4. **No hay FlashAttention.** SDPA nativa elimina conflictos con kernels cuantizados.

5. **APEX FusedRMSNorm opcional.** Solo en tokenizers si OPTIMIZE_FOR_SPEED=1. No afecta LLM ni diffusion head.

6. **_tp_plan** en configuration_vibevoice.py lista targets estandar: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

7. **Targets de LoRA** (train_vibevoice.py:112-115):
   - LLM: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
   - Diffusion head: noisy_images_proj,cond_proj,gate_proj,up_proj,down_proj,linear

### 2.3 Capas NO Cuantizables

| Capa | Tipo | Razon |
|------|------|-------|
| embed_tokens | nn.Embedding | GPTQ/AWQ solo operan sobre nn.Linear |
| SConv1d / SConvTranspose1d | Convolucion custom | No son nn.Linear |
| RMSNorm / LayerNorm | Normalizacion | Parametros element-wise, no proyectivos |
| DPMSolverMultistepScheduler | Algoritmo | Sin parametros de peso |

---

## 3. Diagnostico de Fallos Previos

### 3.1 Sprint 2: RTN-INT4 con bitsandbytes (resultados_bloques.md:262-495)

**v1 (25/06):** BitsAndBytesConfig fallo al cargar VibeVoice: "Some modules dispatched on CPU or disk"

Causa: bnb intenta despachar submodelos (tokenizers) a CPU porque device_map="auto" con cuantizacion no maneja modelos custom con sub-componentes no lineales.

**v2 (26/06):** RTN aplicado con device_map={"": "cuda:0"}, pero:
- VRAM reposo FP16: 5.04 GB
- VRAM reposo RTN: 6.97 GB (+38%)

**Causa raiz del overhead:** bitsandbytes en modo 4-bit NO fusiona dequantizacion en el kernel. Mantiene en VRAM:
- Pesos INT4 comprimidos
- Buffers FP16 para computo forward
- Buffers de quant_state (escalas y zeros)

Ventaja unica: ahorro en disco (~3.2 GB en .safetensors).

**Calidad RTN:** WER=0.38, CER=0.15, PESQ=1.03, MCD=57.1 -- comparable con FP16 (WER=0.54).

### 3.2 Sprint 3: GPTQ Manual - PPL=68,538 (resultados_bloques.md:497-563)

Cuantizo solo Qwen2.5-1.5B standalone extraido de VibeVoice-ES. 197 capas nn.Linear, config: bits=4, g=128, sym=True, damp=0.01.

**Tabla de causas del fallo:**

| Factor | Valor usado | Valor correcto | Gravedad |
|--------|------------|----------------|----------|
| Datos calibracion | 128 textos ~13 tokens avg | 256-512 con 512-1024 tokens | CRITICO |
| damp_percent | 0.01 (1%) | 0.1 (10%) | ALTO |
| Modelo | Qwen2.5 standalone extraido | VibeVoice completo | CRITICO |
| Forward dtype | float32 (probable) | float16/bfloat16 | MEDIO |
| Implementacion | Manual OBS con hooking | GPTQModel / auto-gptq | ALTO |

**Detalle:**

1. **Calibracion insuficiente:** GPTQ necesita secuencias de 512-1024 tokens para que la Hessiana capture correlaciones. Con 13 tokens, la matriz de covarianza es rango-deficiente.

2. **damp_percent muy bajo:** El fine-tuning LoRA introduce estructura de bajo rango -> Hessiana mal condicionada. Requiere damping >= 0.1.

3. **Modelo aislado:** Los pesos del Qwen en VibeVoice-ES fueron fine-tuneados con LoRA + merge. Extraerlos como Qwen standalone:
   - Pierde speech embeddings como contexto de calibracion
   - Rompe alineacion con diffusion head
   - Ignora que lm_head esta tied con embed_tokens

### 3.3 GPTQModel en entorno aislado (NO EJECUTADO)

Script scripts/run_gptq_quantize.py cuantiza Qwen/Qwen2.5-1.5B desde HuggingFace (modelo base, NO fine-tuneado). Incluso si funciona, produce un Qwen cuantizado que NO es el LLM de VibeVoice-ES.

---

## 4. Estrategia de Cuantizacion: Plan de Accion

### 4.1 Principio General

NO cuantizar componentes aislados. Cuantizar VibeVoice COMPLETO, pasando audio+texto real por el forward pipeline. Esto asegura que GPTQ/AWQ capturen distribuciones de activacion correctas considerando la interaccion speech-text.

### 4.2 Que Cuantizar vs. Que Preservar

**CUANTIZAR (INT4, group_size=128):**
- model.language_model.* -- Todas las nn.Linear x 28 capas x 7 proyecciones
- model.prediction_head.* -- Todas las nn.Linear (~26 capas)
- model.acoustic_connector.fc1, .fc2
- model.semantic_connector.fc1, .fc2
- lm_head (considerar tied con embed_tokens)

**PRESERVAR EN FP16:**
- model.acoustic_tokenizer completo (Conv1d/ConvTranspose1d dominan)
- model.semantic_tokenizer completo
- embed_tokens (si no se cuantiza lm_head)
- RMSNorm, LayerNorm

### 4.3 Requisitos de Calibracion

Se necesita un dataloader que pase AUDIO REAL por el forward COMPLETO de VibeVoice. Esto es critico porque:
- Las activaciones en el LLM dependen de speech embeddings inyectados via acoustic_input_mask
- Las activaciones en diffusion head dependen de hidden states del LLM
- Los connectors transforman features acusticos (64-dim) y semanticos (128-dim)

**Formato del dataloader:**

```python
{
    "input_ids": tokenized_text,        # [batch, seq_len] con tokens especiales
    "speech_tensors": audio_waveform,    # [batch, audio_len] float32 24kHz
    "speech_masks": valid_frame_mask,    # [batch, n_frames] bool
    "speech_input_mask": position_mask,  # [batch, seq_len] bool
}
```

**Fuente:** data/calibration_tensor.pt (512 clips, 74.9M samples) + data/calibration_metadata.json (transcripciones).

**Cantidad:** 256-512 muestras, batch_size=1 (VRAM limitada).

### 4.4 Fases de Implementacion

**Fase A: GPTQ del modelo completo**
1. Cargar VibeVoice-ES FP16, device_map={"": "cuda:0"}
2. Construir dataloader audio+texto CV17 validation split
3. Forward de calibracion capturando activaciones en nn.Linear objetivo
4. Aplicar GPTQ per-layer: bits=4, g128, sym=True, damp=0.1, desc_act=False
5. Guardar en weights/vibevoice-1.5b-es-gptq/
VRAM esperada: ~2.5-3.5 GB

**Fase B: AWQ + LoRA (Sprint 4)**
1. Cargar VibeVoice-ES FP16, forward de calibracion
2. Magnitudes de activacion por canal de salida en cada nn.Linear
3. Proteger 1% canales con mayor magnitud via scaling inverso
4. Transformacion equivalente + INT4 g128
5. LoRA rank=8-16 en capas cuantizadas LLM + diffusion head
6. Fine-tunear solo LoRA con CV17 train split
7. Guardar en weights/vibevoice-1.5b-es-awq-lora/

**Fase C: Benchmark (Sprint 5)**
Ejecutar 4 modelos (FP16, GPTQ, AWQ, AWQ+LoRA) sobre CV17 test:
- RTF, VRAM pico, latencia
- WER/CER (Whisper large-v3), PESQ, MCD
- Validacion estadistica (Shapiro-Wilk + Wilcoxon)

---

## 5. Conflicto de Versiones y Entornos

### 5.1 Entorno vibevoice (principal)

| Paquete | Version | Nota |
|---------|---------|------|
| transformers | 4.51.3 | CRITICO: NO actualizar |
| accelerate | 1.6.0 | CRITICO: NO actualizar |
| torch | 2.12.0+cu130 | CUDA 13.0 |
| bitsandbytes | 0.49.2 | Solo baseline RTN |
| peft | 0.19.1 | LoRA |

### 5.2 Entorno gptq (cuantizacion)

| Paquete | Version | Proposito |
|---------|---------|-----------|
| transformers | >=5.4.0 | Requerido por GPTQModel |
| gptqmodel | latest | GPTQ principal |
| auto-gptq | latest | Alternativa |
| autoawq | latest | AWQ (Sprint 4) |

### 5.3 Conflicto transformers

GPTQModel requiere transformers>=5.4.0, INCOMPATIBLE con VibeVoice (4.51.3).

**Opcion A (recomendada):** Cuantizar en entorno gptq con transformers>=5.4.0, guardar .safetensors. Cargar modelo cuantizado en vibevoice con transformers==4.51.3 via wrapper custom. Los .safetensors son agnosticos de version.

**Opcion B:** Implementar GPTQ con PyTorch puro en vibevoice, corrigiendo errores Sprint 3: damp=0.1, calibracion con audio real, modelo completo.

**Opcion C:** Verificar si auto-gptq es compatible con transformers==4.51.3.

---

## 6. Archivos Clave del Repositorio

| Archivo | Proposito | Ref |
|---------|-----------|-----|
| VibeVoice_repo/.../modeling_vibevoice.py | VibeVoiceModel, SpeechConnector, training | L58-69, L107-209 |
| VibeVoice_repo/.../modeling_vibevoice_inference.py | Inferencia autoregresiva + diffusion | L68-712 |
| VibeVoice_repo/.../modular_vibevoice_diffusion_head.py | DiT diffusion head AdaLN-Zero | L191-280 |
| VibeVoice_repo/.../modular_vibevoice_tokenizer.py | Tokenizers convolucionales | L588-596 |
| VibeVoice_repo/.../configuration_vibevoice.py | Configs submodulos | L221 |
| notebooks/tesis_model_cuantization.ipynb | Pipeline Sprints 1-5 | |
| scripts/run_gptq_quantize.py | GPTQ Qwen standalone (entorno gptq) | |
| scripts/run_finetune_es.sh | Fine-tuning LoRA | |
| data/calibration_tensor.pt | 512 clips concatenados (285 MB) | |
| data/calibration_metadata.json | Offsets + transcripciones | |
| weights/vibevoice-1.5b-es/ | Checkpoint fine-tuneado (~10 GB) | |
| resultados_bloques.md | Log historico de ejecuciones | |
| backlog_quantization.md | Planificacion de sprints | |

---

## 7. Lecciones Aprendidas y Directrices

1. **NUNCA cuantizar componentes aislados.** El LLM fine-tuneado tiene pesos distintos al Qwen2.5 base. Extraerlo rompe alineacion con diffusion head y connectors.

2. **Calibracion DEBE usar audio real**, no solo texto. Las activaciones del LLM dependen de speech embeddings inyectados.

3. **damp_percent >= 0.1** para GPTQ en modelos con adaptacion LoRA. La estructura de bajo rango condiciona mal la Hessiana.

4. **Excluir tokenizers convolucionales.** Sus Conv1d no son cuantizables y constituyen el 35% de parametros. Cuantizar solo sus FFN lineales internas tiene bajo impacto.

5. **bitsandbytes NO sirve para despliegue.** Su overhead de VRAM (+38%) lo hace contraproducente para GPUs de 8GB. Usar solo para baseline comparativa.

6. **Verificar tied embeddings.** lm_head.weight = embed_tokens.weight. Cuantizar uno afecta al otro. Considerar mantener embed_tokens en FP16.

7. **El forward de calibracion debe usar el training forward** (modeling_vibevoice.py:332-484), no el inference generate(). El training forward procesa audio por el pipeline completo (tokenizer -> connector -> LLM -> diffusion head) en un solo paso.

8. **Datos de calibracion ya existen:** data/calibration_tensor.pt tiene 512 clips preprocesados a 24kHz, -25dB FS, con transcripciones en data/calibration_metadata.json. Reutilizarlos directamente.
