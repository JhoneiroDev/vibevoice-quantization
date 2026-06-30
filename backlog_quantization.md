### Tablero de Control y Backlog: Cuantización y Benchmark VibeVoice
Este documento registra la planificación por Sprints, el estado de las tareas y la asignación técnica para alcanzar el hito central de la investigación: la optimización multi-técnica post-entrenamiento (PTQ) de la arquitectura VibeVoice 1.5B, preparando los modelos en hardware local (RTX 4060 Ti) para su posterior benchmark en servidores limitados.

--------------------------------------------------------------------------------

#### 📊 Estado General del Proyecto (Resumen Ejecutivo)
| Sprint | Enfoque Principal | Estado | Progreso |
| ------ | ------ | ------ | ------ |
| **Sprint 1** | Ingesta de Datos, Partición y Set de Calibración | COMPLETADO | 100% |
| **Sprint 1.5** | Fine-Tuning Monolingüe Español (LoRA + Diffusion Head) | COMPLETADO | 100% |
| **Sprint 2** | RTN-INT4 via bitsandbytes (split layout) | IMPLEMENTADO | 100% |
| **Sprint 3** | GPTQ-INT4 via GPTQModel (qforge, split layout) | IMPLEMENTADO | 100% |
| **Sprint 4** | AWQ-INT4 Marlin (qforge) + LoRA compensacion | IMPLEMENTADO | 100% |
| **Sprint 5** | INT8 via bitsandbytes (split layout) | IMPLEMENTADO | 100% |
| **Sprint 6** | NF4 + Double Quant via bitsandbytes (split layout) | IMPLEMENTADO | 100% |
| **Sprint 7** | FP8 E4M3FN via torch nativo (split layout) | IMPLEMENTADO | 100% |
| **Sprint 8** | 🏁 Benchmark Final: 7 Modelos + Validación Estadística | IMPLEMENTADO | 100% |

--------------------------------------------------------------------------------

### 🏗️ Entornos de Ejecución

| Entorno | transformers | Propósito | Sprints |
|---------|-------------|-----------|---------|
| `vibevoice` | 4.51.3 | VibeVoice inference, extraccion, reinsercion, bnb, FP8, benchmark | 2, 5, 6, 7, 8 |
| `qforge` | >=5.4.0 | GPTQModel + autoawq (cuantizacion del Qwen2 standalone) | 3, 4 |

```bash
# Setup unico para qforge:
micromamba create -n qforge python=3.12 -y
micromamba run -n qforge pip install gptqmodel autoawq datasets
```

Los entornos se comunican exclusivamente via archivos: `weights/standalone_qwen_fp16/`.

--------------------------------------------------------------------------------

### 🧱 Fase Transversal A: Extraccion del Qwen2 (compartida por todos los sprints)

Ejecutar UNA SOLA VEZ antes de cualquier sprint:
- Carga VibeVoice-ES con `VibeVoiceForConditionalGeneration`
- Extrae `model.model.language_model` (Qwen2Model)
- Crea `Qwen2ForCausalLM` con `lm_head = embed_tokens` (tied)
- Guarda como checkpoint standalone en `weights/standalone_qwen_fp16/`
- Guarda tokenizer para calibracion de texto
- Celda: `fase-a-extract` en el notebook

### 🧱 Fase Transversal B: Split Layout (por sprint)

Para cada tecnica, en `vibevoice`:
1. Crear `weights/vibevoice-1.5b-es-{tecnica}/`
2. Guardar pesos no-decoder (tokenizers, diffusion head, conectores) como FP16 en raiz
3. Crear subdirectorio `decoder-{fmt}/` con pesos cuantizados + `quantization_config.json`
4. Actualizar `config.json` raiz: `"vibevoice_decoder_model_path": "decoder-{fmt}"`

--------------------------------------------------------------------------------

#### 🏃‍♂️ Planificación Detallada por Sprints

##### Sprint 1: Infraestructura de Datos y Pipeline de Calibración ✅
- Dataset: `fsicoli/common_voice_17_0` (config `"es"`), mirror comunitario
- Splits: `train` (fine-tuning), `validation` (calibracion), `test` (benchmark)
- Preprocesamiento: remuestreo 48kHz→24kHz, normalizacion -25 dB FS
- Output: `data/calibration_tensor.pt` (512 clips, 285 MB) + `data/calibration_metadata.json`

##### Sprint 1.5: Fine-Tuning Monolingüe Español con LoRA ✅
- LoRA rank=8 sobre Q/K/V/O/gate/up/down del LLM + diffusion head
- 18.6h, 5263 pasos, 1 epoch. CE loss: 1.93→1.72
- Merge verificado: `weights/vibevoice-1.5b-es/` (10.3 GB, VRAM: 5.04 GB)

##### Sprint 2: RTN-INT4 via bitsandbytes ✅
- **Implementacion:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")` sobre Qwen2 standalone
- **Split layout:** `decoder-rtn/` con pesos FP16 dequantizados
- **Sin calibracion.** Baseline automatico de la industria
- Celda: `s2-rtn-bnb`

##### Sprint 3: GPTQ-INT4 via GPTQModel (qforge) ✅
- **Implementacion:** `GPTQModel.from_pretrained()` + `model.quantize(calib_data)` en qforge
- **Calibracion:** 256 chunks concatenados de CV17 (~512 tokens avg)
- **Config:** bits=4, group_size=128, sym=True, damp_percent=0.1 (>=0.1 para LoRA), desc_act=False
- **Split layout:** `decoder-gptq/` con INT4 empaquetado real (kernels CUDA)
- Celda: `s3-gptqmodel`

##### Sprint 4: AWQ-INT4 Marlin (qforge) + LoRA (vibevoice) ✅
- **Parte A (qforge):** `AutoAWQForCausalLM.from_pretrained()` + `model.quantize(tokenizer, quant_config={"zero_point":True, "q_group_size":128, "w_bit":4, "version":"Marlin"})`
- **Calibracion:** 256 prompts hibridos narracion+dialogo multi-hablante
- **Split layout:** `decoder-awq/` con kernels Marlin para Ada Lovelace
- **Parte B (vibevoice):** LoRA rank=8 via PEFT sobre q_proj/v_proj. Fine-tuning 200 pasos con 10K muestras CV17. `merge_and_unload()`
- Celdas: `s4-awq-marlin`, `s4-lora`

##### Sprint 5: INT8 Selectivo via bitsandbytes (Fabio Sarracino + HelpfulHand3) ✅
- **Implementacion:** `BitsAndBytesConfig(load_in_8bit=True)` sobre Qwen2 standalone. Equivalente a `llm_int8_skip_modules=["diffusion_head","acoustic_connector","semantic_connector","audio_encoder","lm_head"]` de Fabio pero via split layout (los modulos de audio nunca se tocan — vienen de FP16 original).
- **Split layout:** `decoder-int8/` con state_dict FP16 dequantizado
- **Hallazgo HelpfulHand3:** Tokenizers DEBEN permanecer en FP16 (inestabilidad acustica, alucinacion de ruido)
- **Hipotesis:** INT8 preserva calidad (WER ≈ FP16) con compresion ~2x
- Celda: `s5-int8`

##### Sprint 6: NF4 + Double Quant via bitsandbytes ✅
- **Implementacion:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- **Exclusion:** `embed_tokens` excluido del state_dict (protege identidad vocal)
- **Hallazgo Soniqo:** NF4 preserva prosodia >30s. Double quant comprime escalas (~6GB→~4GB VRAM)
- Celda: `s6-nf4`

##### Sprint 7: FP8 E4M3FN via torch nativo ✅
- **Implementacion:** `.to(torch.float8_e4m3fn)` sobre nn.Linear del Qwen2
- **Proteccion:** Si `W.abs().max() > 448`, aplicar scaling simetrico antes del casteo
- **Hallazgo CyberVoice Labs:** FP8 elimina "voz metalica" del DPM-Solver (mantisa+exponente)
- **Ventaja:** Tensor Cores Ada Lovelace 4ta Gen, sin calibracion, ~2.7 GB disco
- Celda: `s7-fp8`

##### Sprint 8: 🏁 Benchmark Final — 7 Modelos + Validación Estadística
- **Modelos:** FP16, RTN, GPTQ, AWQ+LoRA, INT8, NF4, FP8
- **Dataset:** 200 muestras aleatorias del split `test` de CV17
- **Metricas:** VRAM reposo/pico, RTF, WER/CER (Whisper large-v3), PESQ, MCD, espacio en disco
- **Estadistica:** Shapiro-Wilk (normalidad) + Wilcoxon signed-rank (muestras pareadas, p < 0.01)
- **Output:** `outputs/sprint8/metrics_benchmark_report.json`, graficos seaborn, tabla LaTeX booktabs
- Celda: `s8-bench` (placeholder, se completa al tener los 7 modelos)

--------------------------------------------------------------------------------

### 📂 Estructura de Pesos Esperada

```
weights/
├── vibevoice-1.5b/                  # VibeVoice original (base)
├── vibevoice-1.5b-es/               # VibeVoice-ES (fine-tuneado, baseline FP16)
├── standalone_qwen_fp16/             # Qwen2 extraido (Fase A, compartido)
├── vibevoice-1.5b-es-rtn/           # Sprint 2
│   ├── non_decoder_weights.pt
│   ├── config.json (vibevoice_decoder_model_path="decoder-rtn")
│   └── decoder-rtn/
├── vibevoice-1.5b-es-gptq/          # Sprint 3
│   └── decoder-gptq/
├── vibevoice-1.5b-es-awq/           # Sprint 4 (AWQ solo)
│   └── decoder-awq/
├── vibevoice-1.5b-es-awq-lora/      # Sprint 4 (AWQ+LoRA)
├── vibevoice-1.5b-es-int8/          # Sprint 5
│   └── decoder-int8/
├── vibevoice-1.5b-es-nf4/           # Sprint 6
│   └── decoder-nf4/
└── vibevoice-1.5b-es-fp8/           # Sprint 7
    └── decoder-fp8/
```

--------------------------------------------------------------------------------

### 📋 Hallazgos de Arquitectura

- **Split Layout (lemuriandezapada, ComfyUI-VibeVoice):** Cuantizar solo Qwen2 como checkpoint standalone, preservar audio modules en FP16. `vibevoice_decoder_model_path` en config.json permite carga nativa.
- **Conv1D → F16/F32 obligatorio (Mudler):** Si se cuantizan capas Conv1D de tokenizers, el casting inline corrompe las salidas de convolucion acustica.
- **AdaLN fragil bajo INT4 (FluffyBunnies):** Los bloques Adaptive Layer Normalization del diffusion head fallan bajo ciertos esquemas INT4. Mantener prediction_head en FP16.
- **NF4 > RTN para hablantes (DevParker):** NF4 preserva mejor la diarizacion de hablantes que RTN. La distribucion normal se adapta mejor a pesos del LLM congelado.
- **FP8 elimina voz metalica (CyberVoice Labs):** Las tecnicas INT destruyen suavidad del DPM-Solver. FP8 con mantisa+exponente elimina saltos abruptos.
- **embed_tokens excluido (Soniqo):** El multiplicador de escala simetrica en proyecciones de atencion es clave para proteger identidad de voz.
- **lm_head tied con embed_tokens (Directriz 10):** `tie_word_embeddings=True` en Qwen2.5-1.5B. Comparten memoria fisica. NO cuantizar lm_head.
- **Tecnicas NO viables:** GGUF (requiere binarios C++ externos), MLX (Apple Silicon exclusivo), ONNX INT4 (AdaLN blocks fallan).

--------------------------------------------------------------------------------

#### 🛠️ Directrices Tecnicas para la Ejecucion de Tareas
1.  **Defensa de Memoria Local:** `torch.cuda.empty_cache()` antes y despues de cada cuantizacion en frio.
2.  **Dos entornos, dos propositos:** `vibevoice` (transformers 4.51.3) para VibeVoice y benchmark. `qforge` (transformers>=5.4.0) para GPTQModel y autoawq. Se comunican via archivos.
3.  **Preservacion de Estructuras:** No modificar archivos fuente de `VibeVoice_repo/`. Inyectar comportamiento via scripts o herencia.
4.  **Dos LoRAs, dos propositos distintos:** (a) Sprint 1.5 — adaptacion linguistica pre-cuantizacion (mergeada). (b) Sprint 4 — compensacion post-cuantizacion (no mergeada, opera sobre modelo congelado).
5.  **Split Layout como estandar:** Extraer Qwen2 → cuantizar con libreria oficial → reinsertar en VibeVoice via split layout. NUNCA cuantizar el modelo completo de una sola pasada.
6.  **NO cuantizar lm_head** — `tie_word_embeddings=True`. `lm_head.weight` comparte memoria con `embed_tokens.weight`. Excluir en todos los sprints.
7.  **Excluir Conv1d** de tokenizers acustico/semantico durante cuantizacion.
8.  **Calibracion con datos reales** para GPTQ/AWQ: CV17 texto o audio+texto segun la libreria lo requiera.
9.  **bitsandbytes como herramienta valida:** INT8 y NF4 con double quant son tecnicas de produccion. Solo FP4 simple (RTN) tiene overhead problematico.
10. **Benchmark con rigor estadistico:** Minimo 200 muestras. Shapiro-Wilk + Wilcoxon. Graficos + tabla LaTeX para el manuscrito.
