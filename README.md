# VibeVoice Optimization

> Tesis de investigacion — **UPAO 2026** (Universidad Privada Antenor Orrego)

Optimizacion multi-tecnica post-entrenamiento (PTQ) del modelo TTS **VibeVoice 1.5B** para inferencia en hardware limitado (NVIDIA RTX 4060 Ti 8GB VRAM), con especializacion monolingue al espanol mediante fine-tuning con LoRA sobre Common Voice 17.0.

## Resumen

Este proyecto constituye el marco experimental de una tesis que evalua tres tecnicas de cuantizacion a **INT4** sobre el modelo **VibeVoice-ES** (fine-tuneado en espanol), midiendo el *trade-off* entre eficiencia computacional y preservacion de calidad linguistica. El objetivo es validar que una estrategia hibrida de cuantizacion consciente de activaciones (AWQ) con compensacion ligera (LoRA) permite sintesis de voz conversacional de alta calidad en GPU de consumo de 8GB, acercando modelos de frontera a entornos con recursos limitados.

## Estado de Sprints

| Sprint | Enfoque | Estado |
|--------|---------|--------|
| **Sprint 1** | Infraestructura de datos y set de calibracion | Completado |
| **Sprint 1.5** | Fine-tuning monolingue espanol (LoRA + Diffusion Head) | Completado |
| **Sprint 2** | Linea base FP16 + cuantizacion uniforme RTN (INT4) | Pendiente |
| **Sprint 3** | Reconstruccion de 2do orden GPTQ (INT4) | Pendiente |
| **Sprint 4** | Hibrido AWQ (INT4) + compensacion LoRA | Pendiente |
| **Sprint 5** | Benchmark en servidor limitado + validacion estadistica | Pendiente |

## Arquitectura del Modelo

- **Modelo base:** VibeVoice 1.5B (community fork) — TTS conversacional multi-hablante
- **Tokenizadores:** Dos VAE convolucionales continuos (acustico dim=64 + semantico dim=128) operando a **7.5 Hz frame rate** (compresion 3200x)
- **Backbone:** LLM Qwen2.5 + Diffusion Head para decodificacion acustica de alta fidelidad
- **Atencion:** PyTorch SDPA nativa (Flash Attention no instalado por restricciones de compilacion en WSL2)
- **Entrada:** Texto plano con prefijo de hablante + voice prompts (audio de referencia)
- **Dominio temporal puro:** Sin espectrogramas Mel; autoencoder convolucional sobre audio crudo
- **Salida:** Audio mono 24,000 Hz, normalizado a -25 dB FS

### Parametros de Audio

| Parametro | Valor |
|-----------|-------|
| Sample rate | 24,000 Hz |
| Canales | 1 (mono) |
| Normalizacion | -25 dB FS |
| Factor de compresion | 3200x (7.5 Hz frame rate) |
| Dim. latente acustico | 64 |
| Dim. latente semantico | 128 |
| Formatos entrada | .wav, .mp3, .flac, .m4a, .ogg, .pt, .npy |

## Stack Tecnologico

| Componente | Version | Proposito |
|---|---|---|
| Python | 3.12 | Runtime (Micromamba env `vibevoice`) |
| PyTorch | 2.12.0+cu130 | Backend CUDA 13.0 |
| Transformers | 4.51.3 | Carga y ejecucion del modelo |
| Accelerate | 1.6.0 | Device map y distribucion |
| Diffusers | 0.38.0 | Cabezal de difusion acustica |
| PEFT | 0.19.1 | Adaptadores LoRA |
| bitsandbytes | 0.49.2 | Cuantizacion RTN/NF4 |
| auto-gptq | — | Cuantizacion por Hessiana de 2do orden |
| autoawq | — | Cuantizacion consciente de activaciones |
| Numba | 0.65.1 | Tokenizador acustico continuo |
| Datasets | 3.5.0 | Carga de Common Voice 17.0 |
| librosa | 0.11.0 | Remuestreo y preprocesamiento de audio |
| Whisper large-v3 | — | ASR de referencia para WER/CER |
| JiWER | — | Distancia de edicion (Levenshtein) |

## Estructura del Proyecto

```
VibeVoice_Optimization/
├── notebooks/
│   ├── 01_vram_baseline.ipynb           # Diagnostico inicial de VRAM (5.04 GB en FP16)
│   ├── tesis_model_cuantization.ipynb   # Pipeline principal: Sprints 1 al 5
│   └── VibeVoice_Colab.ipynb            # Demo de inferencia en Colab
├── scripts/
│   ├── run_finetune_es.sh               # Fine-tuning monolingue (18h en RTX 4060 Ti)
│   ├── merge_es_checkpoint.sh           # Merge LoRA + Diffusion Head → VibeVoice-ES
│   ├── _fix_pythonpath.py               # Parche para sys.path del repo
│   └── _insert_sprint15.py              # Insercion programatica de celdas Sprint 1.5
├── VibeVoice_repo/                      # Codigo fuente del fork comunitario
│   ├── vibevoice/modular/               # Definicion de arquitectura (.py)
│   ├── vibevoice/finetune/              # Scripts de entrenamiento
│   ├── vibevoice/processor/             # Procesadores de audio y tokenizadores
│   └── demo/                            # Ejemplos y voces de referencia
├── weights/                             # (gitignored) ~15 GB — Checkpoints .safetensors
├── data/                                # (gitignored) ~370 MB — Common Voice 17.0 + calibracion
├── outputs/                             # (gitignored) ~5 GB — Logs de fine-tuning y audio generado
├── verify_stack.py                      # Diagnostico de hardware y dependencias
└── README.md
```

## Uso

### 1. Verificar entorno

```bash
micromamba activate vibevoice
python verify_stack.py
```

### 2. Fine-tuning monolingue (espanol) — Sprint 1.5

Usa el split `train` de Common Voice 17.0 (336,846 registros). Parametros: `lora_r=8`, `bf16=True`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=32`, `voice_prompt_drop_rate=1.0`.

```bash
bash scripts/run_finetune_es.sh
bash scripts/merge_es_checkpoint.sh
```

El checkpoint resultante **VibeVoice-ES** se guarda en `weights/vibevoice-1.5b-es/` (~10 GB, 3 shards `.safetensors`).

### 3. Pipeline de cuantizacion — Sprints 2 al 5

Ejecutar las celdas del notebook principal secuencialmente:

```
notebooks/tesis_model_cuantization.ipynb
```

**Precaucion:** antes de cualquier import del modelo, aplicar el parche obligatorio de `CONFIG_MAPPING` (colision de nombres en `transformers>=4.45.x`):

```python
import transformers
if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)
```

## Notas Tecnicas

- **Flash Attention:** NO instalado. Compilar `flash-attn` desde fuente satura la RAM de WSL2. Se usa PyTorch SDPA nativa con `attn_implementation="sdpa"`.
- **Dos estrategias LoRA independientes:**
  - **Sprint 1.5 — LoRA de adaptacion linguistica:** Fine-tuning pre-cuantizacion sobre datos en espanol; sus pesos se mergean al modelo base.
  - **Sprint 4 — LoRA de compensacion:** Adaptador post-cuantizacion AWQ para recuperar error acustico; opera sobre modelo congelado ya comprimido.
- **Dataset:** `fsicoli/common_voice_17_0` (config `"es"`), mirror comunitario del dataset original de Mozilla (retirado Oct 2025). Audio original a 48kHz → remuestreo a 24kHz + normalizacion a -25 dB FS.
- **Set de calibracion:** 512 muestras del split `validation`, concatenadas en `data/calibration_tensor.pt` (285 MB, `float32`), requerido por GPTQ y AWQ para recolectar estadisticas de activacion.

## Variantes de Cuantizacion (PTQ a INT4)

| Variante | Tecnica | Libreria | Descripcion |
|---|---|---|---|
| **FP16** ($O_1$) | Linea base sin compresion | — | 5.04 GB VRAM en reposo |
| **RTN** ($X_1$) | Round-to-Nearest uniforme | bitsandbytes | Baseline ingenua: redondeo directo, maximo ruido de cuantizacion |
| **GPTQ** ($X_2$) | Reconstruccion por Hessiana de 2do orden | auto-gptq | Correccion fila por fila en bloques g128, minimizando error cuadratico |
| **AWQ+LoRA** ($X_3$) | Proteccion de canales salientes + adaptador | autoawq + PEFT | Estrategia hibrida propuesta: pesos criticos protegidos + compensacion ligera |

## Metricas de Evaluacion

### Eficiencia Computacional
- VRAM pico en inferencia (GB)
- Latencia por segundo de audio generado (ms)
- Real-Time Factor (RTF = $t_{proc} / t_{audio}$)
- Espacio en disco del checkpoint (GB)

### Preservacion de Calidad
- Word Error Rate (WER) — via Whisper large-v3
- Character Error Rate (CER) — distancia de Levenshtein via JiWER
- Perplejidad del modelo (PPL) — CrossEntropyLoss, escala Yao et al. (Clase-1 $\le 0.1$, Clase-2 $\le 0.5$, Clase-3 $> 0.5$)

### Validacion Estadistica (Sprint 5)
- Prueba de normalidad: Shapiro-Wilk
- Prueba no parametrica pareada: Wilcoxon signed-rank para validar $H_1$ (superioridad del modelo hibrido propuesto)

## Hardware de Desarrollo

- **CPU:** Intel Core i5-14400F (8 hilos asignados a WSL2)
- **GPU:** NVIDIA GeForce RTX 4060 Ti 8GB VRAM (Ada Lovelace, Tensor Cores 4ta Gen)
- **RAM:** 32GB DDR5 (16-24GB asignados a WSL2 via `.wslconfig`)
- **OS:** Ubuntu 24.04 LTS sobre WSL2 (Windows 11 Host), sistema de archivos EXT4 nativo
- **IDE:** VS Code con tunel WSL nativo
- **Entorno:** Micromamba env `vibevoice` (Python 3.12, CUDA 13.0)
