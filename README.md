# VibeVoice Optimization

> Tesis de investigacion — **UPAO 2026** (Universidad Privada Antenor Orrego)

Optimizacion multi-tecnica post-entrenamiento (PTQ) del modelo TTS **VibeVoice 1.5B** para inferencia en hardware limitado (NVIDIA RTX 4060 Ti 8GB VRAM), con especializacion monolingue al espanol mediante fine-tuning con LoRA sobre Common Voice 17.0.

## Resumen

Este proyecto constituye el marco experimental de una tesis que evalua tres tecnicas de cuantizacion a **INT4** sobre el modelo **VibeVoice-ES** (fine-tuneado en espanol), midiendo el *trade-off* entre eficiencia computacional y preservacion de calidad linguistica. El objetivo es validar que la cuantizacion consciente de activaciones (AWQ) permite sintesis de voz conversacional de alta calidad en GPU de consumo de 8GB, acercando modelos de frontera a entornos con recursos limitados.

## Estado de Sprints

| Sprint | Enfoque | Estado |
|--------|---------|--------|
| **Sprint 1** | Infraestructura de datos y set de calibracion | Completado |
| **Sprint 1.5** | Correccion linguistica espanola (segundo LoRA CE-only) | Implementado, ejecucion pendiente |
| **Sprint 2** | GGUF IQ4_NL selectivo + runtime C++ CrispASR | Implementado, espera checkpoint corregido |
| **Sprint 2.5** | Analisis de arquitectura y diagnostico de fallos | Completado |
| **Sprint 3** | GPTQ manual (INT4, g128) sobre VibeVoice completo | Completado |
| **Sprint 4** | AWQ via AutoAWQ (vibevoice nativo) | Completado |
| **Sprint 5** | INT8 selectiva (Fabio Sarracino + HelpfulHand3) | Pendiente |
| **Sprint 6** | NF4 + double quant (DevParker/Dubedo + Soniqo) | Implementado, benchmark pendiente |
| **Sprint 7** | FP8 E4M3FN dinamico (Zhao-Kun + CyberVoice) | Implementado, benchmark pendiente |
| **Sprint 8** | 🏁 Benchmark: 6 modelos + validacion estadistica | Pendiente |

Ver [`docs/quantization_architecture_analysis.md`](docs/quantization_architecture_analysis.md) para el analisis completo de la arquitectura, compatibilidad con GPTQ/AWQ, y diagnostico de fallos previos.

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
| auto-gptq / GPTQModel | — | Cuantizacion por Hessiana de 2do orden |
| llmcompressor (NM) | — | AWQ via Neural Magic |
| Numba | 0.65.1 | Tokenizador acustico continuo |
| Datasets | 3.5.0 | Carga de Common Voice 17.0 |
| librosa | 0.11.0 | Remuestreo y preprocesamiento de audio |
| Whisper large-v3 | — | ASR de referencia para WER/CER |
| JiWER | — | Distancia de edicion (Levenshtein) |
| CrispASR | 0.8.23 | Conversor, cuantizador IQ4_NL y runtime GGML/C++ |

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
│   ├── build_vibevoice_iq4_nl.sh        # Conversion y cuantizacion selectiva GGUF
│   ├── validate_vibevoice_gguf.py       # Verificacion tensor por tensor
│   └── smoke_vibevoice_iq4_nl.sh        # Inferencia nativa C++
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

Usa `train` de Common Voice 17.0 con un desarrollo speaker-disjoint excluido del entrenamiento. El split nativo `validation` queda reservado para calibracion PTQ. La correccion usa como base `vibevoice-1.5b-es`, LoRA rank 8, LR `1e-5`, CE weight `1.0` y diffusion weight `0.0`; diffusion head, connectors y tokenizers permanecen congelados.

```bash
bash scripts/run_finetune_es.sh
bash scripts/merge_es_checkpoint.sh
```

El checkpoint resultante **VibeVoice-ES-Corrected** se guarda como BF16 en `weights/vibevoice-1.5b-es-corrected/`. Debe superar el gate WER/CER del notebook antes de regenerar las variantes cuantizadas.

Sprint 2 descarga el runtime CUDA precompilado de CrispASR `v0.8.23`, verifica su SHA-256 y guarda el GGUF validado junto con su manifiesto:

```bash
bash scripts/build_vibevoice_iq4_nl.sh
bash scripts/smoke_vibevoice_iq4_nl.sh
```

### 3. Pipeline de cuantizacion — Sprints 1 al 7

Ejecutar `notebooks/tesis_model_cuantization.ipynb` secuencialmente. Los scripts de shell se ejecutan desde terminal.

### Resultados actuales (Sprints 1-3):

| Sprint | Modelo | WER | PPL |
|--------|--------|-----|-----|
| Sprint 1 | Datos + calibracion | — | — |
| Sprint 1.5 | VibeVoice-ES (fine-tuned) | — | — |
| Sprint 2 | GGUF IQ4_NL selectivo | Pendiente | — |
| Sprint 3 | GPTQ-INT4 (manual) | 0.27 | 12.76 |

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
- **LoRA de adaptacion linguistica (Sprint 1.5):** Fine-tuning pre-cuantizacion sobre datos en espanol; sus pesos se mergean al modelo base. Es el unico LoRA del proyecto: el LoRA de compensacion post-AWQ (Sprint 4) se descarto porque peft no admite adaptadores sobre las capas `WQLinear` de AutoAWQ.
- **Dataset:** `fsicoli/common_voice_17_0` (config `"es"`), mirror comunitario del dataset original de Mozilla (retirado Oct 2025). Audio original a 48kHz → remuestreo a 24kHz + normalizacion a -25 dB FS.
- **Set de calibracion:** 512 muestras del split `validation`, concatenadas en `data/calibration_tensor.pt` (285 MB, `float32`), requerido por GPTQ y AWQ para recolectar estadisticas de activacion.
- **Cuantizabilidad:** Arquitectura confirmada compatible con GPTQ/AWQ. Todas las capas proyectivas son `nn.Linear` estandar. Los tokenizers convolucionales (~35% params) se preservan en FP16. VRAM esperada post-cuantizacion: ~2.5-3.5 GB.
- **IQ4_NL usa un runtime distinto:** Sprint 2 produce un GGUF monolitico para CrispASR. No es compatible con llama.cpp, `ggc v6` ni con el loader Hugging Face del resto de variantes.
- **Cota fisica de IQ4_NL:** proteger 1.394B parametros no-Qwen en F16 requiere al menos 2.596 GiB. Con las 196 matrices Qwen en IQ4_NL, el payload minimo es ~3.28 GiB; una huella total menor a 1.2 GB no es compatible con esta estrategia.
- **Conv1D debe permanecer en FP16/FP32:** Cuantizar capas convolucionales de tokenizers corrompe las salidas acusticas (hallazgo Mudler/LocalAI). Refuerza exclusion de `acoustic_tokenizer` y `semantic_tokenizer`.
- **AdaLN del diffusion head es fragil bajo INT4:** Los bloques Adaptive Layer Normalization fallan bajo ciertos esquemas INT4 (hallazgo FluffyBunnies/ONNX). Considerar FP16 o FP8 para diffusion head.
- **DPM-Solver sensible a INT:** El desnatador de ruido del diffusion head genera "voz metalica" bajo cuantizacion entera. FP8 lo elimina al mantener exponentes flotantes (CyberVoice Labs).
- **embed_tokens debe preservarse para clonacion:** Excluir `embed_tokens` en esquemas NF4 protege la identidad de voz en generaciones largas (>30s) (Soniqo).
- **lm_head tied con embed_tokens:** En Qwen2.5-1.5B, `lm_head.weight` comparte memoria con `embed_tokens.weight`. Cuantizar `lm_head` corrompe embeddings de entrada. Excluir en GPTQ/INT8/NF4/FP8.

## Variantes de Cuantizacion (PTQ a INT4)

| Variante | Tecnica | Libreria | Resultados |
|---|---|---|---|
| **FP16** ($O_1$) | Linea base sin compresion | — | 5.04 GB VRAM, RTF=1.35, WER=0.54 |
| **IQ4_NL** ($X_1$) | LUT no lineal selectiva, Qwen-only | CrispASR/GGML | Implementado; benchmark pendiente |
| **GPTQ** ($X_2$) | Reconstruccion por Hessiana | PyTorch puro | 6.18 GB VRAM, RTF=1.23, WER=0.27, PPL=12.76 |
| **AWQ** ($X_3$) | Proteccion de canales por activaciones | AutoAWQ | Completado (Sprint 4) |
| **INT8** | Selectiva LLM-only | bitsandbytes | Pendiente (Sprint 5) |
| **NF4** | NormalFloat4 + double quant selectivo | bitsandbytes | Implementado; 3.25 GB VRAM de reposo, benchmark pendiente |
| **FP8** | E4M3FN dinamico, salida BF16 | torch._scaled_mm | Implementado; 3.89 GB VRAM tras recarga, benchmark pendiente |

### Tabla Comparativa Final (Sprint 8 — Benchmark)

| Modelo | VRAM (GB) | RTF | WER | CER | PPL |
|--------|-----------|-----|-----|-----|-----|
| FP16 | 5.04 | 1.35 | 0.5385 | 0.3134 | — |
| GGUF IQ4_NL | Pendiente | Pendiente | Pendiente | Pendiente | — |
| GPTQ-INT4 | 6.18 | 1.23 | 0.2746 | 0.1774 | 12.76 |
| AWQ | Pendiente | Pendiente | Pendiente | Pendiente | — |
| INT8 | Pendiente | Pendiente | Pendiente | Pendiente | — |
| NF4 | Pendiente | Pendiente | Pendiente | Pendiente | — |
| FP8 | Pendiente | Pendiente | Pendiente | Pendiente | — |

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
