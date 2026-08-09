# VibeVoice Optimization

> Tesis de investigacion — **UPAO 2026** (Universidad Privada Antenor Orrego)

Optimizacion multi-tecnica post-entrenamiento (PTQ) del modelo TTS **VibeVoice 1.5B** para inferencia en hardware limitado (NVIDIA RTX 4060 Ti 8GB VRAM), con especializacion monolingue al espanol mediante fine-tuning con LoRA sobre Common Voice 17.0.

## Resumen

Este proyecto constituye el marco experimental de una tesis que evalua seis tecnicas de cuantizacion sobre el modelo **VibeVoice-ES** (fine-tuneado en espanol), midiendo el *trade-off* entre eficiencia computacional y preservacion de calidad linguistica. El objetivo es comparar estrategias INT4, INT8 y W8A8 que preservan el pipeline acustico para ejecutar sintesis de voz conversacional en una GPU de consumo de 8 GB.

## Estado de Sprints

| Sprint | Enfoque | Estado |
|--------|---------|--------|
| **Sprint 1** | Infraestructura de datos y set de calibracion | Completado |
| **Sprint 1.5** | Primera adaptacion monolingue: LoRA + diffusion head | Completado; gate con voz WER=0.2321 |
| **Sprint 2** | TorchAO W4A16 HQQ g128 selectivo | Validado; WER=0.1845, pico=3.55 GiB |
| **Sprint 2.5** | Analisis de arquitectura y diagnostico de fallos | Completado |
| **Sprint 3** | GPTQ W4 g128 selectivo con loader hibrido Triton | Rechazado por gate; revision TTS-prefill WER=0.3423 |
| **Sprint 4** | AWQ W4A16 g128 selectivo con AutoAWQ/Triton | Rechazado por gate; WER=0.3095 |
| **Sprint 5** | INT8 selectiva (Fabio Sarracino + HelpfulHand3) | Validado; WER=0.1409, pico=5.18 GiB |
| **Sprint 6** | NF4 + double quant (DevParker/Dubedo + Soniqo) | Rechazado por gate; WER=0.3304 |
| **Sprint 7** | SmoothQuant W8A8 selectivo sobre Qwen2 | Validado; WER=0.0685, pico=3.96 GiB |
| **Sprint 8** | 🏁 Benchmark: 7 modelos + validacion estadistica | Completado; resultados en `outputs/sprint8_benchmark/` |

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
| GPTQModel | 2.2.0 | GPTQ W4 g128 selectivo con backend Triton |
| AutoAWQ | 0.2.9 | AWQ W4A16 GEMM selectivo con backend Triton |
| Numba | 0.65.1 | Tokenizador acustico continuo |
| Datasets | 3.5.0 | Carga de Common Voice 17.0 |
| librosa | 0.11.0 | Remuestreo y preprocesamiento de audio |
| Whisper large-v3 | — | ASR de referencia para WER/CER |
| JiWER | — | Distancia de edicion (Levenshtein) |
| TorchAO | 0.18.0 | W4A16 HQQ g128 selectivo con kernel INT4 PyTorch/CUDA |

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
│   ├── torchao_quantize_vibevoice.py    # Build, recarga y gate TorchAO W4A16
│   ├── build_vibevoice_gptq_tts.sh       # Revision GPTQ con prefills TTS reales
│   ├── gptq_vibevoice.py                 # Export, validacion y loader hibrido
│   ├── smoke_vibevoice_gptq.py           # Recarga limpia y smoke TTS GPTQ
│   ├── setup_autoawq.sh                   # Preflight AutoAWQ 0.2.9 + Triton
│   ├── build_vibevoice_awq.sh             # AWQ W4A16 selectivo y persistente
│   ├── awq_vibevoice.py                   # Export, validacion y loader hibrido
│   └── smoke_vibevoice_awq.py             # Recarga limpia, TTS y gate WER AWQ
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

Existe una sola linea canonica: `weights/vibevoice-1.5b` -> primer LoRA + diffusion head -> `weights/vibevoice-1.5b-es`. Usa `train` de Common Voice 17.0 con un desarrollo speaker-disjoint excluido del entrenamiento; el split nativo `validation` queda reservado para calibracion PTQ.

El checkpoint historico fue entrenado durante 18.6 h con LoRA rank 8, alpha 32, LR `2.5e-5`, CE weight `0.04`, diffusion weight `1.4` y fine-tuning completo del diffusion head. La implementacion publica conserva esos hiperparametros para documentar el experimento, pero corrige la mascara CE, el split, la normalizacion, la semantica y el checkpointing. Una ejecucion nueva es por ello una reproduccion metodologica mejorada, no bit a bit.

```bash
bash scripts/run_finetune_es.sh
bash scripts/merge_es_checkpoint.sh
```

El resultado se guarda en `weights/vibevoice-1.5b-es/`. El gate usa una voz de referencia versionada, seis frases espanolas, Whisper large-v3 y umbral `WER <= 0.30`. El checkpoint existente obtuvo WER `0.2321`, frente a `0.6230` del VibeVoice default bajo el mismo protocolo. El segundo LoRA CE-only posterior fue rechazado con WER `1.0` y no forma parte del pipeline publicado.

Sprint 2 usa TorchAO `0.18.0` para convertir solo las 196 proyecciones Qwen a W4A16 HQQ g128. Conserva embeddings, `lm_head`, prediction head, connectors y tokenizadores en BF16, recarga el artefacto y exige `WER <= 0.30` antes de promoverlo. La ejecucion requiere autorizacion explicita:

```bash
python scripts/torchao_quantize_vibevoice.py
```

Sprint 3 usa GPTQModel en el mismo entorno `vibevoice`. La version conservada guarda por separado el decoder GPTQ y los componentes TTS protegidos, y calibra las 196 proyecciones con 256 activaciones reales del prefill TTS (`inputs_embeds`, 32589 tokens) usando `desc_act=True`:

```bash
bash scripts/setup_gptqmodel.sh
bash scripts/build_vibevoice_gptq_tts.sh
python scripts/smoke_vibevoice_gptq.py \
  --model weights/vibevoice-1.5b-es-gptq-tts \
  --output outputs/sprint3_gptq_tts/smoke-es.wav
```

Esta version obtuvo WER `0.3423` y no aprobo el umbral `0.30`. El artefacto GPTQ anterior con calibracion de texto aislado fue eliminado; sus metricas historicas se conservan para trazabilidad.

Sprint 4 usa AutoAWQ 0.2.9 en el entorno `vibevoice`. Cuantiza exclusivamente las 196 proyecciones Qwen como W4A16 asimetrico g128 GEMM y conserva el pipeline acustico en FP16. La calibracion captura 256 prefills TTS reales con la voz fija y los convierte en bloques de 256 tokens para AutoAWQ. El artefacto se acepta solo despues de la validacion estructural y el gate canonico de seis frases, Whisper large-v3 y `WER <= 0.30`:

```bash
bash scripts/setup_autoawq.sh
bash scripts/build_vibevoice_awq.sh
python scripts/smoke_vibevoice_awq.py \
  --model weights/vibevoice-1.5b-es-awq \
  --output outputs/sprint4_awq/smoke-es.wav
```

El candidato actual completo obtuvo WER `0.3095`, CER `0.2352`, RTF `1.3181` y pico `3.39 GiB`; se conserva con estado `quality_failed` porque no alcanza el umbral `0.30`.

### 3. Pipeline de cuantizacion — Sprints 1 al 7

Ejecutar `notebooks/tesis_model_cuantization.ipynb` secuencialmente. Las celdas correspondientes invocan los scripts de shell en el orden requerido.

Las celdas largas desde `S1.5-03` se ejecutan como jobs desacoplados del kernel. Cada job conserva PID, estado y log en `outputs/.notebook-jobs/<job>/`; si VS Code se desconecta, vuelva a abrir el notebook y reejecute la misma celda para reconectarse. El entrenamiento guarda cada 500 pasos y reanuda automaticamente desde el ultimo `checkpoint-*` despues de una caida completa de WSL.

### Resultados actuales (Sprints 1-4):

| Sprint | Modelo | WER | PPL |
|--------|--------|-----|-----|
| Sprint 1 | Datos + calibracion | — | — |
| Sprint 1.5 | VibeVoice-ES (primer LoRA) | 0.2321 con voice prompt | — |
| Sprint 2 | TorchAO W4A16 HQQ g128 selectivo | 0.1845 | — |
| Sprint 3 | GPTQ W4 g128, revision TTS-prefill | 0.3423 (rechazado) | — |
| Sprint 4 | AWQ W4A16 g128 selectivo | 0.3095 (rechazado) | — |

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
- **Set de calibracion:** 512 muestras reservadas del split `validation`. GPTQ y AWQ seleccionan 256 registros y capturan `inputs_embeds` de prefills TTS reales con voz condicionada; `data/calibration_tensor.pt` conserva el audio preprocesado para otras tecnicas.
- **Cuantizabilidad:** Arquitectura confirmada compatible con GPTQ/AWQ. Todas las capas proyectivas son `nn.Linear` estandar. Los tokenizers convolucionales (~35% params) se preservan en FP16. VRAM esperada post-cuantizacion: ~2.5-3.5 GB.
- **TorchAO preserva la inferencia oficial:** Sprint 2 usa `TorchAoConfig` y `Int4WeightOnlyConfig` con HQQ g128. El manifiesto exige exactamente 196 pesos `Int4TilePackedTo4dTensor`, recarga limpia y gate con voz.
- **GGUF/CrispASR fue descartado:** tanto F16 como IQ4_NL fallaron en la ruta C++ con voz condicionada, mientras que la inferencia oficial de PyTorch funciona con el checkpoint canonico.
- **GPTQ requiere un loader hibrido:** las matrices empaquetadas (`qweight`, `qzeros`, `scales`, `g_idx`) no pueden copiarse a `nn.Linear` mediante `state_dict`. Sprint 3 carga el decoder con GPTQModel/Triton y trasplanta el objeto `Qwen2Model`; los modulos TTS protegidos se cargan por separado en BF16.
- **AWQ usa GEMM/Triton, no Marlin:** AutoAWQ 0.2.9 produce 196 modulos `WQLinear_GEMM` W4A16 asimetricos. `awq_ext` no esta instalado en este host; etiquetar este artefacto como Marlin seria incorrecto.
- **Conv1D debe permanecer en FP16/FP32:** Cuantizar capas convolucionales de tokenizers corrompe las salidas acusticas (hallazgo Mudler/LocalAI). Refuerza exclusion de `acoustic_tokenizer` y `semantic_tokenizer`.
- **AdaLN del diffusion head es fragil bajo INT4:** Los bloques Adaptive Layer Normalization fallan bajo ciertos esquemas INT4 (hallazgo FluffyBunnies/ONNX). El diffusion head permanece en BF16.
- **DPM-Solver sensible a INT:** El diffusion head permanece protegido; SmoothQuant solo se aplica al decoder Qwen.
- **embed_tokens debe preservarse para clonacion:** Excluir `embed_tokens` en esquemas NF4 protege la identidad de voz en generaciones largas (>30s) (Soniqo).
- **lm_head tied con embed_tokens:** En Qwen2.5-1.5B, `lm_head.weight` comparte memoria con `embed_tokens.weight`. Cuantizar `lm_head` corrompe embeddings de entrada. Excluir en GPTQ/INT8/NF4/SmoothQuant.

## Variantes de Cuantizacion (PTQ a INT4)

| Variante | Tecnica | Libreria | Resultados |
|---|---|---|---|
| **FP16** ($O_1$) | Linea base sin compresion | — | 5.04 GB VRAM, RTF=1.35, WER=0.54 |
| **TorchAO W4A16** ($X_1$) | HQQ g128 selectivo, Qwen-only | TorchAO/PyTorch CUDA | Validado; WER=0.1845, pico=3.55 GiB |
| **GPTQ** ($X_2$) | Reconstruccion Hessiana W4 g128, Qwen-only | GPTQModel/Triton | Rechazado; revision TTS-prefill WER=0.3423 |
| **AWQ** ($X_3$) | Proteccion de canales W4A16 g128, Qwen-only | AutoAWQ/Triton | Rechazado; WER=0.3095 |
| **INT8** | Selectiva LLM-only | bitsandbytes | Validado; WER=0.1409, pico=5.18 GiB, RTF=2.3380 |
| **NF4** | NormalFloat4 + double quant selectivo | bitsandbytes | Rechazado; WER=0.3304 |
| **SmoothQuant** | W8A8 selectivo, salida BF16 | `torch._int_mm` | Validado; 3.83 GiB idle, 3.96 GiB pico, WER=0.0685 |

### Tabla Comparativa Final (Sprint 8 — Benchmark)

| Modelo | VRAM (GB) | RTF | WER | CER | PPL |
|--------|-----------|-----|-----|-----|-----|
| FP16 | 5.04 | 1.35 | 0.5385 | 0.3134 | — |
| TorchAO W4A16 | 3.55 | 1.0978 | 0.1845 | 0.1019 | — |
| GPTQ W4 g128 TTS-prefill | 3.37 | 1.1948 | 0.3423 | 0.1783 | — |
| AWQ W4A16 g128 | 3.39 | 1.3181 | 0.3095 | 0.2352 | — |
| INT8 | 5.18 | 2.3380 | 0.1409 | 0.1118 | — |
| NF4 W4A16 double quant | 4.01 | 1.3215 | 0.3304 | 0.2056 | — |
| SmoothQuant W8A8 | 4.04 | 1.5988 | 0.0685 | 0.0419 | — |

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

### Benchmark Final (Sprint 8)

El benchmark reproducible se ejecuta con `scripts/sprint8_benchmark.py`. Cada variante se recarga en un proceso independiente usando seis textos fijos, la voz `en-Alice_woman.wav` y Whisper large-v3. Los resultados completos, muestras individuales, IC95 bootstrap, CSV y tabla Markdown estan en `outputs/sprint8_benchmark/`.

| Modelo | Estado | VRAM pico (GiB) | RTF | WER | CER |
|---|---|---:|---:|---:|---:|
| FP16 | baseline | 5.18 | 0.9475 | 0.2321 | 0.1546 |
| TorchAO INT4 | validado | 3.55 | 1.0133 | 0.1845 | 0.1019 |
| GPTQ TTS | rechazado | 3.37 | 1.1517 | 0.3423 | 0.1783 |
| AWQ | rechazado | 3.39 | 1.2743 | 0.3571 | 0.2544 |
| NF4 | rechazado | 3.37 | 1.1507 | 0.3095 | 0.1880 |
| INT8 | validado | 3.98 | 2.3694 | 0.2083 | 0.1505 |
| SmoothQuant | validado | 4.04 | 1.5988 | 0.0685 | 0.0419 |

PESQ y MCD no se calculan porque el protocolo actual no contiene ondas de referencia emparejadas para los seis textos; se reportan todas las metricas validables sin inventar valores.
`summary.json` incluye comparaciones pareadas contra FP16 mediante Shapiro-Wilk y Wilcoxon. Estas pruebas son exploratorias (`n=6`) y no sustituyen el benchmark estadistico ampliado del split `test`.

## Hardware de Desarrollo

- **CPU:** Intel Core i5-14400F (8 hilos asignados a WSL2)
- **GPU:** NVIDIA GeForce RTX 4060 Ti 8GB VRAM (Ada Lovelace, Tensor Cores 4ta Gen)
- **RAM:** 32GB DDR5 (16-24GB asignados a WSL2 via `.wslconfig`)
- **OS:** Ubuntu 24.04 LTS sobre WSL2 (Windows 11 Host), sistema de archivos EXT4 nativo
- **IDE:** VS Code con tunel WSL nativo
- **Entorno:** Micromamba env `vibevoice` (Python 3.12, CUDA 13.0)
