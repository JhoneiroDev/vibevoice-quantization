# Contexto de Desarrollo: Proyecto VibeVoice Optimization

Este documento sirve como la fuente de verdad absoluta y memoria de contexto para el agente de programación (OpenCode). Contiene el estado actual de la infraestructura local, parches de inicialización corregidos, requerimientos de librerías avanzadas y el diseño de la matriz experimental comparativa basada en el marco teórico de la tesis.

---

## 1. Arquitectura de Hardware y Sistema
* **CPU:** Intel Core i5-14400F (16 hilos / Asignados 8 a WSL2)
* **GPU:** NVIDIA GeForce RTX 4060 Ti (8GB VRAM dedicada / Arquitectura Ada Lovelace / Tensor Cores de 4ta Gen)
* **RAM del Sistema:** 32GB DDR5 (Asignados 16GB-24GB a WSL2 mediante `.wslconfig`)
* **Entorno Operativo (Híbrido de Alto Rendimiento):** Windows 11 Host de Control -> Ejecución Aislada en WSL2 (Ubuntu 24.04 LTS con sistema de archivos EXT4 nativo para mitigar cuellos de botella de I/O).
* **IDE:** VS Code con Túnel WSL Nativo
* **Gestor de Entornos:** Micromamba (Entorno activo: `vibevoice` con Python 3.12)

---

## 2. Estructura del Espacio de Trabajo
El proyecto está alojado estrictamente en el sistema de archivos nativo de Linux (`~/projects/`) para asegurar la velocidad de transferencia y la estabilidad de los compiladores de kernels CUDA.

```text
~/projects/VibeVoice_Optimization/
├── .git/                 # Repositorio Git (rama main) — El proyecto está versionado
├── .gitignore            # Excluye weights/ VibeVoice_repo/ data/ outputs/ __pycache__/
├── VibeVoice_repo/       # Fork de la comunidad (Código fuente del modelo) — IGNORADO por git (tiene su propio .git)
│   └── vibevoice/
│       └── modular/      # Archivos de definición de la arquitectura (.py)
├── weights/              # IGNORADO por git (~5.1 GB, modelos .safetensors)
│   └── vibevoice-1.5b/   # Checkpoint de Hugging Face (.safetensors / config.json)
├── notebooks/
│   ├── 01_vram_baseline.ipynb           # Cuaderno de pruebas y línea base de VRAM
│   └── tesis_model_cuantization.ipynb   # Cuaderno principal: Sprint 1 (datos/calibración) implementado
├── data/                 # IGNORADO por git — Destino para datasets (Mozilla Common Voice)
│   ├── dataset_partitions.json      # Índices de partición test/val/resto
│   ├── calibration_tensor.pt        # Tensor float32 con audio de calibración concatenado
│   └── calibration_metadata.json    # Offsets y metadatos de cada clip de calibración
├── scripts/              # Automatizaciones y scripts de cuantización avanzada
└── outputs/              # IGNORADO por git — Audios generados y logs de benchmarking estadístico
```

---

## 3. Estado del Stack de Software y Restricciones Técnicas

### Aceleración de Atención (Flash Attention)
* **Librería Externa `flash-attn`:** **NO INSTALADA.** Intentar compilarla desde código fuente satura la memoria RAM/Swap de WSL2 provocando fallos críticos en el servicio (`E_UNEXPECTED`).
* **Solución Adoptada:** Se utiliza de forma nativa **PyTorch SDPA** (Scaled Dot Product Attention). El hardware ejecuta los mismos kernels optimizados directamente por hardware sin requerir dependencias externas de compilación.

### Parche Crítico de Carga (Hugging Face Dynamic Mapping Collision)
La librería `transformers` presenta una colisión nativa en el registro de nombres con este fork comunitario. Para evitar un `ValueError` de duplicidad durante la inicialización de `vibevoice_acoustic_tokenizer`, **todo script o notebook debe ejecutar obligatoriamente este bypass previo antes de importar el modelo**:

```python
import transformers
if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)
```

**Explicación técnica:** En `transformers>=4.45.x`, `CONFIG_MAPPING` es una instancia de `_LazyConfigMapping(OrderedDict)` que almacena las entradas en un atributo interno `_mapping`, no en el propio diccionario. El método `__contains__` consulta tanto `_mapping` como `_extra_content`, pero `__delitem__` (heredado de `OrderedDict`) opera sobre el dict vacío. Por eso `del CONFIG_MAPPING[key]` produce un `KeyError` aunque `key in CONFIG_MAPPING` devuelva `True`. La solución es eliminar directamente desde el atributo protegido `_mapping`.

### Versiones Verificadas de Dependencias (Entorno `vibevoice`)
Estas son las versiones exactas contra las que se validó el entorno. Cualquier desviación puede romper la compatibilidad con el modelo (ver `VibeVoice_repo/pyproject.toml`).

| Paquete | Versión instalada | Requerida | Notas |
|----------|-------------------|-----------|-------|
| `torch` | 2.12.0+cu130 | — | CUDA 13.0 |
| `transformers` | **4.51.3** | ==4.51.3 | **Crítico:** Versiones posteriores cambian la API de mapeo |
| `accelerate` | **1.6.0** | ==1.6.0 | **Crítico:** Versiones posteriores rompen el `device_map` automático |
| `diffusers` | 0.38.0 | — | Requerido por el cabezal de difusión acústica del modelo |
| `datasets` | 3.5.0 | ==3.5.0 | Para benchmarks automáticos con datasets remotos |
| `gradio` | 5.50.0 | ==5.50.0 | Para la interfaz gráfica de usuario (opcional) |
| `peft` | 0.19.1 | — | Para la inyección de adaptadores ligeros (LoRA) |
| `numba` | 0.65.1 | >=0.57.0 | Requerido por el tokenizador acústico continuo |
| `llvmlite` | 0.47.0 | >=0.40.0 | Dependencia obligatoria de numba |
| `bitsandbytes` | 0.49.2 | — | Motor backend para cuantización uniforme estándar (RTN/NF4) |
| `librosa` | 0.11.0 | — | Remuestreo de audio para preprocesamiento (TSK-1.3) |
| `auto-gptq` | *Instalar* | — | Pipeline para optimización por Hessiana de segundo orden |
| `autoawq` | *Instalar* | — | Pipeline para optimización consciente de canales de activación |

### Línea Base de Consumo de Memoria (Baseline Obtención Inicial - $O_1$)
* **Modelo:** VibeVoice 1.5B Parámetros.
* **Tipo de Dato Base:** FP16 (Precisión media).
* **Consumo en Reposo (VRAM):** **5.04 GB** netos mapeados en la GPU.
* **Margen de Maniobra Actual:** **2.96 GB libres** en la RTX 4060 Ti para el almacenamiento de activaciones y el KV Cache.

### Parámetros de Audio del Modelo (Arquitectura Temporal Pura — Sin Mel)
VibeVoice **no utiliza espectrogramas de Mel**. Opera sobre audio crudo en el dominio temporal mediante un autoencoder convolucional. Todo preprocesamiento debe ajustarse a estos parámetros:

| Parámetro | Valor | Fuente en el código |
|-----------|-------|---------------------|
| **Sample rate** | 24,000 Hz | `VibeVoiceTokenizerProcessor.sampling_rate` |
| **Canales** | 1 (mono) | Conversión automática con `_ensure_mono()` |
| **Normalización** | -25 dB FS, prevención de clipping | `AudioNormalizer` |
| **Factor de compresión** | 3200× (7.5 Hz frame rate @ 24kHz) | `speech_tok_compress_ratio` |
| **Encoder ratios** | [8, 5, 5, 4, 2, 2] | Ambos tokenizers (acústico y semántico) |
| **Dim. latente acústico** | 64 | `acoustic_tokenizer_config.vae_dim` |
| **Dim. latente semántico** | 128 | `semantic_tokenizer_config.vae_dim` |
| **Latencia VAE acústico** | Gaussian (`fix_std=0.5`) | `std_dist_type='gaussian'` |
| **Latencia VAE semántico** | Sin muestreo (`fix_std=0`) | `std_dist_type='none'` |
| **Formato de entrada** | `(batch, 1, time)` float32 | Procesador `__call__()` |
| **Formatos de archivo** | .wav, .mp3, .flac, .m4a, .ogg, .pt, .npy | `_load_audio_from_path()` |

> **Implicacion para el dataset:** El dataset `fsicoli/common_voice_17_0` (config `"es"`) proporciona audio a 48kHz (tasa original de CV17). El pipeline de preprocesamiento (TSK-1.3) aplica `librosa.resample(y, orig_sr=48000, target_sr=24000)` seguido de normalizacion a -25 dB FS.

---

## 4. Objetivos del Proyecto y Tareas Asignadas a OpenCode

OpenCode deberá diseñar el código enfocándose en maximizar la eficiencia y preservar la calidad del modelo bajo un **Diseño Experimental Comparativo Multi-Técnica** (Variable Independiente $X$):

### Objetivo Transversal: Especialización Monolingüe Español (Sprint 1.5)
*   **Meta:** Adaptar VibeVoice 1.5B para funcionar exclusivamente como TTS en español mediante fine-tuning con LoRA sobre el dataset `fsicoli/common_voice_17_0` (config `"es"`, split `train`).
*   **Estrategia:** Fine-tuning del LLM backbone (LoRA rank=8 sobre proyecciones Q/K/V/O y gate/up/down) + entrenamiento completo del diffusion head. Los tokenizadores acústico y semántico permanecen congelados por ser lenguaje-agnósticos.
*   **Resultado esperado:** El modelo **VibeVoice-ES** (checkpoint en `weights/vibevoice-1.5b-es/`) sintetiza voz natural en español y degrada otros idiomas por *catastrophic forgetting*. Este modelo es la nueva línea base para todos los experimentos de cuantización (Sprints 2-5).

### Tarea 1: Script de Inferencia de Línea Base ($O_1$) y Telemetría CUDA ✅
*   Desarrollar la lógica de inferencia de Texto a Voz (TTS) usando la clase `VibeVoiceForConditionalGeneration` en FP16, cargando el checkpoint **VibeVoice-ES** (`weights/vibevoice-1.5b-es/`).
*   Implementar funciones de telemetría precisas mediante llamadas directas a la API de CUDA (`torch.cuda.memory_allocated() / 1024**3`) para registrar el consumo pico neto en ejecución.

### Tarea 2: Implementación de la Variable Independiente ($X$) - Pipeline Comparativo PTQ
OpenCode debe construir un script unificado de benchmarking que someta al modelo VibeVoice 1.5B a tres variantes de cuantización Post-Training (PTQ) a niveles de **INT8** e **INT4**, evaluando la velocidad de inferencia y la retención de calidad:

1. **Variante $X_1$ - Cuantización Uniforme Estándar (RTN / NormalFloat4):** Redondeo directo al entero más cercano mediante `bitsandbytes`. Servirá como la línea base comprimida para documentar el impacto del ruido de cuantización directo.
2. **Variante $X_2$ - Reconstrucción por Capas de Segundo Orden (GPTQ):** Implementar la cuantización empleando la inversa de la matriz Hessiana para corregir los pesos fila por fila en bloques de 128 (*g128*), minimizando el error cuadrático total de reconstrucción.
3. **Variante $X_3$ - AWQ via llmcompressor (Ncoder-ai):** `llmcompressor.compress()` con `AWQModifier(w_bit=4, group_size=128, zero_point=True)`. Calibracion con 256 prompts hibridos de dialogo multi-hablante. Target: ~50% reduccion VRAM con calidad preservada.
4. **Variante $X_4$ - Cuantización Selectiva INT8 (Fabio Sarracino):** `BitsAndBytesConfig(load_in_8bit=True)` solo sobre el LLM (Qwen2), preservando audio modules en FP16. Minimiza degradación acústica.
5. **Variante $X_5$ - Cuantización NF4 + Double Quant (DevParker/Dubedo):** `bnb_4bit_quant_type="nf4"` con double quantization. Distribución normal de pesos para mejor ajuste que RTN.
6. **Variante $X_6$ - Cuantización FP8 E4M3FN (Zhao-Kun):** `torch.float8_e4m3fn` nativo en PyTorch 2.12+. Ada Lovelace (RTX 4060 Ti) con soporte hardware FP8 vía Tensor Cores 4ta Gen.

> **No viables para este hardware:** GGUF Q4_K_M (CrispStrobe) requiere binario C++ externo. MLX INT4 (Aufklarer) es exclusivo Apple Silicon.

### Tarea 3: Set de Calibración e Ingesta del Dataset (Mozilla Common Voice 17.0)
*   Automatizar la descarga del subconjunto en español de Common Voice 17.0 desde el mirror comunitario `fsicoli/common_voice_17_0` (config `"es"`) en la carpeta `data/`. El repositorio original de Mozilla en HuggingFace fue retirado en Oct 2025; este mirror almacena los audios directamente.
*   Usar los splits nativos de CV17: `train` para fine-tuning, `test` para benchmark, `validation` para calibración.
*   Utilizar el split `validation` como el **Set de Calibración Offline** obligatorio para recolectar las estadísticas de activación requeridas por AWQ y la matriz de covarianzas de GPTQ.

### Tarea 4: Fine-Tuning Eficiente con PEFT/LoRA (Estrategia de Preservación de Calidad)
* Diseñar un pipeline de entrenamiento ligero utilizando la librería `peft` para inyectar adaptadores de bajo rango (LoRA) en los módulos lineales del modelo cuantizado.
* El objetivo del ajuste fino es doble: actuar como un compensador de ruido post-cuantización extrema (INT4) y realizar la adaptación fonética y semántica específica de la síntesis de voz al idioma español.
* Congelar los pesos base optimizados de VibeVoice y entrenar exclusivamente los parámetros del adaptador (*Style Decorator approach*).

---

## 5. Marco Métrico e Instrumentos de Validación de Tesis (UPAO 2026)

Para que el benchmark sea válido científicamente frente al jurado evaluador, OpenCode debe implementar módulos de recolección de datos que alimenten de forma automatizada los siguientes indicadores en un reporte final `.csv` o `.json`:

### A. Dimensión de Eficiencia Computacional (Variables Dependientes)
1. **Latencia de Inferencia (ms):** Tiempo promedio requerido por el hardware para generar exactamente 1 segundo de audio.
2. **Real-Time Factor (RTF):** Relación matemática directa entre el tiempo empleado en el procesamiento ($t_{proc}$) y la duración del audio sintetizado ($t_{audio}$). Un valor de $RTF < 1$ es obligatorio para entornos productivos locales.
3. **Memory Footprint (MB):** Consumo pico de VRAM detectado en la GPU durante la fase de inferencia masiva.
4. **Espacio en Disco (GB):** Tamaño neto del archivo de pesos en formato `.safetensors` y su porcentaje de reducción espacial frente al original.

### B. Dimensión de Preservación de Calidad (Exactitud Lingüística)
1. **Word Error Rate (WER):** Tasa de palabras incorrectas generadas al procesar el audio sintetizado mediante el modelo ASR externo de referencia **Whisper Large-v3**.
2. **Character Error Rate (CER):** Porcentaje de error calculado a nivel de grafemas utilizando la distancia de edición de Levenshtein (vía librería `JiWER`) para capturar fallos fonéticos finos.
3. **Perplejidad del Modelo (PPL):** Medición de la coherencia probabilística de los tokens acústicos calculada a partir de la entropía cruzada (`CrossEntropyLoss`) de PyTorch. Los resultados se clasificarán según la escala de Yao et al. (Clase-1 $\le0.1$; Clase-2 $\le0.5$; Clase-3 $>0.5$).

---

## 6. Directrices Estrictas de Codificación para el Agente
1. **Preservar Rutas Locales:** El repositorio local se inyecta dinámicamente con `sys.path.insert(0, os.path.abspath("../VibeVoice_repo"))`. Queda estrictamente prohibido usar rutas absolutas de Windows o rutas fijas ajenas al entorno WSL.
2. **Manejo de Memoria CUDA:** Forzar un `torch.cuda.empty_cache()` y limpiar la caché de asignación antes y después de cada iteración de inferencia o inicialización de algoritmos.
3.  **Uso de SDPA:** Configurar de forma explícita el parámetro `attn_implementation="sdpa"` en el método `.from_pretrained()` de la arquitectura.

---

## 7. Analisis de Cuantizabilidad y Estado Real de Sprints

> Documento completo en: [`docs/quantization_architecture_analysis.md`](docs/quantization_architecture_analysis.md)

### 7.1 Veredicto de Cuantizabilidad

**La cuantizacion GPTQ/AWQ de VibeVoice 1.5B es TOTALMENTE VIABLE.** El modelo utiliza exclusivamente `nn.Linear` estandar de PyTorch en todas sus capas proyectivas (LLM Qwen2.5, Diffusion Head, Connectors, lm_head). No hay kernels fusionados, subclases de Linear ni operaciones custom que bloqueen GPTQ o AWQ.

### 7.2 Estado Real de Sprints (Junio 26, 2026)

| Sprint | Estado Planeado | Estado REAL | Resultado |
|--------|----------------|-------------|-----------|
| Sprint 1 | COMPLETADO | COMPLETADO | Dataset CV17 descargado, particionado, calibracion generada (512 clips) |
| Sprint 1.5 | COMPLETADO | COMPLETADO | Fine-tuning LoRA 18.6h exitoso. VibeVoice-ES en weights/vibevoice-1.5b-es/ (10.3 GB, VRAM 5.04 GB) |
| Sprint 2 | COMPLETADO | COMPLETADO | FP16 baseline (RTF=1.35, WER=0.54). RTN INT4 via bnb: CARGADO pero +38% VRAM overhead (6.97 GB) |
| Sprint 3 | POR EMPEZAR | PARCIAL (FALLIDO) | GPTQ manual sobre Qwen standalone: PPL=68,538 (Clase-3). Causa: modelo aislado, calibracion insuficiente (128 textos de ~13 tokens), damp=0.01 |
| Sprint 4 | POR EMPEZAR | POR EMPEZAR | Sin iniciar |
| Sprint 5 | POR EMPEZAR | POR EMPEZAR | Sin iniciar |

### 7.3 Causas de Fallos en Sprint 3

1. **Se cuantizo solo el LLM Qwen2.5 standalone**, extraido de VibeVoice-ES, en lugar del modelo completo.
2. **Datos de calibracion solo texto** (128 frases cortas de ~13 tokens). GPTQ necesita secuencias de 512-1024 tokens para Hessianas bien condicionadas.
3. **damp_percent=0.01** (debio ser >=0.1). El fine-tuning LoRA hace la Hessiana mal condicionada.
4. **Implementacion manual OBS** sin optimizaciones numericas (Cholesky con damping, reordenamiento de columnas).

### 7.4 Estrategia Correcta de Cuantizacion

**Principio:** NO cuantizar componentes aislados. Cuantizar VibeVoice completo con datos de audio+texto reales.

**Capas a CUANTIZAR (INT4, g128):**
- `model.language_model.*` — q/k/v/o/gate/up/down_proj x 28 capas (~1,310M params)
- `model.prediction_head.*` — Todas las nn.Linear (~26 capas, ~123M params)
- `model.acoustic_connector.fc1, .fc2` + `model.semantic_connector.fc1, .fc2`
- `lm_head` (considerar tied con embed_tokens)

**Capas a PRESERVAR EN FP16:**
- `model.acoustic_tokenizer` + `model.semantic_tokenizer` — Conv1d/ConvTranspose1d no cuantizables (~950M params)
- `embed_tokens`, RMSNorm, LayerNorm

**VRAM esperada post-cuantizacion:** ~2.5-3.5 GB (vs 5.04 GB FP16).

**Datos de calibracion:** `data/calibration_tensor.pt` (512 clips, 74.9M samples, 3120s a 24kHz) con transcripciones en `data/calibration_metadata.json`. Pasar audio real por el forward COMPLETO (training forward, no inference generate).

**Config GPTQ:** bits=4, group_size=128, sym=True, damp_percent=0.1, desc_act=False.

### 7.5 Conflicto de Entornos

GPTQModel requiere `transformers>=5.4.0`, INCOMPATIBLE con VibeVoice (`transformers==4.51.3`). Tres opciones:
- **A (recomendada):** Cuantizar en entorno `gptq`, guardar .safetensors, cargar en `vibevoice` con wrapper custom.
- **B:** GPTQ con PyTorch puro en `vibevoice` (corrigiendo errores del Sprint 3).
- **C:** Verificar compatibilidad de `auto-gptq` con `transformers==4.51.3`.

## 8. Resultados del Benchmark PTQ (Sprints 1-3)

| Modelo | VRAM (GB) | RTF | WER | CER | PPL |
|--------|-----------|-----|-----|-----|-----|
| FP16 (VibeVoice-ES) | 5.04 | 1.35 | 0.5385 | 0.3134 | — |
| RTN-INT4 (bitsandbytes) | 6.97 | 1.48 | 0.3846 | 0.1493 | — |
| GPTQ-INT4 (manual) | 6.18 | 1.23 | 0.2746 | 0.1774 | 12.76 |

### Hallazgos:
1. RTN +38% VRAM por buffers FP16 no fusionados (bnb). Solo baseline.
2. GPTQ 220/222 capas cuantizadas, WER mejora 0.54→0.27 vs FP16.
3. Sin empaquetado INT4 nativo → sin ahorro real de VRAM. Requiere GPTQModel/AWQ para INT4 nativo.
4. Bug VibeVoice: `forward_speech_features` con `"audio"` falla (encode()[0][0] en objeto no subscriptable). Workaround: `"vae"` + pre-encoding.
5. Ver `docs/quantization_architecture_analysis.md` para analisis completo.
```