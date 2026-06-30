### Tablero de Control y Backlog: Objetivos Específicos 2 al 5 (Cuantización y Benchmark VibeVoice)
Este documento registra la planificación por Sprints, el estado de las tareas y la asignación técnica para alcanzar el hito central de la investigación: la optimización multi-técnica post-entrenamiento (PTQ) de la arquitectura VibeVoice 1.5B, preparando los modelos en hardware local (RTX 4060 Ti) para su posterior benchmark en servidores limitados.

--------------------------------------------------------------------------------

#### 📊 Estado General del Proyecto (Resumen Ejecutivo)
| Sprint | Enfoque Principal | Estado | Progreso |
| ------ | ------ | ------ | ------ |
| **Sprint 1** | Ingesta de Datos, Partición y Set de Calibración | COMPLETADO | 100% |
| **Sprint 1.5** | Fine-Tuning Monolingüe Español (LoRA + Diffusion Head) | COMPLETADO | 100% |
| **Sprint 2** | Línea Base FP16 ($O_1$) y Baseline Ingenuo (RTN 4-bits) | COMPLETADO | 100% |
| **Sprint 2.5** | Analisis de Arquitectura y Diagnostico de Fallos | COMPLETADO | 100% |
| **Sprint 3** | Baseline Avanzado: GPTQ 4-bits (VibeVoice COMPLETO) | COMPLETADO | 100% |
| **Sprint 4** | Enfoque Propuesto: Híbrido AWQ (4-bits) + Compensación LoRA | POR EMPEZAR | 0% |
| **Sprint 5** | Benchmark en Servidor Limitado y Validación Estadística | POR EMPEZAR | 0% |

--------------------------------------------------------------------------------

#### 🏃‍♂️ Planificación Detallada por Sprints
##### Sprint 1: Infraestructura de Datos y Pipeline de Calibración (Semanas 1-2) ✅
*   **Objetivo del Sprint:** Consolidar el entorno de datos local en WSL2 y construir el cargador de tensores que alimentará a los algoritmos avanzados de cuantización offline en la RTX 4060 Ti.
*   **Nota de Arquitectura:** VibeVoice 1.5B procesa **audio crudo monofónico a 24,000 Hz** (no espectrogramas de Mel). La arquitectura usa un autoencoder convolucional puramente temporal con factor de compresión 3200× (7.5 Hz frame rate). Ver `agents.md` sección 3 — Parámetros de Audio del Modelo.
*   **Tareas Asignadas a OpenCode:**
    *  [x]  **TSK-1.1:** Desarrollar el script de descarga automatizada del subconjunto en español de Common Voice 17.0 usando el mirror comunitario `fsicoli/common_voice_17_0` (config `"es"`). El dataset original de Mozilla fue retirado de HuggingFace en Oct 2025. → `notebooks/tesis_model_cuantization.ipynb` Bloque TSK-1.1
    *  [x]  **TSK-1.2:** Implementar la lógica de partición usando los splits nativos de CV17: `train` para fine-tuning, `test` para benchmark, `validation` para calibración. → `data/dataset_partitions.json`
    *  [x]  **TSK-1.3:** Construir el módulo de preprocesamiento acústico: remuestreo de 16kHz → 24kHz, normalización de amplitud a -25 dB FS y prevención de clipping, generando tensores `float32` compatibles con el encoder convolucional del modelo. → `notebooks/tesis_model_cuantization.ipynb` Bloque TSK-1.3
    *  [x]  **TSK-1.4:** Agrupar 512 muestras del 10% de validación en un tensor secuencial unificado (`data/calibration_tensor.pt`) con metadatos de offsets (`data/calibration_metadata.json`) para servir como el **Set de Calibración Offline** requerido por GPTQ y AWQ. → `notebooks/tesis_model_cuantization.ipynb` Bloque TSK-1.4

--------------------------------------------------------------------------------

##### Sprint 1.5: Fine-Tuning Monolingüe Español con LoRA (Semanas 2-3) ✅
*   **Objetivo del Sprint:** Adaptar VibeVoice 1.5B a español exclusivamente mediante fine-tuning eficiente con LoRA sobre el dataset `fsicoli/common_voice_17_0` (config `"es"`, split `train`). El modelo resultante (**VibeVoice-ES**) será la nueva línea base FP16 sobre la cual se aplicarán todas las técnicas de cuantización en los Sprints 2-4. La especialización monolingüe ocurre por *catastrophic forgetting* natural de los otros idiomas al entrenar exclusivamente con datos en español.
*   **Resultado:** Entrenamiento completado (18.6h, 5263 pasos, 1 epoch). CE loss: 1.93→1.72. Merge verificado: LLM LoRA + diffusion head (22/26 params) fusionados correctamente en `weights/vibevoice-1.5b-es/` (10.3 GB, VRAM reposo: 5.04 GB).
*   **Tareas Asignadas a OpenCode:**
    *  [x]  **TSK-1.5.1:** Formatear el dataset CV17 al esquema VibeVoice: 336,846 registros en `data/finetune/cv17_es_train.jsonl` (85 MB). → `notebooks/tesis_model_cuantization.ipynb`
    *  [x]  **TSK-1.5.2:** Configurar y ejecutar fine-tuning con `train_vibevoice.py`. 18.6h en RTX 4060 Ti. Checkpoint en `outputs/finetune_vibevoice_es/`. → `scripts/run_finetune_es.sh`
    *  [x]  **TSK-1.5.3:** Merge de LoRA + diffusion head → **VibeVoice-ES** en `weights/vibevoice-1.5b-es/`. Verificación: 26/26 parámetros correctos. → `scripts/merge_es_checkpoint.sh`
    *  [x]  **TSK-1.5.4:** Validación: VRAM reposo 5.04 GB, tamaño disco 10.3 GB. Margen suficiente para cuantización. → `notebooks/tesis_model_cuantization.ipynb`

--------------------------------------------------------------------------------

##### Sprint 2: Línea Base FP16 ($O_1$) y Baseline Ingenuo (RTN 4-bits) ✅
*   **Objetivo del Sprint:** Documentar métricas de VibeVoice-ES sin compresión y ejecutar cuantización RTN como baseline comparativa.
*   **Resultados:** RTF=1.35 (FP16) / 1.48 (RTN), WER=0.54 / 0.38, CER=0.31 / 0.15. VRAM RTN: +38% (6.97 GB) por buffers FP16 no fusionados de bnb.
*   **Leccion (Directriz 5):** bitsandbytes no fusiona dequantizacion en kernel. Overhead de VRAM lo hace inadecuado para despliegue en 8GB. Util solo como baseline.

--------------------------------------------------------------------------------

##### Sprint 3: Optimización por Hessiana de Segundo Orden (GPTQ) ✅
*   **Resultado final:** 220/222 capas cuantizadas (26 min). GPTQ manual PyTorch puro.
*   **Métricas:** VRAM=6.18 GB, RTF=1.23, WER=0.2746 (multi-sample), CER=0.1774, PPL=12.76 (Clase-3).
*   **Comparativa:** WER mejoro 0.54→0.27 vs FP16. RTF similar (1.23 vs 1.35). VRAM aumento +22% por falta de empaquetado INT4 nativo.
*   **Lecciones:** forward_speech_features con "audio" tiene bug encode()[0][0]; usar "vae" con pre-encoding. Calibracion audio+texto con semantic features reales. damp=0.1 requerido por LoRA.

--------------------------------------------------------------------------------

##### Sprint 4: Enfoque Propuesto - Híbrido AWQ + Compensación LoRA (Semanas 7-8)
*   **Objetivo del Sprint:** Aplicar AWQ sobre VibeVoice-ES COMPLETO (misma estrategia que GPTQ: modelo integrado, calibracion audio+texto, excluir Conv1d) + LoRA de compensacion post-cuantizacion.
*   **Estrategia:** Seguir Fase B del plan de accion en `docs/quantization_architecture_analysis.md`
*   **Tareas Asignadas a OpenCode:**
    *  [ ]  **TSK-4.1:** Instalar y validar la librería `autoawq` interactuando con los núcleos CUDA Ada Lovelace nativos del sistema.
    *  [ ]  **TSK-4.2:** Desarrollar el pipeline de inspección offline que analice la magnitud de las activaciones para aislar los canales de peso salientes.
    *  [ ]  **TSK-4.3:** Aplicar la transformación equivalente y ejecutar compresión a **INT4 (g128)**.
    *  [ ]  **TSK-4.4:** **[IMPLEMENTACIÓN HÍBRIDA]** Desarrollar el mecanismo de adaptación de bajo rango (LoRA / Style Decorator). Cargar el modelo AWQ congelado y aplicar factorizaciones a las capas atencionales para compensar el error de cuantización.
    *  [ ]  **TSK-4.5:** Correr el módulo evaluador de calidad fonética (librería JiWER) y guardar el checkpoint optimizado final.

--------------------------------------------------------------------------------

##### Sprint 5: Consolidación, Benchmark en Servidor y Reporte Estadístico ($O_2$) (Semanas 9-10)
*   **Objetivo del Sprint:** Desplegar los 4 modelos derivados de VibeVoice-ES (FP16, RTN, GPTQ, AWQ+LoRA) en un entorno de servidor limitado, unificar métricas y aplicar pruebas de significancia para el Capítulo 6 del manuscrito.
*   **Tareas Asignadas a OpenCode:**
    *  [ ]  **TSK-5.1:** Escribir y ejecutar el script de inferencia automatizada ("Arena de Pruebas") en el servidor limitado, forzando a los 4 modelos a sintetizar el set de pruebas en español (20% partición `test_indices`).
    *  [ ]  **TSK-5.2:** Utilizar *profilers* (como `psutil`) en el servidor para compilar los logs históricos de inferencia (Latencia, huella de RAM/VRAM, % CPU) en un archivo `metrics_benchmark_report.json`.
    *  [ ]  **TSK-5.3:** Generar la tabla comparativa final y utilizar matplotlib / seaborn para trazar gráficos de dispersión que expongan el *trade-off* entre la velocidad (RTF) y el error lingüístico (WER/CER/PESQ).
    *  [ ]  **TSK-5.4:** **[VALIDACIÓN]** Desarrollar el script estadístico con `scipy.stats`. Aplicar la prueba de normalidad de **Shapiro-Wilk** y la prueba no paramétrica de **Wilcoxon para muestras pareadas** para respaldar la aceptación de la hipótesis ($H_1$) de superioridad del modelo propuesto.

--------------------------------------------------------------------------------

#### 🛠️ Directrices Técnicas para la Ejecución de Tareas
1.  **Defensa de Memoria Local:** Cada tarea que implique cuantización en frío (GPTQ/AWQ) debe precederse obligatoriamente por una limpieza explícita de caché mediante `torch.cuda.empty_cache()` para evitar desbordamientos en la RTX 4060 Ti de 8GB.
2.  **Entorno Separado para Benchmark:** La creación de los modelos se realiza en WSL2 / RTX 4060 Ti. La recolección de métricas del Sprint 5 debe ser ejecutada simulando o usando las restricciones del servidor de inferencia.
3.  **Preservación de Estructuras:** Ningún script generado por OpenCode puede modificar de manera directa los archivos fuente alojados en `VibeVoice_repo/`. Las alteraciones de comportamiento deben inyectarse mediante scripts de inicialización o herencia de clases desde la carpeta `scripts/`.
4.  **Dos LoRAs, dos propósitos distintos:** El proyecto utiliza LoRA en dos momentos independientes que no deben confundirse: (a) **Sprint 1.5 — LoRA de adaptación lingüística:** fine-tuning pre-cuantización sobre datos en español para especializar el modelo; sus pesos se mergean al modelo base. (b) **Sprint 4 — LoRA de compensación:** adaptador post-cuantización AWQ para recuperar error acústico; opera sobre el modelo ya comprimido y congelado.
5.  **SIEMPRE cuantizar VibeVoice completo**, nunca componentes aislados. Ver `docs/quantization_architecture_analysis.md` Directrices 1-8.
6.  **Calibracion con audio real** para GPTQ/AWQ: usar `data/calibration_tensor.pt` + `data/calibration_metadata.json` con training forward (no inference generate).
7.  **damp_percent >= 0.1** para cualquier GPTQ sobre modelo con adaptacion LoRA.
8.  **Excluir Conv1d** de tokenizers acustico/semantico durante cuantizacion.
9.  **bitsandbytes solo para baseline comparativa** — no usar para despliegue en 8GB por overhead de VRAM.