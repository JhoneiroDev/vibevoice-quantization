"""Insert Sprint 1.5 cells into the notebook before the Sprint 1 summary."""
import json

NB_PATH = "/home/alfrog/projects/VibeVoice_Optimization/notebooks/tesis_model_cuantization.ipynb"
with open(NB_PATH) as f:
    nb = json.load(f)

sprint15_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "id": "sprint15-header",
        "metadata": {},
        "source": [
            "---\n",
            "# Sprint 1.5: Fine-Tuning Monolingüe Español con LoRA\n",
            "\n",
            "**Objetivo:** Adaptar VibeVoice 1.5B a español exclusivamente mediante LoRA fine-tuning\n",
            "sobre el split `train` de Common Voice 17.0. El modelo resultante (**VibeVoice-ES**)\n",
            "será la nueva línea base FP16 para todos los experimentos de cuantización.\n",
            "\n",
            "**Estrategia:**\n",
            "- LoRA rank=8 sobre proyecciones Q/K/V/O + gate/up/down del LLM (Qwen2.5)\n",
            "- Entrenamiento completo del *diffusion head* para adaptar prosodia española\n",
            "- Tokenizadores acústico y semántico congelados (lenguaje-agnósticos)\n",
            "- `voice_prompt_drop_rate=1.0` para priorizar síntesis sobre clonación\n",
            "- `per_device_train_batch_size=1`, `gradient_accumulation_steps=64` para 8GB VRAM\n",
            "\n",
            "> **Nota:** El entrenamiento ejecuta en terminal vía `python -m vibevoice.finetune.train_vibevoice`.\n",
            "> Las celdas de este bloque preparan los datos y generan el script de lanzamiento.\n",
            "> La ejecución del entrenamiento (~7-15h) se realiza fuera del notebook."
        ]
    },
    # TSK-1.5.1: Format dataset markdown
    {
        "cell_type": "markdown",
        "id": "tsk151-md",
        "metadata": {},
        "source": [
            "---\n",
            "## TSK-1.5.1: Formatear Dataset al Esquema VibeVoice\n",
            "\n",
            "El script de fine-tuning espera:\n",
            "- Columna `text` con formato `\"Speaker 1: {sentence}\"`\n",
            "- Columna `audio` como path a archivo de audio (str) o dict con array+sampling_rate\n",
            "- Columna `voice_prompts` opcional (se auto-genera si no existe)\n",
            "\n",
            "Guardamos los datos formateados como JSONL para pasarlos con `--train_jsonl` / `--validation_jsonl`.\n",
            "Los paths de audio apuntan a la caché local de HuggingFace (archivos MP3 descargados)."
        ]
    },
    # TSK-1.5.1 code cell
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "format-dataset-jsonl",
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "FINETUNE_DATA_DIR = DATA_DIR / \"finetune\"\n",
            "FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(\"Formateando train split a JSONL...\")\n",
            "train_jsonl_path = FINETUNE_DATA_DIR / \"cv17_es_train.jsonl\"\n",
            "\n",
            "with open(train_jsonl_path, \"w\", encoding=\"utf-8\") as f:\n",
            "    for i in tqdm(range(len(train_dataset)), desc=\"Escribiendo train JSONL\"):\n",
            "        sample = train_dataset[i]\n",
            "        record = {\n",
            "            \"text\": f\"Speaker 1: {sample['sentence']}\",\n",
            "            \"audio\": sample[\"audio\"][\"path\"],\n",
            "        }\n",
            "        f.write(json.dumps(record, ensure_ascii=False) + \"\\n\")\n",
            "\n",
            "print(f\"Train JSONL guardado: {train_jsonl_path}\")\n",
            "print(f\"Tamaño: {train_jsonl_path.stat().st_size / (1024**2):.1f} MB\")\n",
            "\n",
            "# --- Validation subset (2K del train para monitoreo) ---\n",
            "print(\"\\nCreando validation JSONL (muestra de 2000 registros)...\")\n",
            "val_jsonl_path = FINETUNE_DATA_DIR / \"cv17_es_val.jsonl\"\n",
            "N_VAL = min(2000, len(train_dataset))\n",
            "val_indices_sample = sorted(random.sample(range(len(train_dataset)), N_VAL))\n",
            "\n",
            "with open(val_jsonl_path, \"w\", encoding=\"utf-8\") as f:\n",
            "    for idx in tqdm(val_indices_sample, desc=\"Escribiendo val JSONL\"):\n",
            "        sample = train_dataset[idx]\n",
            "        record = {\n",
            "            \"text\": f\"Speaker 1: {sample['sentence']}\",\n",
            "            \"audio\": sample[\"audio\"][\"path\"],\n",
            "        }\n",
            "        f.write(json.dumps(record, ensure_ascii=False) + \"\\n\")\n",
            "\n",
            "print(f\"Validation JSONL guardado: {val_jsonl_path}\")\n",
            "print(f\"Tamaño: {val_jsonl_path.stat().st_size / (1024**2):.1f} MB\")"
        ]
    },
    # TSK-1.5.2: Training config markdown
    {
        "cell_type": "markdown",
        "id": "tsk152-md",
        "metadata": {},
        "source": [
            "---\n",
            "## TSK-1.5.2: Script de Fine-Tuning para 8GB VRAM\n",
            "\n",
            "Parámetros optimizados para RTX 4060 Ti (8GB):\n",
            "\n",
            "| Parámetro | Valor | Justificación |\n",
            "|-----------|-------|---------------|\n",
            "| `per_device_train_batch_size` | 1 | Limitar VRAM de activaciones |\n",
            "| `gradient_accumulation_steps` | 64 | Batch efectivo = 64 |\n",
            "| `bf16` | True | Mixed precision (Ada Lovelace) |\n",
            "| `ddpm_batch_mul` | 1 | Mínimo; sin amplificación de difusión |\n",
            "| `lora_r` | 8 | Rank estándar para adaptación |\n",
            "| `learning_rate` | 2.5e-5 | Recomendado por la comunidad |\n",
            "| `num_train_epochs` | 1 | ~5,263 pasos (~7-15h) |\n",
            "| `voice_prompt_drop_rate` | 1.0 | Drop siempre; prioriza síntesis |\n",
            "\n",
            "El script generado se guarda en `scripts/run_finetune_es.sh`."
        ]
    },
    # TSK-1.5.2 code cell - generate training script
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "generate-train-script",
        "metadata": {},
        "outputs": [],
        "source": [
            "SCRIPTS_DIR = PROJECT_ROOT / \"scripts\"\n",
            "SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "VENV_PYTHON = \"/home/alfrog/micromamba/envs/vibevoice/bin/python\"\n",
            "MODEL_PATH = str(WEIGHTS_DIR / \"vibevoice-1.5b\")\n",
            "OUTPUT_DIR = str(PROJECT_ROOT / \"outputs\" / \"finetune_vibevoice_es\")\n",
            "TRAIN_JSONL = str(FINETUNE_DATA_DIR / \"cv17_es_train.jsonl\")\n",
            "VAL_JSONL = str(FINETUNE_DATA_DIR / \"cv17_es_val.jsonl\")\n",
            "\n",
            "train_script = f\"\"\"#!/bin/bash\n",
            "# ============================================================\n",
            "# Sprint 1.5: Fine-Tuning VibeVoice 1.5B -> Espanol (LoRA)\n",
            "# Hardware: RTX 4060 Ti 8GB VRAM\n",
            "# Duracion estimada: 7-15 horas (1 epoch, 336K muestras)\n",
            "# ============================================================\n",
            "set -e\n",
            "\n",
            "cd {PROJECT_ROOT}\n",
            "\n",
            "echo \"=== Iniciando Fine-Tuning VibeVoice-ES ===\"\n",
            "echo \"Modelo base : {MODEL_PATH}\"\n",
            "echo \"Train JSONL : {TRAIN_JSONL}\"\n",
            "echo \"Val JSONL   : {VAL_JSONL}\"\n",
            "echo \"Output      : {OUTPUT_DIR}\"\n",
            "echo \"\"\n",
            "\n",
            "# Limpiar cache CUDA antes de empezar\n",
            "{VENV_PYTHON} -c \"import torch; torch.cuda.empty_cache(); print(f'VRAM libre: {{torch.cuda.memory_allocated()/1024**3:.2f}} GB')\"\n",
            "\n",
            "{VENV_PYTHON} -m vibevoice.finetune.train_vibevoice \\\\\n",
            "    --model_name_or_path {MODEL_PATH} \\\\\n",
            "    --train_jsonl {TRAIN_JSONL} \\\\\n",
            "    --validation_jsonl {VAL_JSONL} \\\\\n",
            "    --output_dir {OUTPUT_DIR} \\\\\n",
            "    --text_column_name text \\\\\n",
            "    --audio_column_name audio \\\\\n",
            "    --per_device_train_batch_size 1 \\\\\n",
            "    --gradient_accumulation_steps 64 \\\\\n",
            "    --learning_rate 2.5e-5 \\\\\n",
            "    --lr_scheduler_type cosine \\\\\n",
            "    --warmup_ratio 0.03 \\\\\n",
            "    --num_train_epochs 1 \\\\\n",
            "    --logging_steps 50 \\\\\n",
            "    --save_steps 500 \\\\\n",
            "    --eval_steps 500 \\\\\n",
            "    --save_total_limit 3 \\\\\n",
            "    --bf16 True \\\\\n",
            "    --do_train \\\\\n",
            "    --do_eval \\\\\n",
            "    --dataloader_num_workers 2 \\\\\n",
            "    --remove_unused_columns False \\\\\n",
            "    --gradient_checkpointing False \\\\\n",
            "    --ddpm_batch_mul 1 \\\\\n",
            "    --diffusion_loss_weight 1.4 \\\\\n",
            "    --train_diffusion_head True \\\\\n",
            "    --ce_loss_weight 0.04 \\\\\n",
            "    --voice_prompt_drop_rate 1.0 \\\\\n",
            "    --lora_r 8 \\\\\n",
            "    --lora_alpha 32 \\\\\n",
            "    --lora_dropout 0.05 \\\\\n",
            "    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\\\\n",
            "    --max_grad_norm 0.8 \\\\\n",
            "    --gradient_clipping \\\\\n",
            "    --seed 42\n",
            "\n",
            "echo \"\"\n",
            "echo \"=== Fine-Tuning completado ===\"\n",
            "echo \"Checkpoint en: {OUTPUT_DIR}\"\n",
            "\"\"\"\n",
            "\n",
            "script_path = SCRIPTS_DIR / \"run_finetune_es.sh\"\n",
            "script_path.write_text(train_script)\n",
            "script_path.chmod(0o755)\n",
            "\n",
            "print(f\"Script de entrenamiento generado: {script_path}\")\n",
            "print(f\"\\nPara ejecutar:\\n  bash {script_path}\")"
        ]
    },
    # TSK-1.5.3: Merge markdown
    {
        "cell_type": "markdown",
        "id": "tsk153-md",
        "metadata": {},
        "source": [
            "---\n",
            "## TSK-1.5.3: Merge de LoRA al Modelo Base\n",
            "\n",
            "Al finalizar el fine-tuning, los adaptadores LoRA y el diffusion head entrenado\n",
            "deben fusionarse al modelo base para obtener el checkpoint **VibeVoice-ES** en\n",
            "`weights/vibevoice-1.5b-es/`. Se usa el script nativo `merge_vibevoice_models.py`."
        ]
    },
    # TSK-1.5.3 code cell
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "generate-merge-script",
        "metadata": {},
        "outputs": [],
        "source": [
            "MERGE_OUTPUT = str(WEIGHTS_DIR / \"vibevoice-1.5b-es\")\n",
            "\n",
            "merge_script = f\"\"\"#!/bin/bash\n",
            "# ============================================================\n",
            "# Sprint 1.5: Merge LoRA -> VibeVoice-ES\n",
            "# ============================================================\n",
            "set -e\n",
            "\n",
            "cd {PROJECT_ROOT}\n",
            "\n",
            "echo \"=== Mergeando LoRA + Diffusion Head al modelo base ===\"\n",
            "echo \"Checkpoint LoRA : {OUTPUT_DIR}\"\n",
            "echo \"Modelo base     : {MODEL_PATH}\"\n",
            "echo \"Output (ES)     : {MERGE_OUTPUT}\"\n",
            "echo \"\"\n",
            "\n",
            "{VENV_PYTHON} -m vibevoice.scripts.merge_vibevoice_models \\\\\n",
            "    --lora_model_name_or_path {OUTPUT_DIR} \\\\\n",
            "    --base_model_name_or_path {MODEL_PATH} \\\\\n",
            "    --output_dir {MERGE_OUTPUT} \\\\\n",
            "    --tokenizer_name {MODEL_PATH}\n",
            "\n",
            "echo \"\"\n",
            "echo \"=== VibeVoice-ES generado en: {MERGE_OUTPUT} ===\"\n",
            "ls -lh {MERGE_OUTPUT}/*.safetensors 2>/dev/null || echo \"(verificar estructura)\"\n",
            "\"\"\"\n",
            "\n",
            "merge_script_path = SCRIPTS_DIR / \"merge_es_checkpoint.sh\"\n",
            "merge_script_path.write_text(merge_script)\n",
            "merge_script_path.chmod(0o755)\n",
            "\n",
            "print(f\"Script de merge generado: {merge_script_path}\")\n",
            "print(f\"\\nPara ejecutar (despues del entrenamiento):\\n  bash {merge_script_path}\")"
        ]
    },
    # TSK-1.5.4: Validation markdown
    {
        "cell_type": "markdown",
        "id": "tsk154-md",
        "metadata": {},
        "source": [
            "---\n",
            "## TSK-1.5.4: Validación del Modelo Fine-Tuneado\n",
            "\n",
            "Verificamos que **VibeVoice-ES** carga correctamente en VRAM y registramos\n",
            "el nuevo consumo base. La evaluación completa de calidad se realiza en Sprint 2."
        ]
    },
    # TSK-1.5.4 code cell
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "validate-finetuned",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, os\n",
            "repo_path = os.path.abspath(\"../VibeVoice_repo\")\n",
            "if repo_path not in sys.path:\n",
            "    sys.path.insert(0, repo_path)\n",
            "\n",
            "import torch\n",
            "import transformers\n",
            "if hasattr(transformers, \"CONFIG_MAPPING\"):\n",
            "    transformers.CONFIG_MAPPING._extra_content.pop(\"vibevoice_acoustic_tokenizer\", None)\n",
            "    if hasattr(transformers.CONFIG_MAPPING, \"_mapping\"):\n",
            "        transformers.CONFIG_MAPPING._mapping.pop(\"vibevoice_acoustic_tokenizer\", None)\n",
            "\n",
            "from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration\n",
            "\n",
            "MODEL_ES_PATH = \"../weights/vibevoice-1.5b-es\"\n",
            "\n",
            "print(\"Cargando VibeVoice-ES...\")\n",
            "torch.cuda.empty_cache()\n",
            "model_es = VibeVoiceForConditionalGeneration.from_pretrained(\n",
            "    MODEL_ES_PATH,\n",
            "    torch_dtype=torch.float16,\n",
            "    device_map=\"auto\",\n",
            "    attn_implementation=\"sdpa\",\n",
            ")\n",
            "\n",
            "vram_es = torch.cuda.memory_allocated() / 1024**3\n",
            "print(f\"VibeVoice-ES cargado. VRAM: {vram_es:.2f} GB\")\n",
            "\n",
            "# --- Verificacion de tamano en disco ---\n",
            "import pathlib\n",
            "model_size_mb = sum(f.stat().st_size for f in pathlib.Path(MODEL_ES_PATH).rglob(\"*.safetensors\")) / (1024**2)\n",
            "print(f\"Tamano en disco (VibeVoice-ES): {model_size_mb:.1f} MB\")\n",
            "\n",
            "if vram_es < 6.0:\n",
            "    print(\"VRAM OK - margen suficiente para cuantizacion.\")\n",
            "else:\n",
            "    print(\"ADVERTENCIA: VRAM elevada. Verificar que el merge fue exitoso.\")"
        ]
    },
]

# Find summary cell index
summary_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "summary-sprint1":
        summary_idx = i
        break

if summary_idx:
    nb["cells"] = nb["cells"][:summary_idx] + sprint15_cells + nb["cells"][summary_idx:]
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Inserted {len(sprint15_cells)} Sprint 1.5 cells before summary. Total cells: {len(nb['cells'])}")
else:
    print("ERROR: summary cell not found")
