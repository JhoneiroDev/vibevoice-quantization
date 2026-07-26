#!/bin/bash
# ============================================================
# Sprint 1.5: Correccion linguistica VibeVoice-ES (segundo LoRA)
# Hardware: RTX 4060 Ti 8GB VRAM
# Base: VibeVoice-ES existente; diffusion head y connectors congelados
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/alfrog/micromamba/envs/vibevoice/bin/python"
MODEL_PATH="$PROJECT_ROOT/weights/vibevoice-1.5b-es"
TRAIN_JSONL="$PROJECT_ROOT/data/finetune/cv17_es_train.jsonl"
VAL_JSONL="$PROJECT_ROOT/data/finetune/cv17_es_val.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/finetune_vibevoice_es_correction"
cd "$PROJECT_ROOT"

# Inyectar VibeVoice_repo en PYTHONPATH (requerido para importar vibevoice)
export PYTHONPATH="$PROJECT_ROOT/VibeVoice_repo:$PYTHONPATH"

echo "[S1.5-03] Iniciando correccion linguistica VibeVoice-ES"
echo "Modelo base : $MODEL_PATH"
echo "Train JSONL : $TRAIN_JSONL"
echo "Val JSONL   : $VAL_JSONL"
echo "Output      : $OUTPUT_DIR"
echo ""

# Limpiar cache CUDA antes de empezar
$PYTHON_BIN -c "import torch; torch.cuda.empty_cache(); print(f'VRAM libre: {torch.cuda.memory_allocated()/1024**3:.2f} GB')"

$PYTHON_BIN -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path "$MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --validation_jsonl "$VAL_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --text_column_name text \
    --audio_column_name audio \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 2 \
    --prediction_loss_only True \
    --bf16 True \
    --do_train \
    --do_eval \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --ddpm_batch_mul 1 \
    --diffusion_loss_weight 0.0 \
    --train_diffusion_head False \
    --train_connectors False \
    --ce_loss_weight 1.0 \
    --voice_prompt_drop_rate 1.0 \
    --target_audio_dbfs -25.0 \
    --augment_target_silence False \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --max_grad_norm 0.8 \
    --gradient_clipping \
    --seed 42

echo ""
echo "[S1.5-03] Fine-tuning completado"
echo "Checkpoint en: $OUTPUT_DIR"
