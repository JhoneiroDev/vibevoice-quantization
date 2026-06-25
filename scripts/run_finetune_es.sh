#!/bin/bash
# ============================================================
# Sprint 1.5: Fine-Tuning VibeVoice 1.5B -> Espanol (LoRA)
# Hardware: RTX 4060 Ti 8GB VRAM
# Duracion estimada: 7-15 horas (1 epoch, 336K muestras)
# ============================================================
set -e

cd /home/alfrog/projects/VibeVoice_Optimization

# Inyectar VibeVoice_repo en PYTHONPATH (requerido para importar vibevoice)
export PYTHONPATH="/home/alfrog/projects/VibeVoice_Optimization/VibeVoice_repo:$PYTHONPATH"

echo "=== Iniciando Fine-Tuning VibeVoice-ES ==="
echo "Modelo base : /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b"
echo "Train JSONL : /home/alfrog/projects/VibeVoice_Optimization/data/finetune/cv17_es_train.jsonl"
echo "Val JSONL   : /home/alfrog/projects/VibeVoice_Optimization/data/finetune/cv17_es_val.jsonl"
echo "Output      : /home/alfrog/projects/VibeVoice_Optimization/outputs/finetune_vibevoice_es"
echo ""

# Limpiar cache CUDA antes de empezar
/home/alfrog/micromamba/envs/vibevoice/bin/python -c "import torch; torch.cuda.empty_cache(); print(f'VRAM libre: {torch.cuda.memory_allocated()/1024**3:.2f} GB')"

/home/alfrog/micromamba/envs/vibevoice/bin/python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b \
    --train_jsonl /home/alfrog/projects/VibeVoice_Optimization/data/finetune/cv17_es_train.jsonl \
    --validation_jsonl /home/alfrog/projects/VibeVoice_Optimization/data/finetune/cv17_es_val.jsonl \
    --output_dir /home/alfrog/projects/VibeVoice_Optimization/outputs/finetune_vibevoice_es \
    --text_column_name text \
    --audio_column_name audio \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2.5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --bf16 True \
    --do_train \
    --do_eval \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --ddpm_batch_mul 1 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 1.0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --max_grad_norm 0.8 \
    --gradient_clipping \
    --seed 42

echo ""
echo "=== Fine-Tuning completado ==="
echo "Checkpoint en: /home/alfrog/projects/VibeVoice_Optimization/outputs/finetune_vibevoice_es"
