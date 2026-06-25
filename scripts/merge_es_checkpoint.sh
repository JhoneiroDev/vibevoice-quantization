#!/bin/bash
# ============================================================
# Sprint 1.5: Merge LoRA -> VibeVoice-ES
# ============================================================
set -e

cd /home/alfrog/projects/VibeVoice_Optimization

# Inyectar VibeVoice_repo en PYTHONPATH
export PYTHONPATH="/home/alfrog/projects/VibeVoice_Optimization/VibeVoice_repo:$PYTHONPATH"

echo "=== Mergeando LoRA + Diffusion Head al modelo base ==="
echo "Checkpoint LoRA : /home/alfrog/projects/VibeVoice_Optimization/outputs/finetune_vibevoice_es"
echo "Modelo base     : /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b"
echo "Output (ES)     : /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b-es"
echo ""

/home/alfrog/micromamba/envs/vibevoice/bin/python -m vibevoice.scripts.merge_vibevoice_models \
    --base_model_path /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b \
    --checkpoint_path /home/alfrog/projects/VibeVoice_Optimization/outputs/finetune_vibevoice_es/lora \
    --output_path /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b-es \
    --output_format safetensors

echo ""
echo "=== VibeVoice-ES generado en: /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b-es ==="
ls -lh /home/alfrog/projects/VibeVoice_Optimization/weights/vibevoice-1.5b-es/*.safetensors 2>/dev/null || echo "(verificar estructura)"
