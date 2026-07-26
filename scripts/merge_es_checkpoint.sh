#!/bin/bash
# ============================================================
# Sprint 1.5: Merge LoRA correctivo -> VibeVoice-ES corregido
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/alfrog/micromamba/envs/vibevoice/bin/python"
BASE_MODEL="$PROJECT_ROOT/weights/vibevoice-1.5b-es"
CHECKPOINT="$PROJECT_ROOT/outputs/finetune_vibevoice_es_correction/lora"
MERGE_OUTPUT="$PROJECT_ROOT/weights/vibevoice-1.5b-es-corrected"
cd "$PROJECT_ROOT"

# Inyectar VibeVoice_repo en PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/VibeVoice_repo:$PYTHONPATH"

echo "[S1.5-05] Mergeando LoRA correctivo sobre VibeVoice-ES"
echo "Checkpoint LoRA : $CHECKPOINT"
echo "Modelo base     : $BASE_MODEL"
echo "Output (ES)     : $MERGE_OUTPUT"
echo ""

$PYTHON_BIN -m vibevoice.scripts.merge_vibevoice_models \
    --base_model_path "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT" \
    --output_path "$MERGE_OUTPUT" \
    --output_format safetensors \
    --output_dtype bfloat16

echo ""
echo "[S1.5-05] VibeVoice-ES corregido en: $MERGE_OUTPUT"
ls -lh "$MERGE_OUTPUT"/*.safetensors 2>/dev/null || echo "(verificar estructura)"
