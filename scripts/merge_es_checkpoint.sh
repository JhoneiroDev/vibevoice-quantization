#!/bin/bash
# ============================================================
# Sprint 1.5: Merge de la primera adaptacion -> VibeVoice-ES
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-${PYTHON_BIN:-python}}"
BASE_MODEL="$PROJECT_ROOT/weights/vibevoice-1.5b"
CHECKPOINT="$PROJECT_ROOT/outputs/finetune_vibevoice_es/lora"
MERGE_OUTPUT="$PROJECT_ROOT/weights/vibevoice-1.5b-es"
STAGING_OUTPUT="$MERGE_OUTPUT.partial"
cd "$PROJECT_ROOT"

# Inyectar VibeVoice_repo en PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/VibeVoice_repo:$PYTHONPATH"

echo "[S1.5-05] Mergeando LoRA y diffusion head sobre VibeVoice base"
echo "Checkpoint LoRA : $CHECKPOINT"
echo "Modelo base     : $BASE_MODEL"
echo "Output (ES)     : $MERGE_OUTPUT"
echo ""

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "ERROR: falta el modelo base en $BASE_MODEL" >&2
    exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: falta el checkpoint de adaptacion en $CHECKPOINT" >&2
    echo "Ejecute primero: bash scripts/run_finetune_es.sh" >&2
    exit 1
fi
if [[ -f "$MERGE_OUTPUT/config.json" ]] && compgen -G "$MERGE_OUTPUT/*.safetensors" > /dev/null; then
    echo "Checkpoint VibeVoice-ES ya existente y completo: $MERGE_OUTPUT"
    exit 0
fi
if [[ -e "$MERGE_OUTPUT" ]]; then
    echo "ERROR: existe un checkpoint VibeVoice-ES incompleto: $MERGE_OUTPUT" >&2
    echo "Retirelo o inspeccionelo antes de reintentar; no se sobrescribira automaticamente." >&2
    exit 1
fi
rm -rf "$STAGING_OUTPUT"

$PYTHON_BIN -m vibevoice.scripts.merge_vibevoice_models \
    --base_model_path "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT" \
    --output_path "$STAGING_OUTPUT" \
    --output_format safetensors \
    --output_dtype float32

if [[ ! -f "$STAGING_OUTPUT/config.json" ]] || ! compgen -G "$STAGING_OUTPUT/*.safetensors" > /dev/null; then
    echo "ERROR: el merge no produjo un checkpoint completo en $STAGING_OUTPUT" >&2
    exit 1
fi
mv "$STAGING_OUTPUT" "$MERGE_OUTPUT"

echo ""
echo "[S1.5-05] VibeVoice-ES generado en: $MERGE_OUTPUT"
ls -lh "$MERGE_OUTPUT"/*.safetensors 2>/dev/null || echo "(verificar estructura)"
