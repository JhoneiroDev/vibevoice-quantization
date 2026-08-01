#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}
SOURCE=${1:-"$ROOT/weights/vibevoice-1.5b-es"}
OUTPUT=${2:-"$ROOT/weights/vibevoice-1.5b-es-gptq"}
METADATA=${3:-"$ROOT/data/calibration_metadata.json"}
STAGING="$OUTPUT.partial"

echo "[S3-02] GPTQ W4 g128 selectivo sobre Qwen2"
if [[ -d "$OUTPUT" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" validate --output "$OUTPUT" --remove-work
    echo "Modelo GPTQ existente validado: $OUTPUT"
    exit 0
fi
rm -rf "$STAGING"
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" prepare \
    --source "$SOURCE" --output "$STAGING" --metadata "$METADATA"
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" quantize --output "$STAGING"
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" validate --output "$STAGING" --remove-work
rm -rf "$OUTPUT"
mv "$STAGING" "$OUTPUT"
echo "Modelo GPTQ guardado y validado estructuralmente: $OUTPUT"
