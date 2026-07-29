#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}
SOURCE=${1:-"$ROOT/weights/vibevoice-1.5b-es-corrected"}
OUTPUT=${2:-"$ROOT/weights/vibevoice-1.5b-es-gptq"}
METADATA=${3:-"$ROOT/data/calibration_metadata.json"}

echo "[S3-02] GPTQ W4 g128 selectivo sobre Qwen2"
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" prepare \
    --source "$SOURCE" --output "$OUTPUT" --metadata "$METADATA" --force
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" quantize --output "$OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" validate --output "$OUTPUT" --remove-work
echo "Modelo GPTQ guardado y validado estructuralmente: $OUTPUT"
