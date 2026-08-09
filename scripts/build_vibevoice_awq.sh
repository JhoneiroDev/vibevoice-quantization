#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}
SOURCE=${1:-"$ROOT/weights/vibevoice-1.5b-es"}
OUTPUT=${2:-"$ROOT/weights/vibevoice-1.5b-es-awq"}
METADATA=${3:-"$ROOT/data/calibration_metadata.json"}
STAGING="$OUTPUT.partial"
CANONICAL_SOURCE="$ROOT/weights/vibevoice-1.5b-es"
CANONICAL_OUTPUT="$ROOT/weights/vibevoice-1.5b-es-awq"
CANONICAL_METADATA="$ROOT/data/calibration_metadata.json"

[[ "$(realpath -m "$SOURCE")" == "$CANONICAL_SOURCE" ]] || {
    echo "La fuente AWQ debe ser el checkpoint canonico: $CANONICAL_SOURCE" >&2
    exit 1
}
[[ "$(realpath -m "$OUTPUT")" == "$CANONICAL_OUTPUT" ]] || {
    echo "La salida AWQ debe ser la ruta canonica: $CANONICAL_OUTPUT" >&2
    exit 1
}
[[ "$(realpath -m "$METADATA")" == "$CANONICAL_METADATA" ]] || {
    echo "La metadata AWQ debe ser la reservada: $CANONICAL_METADATA" >&2
    exit 1
}

echo "[S4-02] AWQ W4A16 asimetrico g128 selectivo sobre Qwen2"
if [[ -e "$OUTPUT" ]]; then
    [[ -d "$OUTPUT" ]] || { echo "La salida AWQ no es un directorio: $OUTPUT" >&2; exit 1; }
    "$PYTHON_BIN" "$ROOT/scripts/awq_vibevoice.py" validate --output "$OUTPUT" --verify-source
    echo "Modelo AWQ existente validado: $OUTPUT"
    exit 0
fi
if [[ ! -e "$STAGING" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/awq_vibevoice.py" prepare \
        --source "$SOURCE" --output "$STAGING" --metadata "$METADATA"
elif [[ ! -d "$STAGING" ]]; then
    echo "El staging AWQ no es un directorio: $STAGING" >&2
    exit 1
elif [[ ! -f "$STAGING/manifest.json" ]]; then
    echo "El staging AWQ esta incompleto y requiere revision manual: falta manifest.json" >&2
    exit 1
else
    echo "Reanudando staging AWQ existente: $STAGING"
fi
if [[ ! -d "$STAGING/decoder-awq" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/awq_vibevoice.py" quantize --output "$STAGING"
elif [[ ! -f "$STAGING/decoder-awq/config.json" ]] || \
     { [[ ! -f "$STAGING/decoder-awq/model.safetensors" ]] && \
       [[ ! -f "$STAGING/decoder-awq/model.safetensors.index.json" ]]; }; then
    echo "El decoder AWQ del staging esta incompleto y requiere revision manual" >&2
    exit 1
fi
"$PYTHON_BIN" "$ROOT/scripts/awq_vibevoice.py" validate \
    --output "$STAGING" --verify-source --remove-work
[[ ! -e "$OUTPUT" ]] || { echo "La salida AWQ aparecio durante el build: $OUTPUT" >&2; exit 1; }
mv "$STAGING" "$OUTPUT"
echo "Modelo AWQ guardado y validado estructuralmente: $OUTPUT"
