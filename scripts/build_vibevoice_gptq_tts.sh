#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}
SOURCE="$ROOT/weights/vibevoice-1.5b-es"
OUTPUT="$ROOT/weights/vibevoice-1.5b-es-gptq-tts"
METADATA="$ROOT/data/calibration_metadata.json"
STAGING="$OUTPUT.partial"

echo "[S3R-02] GPTQ W4 g128 con calibracion TTS-prefill real"
if [[ -e "$OUTPUT" ]]; then
    [[ -d "$OUTPUT" ]] || { echo "La salida GPTQ-TTS no es un directorio: $OUTPUT" >&2; exit 1; }
    "$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" validate --output "$OUTPUT"
    echo "Modelo GPTQ-TTS existente validado: $OUTPUT"
    exit 0
fi
if [[ ! -e "$STAGING" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" prepare \
        --source "$SOURCE" --output "$STAGING" --metadata "$METADATA" \
        --tts-prefill --desc-act
elif [[ ! -d "$STAGING" ]]; then
    echo "El staging GPTQ-TTS no es un directorio: $STAGING" >&2
    exit 1
else
    echo "Reanudando staging GPTQ-TTS existente: $STAGING"
fi
if [[ ! -d "$STAGING/decoder-gptq" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" quantize --output "$STAGING"
fi
"$PYTHON_BIN" "$ROOT/scripts/gptq_vibevoice.py" validate --output "$STAGING" --remove-work
[[ ! -e "$OUTPUT" ]] || { echo "La salida GPTQ-TTS aparecio durante el build: $OUTPUT" >&2; exit 1; }
mv "$STAGING" "$OUTPUT"
echo "Modelo GPTQ-TTS guardado y validado estructuralmente: $OUTPUT"
