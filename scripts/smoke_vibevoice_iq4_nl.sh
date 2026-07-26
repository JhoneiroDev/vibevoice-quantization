#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-"$ROOT/weights/vibevoice-1.5b-es-iq4_nl/vibevoice-1.5b-iq4_nl.gguf"}
OUTPUT=${2:-"$ROOT/outputs/sprint2_iq4_nl/smoke-es.wav"}
VOICE=${3:-"$ROOT/VibeVoice_repo/demo/voices/en-Alice_woman.wav"}
RUNTIME_FILE="$(dirname "$MODEL")/crispasr-runtime.txt"

echo "[S2-03] Smoke test nativo de VibeVoice IQ4_NL"
[[ -f "$MODEL" ]] || { echo "ERROR: no existe $MODEL" >&2; exit 2; }
[[ -f "$RUNTIME_FILE" ]] || { echo "ERROR: no existe $RUNTIME_FILE" >&2; exit 3; }
[[ -f "$VOICE" ]] || { echo "ERROR: no existe $VOICE" >&2; exit 4; }
CRISPASR_BIN=$(<"$RUNTIME_FILE")
[[ -x "$CRISPASR_BIN" ]] || { echo "ERROR: runtime no ejecutable: $CRISPASR_BIN" >&2; exit 5; }
mkdir -p "$(dirname "$OUTPUT")"

CRISPASR_VIBEVOICE_VOICE_AUDIO="$VOICE" "$CRISPASR_BIN" \
    --backend vibevoice-1.5b -m "$MODEL" \
    --tts "Hola, esta es una prueba de sintesis de voz en espanol." \
    --tts-output "$OUTPUT" --seed 42

[[ -s "$OUTPUT" ]] || { echo "ERROR: no se genero audio" >&2; exit 6; }
echo "Audio guardado: $OUTPUT"
