#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_MODEL=${1:-"$ROOT/weights/vibevoice-1.5b-es"}
OUTPUT_DIR=${2:-"$ROOT/weights/vibevoice-1.5b-es-iq4_nl"}
OUTPUT_MODEL="$OUTPUT_DIR/vibevoice-1.5b-iq4_nl.gguf"
MANIFEST="$OUTPUT_DIR/manifest.json"
STAGING_DIR="$OUTPUT_DIR.partial"

CRISP_VERSION=v0.8.23
CRISP_COMMIT=d80ed4b98b52062264c278320b1c0889603f0034
CRISP_SOURCE="$ROOT/.cache/CrispASR-$CRISP_COMMIT"
CRISP_RUNTIME="$ROOT/.cache/crispasr-$CRISP_VERSION-cuda"
RUNTIME_ARCHIVE="$ROOT/.cache/crispasr-linux-x86_64-cuda-$CRISP_VERSION.tar.gz"
RUNTIME_URL="https://github.com/CrispStrobe/CrispASR/releases/download/$CRISP_VERSION/crispasr-linux-x86_64-cuda.tar.gz"
RUNTIME_SHA256=6eda215837da0ffbbd5f362945b0197ea81d9e29b7fd5e9889dc237e222539ac

echo "[S2-02] Construccion selectiva VibeVoice IQ4_NL con CrispASR"
if [[ ! -f "$SOURCE_MODEL/config.json" ]] || ! compgen -G "$SOURCE_MODEL/*.safetensors" >/dev/null; then
    echo "ERROR: falta el checkpoint canonico: $SOURCE_MODEL" >&2
    echo "Complete Sprint 1 y apruebe S1.5-07 antes de Sprint 2." >&2
    exit 2
fi

mkdir -p "$ROOT/.cache" "$(dirname "$OUTPUT_DIR")"

if [[ ! -d "$CRISP_SOURCE/.git" ]]; then
    git clone --recursive https://github.com/CrispStrobe/CrispASR.git "$CRISP_SOURCE"
    git -C "$CRISP_SOURCE" checkout --detach "$CRISP_COMMIT"
    git -C "$CRISP_SOURCE" submodule update --init --recursive
fi
actual_commit=$(git -C "$CRISP_SOURCE" rev-parse HEAD)
if [[ "$actual_commit" != "$CRISP_COMMIT" ]]; then
    echo "ERROR: CrispASR debe estar fijado en $CRISP_COMMIT, encontrado $actual_commit" >&2
    exit 3
fi

if [[ ! -d "$CRISP_RUNTIME" ]]; then
    if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
        curl --fail --location "$RUNTIME_URL" --output "$RUNTIME_ARCHIVE"
    fi
    echo "$RUNTIME_SHA256  $RUNTIME_ARCHIVE" | sha256sum --check --status || {
        echo "ERROR: SHA-256 invalido para el runtime CrispASR" >&2
        exit 4
    }
    mkdir -p "$CRISP_RUNTIME"
    tar -xzf "$RUNTIME_ARCHIVE" -C "$CRISP_RUNTIME"
fi

CRISPASR_BIN=$(find "$CRISP_RUNTIME" -type f -name crispasr -perm -u+x | head -n 1)
QUANTIZE_BIN=$(find "$CRISP_RUNTIME" -type f -name crispasr-quantize -perm -u+x | head -n 1)
if [[ -z "$CRISPASR_BIN" ]] || [[ -z "$QUANTIZE_BIN" ]]; then
    echo "ERROR: el release no contiene crispasr y crispasr-quantize" >&2
    exit 5
fi

if [[ -f "$OUTPUT_MODEL" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "El modelo ya existe; validando sin sobrescribir: $OUTPUT_MODEL"
    "$PYTHON_BIN" "$ROOT/scripts/validate_vibevoice_gguf.py" "$OUTPUT_MODEL" \
        --gguf-python "$CRISP_SOURCE/ggml/python" --manifest "$MANIFEST" --source "$SOURCE_MODEL"
    exit 0
fi

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
STAGING_MODEL="$STAGING_DIR/vibevoice-1.5b-iq4_nl.gguf"
STAGING_MANIFEST="$STAGING_DIR/manifest.json"
F16_MODEL="$STAGING_DIR/.vibevoice-1.5b-es-f16.gguf"
PARTIAL_MODEL="$STAGING_MODEL.partial"
cleanup() { rm -f "$F16_MODEL" "$PARTIAL_MODEL"; }
trap cleanup EXIT

"$PYTHON_BIN" "$CRISP_SOURCE/models/convert-vibevoice-to-gguf.py" \
    --input "$SOURCE_MODEL" --output "$F16_MODEL" --include-decoder

env -u CRISPASR_VIBEVOICE_QUANT_ALL -u CRISPASR_QUANT_LMHEAD \
    "$QUANTIZE_BIN" "$F16_MODEL" "$PARTIAL_MODEL" iq4_nl \
    --tensor-type '^lm\.layers\.[0-9]+\.(attn\.(q_proj|k_proj|v_proj|o_proj)|ffn\.(gate|up|down))\.weight$=iq4_nl' \
    --tensor-type '^(at_enc|at_dec|st_enc|st_dec)\..*dw_conv\.weight$=f32' \
    --tensor-type '^lm\.(tok_emb|lm_head|output)\..*=f16' \
    --tensor-type '^(at_enc|at_dec|st_enc|st_dec|at_conn|se_conn|pred|tts_eos|tts_types|tts_lm)\..*\.weight$=f16'

"$PYTHON_BIN" "$ROOT/scripts/validate_vibevoice_gguf.py" "$PARTIAL_MODEL" \
    --gguf-python "$CRISP_SOURCE/ggml/python" --source "$SOURCE_MODEL"
mv "$PARTIAL_MODEL" "$STAGING_MODEL"
"$PYTHON_BIN" "$ROOT/scripts/validate_vibevoice_gguf.py" "$STAGING_MODEL" \
    --gguf-python "$CRISP_SOURCE/ggml/python" --manifest "$STAGING_MANIFEST" --source "$SOURCE_MODEL"

rm -f "$F16_MODEL"
printf '%s\n' "$CRISPASR_BIN" > "$STAGING_DIR/crispasr-runtime.txt"
rm -rf "$OUTPUT_DIR"
mv "$STAGING_DIR" "$OUTPUT_DIR"
echo "Modelo guardado: $OUTPUT_MODEL"
