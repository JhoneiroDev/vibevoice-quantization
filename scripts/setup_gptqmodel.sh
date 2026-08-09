#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}

echo "[S3-00] Instalando GPTQModel compatible con VibeVoice"
"$PYTHON_BIN" -m pip install \
    "device-smi==0.4.1" "random-word==1.0.13" "tokenicer==0.0.4" \
    "logbar==0.0.4" "hf-transfer>=0.1.9" \
    "accelerate>=1.3.0,<2" "datasets>=3.2.0,<4" "safetensors>=0.5.2" \
    "huggingface-hub>=0.28.1" "threadpoolctl>=3.6.0" \
    "protobuf>=5.29.3" "pillow>=11.1.0" "triton==3.7.0"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation "gptqmodel==2.2.0"

"$PYTHON_BIN" "$(dirname "$0")/gptq_vibevoice.py" preflight

"$PYTHON_BIN" -m pip check
