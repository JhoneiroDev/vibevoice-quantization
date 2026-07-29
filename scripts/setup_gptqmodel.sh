#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}

echo "[S3-00] Instalando GPTQModel compatible con VibeVoice"
"$PYTHON_BIN" -m pip install \
    "device-smi==0.4.1" "random-word==1.0.13" "tokenicer==0.0.4" \
    "logbar==0.0.4" "hf-transfer>=0.1.9"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation "gptqmodel==2.2.0"

"$PYTHON_BIN" - <<'PY'
import torch
import transformers
import gptqmodel
from gptqmodel.utils.backend import BACKEND

assert transformers.__version__ == "4.51.3", transformers.__version__
assert gptqmodel.__version__ == "2.2.0", gptqmodel.__version__
assert torch.cuda.is_available()
assert hasattr(BACKEND, "TRITON")
print(f"GPTQModel {gptqmodel.__version__}, Transformers {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

"$PYTHON_BIN" -m pip check
