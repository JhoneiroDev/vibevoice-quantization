#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}

echo "[S4-00] Verificando AutoAWQ GEMM/Triton"
"$PYTHON_BIN" -m pip install --no-deps "autoawq==0.2.9"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import torch
import transformers
import triton
import awq
from awq.modules.linear.gemm import WQLinear_GEMM

assert awq.__version__ == "0.2.9", awq.__version__
assert transformers.__version__ == "4.51.3", transformers.__version__
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 9)
assert importlib.util.find_spec("awq_ext") is None, (
    "Sprint 4 pins the verified Triton path; remove incompatible awq_ext kernels"
)
assert WQLinear_GEMM is not None and triton.__version__
print(f"AutoAWQ {awq.__version__}, Triton {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}, SM{torch.cuda.get_device_capability(0)}")
PY
"$PYTHON_BIN" -m pip check
