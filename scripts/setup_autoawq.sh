#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/alfrog/micromamba/envs/vibevoice/bin/python}

echo "[S4-00] Verificando AutoAWQ GEMM/Triton"
"$PYTHON_BIN" -m pip install --no-deps "autoawq==0.2.9"
"$PYTHON_BIN" "$(dirname "$0")/awq_vibevoice.py" preflight
"$PYTHON_BIN" -m pip check
