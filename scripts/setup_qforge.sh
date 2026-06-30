#!/bin/bash
# Setup qforge environment for GPTQModel + autoawq
set -e

echo "=== Creando entorno qforge ==="
micromamba create -n qforge python=3.12 -y

echo ""
echo "=== Instalando GPTQModel + autoawq ==="
micromamba run -n qforge pip install gptqmodel autoawq datasets

echo ""
echo "=== Verificando ==="
micromamba run -n qforge python -c "
import torch; print(f'Torch: {torch.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
from gptqmodel import GPTQModel; print('GPTQModel: OK')
from awq import AutoAWQForCausalLM; print('AutoAWQ: OK')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"

echo ""
echo "=== qforge listo ==="
