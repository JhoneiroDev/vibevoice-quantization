import torch
import bitsandbytes as bnb
try:
    from flash_attn import flash_attn_func
    flash_ok = True
except ImportError:
    flash_ok = False

print(f"--- Diagnóstico de Hardware ---")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Flash Attention 2: {'✅' if flash_ok else '❌'}")
print(f"Bitsandbytes (8-bit): ✅")

# Pequeña prueba de carga en VRAM
x = torch.randn(100, 100).cuda()
print("Prueba de tensor en CUDA: Exitosa")