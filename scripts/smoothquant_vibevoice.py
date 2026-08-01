#!/usr/bin/env python3
"""Selective SmoothQuant W8A8 inference for VibeVoice's Qwen decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import transformers  # noqa: E402

if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig  # noqa: E402
from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: E402
    VibeVoiceForConditionalGenerationInference,
)

try:  # noqa: E402
    from quantization_common import expected_qwen_weights
except ModuleNotFoundError:  # noqa: E402
    from scripts.quantization_common import expected_qwen_weights


SCHEMA = "vibevoice-selective-smoothquant-v1"
ALPHA = 0.5
TARGETS = {
    name.removesuffix(".weight") for name in expected_qwen_weights()
}


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


class SmoothQuantLinear(nn.Module):
    """W8A8 linear using SmoothQuant channel migration and CUDA int8 GEMM."""

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("input_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_scale: torch.Tensor) -> "SmoothQuantLinear":
        result = cls(linear.in_features, linear.out_features, linear.bias is not None).to(
            device=linear.weight.device
        )
        weight = linear.weight.detach().float()
        migrated = weight * input_scale.to(weight.device).unsqueeze(0)
        weight_scale = migrated.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        qweight = torch.round(migrated / weight_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        result.qweight.copy_(qweight)
        result.input_scale.copy_(input_scale.float())
        result.weight_scale.copy_(weight_scale)
        if linear.bias is not None:
            result.bias.copy_(linear.bias.detach().to(torch.bfloat16))
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape[:-1]
        flat = inputs.float().reshape(-1, self.in_features)
        scaled = flat / self.input_scale.to(flat.device)
        activation_scale = scaled.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        qactivation = torch.round(scaled / activation_scale).clamp(-127, 127).to(torch.int8)
        if qactivation.device.type == "cuda" and hasattr(torch, "_int_mm"):
            # cuBLAS INT8 requires aligned sequence dimensions on this GPU.
            rows = qactivation.shape[0]
            padded_rows = max(32, (rows + 7) // 8 * 8)
            if padded_rows != rows:
                qactivation = torch.nn.functional.pad(qactivation, (0, 0, 0, padded_rows - rows))
            product = torch._int_mm(qactivation, self.qweight.t().contiguous()).float()[:rows]
        else:
            product = qactivation.float() @ self.qweight.float().t()
        output = product * activation_scale * self.weight_scale.to(product.device).unsqueeze(0)
        if self.bias is not None:
            output = output + self.bias.to(output.device)
        return output.to(inputs.dtype).reshape(*original_shape, self.out_features)


def replace_qwen_linears_with_smoothquant(
    model: nn.Module, activation_scales: dict[str, torch.Tensor], *, empty: bool = False
) -> list[str]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name in TARGETS and isinstance(module, nn.Linear)
    ]
    names = {name for name, _ in targets}
    if names != TARGETS:
        raise RuntimeError(
            f"Invalid SmoothQuant targets: missing={sorted(TARGETS - names)[:5]}, "
            f"unexpected={sorted(names - TARGETS)[:5]}"
        )
    for name, linear in targets:
        if empty:
            replacement = SmoothQuantLinear(linear.in_features, linear.out_features, linear.bias is not None)
        else:
            replacement = SmoothQuantLinear.from_linear(linear, activation_scales[name])
        _set_module(model, name, replacement)
    return sorted(names)


def collect_activation_scales(model, tokenizer, texts: list[str], device: str) -> dict[str, torch.Tensor]:
    maxima: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name):
        def capture(_module, inputs, _output):
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            current = values.abs().amax(dim=0).cpu()
            maxima[name] = torch.maximum(maxima.get(name, torch.zeros_like(current)), current)
        return capture

    for name, module in model.named_modules():
        if name in TARGETS:
            handles.append(module.register_forward_hook(hook(name)))
    encoded = [tokenizer(f"Speaker 1: {text}", return_tensors="pt") for text in texts]
    with torch.inference_mode():
        for item in encoded:
            inputs = {key: value.to(device) for key, value in item.items()}
            model.model.language_model(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    if set(maxima) != TARGETS:
        raise RuntimeError(f"Activation calibration missed targets: {sorted(TARGETS - set(maxima))[:5]}")
    result = {}
    for name, module in model.named_modules():
        if name in TARGETS:
            weight_max = module.weight.detach().float().abs().amax(dim=0).cpu().clamp_min(1e-8)
            activation_max = maxima[name].clamp_min(1e-8)
            result[name] = (activation_max.pow(ALPHA) / weight_max.pow(1 - ALPHA)).clamp(1e-3, 1e3)
    return result


def load_smoothquant_vibevoice(model_path: str | Path, device_map=None, max_memory=None):
    model_path = Path(model_path)
    config = VibeVoiceConfig.from_pretrained(str(model_path))
    with init_empty_weights():
        model = VibeVoiceForConditionalGenerationInference(config)
        replaced = replace_qwen_linears_with_smoothquant(model, {}, empty=True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    if not replaced:
        raise RuntimeError("No SmoothQuant targets installed")
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(model_path), device_map=device_map or {"": "cuda:0"},
        max_memory=max_memory, dtype=None,
    )
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.eval()
    return model
