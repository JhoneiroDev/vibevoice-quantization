"""Selective dynamic FP8 inference helpers for VibeVoice."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_DTYPE = torch.float8_e4m3fn
DEFAULT_RANGE_LIMIT = 240.0


def expected_qwen_linears() -> set[str]:
    names = set()
    for layer in range(28):
        prefix = f"model.language_model.layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{projection}")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"{prefix}.mlp.{projection}")
    return names


def probe_native_fp8(device: str = "cuda:0") -> None:
    """Fail before model loading when the GPU lacks native scaled FP8 matmul."""
    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("Native FP8 requires CUDA and torch._scaled_mm")
    try:
        left = torch.ones((16, 16), device=device, dtype=FP8_DTYPE)
        right = torch.ones((16, 16), device=device, dtype=FP8_DTYPE)
        scale = torch.ones((), device=device, dtype=torch.float32)
        output = torch._scaled_mm(
            left,
            right,
            scale_a=scale,
            scale_b=scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        torch.cuda.synchronize()
        if output.shape != (16, 16) or not torch.isfinite(output).all():
            raise RuntimeError("torch._scaled_mm returned an invalid result")
    except Exception as error:
        raise RuntimeError(
            f"Native FP8 torch._scaled_mm is unsupported on {torch.cuda.get_device_name(0)}"
        ) from error


class DynamicScaledFP8Linear(nn.Module):
    """FP8 weight linear with per-tensor dynamic activation scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        device: torch.device | str | None = None,
        range_limit: float = DEFAULT_RANGE_LIMIT,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.range_limit = float(range_limit)
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.ones((), dtype=torch.float32, device=device),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        range_limit: float = DEFAULT_RANGE_LIMIT,
    ) -> "DynamicScaledFP8Linear":
        converted = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
            range_limit=range_limit,
        )
        weight = linear.weight.detach().float()
        max_abs = torch.nan_to_num(weight.abs().amax(), nan=0.0, posinf=0.0)
        scale = torch.where(
            max_abs > 0,
            torch.tensor(range_limit, device=weight.device) / max_abs,
            torch.ones((), device=weight.device),
        )
        converted.weight.copy_(
            (weight * scale).clamp(-range_limit, range_limit).to(FP8_DTYPE)
        )
        converted.weight_scale_inv.copy_(scale.reciprocal())
        if linear.bias is not None:
            converted.bias.copy_(linear.bias.detach().to(torch.bfloat16))
        return converted

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_bf16 = inputs.to(torch.bfloat16)
        original_shape = inputs_bf16.shape[:-1]
        flat_inputs = inputs_bf16.reshape(-1, self.in_features)

        if self.weight.device.type != "cuda" or not hasattr(torch, "_scaled_mm"):
            weight_bf16 = self.weight.to(torch.bfloat16) * self.weight_scale_inv
            return F.linear(inputs_bf16, weight_bf16, self.bias)

        max_abs = torch.nan_to_num(
            flat_inputs.detach().abs().amax().float(), nan=0.0, posinf=0.0
        )
        activation_scale = torch.where(
            max_abs > 0,
            torch.tensor(self.range_limit, device=flat_inputs.device) / max_abs,
            torch.ones((), device=flat_inputs.device),
        )
        inputs_fp8 = (
            flat_inputs.float()
            .mul(activation_scale)
            .clamp(-self.range_limit, self.range_limit)
            .to(FP8_DTYPE)
        )
        output = torch._scaled_mm(
            inputs_fp8,
            self.weight.t(),
            scale_a=activation_scale.reciprocal(),
            scale_b=self.weight_scale_inv,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        if self.bias is not None:
            output.add_(self.bias)
        return output.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, range_limit={self.range_limit}"
        )


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def replace_qwen_linears_with_fp8(
    model: nn.Module,
    *,
    range_limit: float = DEFAULT_RANGE_LIMIT,
) -> list[str]:
    """Replace only VibeVoice's Qwen linear projections in-place."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.") and isinstance(module, nn.Linear)
    ]
    names = {name for name, _ in targets}
    if names != expected_qwen_linears():
        raise RuntimeError(
            f"Invalid FP8 targets: missing={sorted(expected_qwen_linears() - names)[:5]}, "
            f"unexpected={sorted(names - expected_qwen_linears())[:5]}"
        )
    for name, linear in targets:
        _set_module(
            model,
            name,
            DynamicScaledFP8Linear.from_linear(linear, range_limit=range_limit),
        )
    return [name for name, _ in targets]


def replace_qwen_linears_with_empty_fp8(
    model: nn.Module,
    *,
    range_limit: float = DEFAULT_RANGE_LIMIT,
) -> list[str]:
    """Install empty FP8 modules in a meta-initialized model before loading."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.") and isinstance(module, nn.Linear)
    ]
    names = {name for name, _ in targets}
    if names != expected_qwen_linears():
        raise RuntimeError(
            f"Invalid FP8 reload targets: missing={sorted(expected_qwen_linears() - names)[:5]}, "
            f"unexpected={sorted(names - expected_qwen_linears())[:5]}"
        )
    for name, linear in targets:
        _set_module(
            model,
            name,
            DynamicScaledFP8Linear(
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
                device=linear.weight.device,
                range_limit=range_limit,
            ),
        )
    return [name for name, _ in targets]


def load_fp8_vibevoice(
    model_path: str | Path,
    *,
    device_map: dict | str = None,
    max_memory: dict | None = None,
):
    """Load a checkpoint saved with DynamicScaledFP8Linear modules."""
    if device_map == {"": "cuda:0"} or device_map == "cuda:0":
        probe_native_fp8("cuda:0")
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    model_path = Path(model_path)
    config = VibeVoiceConfig.from_pretrained(str(model_path))
    fp8_config = getattr(config, "vibevoice_fp8_config", {})
    range_limit = float(fp8_config.get("range_limit", DEFAULT_RANGE_LIMIT))

    with init_empty_weights():
        model = VibeVoiceForConditionalGenerationInference(config)
        replaced = replace_qwen_linears_with_empty_fp8(
            model, range_limit=range_limit
        )
    # Do this outside init_empty_weights: its register_parameter hook creates
    # a new meta Parameter for assignments and would silently break the tie.
    model.lm_head.weight = model.model.language_model.embed_tokens.weight

    if not replaced:
        raise RuntimeError("No Qwen linear layers were replaced in the FP8 skeleton.")
    if device_map is None:
        device_map = {"": "cuda:0"}
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(model_path),
        device_map=device_map,
        max_memory=max_memory,
        dtype=None,
    )
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.eval()
    return model
