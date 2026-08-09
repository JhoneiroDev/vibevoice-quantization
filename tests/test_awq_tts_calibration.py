from types import SimpleNamespace

import torch

from scripts.awq_vibevoice import (
    TTS_CALIBRATION_SCHEMA,
    TTSPrefillAwqQuantizer,
)
from scripts.bnb_quantize_vibevoice import _validate_nf4_state


def test_tts_prefill_quantizer_builds_fixed_blocks(monkeypatch):
    payload = {
        "schema": TTS_CALIBRATION_SCHEMA,
        "samples": [torch.ones(1, 1536, dtype=torch.float16) for _ in range(256)],
    }
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)

    quantizer = object.__new__(TTSPrefillAwqQuantizer)
    quantizer.calib_data = "calibration.pt"
    quantizer.model = object()
    quantizer.awq_model = SimpleNamespace(get_model_layers=lambda _model: ["layer-0"])

    modules, kwargs, inputs = quantizer.init_quant(n_samples=256, max_seq_len=2)

    assert modules == ["layer-0"]
    assert inputs.shape == (128, 2, 1536)
    assert inputs.dtype == torch.float16
    assert kwargs["attention_mask"] is None
    assert kwargs["position_ids"].tolist() == [[0, 1]]
    assert kwargs["use_cache"] is False


def test_nf4_validation_requires_nested_nf4_state():
    nested = SimpleNamespace(absmax=torch.ones(2))
    state = SimpleNamespace(
        nested=True,
        quant_type="nf4",
        blocksize=64,
        absmax=torch.ones(2),
        state2=nested,
    )
    module = SimpleNamespace(
        weight=SimpleNamespace(quant_state=state, dtype=torch.uint8)
    )

    _validate_nf4_state(module, "model.language_model.layers.0.self_attn.q_proj")
