"""Shared provenance and structural checks for selective quantization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

try:
    from quality_gate import validate_quality_gate
except ModuleNotFoundError:  # Imported as scripts.quantization_common.
    from scripts.quality_gate import validate_quality_gate


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODEL = ROOT / "weights" / "vibevoice-1.5b-es"
QUALITY_METRICS = ROOT / "outputs" / "sprint1_5_quality_gate" / "metrics.json"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(path: Path) -> list[Path]:
    index_path = path / "model.safetensors.index.json"
    all_files = set(path.glob("*.safetensors"))
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = [path / name for name in sorted(set(index["weight_map"].values()))]
        missing = [file for file in files if not file.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing indexed shards: {missing}")
        unexpected = all_files - set(files)
        if unexpected:
            raise ValueError(f"Unindexed safetensors found in {path}: {sorted(unexpected)}")
        return files
    model_file = path / "model.safetensors"
    if all_files != {model_file}:
        raise ValueError(
            f"Expected model.safetensors or an index in {path}, found {sorted(all_files)}"
        )
    return [model_file]


def source_hashes(path: Path) -> dict[str, str]:
    files = [path / "config.json", *checkpoint_files(path)]
    return {file.name: sha256(file) for file in files}


def validate_finite_checkpoint(path: Path) -> None:
    for shard in checkpoint_files(path):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite tensor in {path}: {name}")


def expected_qwen_weights() -> dict[str, tuple[int, int]]:
    expected = {}
    for layer in range(28):
        prefix = f"model.language_model.layers.{layer}"
        for projection in ("q_proj", "o_proj"):
            expected[f"{prefix}.self_attn.{projection}.weight"] = (1536, 1536)
        for projection in ("k_proj", "v_proj"):
            expected[f"{prefix}.self_attn.{projection}.weight"] = (256, 1536)
        for projection in ("gate_proj", "up_proj"):
            expected[f"{prefix}.mlp.{projection}.weight"] = (8960, 1536)
        expected[f"{prefix}.mlp.down_proj.weight"] = (1536, 8960)
    return expected


def _validate_qwen_inventory(path: Path) -> None:
    expected = expected_qwen_weights()
    found: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    seen = set()
    for shard in checkpoint_files(path):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"Duplicate tensor {name!r} across checkpoint shards")
                seen.add(name)
                if name in expected:
                    tensor = handle.get_tensor(name)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite canonical Qwen projection: {name}")
                    found[name] = (tuple(tensor.shape), tensor.dtype)
    if set(found) != set(expected):
        raise ValueError(
            "Canonical Qwen projection inventory mismatch: "
            f"missing={sorted(set(expected) - set(found))[:5]}, "
            f"unexpected={sorted(set(found) - set(expected))[:5]}"
        )
    invalid = [
        (name, shape, str(dtype))
        for name, (shape, dtype) in found.items()
        if shape != expected[name] or dtype != torch.float32
    ]
    if invalid:
        raise ValueError(f"Invalid canonical Qwen projection tensors: {invalid[:5]}")


def validate_canonical_source(source: str | Path) -> dict:
    source = Path(source).resolve()
    if source != CANONICAL_MODEL.resolve():
        raise ValueError(f"Quantization source must be the canonical checkpoint: {CANONICAL_MODEL}")
    gate = validate_quality_gate(source, QUALITY_METRICS)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    decoder = config.get("decoder_config", {})
    expected_architecture = {
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    }
    actual = {key: decoder.get(key) for key in expected_architecture}
    if actual != expected_architecture:
        raise ValueError(f"Unsupported canonical decoder architecture: {actual}")
    _validate_qwen_inventory(source)
    return gate


def validate_manifest_source(manifest: dict) -> Path:
    source_value = manifest.get("source")
    if not source_value:
        raise ValueError("Artifact manifest does not identify its source checkpoint")
    source = Path(source_value).resolve()
    validate_canonical_source(source)
    current = source_hashes(source)
    if current != manifest.get("source_hashes"):
        raise ValueError("Artifact source hashes do not match the canonical checkpoint")
    return source
