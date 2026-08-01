"""Validate that a passed quality gate belongs to the current model files."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_quality_gate(model_path: str | Path, metrics_path: str | Path) -> dict:
    model = Path(model_path)
    metrics_file = Path(metrics_path)
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Missing Sprint 1.5 quality gate: {metrics_file}")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    if not metrics.get("passed"):
        raise ValueError(f"Sprint 1.5 quality gate did not pass: WER={metrics.get('wer')}")
    files = [model / "config.json", *sorted(model.glob("*.safetensors"))]
    current = {path.name: _sha256(path) for path in files if path.is_file()}
    if not current or current != metrics.get("source_hashes"):
        raise ValueError("Sprint 1.5 quality gate hashes do not match the Spanish checkpoint")
    return metrics
