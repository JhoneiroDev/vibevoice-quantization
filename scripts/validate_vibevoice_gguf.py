#!/usr/bin/env python3
"""Validate the selective IQ4_NL layout of a CrispASR VibeVoice GGUF."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


TARGET = re.compile(
    r"^lm\.layers\.\d+\."
    r"(?:attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"ffn\.(?:gate|up|down))\.weight$"
)
AUDIO_PREFIXES = ("at_enc.", "at_dec.", "st_enc.", "st_dec.", "at_conn.", "se_conn.", "pred.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    sys.path.insert(0, str(args.gguf_python.resolve()))
    import gguf  # type: ignore[import-not-found]

    reader = gguf.GGUFReader(str(args.model))
    type_counts: Counter[str] = Counter()
    quantized = []
    wrong_targets = []
    audio_tensors = []

    for tensor in reader.tensors:
        name = tensor.name
        dtype = tensor.tensor_type.name
        type_counts[dtype] += 1
        is_quantized = dtype.startswith(("Q", "IQ"))
        if is_quantized:
            quantized.append((name, dtype))
        if TARGET.fullmatch(name) and dtype != "IQ4_NL":
            wrong_targets.append((name, dtype))
        if name.startswith(AUDIO_PREFIXES):
            audio_tensors.append((name, dtype))

    unexpected = [(name, dtype) for name, dtype in quantized if not TARGET.fullmatch(name)]
    quantized_targets = [(name, dtype) for name, dtype in quantized if TARGET.fullmatch(name)]
    if len(quantized_targets) != 196:
        raise AssertionError(f"Expected 196 IQ4_NL Qwen matrices, found {len(quantized_targets)}")
    if wrong_targets:
        raise AssertionError(f"Qwen targets not stored as IQ4_NL: {wrong_targets[:10]}")
    if unexpected:
        raise AssertionError(f"Unexpected quantized tensors: {unexpected[:10]}")
    if not any(name.startswith("at_dec.") for name, _ in audio_tensors):
        raise AssertionError("Acoustic decoder missing; conversion must use --include-decoder")
    bad_audio = [(name, dtype) for name, dtype in audio_tensors if dtype not in {"F16", "F32"}]
    if bad_audio:
        raise AssertionError(f"Audio tensor not protected in F16/F32: {bad_audio[:10]}")

    result = {
        "format": "GGUF",
        "runtime": "CrispASR v0.8.23",
        "quantization": "selective IQ4_NL",
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()) if args.source else None,
        "size_bytes": args.model.stat().st_size,
        "sha256": sha256(args.model),
        "tensor_types": dict(sorted(type_counts.items())),
        "iq4_nl_qwen_matrices": len(quantized_targets),
        "protected_audio_tensors": len(audio_tensors),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
