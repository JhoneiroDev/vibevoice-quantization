#!/usr/bin/env python3
# ruff: noqa: E402
"""Build, validate, and load a selective GPTQ VibeVoice checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
import types
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import transformers

if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TritonV2QuantLinear
from gptqmodel.quantization.config import FORMAT
from gptqmodel.models._const import DEVICE
from gptqmodel.utils.backend import BACKEND
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
try:
    from quantization_common import (
        atomic_json,
        validate_canonical_source,
        validate_finite_checkpoint,
        validate_manifest_source,
    )
except ModuleNotFoundError:
    from scripts.quantization_common import (
        atomic_json,
        validate_canonical_source,
        validate_finite_checkpoint,
        validate_manifest_source,
    )


TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
PACKED_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")
SCHEMA = "vibevoice-selective-gptq-v1"
QWEN_TOKENIZER = "Qwen/Qwen2.5-1.5B"
TTS_CALIBRATION_FILENAME = "tts_prefill_inputs.pt"
FIXED_VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"


def expected_targets() -> set[str]:
    names = set()
    for layer in range(28):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"model.layers.{layer}.self_attn.{projection}")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"model.layers.{layer}.mlp.{projection}")
    return names


def preflight() -> dict:
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gptqmodel": importlib.metadata.version("gptqmodel"),
        "triton": importlib.metadata.version("triton"),
    }
    if versions["transformers"] != "4.51.3" or versions["gptqmodel"] != "2.2.0":
        raise RuntimeError(f"Unsupported GPTQ environment: {versions}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("Sprint 3 requires a CUDA GPU with compute capability >= 8.0")
    if not TRITON_AVAILABLE:
        raise RuntimeError("GPTQModel Triton backend is unavailable")
    for in_features, out_features in {(1536, 1536), (1536, 256), (1536, 8960), (8960, 1536)}:
        valid, error = TritonV2QuantLinear.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            pack_dtype=torch.int32,
            in_features=in_features,
            out_features=out_features,
            device=DEVICE.CUDA,
            trainable=False,
        )
        if not valid:
            raise RuntimeError(
                f"TritonV2 does not support shape {(in_features, out_features)}"
            ) from error

    dense = torch.nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
    scales = torch.full((32, 1), 0.1, dtype=torch.float32)
    zeros = torch.full((32, 1), 8, dtype=torch.float32)
    quantized = TritonV2QuantLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=32,
        bias=False,
        pack_dtype=torch.int32,
    )
    quantized.pack(dense, scales, zeros)
    quantized = quantized.to("cuda:0").eval()
    quantized.post_init()
    probe = torch.randn(2, 128, device="cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        output = quantized(probe)
    if output.shape != (2, 32) or not torch.isfinite(output).all():
        raise RuntimeError("TritonV2 synthetic kernel produced an invalid output")
    del dense, scales, zeros, quantized, probe, output
    torch.cuda.empty_cache()
    report = {
        **versions,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "triton_kernel": "passed",
    }
    print(json.dumps(report, indent=2))
    return report


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
            f"Expected exactly model.safetensors or an index in {path}, found {sorted(all_files)}"
        )
    return [model_file]


def iter_tensors(path: Path):
    seen = set()
    for shard in checkpoint_files(path):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"Duplicate tensor {name!r} across checkpoint shards")
                seen.add(name)
                yield name, handle.get_tensor(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(path: Path) -> dict[str, str]:
    files = [path / "config.json", *checkpoint_files(path)]
    return {file.name: sha256(file) for file in files}


def build_calibration(tokenizer, metadata_path: Path, output_path: Path, sequence_length: int = 256):
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("split_origin") != "validation":
        raise ValueError("GPTQ calibration must use the reserved validation split")

    token_ids: list[int] = []
    records = metadata["records"]
    texts = [record.get("sentence", "").strip() for record in records]
    if metadata.get("n_samples") != 512 or len(records) != 512:
        raise ValueError(f"GPTQ calibration requires exactly 512 records, found {len(records)}")
    if any(not text for text in texts) or len(set(texts)) != len(texts):
        raise ValueError("GPTQ calibration contains empty or duplicate transcriptions")
    eos = tokenizer.eos_token_id
    for sentence in texts:
        text = f"Speaker 1: {sentence}\n"
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            token_ids.append(eos)

    samples = []
    for start in range(0, len(token_ids), sequence_length):
        ids = token_ids[start : start + sequence_length]
        if len(ids) < sequence_length // 2:
            break
        samples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    if len(samples) < 16:
        raise ValueError(f"Insufficient calibration coverage: {len(samples)} chunks")
    if len(token_ids) < 10_000:
        raise ValueError(f"Insufficient calibration tokens: {len(token_ids)}")

    report = {
        "dataset_id": metadata["dataset_id"],
        "dataset_config": metadata["dataset_config"],
        "split_origin": metadata["split_origin"],
        "source_records": len(records),
        "sequence_length": sequence_length,
        "chunks": len(samples),
        "total_tokens": sum(len(item["input_ids"]) for item in samples),
        "metadata_sha256": sha256(metadata_path),
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def prepare_qwen_tokenizer(work: Path):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    tokenizer.save_pretrained(work)
    tokenizer_config = json.loads((work / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("tokenizer_class") not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}:
        raise RuntimeError(f"Unexpected GPTQ tokenizer config: {tokenizer_config.get('tokenizer_class')}")
    return tokenizer


def capture_tts_prefill_calibration(
    source: Path,
    metadata_path: Path,
    output_path: Path,
    sample_count: int = 256,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records", [])
    if len(records) != 512 or sample_count != 256:
        raise ValueError("TTS-prefill GPTQ calibration requires 256 samples from 512 records")
    if not FIXED_VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {FIXED_VOICE}")
    selected = records[::2]
    if len(selected) != sample_count:
        raise AssertionError(f"Unexpected TTS calibration selection: {len(selected)}")

    started = time.perf_counter()
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    processor = VibeVoiceProcessor.from_pretrained(source, local_files_only=True)
    voice = processor.audio_processor._load_audio_from_path(str(FIXED_VOICE))
    samples = []
    cached_speech_embeds = None
    expected_speech_positions = None
    lengths = []
    with torch.inference_mode():
        for index, record in enumerate(selected):
            inputs = processor(
                text=[f"Speaker 1: {record['sentence'].strip()}"],
                voice_samples=[[voice]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = inputs["input_ids"].to("cuda:0")
            attention_mask = inputs["attention_mask"]
            speech_input_mask = inputs["speech_input_mask"].to("cuda:0")
            inputs_embeds = model.get_input_embeddings()(input_ids)
            speech_positions = int(speech_input_mask.sum().item())
            if cached_speech_embeds is None:
                torch.manual_seed(42)
                _, cached_speech_embeds = model._process_speech_inputs(
                    inputs["speech_tensors"].to("cuda:0", dtype=torch.bfloat16),
                    inputs["speech_masks"],
                )
                cached_speech_embeds = cached_speech_embeds.detach()
                expected_speech_positions = speech_positions
            if speech_positions != expected_speech_positions:
                raise AssertionError(
                    f"Voice prompt length changed at sample {index}: {speech_positions}"
                )
            inputs_embeds[speech_input_mask] = cached_speech_embeds
            sample = {
                "inputs_embeds": inputs_embeds.squeeze(0).cpu().contiguous(),
                "attention_mask": attention_mask.squeeze(0).cpu().contiguous(),
            }
            if sample["inputs_embeds"].dtype != torch.bfloat16:
                raise AssertionError("TTS-prefill embeddings must be BF16")
            if not torch.isfinite(sample["inputs_embeds"]).all():
                raise AssertionError(f"Non-finite TTS-prefill embeddings at sample {index}")
            samples.append(sample)
            lengths.append(int(sample["attention_mask"].sum().item()))

    payload = {
        "schema": "vibevoice-gptq-tts-prefill-v1",
        "samples": samples,
        "voice": str(FIXED_VOICE.resolve()),
        "voice_sha256": sha256(FIXED_VOICE),
        "metadata_sha256": sha256(metadata_path),
        "source_indices": [record["local_idx"] for record in selected],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    report = {
        "mode": "tts_prefill_inputs_embeds",
        "tts_samples": len(samples),
        "tokens": sum(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "hidden_size": samples[0]["inputs_embeds"].shape[-1],
        "voice": payload["voice"],
        "voice_sha256": payload["voice_sha256"],
        "metadata_sha256": payload["metadata_sha256"],
        "capture_seconds": time.perf_counter() - started,
    }
    del model, processor, cached_speech_embeds, samples, payload
    gc.collect()
    torch.cuda.empty_cache()
    return report


class _TTSCalibrationExample(dict):
    def __init__(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        super().__init__(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        self._input_ids = [0] * inputs_embeds.shape[0]

    def __getitem__(self, key):
        if key == "input_ids":
            return self._input_ids
        return super().__getitem__(key)


def load_tts_calibration(path: Path) -> list[_TTSCalibrationExample]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "vibevoice-gptq-tts-prefill-v1":
        raise ValueError(f"Unsupported TTS-prefill calibration: {path}")
    samples = payload.get("samples", [])
    if len(samples) != 256:
        raise ValueError(f"Expected 256 TTS-prefill samples, found {len(samples)}")
    prepared = []
    for index, sample in enumerate(samples):
        embeds = sample["inputs_embeds"]
        mask = sample["attention_mask"]
        if embeds.ndim != 2 or embeds.shape[-1] != 1536 or mask.shape != embeds.shape[:1]:
            raise ValueError(f"Invalid TTS-prefill sample {index}: {embeds.shape}, {mask.shape}")
        if embeds.dtype != torch.bfloat16 or not torch.isfinite(embeds).all():
            raise ValueError(f"Invalid TTS-prefill embeddings at sample {index}")
        prepared.append(_TTSCalibrationExample(embeds, mask))
    return prepared


def install_tts_calibration_adapter(model, calibration) -> None:
    prepared_calibration = calibration

    def prepare_tts_dataset(
        _self,
        calibration_dataset,
        calibration_dataset_concat_size=None,
        batch_size=1,
    ):
        if calibration_dataset_concat_size is not None or batch_size != 1:
            raise ValueError(
                "TTS-prefill calibration requires batch_size=1 without concatenation"
            )
        return prepared_calibration

    model.prepare_dataset = types.MethodType(prepare_tts_dataset, model)


def prepare(
    source: Path,
    output: Path,
    metadata: Path,
    force: bool,
    tts_prefill: bool = False,
    desc_act: bool = False,
) -> None:
    validate_canonical_source(source)
    if output.exists():
        suffix = "; --force does not delete artifacts" if force else ""
        raise FileExistsError(f"Output requires manual review or removal: {output}{suffix}")

    work = output / ".work" / "decoder-bf16"
    protected_dir = output / "protected"
    calibration_dir = output / "calibration"
    work.mkdir(parents=True)
    protected_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)

    config_data = json.loads((source / "config.json").read_text(encoding="utf-8"))
    decoder_config = Qwen2Config.from_dict(config_data["decoder_config"])
    if decoder_config.num_hidden_layers != 28 or decoder_config.hidden_size != 1536:
        raise ValueError("Unsupported VibeVoice decoder architecture")

    qwen_state = {}
    protected_state = {}
    for name, tensor in iter_tensors(source):
        if name.startswith("model.language_model."):
            mapped = name.removeprefix("model.language_model.")
            qwen_state[mapped] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        elif name != "lm_head.weight":
            protected_state[name] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

    with init_empty_weights():
        qwen = Qwen2ForCausalLM(decoder_config)
    missing, unexpected = qwen.model.load_state_dict(qwen_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Qwen extraction mismatch: missing={missing}, unexpected={unexpected}")
    qwen.lm_head.weight = qwen.model.embed_tokens.weight
    qwen.config.torch_dtype = torch.bfloat16
    qwen.save_pretrained(work, safe_serialization=True, max_shard_size="4GB")
    del qwen, qwen_state
    gc.collect()

    save_file(protected_state, protected_dir / "model.safetensors", metadata={"format": "pt"})
    del protected_state
    gc.collect()

    processor = VibeVoiceProcessor.from_pretrained(source)
    processor.save_pretrained(output)
    shutil.copy2(source / "preprocessor_config.json", output / "preprocessor_config.json")
    qwen_tokenizer = prepare_qwen_tokenizer(work)
    calibration = build_calibration(qwen_tokenizer, metadata, calibration_dir / "samples.json")
    del qwen_tokenizer
    del processor
    if tts_prefill:
        tts_report = capture_tts_prefill_calibration(
            source,
            metadata,
            calibration_dir / TTS_CALIBRATION_FILENAME,
        )
        calibration = {**calibration, **tts_report}

    for filename in ("config.json", "generation_config.json"):
        src = source / filename
        if src.is_file():
            shutil.copy2(src, output / filename)

    manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "source": str(source.resolve()),
        "source_hashes": source_hashes(source),
        "dtype_policy": "audio/connectors/prediction_head BF16; Qwen projections GPTQ W4",
        "gptq": {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": desc_act,
            "damp_percent": 0.1,
            "format": "gptq",
            "backend": "triton",
            "parallel_packing": False,
            "expected_modules": 196,
        },
        "calibration": {key: value for key, value in calibration.items() if key != "samples"},
        "build_environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gptqmodel": importlib.metadata.version("gptqmodel"),
            "triton": importlib.metadata.version("triton"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"Prepared BF16 decoder and protected VibeVoice modules in {output}")


def quantize(output: Path) -> None:
    preflight()
    free_vram, _ = torch.cuda.mem_get_info()
    if free_vram < 6 * 1024**3:
        raise RuntimeError(
            f"GPTQ requires at least 6 GiB free VRAM, found {free_vram / 1024**3:.2f} GiB"
        )
    work = output / ".work" / "decoder-bf16"
    decoder = output / "decoder-gptq"
    calibration_path = output / "calibration" / "samples.json"
    tts_calibration_path = output / "calibration" / TTS_CALIBRATION_FILENAME
    if not work.is_dir() or not calibration_path.is_file():
        raise FileNotFoundError("Run prepare before quantize")
    if decoder.exists():
        raise FileExistsError(f"Existing GPTQ decoder requires manual review: {decoder}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = prepare_qwen_tokenizer(work)
    if tts_calibration_path.is_file():
        calibration = load_tts_calibration(tts_calibration_path)
    else:
        calibration_report = build_calibration(
            tokenizer, ROOT / "data" / "calibration_metadata.json", calibration_path
        )
        expected_calibration = {
            key: value for key, value in calibration_report.items() if key != "samples"
        }
        if manifest.get("calibration") != expected_calibration:
            raise RuntimeError("Regenerated GPTQ calibration does not match the prepared manifest")
        calibration = calibration_report["samples"]
    del tokenizer
    desc_act = bool(manifest.get("gptq", {}).get("desc_act", False))

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=desc_act,
        damp_percent=0.1,
        true_sequential=True,
        lm_head=False,
        format=FORMAT.GPTQ,
        device="cuda:0",
        parallel_packing=False,
    )
    model = GPTQModel.load(work, quantize_config=quant_config)
    if tts_calibration_path.is_file():
        install_tts_calibration_adapter(model, calibration)
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TRITON,
        calibration_enable_gpu_cache=False,
        buffered_fwd=True,
        auto_gc=True,
    )
    model.save(decoder, max_shard_size="4GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPTQ decoder saved in {decoder}")


def tensor_inventory(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    inventory = {}
    for name, tensor in iter_tensors(path):
        inventory[name] = (tuple(tensor.shape), str(tensor.dtype))
    return inventory


def validate(output: Path, remove_work: bool = False) -> dict:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported or legacy GPTQ artifact")
    validate_manifest_source(manifest)

    decoder_inventory = tensor_inventory(output / "decoder-gptq")
    validate_finite_checkpoint(output / "decoder-gptq")
    packed = {suffix: set() for suffix in PACKED_SUFFIXES}
    for name in decoder_inventory:
        for suffix in PACKED_SUFFIXES:
            marker = f".{suffix}"
            if name.endswith(marker):
                packed[suffix].add(name[: -len(marker)])
    expected = expected_targets()
    for suffix, names in packed.items():
        if names != expected:
            raise AssertionError(
                f"Invalid {suffix} target set: missing={sorted(expected - names)[:5]}, "
                f"unexpected={sorted(names - expected)[:5]}"
            )
    dense_targets = [
        name
        for name in decoder_inventory
        if name.endswith(".weight") and name.removesuffix(".weight") in expected
    ]
    if dense_targets:
        raise AssertionError(f"Dense GPTQ target weights found: {dense_targets[:5]}")
    if any("acoustic" in name or "semantic" in name or "prediction_head" in name for name in decoder_inventory):
        raise AssertionError("Audio tensor found inside GPTQ decoder")

    protected_inventory = tensor_inventory(output / "protected")
    validate_finite_checkpoint(output / "protected")
    if any(name.startswith("model.language_model.") or name == "lm_head.weight" for name in protected_inventory):
        raise AssertionError("Dense language-model tensor found in protected artifact")
    required_prefixes = (
        "model.acoustic_tokenizer.",
        "model.semantic_tokenizer.",
        "model.acoustic_connector.",
        "model.semantic_connector.",
        "model.prediction_head.",
    )
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in protected_inventory):
            raise AssertionError(f"Missing protected component: {prefix}")
    with safe_open(output / "protected" / "model.safetensors", framework="pt", device="cpu") as handle:
        for buffer_name in ("model.speech_scaling_factor", "model.speech_bias_factor"):
            if buffer_name not in protected_inventory:
                raise AssertionError(f"Missing protected buffer: {buffer_name}")
            if not torch.isfinite(handle.get_tensor(buffer_name)).all():
                raise AssertionError(f"Non-finite protected buffer: {buffer_name}")
    invalid_dtype = [
        (name, dtype)
        for name, (_, dtype) in protected_inventory.items()
        if dtype.startswith("torch.float") and dtype != "torch.bfloat16"
    ]
    if invalid_dtype:
        raise AssertionError(f"Unexpected protected dtypes: {invalid_dtype[:5]}")
    quant_config_path = output / "decoder-gptq" / "quantize_config.json"
    quant_config = json.loads(quant_config_path.read_text(encoding="utf-8"))
    expected_quant_config = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": bool(manifest.get("gptq", {}).get("desc_act", False)),
        "lm_head": False,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "pack_dtype": "int32",
    }
    actual_quant_config = {key: quant_config.get(key) for key in expected_quant_config}
    if actual_quant_config != expected_quant_config:
        raise AssertionError(f"Unexpected saved GPTQ config: {actual_quant_config}")

    files = [
        *[file for file in (output / "decoder-gptq").rglob("*") if file.is_file()],
        *(output / "protected").glob("*.safetensors"),
        *[
            file
            for name in ("config.json", "generation_config.json", "preprocessor_config.json")
            if (file := output / name).is_file()
        ],
        *[file for file in (output / "tokenizer").rglob("*") if file.is_file()],
        *[file for file in (output / "calibration").rglob("*") if file.is_file()],
    ]
    current_artifacts = {
        str(file.relative_to(output)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(files)
    }
    current_validation = {
        "gptq_modules": len(packed["qweight"]),
        "protected_tensors": len(protected_inventory),
        "total_bytes": sum(file.stat().st_size for file in files),
    }
    if manifest.get("status") in {"structure_validated", "validated", "quality_failed"}:
        if manifest.get("artifacts") != current_artifacts:
            raise AssertionError("Artifact size or SHA-256 differs from the validated manifest")
        if manifest.get("validation") != current_validation:
            raise AssertionError("Structural validation differs from the validated manifest")
    else:
        manifest["status"] = "structure_validated"
        manifest["artifacts"] = current_artifacts
        manifest["validation"] = current_validation
        atomic_json(manifest_path, manifest)
    if remove_work:
        shutil.rmtree(output / ".work", ignore_errors=True)
    print(json.dumps(current_validation, indent=2))
    return manifest


def load_gptq_vibevoice(
    artifact: str | Path,
    device: str = "cuda:0",
    backend: str = "triton",
) -> VibeVoiceForConditionalGenerationInference:
    artifact = Path(artifact)
    validate(artifact)
    selected_backend = {"triton": BACKEND.TRITON, "torch": BACKEND.TORCH}[backend]
    decoder_wrapper = GPTQModel.load(
        artifact / "decoder-gptq",
        device=device,
        backend=selected_backend,
    )
    decoder = decoder_wrapper.model.model

    config = VibeVoiceConfig.from_pretrained(artifact)
    # Keep deterministic/non-persistent buffers materialized; only parameters
    # belong on meta because those buffers are not present in safetensors.
    with init_empty_weights(include_buffers=False):
        model = VibeVoiceForConditionalGenerationInference(config)
    model.model.language_model = decoder

    protected = load_file(artifact / "protected" / "model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(protected, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected protected keys: {unexpected[:10]}")
    invalid_missing = [
        name
        for name in missing
        if not name.startswith("model.language_model.") and name != "lm_head.weight"
    ]
    if invalid_missing:
        raise RuntimeError(f"Missing protected keys: {invalid_missing[:10]}")
    del protected

    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    for module in (
        model.model.acoustic_tokenizer,
        model.model.semantic_tokenizer,
        model.model.acoustic_connector,
        model.model.semantic_connector,
        model.model.prediction_head,
    ):
        module.to(device)
    model.model.speech_scaling_factor = model.model.speech_scaling_factor.to(device)
    model.model.speech_bias_factor = model.model.speech_bias_factor.to(device)

    meta = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if meta:
        raise RuntimeError(f"Unmaterialized tensors after hybrid load: {meta[:10]}")
    packed_modules = [
        name
        for name, module in model.model.language_model.named_modules()
        if type(module).__name__ == "TritonV2QuantLinear"
    ]
    if len(packed_modules) != 196:
        raise RuntimeError(f"Expected 196 GPTQ modules after reload, found {len(packed_modules)}")
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise RuntimeError("lm_head and embed_tokens are not tied")
    invalid_dense_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if not hasattr(parameter, "qweight")
        and parameter.is_floating_point()
        and parameter.dtype != torch.bfloat16
    ]
    if invalid_dense_dtypes:
        raise RuntimeError(f"Non-GPTQ parameters are not BF16: {invalid_dense_dtypes[:5]}")

    object.__setattr__(model, "_gptq_decoder_wrapper", decoder_wrapper)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--metadata", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--tts-prefill", action="store_true")
    prepare_parser.add_argument("--desc-act", action="store_true")

    quantize_parser = subparsers.add_parser("quantize")
    quantize_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--remove-work", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "prepare":
        prepare(
            args.source.resolve(),
            args.output.resolve(),
            args.metadata.resolve(),
            args.force,
            args.tts_prefill,
            args.desc_act,
        )
    elif args.command == "quantize":
        quantize(args.output.resolve())
    elif args.command == "validate":
        validate(args.output.resolve(), args.remove_work)


if __name__ == "__main__":
    main()
