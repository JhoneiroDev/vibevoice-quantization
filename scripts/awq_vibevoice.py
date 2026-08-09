#!/usr/bin/env python3
# ruff: noqa: E402
"""Build, validate, and load a selective AutoAWQ VibeVoice checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
import time
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

from awq import AutoAWQForCausalLM
from awq.modules.linear.gemm import TRITON_AVAILABLE, WQLinear_GEMM
from awq.quantize.quantizer import AwqQuantizer
from transformers import Qwen2Config, Qwen2ForCausalLM

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
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


PACKED_SUFFIXES = ("qweight", "qzeros", "scales")
SCHEMA = "vibevoice-selective-awq-v2"
TTS_CALIBRATION_FILENAME = "tts_prefill_inputs.pt"
TTS_CALIBRATION_SCHEMA = "vibevoice-awq-tts-prefill-v1"
FIXED_VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"


def validate_runtime() -> dict:
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "autoawq": importlib.metadata.version("autoawq"),
        "triton": importlib.metadata.version("triton"),
    }
    if versions["transformers"] != "4.51.3" or versions["autoawq"] != "0.2.9":
        raise RuntimeError(f"Unsupported AWQ environment: {versions}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("Sprint 4 requires a CUDA GPU with compute capability >= 8.0")
    if importlib.util.find_spec("awq_ext") is not None:
        raise RuntimeError("Sprint 4 requires the reproducible Triton path, not awq_ext")
    if not TRITON_AVAILABLE:
        raise RuntimeError("AutoAWQ Triton kernels are unavailable")
    return versions


def preflight() -> dict:
    versions = validate_runtime()

    dense = torch.nn.Linear(128, 32, bias=False, device="cuda:0", dtype=torch.float16)
    scales = torch.full((1, 32), 0.1, device="cuda:0", dtype=torch.float16)
    zeros = torch.full((1, 32), 8, device="cuda:0", dtype=torch.float16)
    quantized = WQLinear_GEMM.from_linear(
        dense, w_bit=4, group_size=128, scales=scales, zeros=zeros
    ).eval()
    probe = torch.randn(1, 2, 128, device="cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        result = quantized(probe)
    if result.shape != (1, 2, 32) or not torch.isfinite(result).all():
        raise RuntimeError("AutoAWQ Triton synthetic kernel produced an invalid output")
    del dense, scales, zeros, quantized, probe, result
    torch.cuda.empty_cache()
    report = {
        **versions,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "triton_kernel": "passed",
    }
    print(json.dumps(report, indent=2))
    return report


def expected_targets() -> set[str]:
    names = set()
    for layer in range(28):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"model.layers.{layer}.self_attn.{projection}")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"model.layers.{layer}.mlp.{projection}")
    return names


def expected_target_shapes() -> dict[str, tuple[int, int]]:
    shapes = {}
    for layer in range(28):
        prefix = f"model.layers.{layer}"
        for projection in ("q_proj", "o_proj"):
            shapes[f"{prefix}.self_attn.{projection}"] = (1536, 1536)
        for projection in ("k_proj", "v_proj"):
            shapes[f"{prefix}.self_attn.{projection}"] = (1536, 256)
        for projection in ("gate_proj", "up_proj"):
            shapes[f"{prefix}.mlp.{projection}"] = (1536, 8960)
        shapes[f"{prefix}.mlp.down_proj"] = (8960, 1536)
    return shapes


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


def validate_calibration_metadata(metadata_path: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split_origin") != "validation":
        raise ValueError("AWQ calibration must use the reserved validation split")
    records = metadata.get("records", [])
    if metadata.get("n_samples") != 512 or len(records) != 512:
        raise ValueError(f"AWQ calibration requires exactly 512 reserved records, found {len(records)}")
    local_indices = [record["local_idx"] for record in records]
    if len(set(local_indices)) != len(local_indices):
        raise ValueError("AWQ calibration contains duplicate source indices")
    texts = [record.get("sentence", "").strip() for record in records]
    if any(not text for text in texts) or len(set(texts)) != len(texts):
        raise ValueError("AWQ calibration contains empty or duplicate transcriptions")
    return metadata, records


def capture_tts_prefill_calibration(
    source: Path,
    metadata_path: Path,
    output_path: Path,
    sample_count: int = 256,
) -> dict:
    metadata, records = validate_calibration_metadata(metadata_path)
    if sample_count != 256:
        raise ValueError("TTS-prefill AWQ calibration requires exactly 256 samples")
    if not FIXED_VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {FIXED_VOICE}")
    selected = records[::2]
    if len(selected) != sample_count:
        raise AssertionError(f"Unexpected TTS calibration selection: {len(selected)}")

    started = time.perf_counter()
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        source,
        torch_dtype=torch.float16,
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
            speech_input_mask = inputs["speech_input_mask"].to("cuda:0")
            inputs_embeds = model.get_input_embeddings()(input_ids)
            speech_positions = int(speech_input_mask.sum().item())
            if cached_speech_embeds is None:
                torch.manual_seed(42)
                _, cached_speech_embeds = model._process_speech_inputs(
                    inputs["speech_tensors"].to("cuda:0", dtype=torch.float16),
                    inputs["speech_masks"],
                )
                cached_speech_embeds = cached_speech_embeds.detach()
                expected_speech_positions = speech_positions
            if speech_positions != expected_speech_positions:
                raise AssertionError(
                    f"Voice prompt length changed at sample {index}: {speech_positions}"
                )
            inputs_embeds[speech_input_mask] = cached_speech_embeds
            sample = inputs_embeds.squeeze(0).cpu().contiguous()
            if sample.dtype != torch.float16 or not torch.isfinite(sample).all():
                raise AssertionError(f"Invalid TTS-prefill embeddings at sample {index}")
            samples.append(sample)
            lengths.append(sample.shape[0])

    payload = {
        "schema": TTS_CALIBRATION_SCHEMA,
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
        "dataset_id": metadata["dataset_id"],
        "dataset_config": metadata["dataset_config"],
        "split_origin": metadata["split_origin"],
        "source_records": len(records),
        "mode": "tts_prefill_inputs_embeds",
        "tts_samples": len(samples),
        "tokens": sum(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "hidden_size": samples[0].shape[-1],
        "voice": payload["voice"],
        "voice_sha256": payload["voice_sha256"],
        "metadata_sha256": payload["metadata_sha256"],
        "capture_seconds": time.perf_counter() - started,
    }
    del model, processor, cached_speech_embeds, samples, payload
    gc.collect()
    torch.cuda.empty_cache()
    return report


class TTSPrefillAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=128, max_seq_len=256):
        calibration_path = Path(self.calib_data)
        payload = torch.load(calibration_path, map_location="cpu", weights_only=True)
        if payload.get("schema") != TTS_CALIBRATION_SCHEMA:
            raise ValueError(f"Unsupported TTS-prefill calibration: {calibration_path}")
        samples = payload.get("samples", [])
        if len(samples) != 256:
            raise ValueError(f"Expected 256 TTS-prefill samples, found {len(samples)}")
        for index, sample in enumerate(samples):
            if sample.ndim != 2 or sample.shape[-1] != 1536:
                raise ValueError(f"Invalid TTS-prefill shape at sample {index}: {sample.shape}")
            if sample.dtype != torch.float16 or not torch.isfinite(sample).all():
                raise ValueError(f"Invalid TTS-prefill values at sample {index}")

        token_stream = torch.cat(samples, dim=0)
        block_count = token_stream.shape[0] // max_seq_len
        if block_count < 64 or block_count > n_samples:
            raise ValueError(
                f"Invalid AWQ calibration coverage: {block_count} blocks of {max_seq_len}"
            )
        inps = token_stream[: block_count * max_seq_len].reshape(
            block_count, max_seq_len, 1536
        )
        modules = self.awq_model.get_model_layers(self.model)
        layer_kwargs = {
            "attention_mask": None,
            "position_ids": torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0),
            "use_cache": False,
        }
        del payload, samples, token_stream
        return modules, layer_kwargs, inps.contiguous()


def prepare(source: Path, output: Path, metadata: Path, force: bool) -> None:
    validate_canonical_source(source)
    if output.exists():
        suffix = "; --force does not delete artifacts" if force else ""
        raise FileExistsError(f"Output requires manual review or removal: {output}{suffix}")

    work = output / ".work" / "decoder-fp16"
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
            qwen_state[mapped] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
        elif name != "lm_head.weight":
            protected_state[name] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor

    with init_empty_weights():
        qwen = Qwen2ForCausalLM(decoder_config)
    missing, unexpected = qwen.model.load_state_dict(qwen_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Qwen extraction mismatch: missing={missing}, unexpected={unexpected}")
    qwen.lm_head.weight = qwen.model.embed_tokens.weight
    qwen.config.torch_dtype = torch.float16
    qwen.config.use_cache = False
    qwen.save_pretrained(work, safe_serialization=True, max_shard_size="4GB")
    del qwen, qwen_state
    gc.collect()

    save_file(protected_state, protected_dir / "model.safetensors", metadata={"format": "pt"})
    del protected_state
    gc.collect()

    processor = VibeVoiceProcessor.from_pretrained(source, local_files_only=True)
    processor.save_pretrained(output)
    processor.tokenizer.save_pretrained(work)
    del processor
    calibration = capture_tts_prefill_calibration(
        source, metadata, calibration_dir / TTS_CALIBRATION_FILENAME
    )
    for filename in ("config.json", "generation_config.json"):
        src = source / filename
        if src.is_file():
            shutil.copy2(src, output / filename)

    manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "source": str(source.resolve()),
        "source_hashes": source_hashes(source),
        "dtype_policy": "audio/connectors/prediction_head FP16; Qwen projections AWQ W4A16",
        "awq": {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
            "runtime": "triton",
            "expected_modules": 196,
            "duo_scaling": True,
            "apply_clip": True,
        },
        "calibration": calibration,
        "build_environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "autoawq": importlib.metadata.version("autoawq"),
            "triton": importlib.metadata.version("triton"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"Prepared FP16 decoder and protected VibeVoice modules in {output}")


def quantize(output: Path) -> None:
    validate_runtime()
    work = output / ".work" / "decoder-fp16"
    decoder = output / "decoder-awq"
    calibration_path = output / "calibration" / TTS_CALIBRATION_FILENAME
    if not work.is_dir() or not calibration_path.is_file():
        raise FileNotFoundError("Run prepare before quantize")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        raise ValueError(f"Cannot quantize AWQ artifact in state {manifest.get('status')!r}")
    calibration_payload = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if calibration_payload.get("metadata_sha256") != manifest["calibration"]["metadata_sha256"]:
        raise ValueError("AWQ calibration metadata hash differs from the manifest")
    if calibration_payload.get("voice_sha256") != manifest["calibration"]["voice_sha256"]:
        raise ValueError("AWQ calibration voice hash differs from the manifest")
    del calibration_payload
    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
        output / "tokenizer", local_files_only=True
    )
    model = AutoAWQForCausalLM.from_pretrained(
        work,
        safetensors=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
    )
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
        "modules_to_not_convert": ["lm_head"],
    }
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=str(calibration_path),
        max_calib_samples=128,
        max_calib_seq_len=256,
        n_parallel_calib_samples=1,
        max_chunk_memory=512 * 1024 * 1024,
        duo_scaling=True,
        apply_clip=True,
        quantizer_cls=TTSPrefillAwqQuantizer,
    )
    model.model.config.torch_dtype = torch.float16
    model.model.config.use_cache = True
    model.save_quantized(str(decoder), shard_size="4GB")
    tokenizer.save_pretrained(decoder)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"AWQ decoder saved in {decoder}")


def tensor_inventory(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        name: (tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in iter_tensors(path)
    }


def validate(
    output: Path,
    remove_work: bool = False,
    verify_source: bool = False,
) -> dict:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported or legacy AWQ artifact")
    if verify_source:
        validate_manifest_source(manifest)
    elif not manifest.get("source") or not manifest.get("source_hashes"):
        raise ValueError("AWQ manifest lacks source provenance")
    if manifest.get("status") not in {
        "prepared",
        "structure_validated",
        "validated",
        "quality_failed",
    }:
        raise ValueError(f"Invalid AWQ artifact status: {manifest.get('status')!r}")

    decoder_config = json.loads((output / "decoder-awq" / "config.json").read_text(encoding="utf-8"))
    quant_config = decoder_config.get("quantization_config", {})
    expected_config = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    }
    for key, expected_value in expected_config.items():
        if quant_config.get(key) != expected_value:
            raise AssertionError(f"Invalid AWQ config {key}: {quant_config.get(key)!r}")

    decoder_inventory = tensor_inventory(output / "decoder-awq")
    validate_finite_checkpoint(output / "decoder-awq")
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
                f"Invalid {suffix} targets: missing={sorted(expected - names)[:5]}, "
                f"unexpected={sorted(names - expected)[:5]}"
            )
    for target in expected:
        if f"{target}.weight" in decoder_inventory:
            raise AssertionError(f"Dense target remains in AWQ decoder: {target}")
        if decoder_inventory[f"{target}.qweight"][1] != "torch.int32":
            raise AssertionError(f"qweight is not INT32: {target}")
        if decoder_inventory[f"{target}.qzeros"][1] != "torch.int32":
            raise AssertionError(f"qzeros is not INT32: {target}")
        if decoder_inventory[f"{target}.scales"][1] != "torch.float16":
            raise AssertionError(f"scales are not FP16: {target}")
        in_features, out_features = expected_target_shapes()[target]
        expected_shapes = {
            "qweight": (in_features, out_features // 8),
            "qzeros": (in_features // 128, out_features // 8),
            "scales": (in_features // 128, out_features),
        }
        for suffix, shape in expected_shapes.items():
            actual_shape = decoder_inventory[f"{target}.{suffix}"][0]
            if actual_shape != shape:
                raise AssertionError(
                    f"Invalid {suffix} shape for {target}: {actual_shape} != {shape}"
                )
    if any("acoustic" in name or "semantic" in name or "prediction_head" in name for name in decoder_inventory):
        raise AssertionError("Audio tensor found inside AWQ decoder")

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
        for name in ("model.speech_scaling_factor", "model.speech_bias_factor"):
            if name not in protected_inventory or not torch.isfinite(handle.get_tensor(name)).all():
                raise AssertionError(f"Missing or non-finite protected buffer: {name}")
    invalid_dtype = [
        (name, dtype)
        for name, (_, dtype) in protected_inventory.items()
        if dtype.startswith("torch.float") and dtype != "torch.float16"
    ]
    if invalid_dtype:
        raise AssertionError(f"Protected modules are not FP16: {invalid_dtype[:5]}")

    files = [
        *[file for file in (output / "decoder-awq").rglob("*") if file.is_file()],
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
        "awq_modules": len(packed["qweight"]),
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


def load_awq_vibevoice(
    artifact: str | Path,
    device: str = "cuda:0",
) -> VibeVoiceForConditionalGenerationInference:
    artifact = Path(artifact)
    validate_runtime()
    validate(artifact)
    decoder_wrapper = AutoAWQForCausalLM.from_quantized(
        artifact / "decoder-awq",
        device_map={"": device},
        torch_dtype=torch.float16,
        safetensors=True,
        fuse_layers=False,
        use_exllama=False,
        use_exllama_v2=False,
    )
    decoder = decoder_wrapper.model.model

    config = VibeVoiceConfig.from_pretrained(artifact)
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
        raise RuntimeError(f"Unmaterialized tensors after AWQ load: {meta[:10]}")
    modules = [module for module in model.model.language_model.modules() if isinstance(module, WQLinear_GEMM)]
    if len(modules) != 196:
        raise RuntimeError(f"Expected 196 WQLinear_GEMM modules, found {len(modules)}")
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise RuntimeError("lm_head and embed_tokens are not tied")
    invalid_dense_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float16
    ]
    if invalid_dense_dtypes:
        raise RuntimeError(f"Non-AWQ parameters are not FP16: {invalid_dense_dtypes[:5]}")
    object.__setattr__(model, "_awq_decoder_wrapper", decoder_wrapper)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--metadata", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    quantize_parser = subparsers.add_parser("quantize")
    quantize_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--remove-work", action="store_true")
    validate_parser.add_argument("--verify-source", action="store_true")
    subparsers.add_parser("preflight")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare(args.source.resolve(), args.output.resolve(), args.metadata.resolve(), args.force)
    elif args.command == "quantize":
        quantize(args.output.resolve())
    elif args.command == "validate":
        validate(args.output.resolve(), args.remove_work, args.verify_source)
    elif args.command == "preflight":
        preflight()


if __name__ == "__main__":
    main()
