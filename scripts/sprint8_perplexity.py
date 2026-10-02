#!/usr/bin/env python3
"""Teacher-forced text perplexity for every VibeVoice benchmark variant."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.finetune.data_vibevoice import VibeVoiceCollator
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


OUTPUT = ROOT / "outputs" / "sprint8_perplexity"
CALIBRATION_METADATA = ROOT / "data" / "calibration_metadata.json"
DATASET_ID = "fsicoli/common_voice_17_0"
DATASET_CONFIG = "es"
SPLIT = "validation"
PREFIX = "Speaker 1: "
VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"
SEED = 42
DEFAULT_SAMPLES = 100
PROTOCOL = "common-voice-17-es-validation-heldout-acoustic-control-ppl-v1"

MODELS = {
    "fp16": (ROOT / "weights" / "vibevoice-1.5b-es", "native"),
    "torchao_int4": (ROOT / "weights" / "vibevoice-1.5b-es-torchao-int4", "native"),
    "gptq_tts": (ROOT / "weights" / "vibevoice-1.5b-es-gptq-tts", "gptq"),
    "awq": (ROOT / "weights" / "vibevoice-1.5b-es-awq", "awq"),
    "int8": (ROOT / "weights" / "vibevoice-1.5b-es-int8", "bnb_int8"),
    "nf4": (ROOT / "weights" / "vibevoice-1.5b-es-nf4.partial", "bnb_nf4"),
    "smoothquant": (ROOT / "weights" / "vibevoice-1.5b-es-smoothquant", "smoothquant"),
}


def load_variant(path: Path, kind: str):
    if kind == "gptq":
        from gptq_vibevoice import load_gptq_vibevoice

        return load_gptq_vibevoice(path, device="cuda:0", backend="triton")
    if kind == "awq":
        from awq_vibevoice import load_awq_vibevoice

        return load_awq_vibevoice(path, device="cuda:0")
    if kind == "smoothquant":
        from smoothquant_vibevoice import load_smoothquant_vibevoice

        return load_smoothquant_vibevoice(path, device_map={"": "cuda:0"})
    if kind.startswith("bnb_"):
        from bnb_quantize_vibevoice import _configuration

        config, _, dtype = _configuration(kind.removeprefix("bnb_"))
        return VibeVoiceForConditionalGenerationInference.from_pretrained(
            path,
            quantization_config=config,
            torch_dtype=dtype,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
    return VibeVoiceForConditionalGenerationInference.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )


def select_sentences(sample_count: int, seed: int) -> dict:
    metadata = json.loads(CALIBRATION_METADATA.read_text(encoding="utf-8"))
    excluded = {int(record["local_idx"]) for record in metadata["records"]}
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        download_mode="reuse_dataset_if_exists",
    )
    sentences = dataset["sentence"]
    paths = dataset["path"]
    candidates = [
        (index, sentence.strip(), paths[index])
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and isinstance(sentence, str)
        and sentence.strip()
        and isinstance(paths[index], str)
    ]
    if len(candidates) < sample_count:
        raise ValueError(f"Requested {sample_count} held-out sentences, found {len(candidates)}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=sample_count, replace=False)
    records = [
        {
            "dataset_index": candidates[int(position)][0],
            "sentence": candidates[int(position)][1],
            "audio": candidates[int(position)][2],
        }
        for position in chosen
    ]
    serialized = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "split": SPLIT,
        "excluded_calibration_records": len(excluded),
        "sample_count": sample_count,
        "seed": seed,
        "selection_sha256": hashlib.sha256(serialized).hexdigest(),
        "records": records,
    }


def evaluate(model_id: str, sample_count: int, seed: int, batch_size: int) -> dict:
    path, kind = MODELS[model_id]
    selection = select_sentences(sample_count, seed)
    model = load_variant(path, kind)
    model.eval()
    processor = VibeVoiceProcessor.from_pretrained(path, local_files_only=True)
    collator = VibeVoiceCollator(
        processor=processor,
        speech_compress_ratio=processor.speech_tok_compress_ratio,
        semantic_vae_dim=model.config.semantic_vae_dim,
        compute_semantics=False,
        debug_checks=True,
        voice_prompt_drop_rate=0.0,
        target_audio_dbfs=-25.0,
        augment_target_silence=False,
    )
    sentence_results = []
    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        records = selection["records"]
        for start in range(0, len(records), batch_size):
            batch_seed = seed + start
            random.seed(batch_seed)
            np.random.seed(batch_seed)
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            batch = records[start : start + batch_size]
            encoded = collator(
                [
                    {
                        "text": PREFIX + record["sentence"],
                        "audio": record["audio"],
                        "voice_prompts": [str(VOICE)],
                    }
                    for record in batch
                ]
            )
            encoded = {
                key: value.to("cuda:0") if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            speech_tensors = encoded["speech_tensors"]
            speech_masks = encoded["speech_masks"]
            acoustic_input_mask = encoded["acoustic_input_mask"]
            acoustic_loss_mask = encoded["acoustic_loss_mask"]

            embeddings = model.get_input_embeddings()(input_ids)
            semantic_dtype = next(model.model.semantic_tokenizer.parameters()).dtype
            semantic_output = model.model.semantic_tokenizer.encode(
                speech_tensors.unsqueeze(1).to(dtype=semantic_dtype)
            )
            semantic_connected = model.model.semantic_connector(semantic_output.mean)
            _, acoustic_connected = model._process_speech_inputs(
                speech_tensors.type_as(embeddings),
                speech_masks,
                speech_type="audio",
            )
            embeddings[acoustic_input_mask] = (
                acoustic_connected + semantic_connected[speech_masks]
            )
            outputs = model.model(
                input_ids=None,
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[:, :-1, :]
            labels = input_ids[:, 1:]
            target_continuation = acoustic_loss_mask[:, 1:]
            target_end = acoustic_loss_mask[:, :-1] & ~target_continuation
            score_mask = attention_mask[:, 1:].bool() & (target_continuation | target_end)
            if not score_mask.any():
                raise RuntimeError("Collator produced no acoustic-control targets for perplexity")
            rows = torch.arange(len(batch), device="cuda:0").unsqueeze(1).expand_as(score_mask)
            selected_rows = rows[score_mask]
            selected_hidden = hidden[score_mask]
            selected_labels = labels[score_mask]
            logits = model.lm_head(selected_hidden)
            losses = F.cross_entropy(logits.float(), selected_labels, reduction="none")

            for row, record in enumerate(batch):
                row_losses = losses[selected_rows == row]
                n_tokens = int(row_losses.numel())
                nll = float(row_losses.sum().item())
                sentence_results.append(
                    {
                        "dataset_index": record["dataset_index"],
                        "sentence": record["sentence"],
                        "tokens": n_tokens,
                        "mean_nll": nll / n_tokens,
                        "perplexity": math.exp(nll / n_tokens),
                    }
                )
                total_nll += nll
                total_tokens += n_tokens
            del outputs, hidden, labels, logits, losses, selected_hidden, selected_labels
            del embeddings, semantic_output, semantic_connected, acoustic_connected, encoded

    mean_nll = total_nll / total_tokens
    result = {
        "model_id": model_id,
        "model": str(path.resolve()),
        "kind": kind,
        "protocol": PROTOCOL,
        "prefix": PREFIX,
        "loss": "teacher-forced causal cross-entropy over target speech-diffusion continuations and terminal token",
        "metric_scope": "acoustic-control-token perplexity; not lexical text perplexity",
        "sample_count": sample_count,
        "token_count": total_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "mean_sentence_perplexity": float(
            np.mean([item["perplexity"] for item in sentence_results])
        ),
        "selection": {key: value for key, value in selection.items() if key != "records"},
        "sentences": sentence_results,
    }
    del model, processor, collator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_reports(results: list[dict], selection: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = ["model_id", "kind", "sample_count", "token_count", "mean_nll", "perplexity"]
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result.get(field) for field in fields} for result in results)
    lines = [
        "# Sprint 8: Perplejidad textual",
        "",
        f"Protocolo: `{PROTOCOL}`",
        "",
        "| Modelo | Oraciones | Tokens | NLL media | PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        if "error" in result:
            lines.append(f"| {result['model_id']} | ERROR | ERROR | ERROR | ERROR |")
        else:
            lines.append(
                f"| {result['model_id']} | {result['sample_count']} | {result['token_count']} | "
                f"{result['mean_nll']:.6f} | {result['perplexity']:.4f} |"
            )
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_into_sprint8_benchmark(results: list[dict]) -> None:
    if len(results) != len(MODELS) or any("error" in result for result in results):
        return
    perplexity = {result["model_id"]: result for result in results}
    for benchmark_output in (
        ROOT / "outputs" / "sprint8_benchmark",
        ROOT / "outputs" / "sprint8_benchmark_n100",
    ):
        benchmark_path = benchmark_output / "results.json"
        if not benchmark_path.is_file():
            continue
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for result in benchmark:
            ppl = perplexity.get(result.get("model_id"))
            if ppl is None:
                continue
            result["perplexity"] = ppl["perplexity"]
            result["perplexity_mean_nll"] = ppl["mean_nll"]
            result["perplexity_samples"] = ppl["sample_count"]
            result["perplexity_tokens"] = ppl["token_count"]
            result["perplexity_protocol"] = ppl["protocol"]
            result["perplexity_scope"] = ppl["metric_scope"]
        from sprint8_benchmark import write_reports as write_benchmark_reports

        protocol = benchmark[0].get("protocol", "unknown") if benchmark else "unknown"
        write_benchmark_reports(benchmark, benchmark_output, protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=sorted(MODELS))
    parser.add_argument("--models", nargs="*", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--append", action="store_true", help="Merge selected models into existing results")
    parser.add_argument("--merge-only", action="store_true", help="Regenerate and merge existing reports")
    parser.add_argument("--selection-only", type=Path, help="Write held-out selection JSON without models")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("--samples and --batch-size must be positive")
    if args.selection_only is not None:
        selection = select_sentences(args.samples, args.seed)
        args.selection_only.parent.mkdir(parents=True, exist_ok=True)
        args.selection_only.write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Held-out selection written to {args.selection_only}")
        return
    if args.merge_only:
        results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
        selection = json.loads((OUTPUT / "selection.json").read_text(encoding="utf-8"))
        write_reports(results, selection)
        merge_into_sprint8_benchmark(results)
        print(f"Existing perplexity reports merged from {OUTPUT}")
        return
    if args.worker:
        print(json.dumps(evaluate(args.worker, args.samples, args.seed, args.batch_size), ensure_ascii=False))
        return

    selection = select_sentences(args.samples, args.seed)
    results = []
    for model_id in args.models:
        process = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                model_id,
                "--samples",
                str(args.samples),
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            results.append({"model_id": model_id, "error": process.stderr[-4000:]})
            continue
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1])
        if result["selection"]["selection_sha256"] != selection["selection_sha256"]:
            raise RuntimeError(f"Selection mismatch for {model_id}")
        results.append(result)
        print(f"{model_id}: PPL={result['perplexity']:.4f}, tokens={result['token_count']}")
    if args.append and (OUTPUT / "results.json").is_file():
        previous = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
        selected = set(args.models)
        results = [result for result in previous if result.get("model_id") not in selected] + results
        results.sort(key=lambda result: result.get("model_id", ""))
    write_reports(results, selection)
    merge_into_sprint8_benchmark(results)
    print(f"Perplexity reports written to {OUTPUT}")


if __name__ == "__main__":
    main()
