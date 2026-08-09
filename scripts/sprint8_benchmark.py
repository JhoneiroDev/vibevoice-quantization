#!/usr/bin/env python3
"""Fresh-process final benchmark for every available VibeVoice variant."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
import whisper


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validate_vibevoice_es_quality import TEXTS, normalize
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


VOICE = REPO / "demo" / "voices" / "en-Alice_woman.wav"
OUTPUT = ROOT / "outputs" / "sprint8_benchmark"
TARGET_SR = 24000
MAX_WER = 0.30
PROTOCOL = "six-texts-en-Alice-large-v3-WER-0.30-v1"

MODELS = {
    "fp16": (ROOT / "weights" / "vibevoice-1.5b-es", "native"),
    "torchao_int4": (ROOT / "weights" / "vibevoice-1.5b-es-torchao-int4", "native"),
    "gptq_tts": (ROOT / "weights" / "vibevoice-1.5b-es-gptq-tts", "gptq"),
    "awq": (ROOT / "weights" / "vibevoice-1.5b-es-awq", "awq"),
    "int8": (ROOT / "weights" / "vibevoice-1.5b-es-int8", "bnb_int8"),
    "nf4": (ROOT / "weights" / "vibevoice-1.5b-es-nf4.partial", "bnb_nf4"),
    "smoothquant": (ROOT / "weights" / "vibevoice-1.5b-es-smoothquant", "smoothquant"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def artifact_status(path: Path) -> str:
    manifest = path / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8")).get("status", "unknown")
    return "dense_baseline"


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

        mode = kind.removeprefix("bnb_")
        config, _, dtype = _configuration(mode)
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


def bootstrap_ci(values: list[float], seed: int = 42, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def run_worker(model_id: str) -> dict:
    path, kind = MODELS[model_id]
    if not path.is_dir():
        raise FileNotFoundError(f"Missing benchmark artifact: {path}")
    if not VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice: {VOICE}")
    torch.cuda.empty_cache()
    load_started = time.perf_counter()
    model = load_variant(path, kind)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    load_seconds = time.perf_counter() - load_started
    processor = VibeVoiceProcessor.from_pretrained(path, local_files_only=True)
    idle_vram = torch.cuda.memory_allocated() / 1024**3
    generation_seconds, audio_seconds, audio_paths = [], [], []
    sample_stats = []
    model_output = OUTPUT / model_id
    model_output.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(TEXTS, start=1):
        torch.manual_seed(42 + index - 1)
        inputs = processor(
            text=[f"Speaker 1: {text}"], voice_samples=[[str(VOICE)],],
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        inputs = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                  for key, value in inputs.items() if value is not None}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=600, tokenizer=processor.tokenizer,
                generation_config={"do_sample": False}, verbose=False, is_prefill=True,
            )
        elapsed = time.perf_counter() - started
        if not generated.speech_outputs or generated.speech_outputs[0] is None:
            raise RuntimeError(f"No audio for {model_id} sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all() or len(audio) == 0:
            raise RuntimeError(f"Invalid audio for {model_id} sample {index}")
        duration = len(audio) / TARGET_SR
        wav = model_output / f"sample_{index:02d}.wav"
        sf.write(wav.with_suffix(".wav.partial"), audio, TARGET_SR, subtype="PCM_16", format="WAV")
        wav.with_suffix(".wav.partial").replace(wav)
        generation_seconds.append(elapsed)
        audio_seconds.append(duration)
        audio_paths.append(wav)
        sample_stats.append({
            "index": index, "generation_seconds": elapsed, "audio_seconds": duration,
            "rtf": elapsed / duration,
            "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
            "peak": float(np.max(np.abs(audio))),
            "vram_peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        })
    peak_vram = max(item["vram_peak_gib"] for item in sample_stats)
    del model, generated, inputs, processor
    gc.collect()
    torch.cuda.empty_cache()
    asr = whisper.load_model("large-v3", device="cpu", download_root=str(Path.home() / ".cache" / "whisper"))
    samples = []
    for text, wav, stats in zip(TEXTS, audio_paths, sample_stats):
        audio, _ = librosa.load(wav, sr=16000, mono=True)
        transcript = asr.transcribe(
            audio, language="es", fp16=False, verbose=False,
            condition_on_previous_text=False, temperature=0.0, beam_size=1, sample_len=64,
        )["text"].strip()
        samples.append({
            "reference": text, "transcript": transcript,
            "wer": float(jiwer.wer(normalize(text), normalize(transcript))),
            "cer": float(jiwer.cer(normalize(text), normalize(transcript))), **stats,
        })
    del asr
    wers = [sample["wer"] for sample in samples]
    cers = [sample["cer"] for sample in samples]
    rtfs = [sample["rtf"] for sample in samples]
    result = {
        "model_id": model_id, "model": str(path.resolve()), "kind": kind,
        "artifact_status": artifact_status(path), "artifact_bytes": artifact_bytes(path),
        "artifact_gib": artifact_bytes(path) / 1024**3,
        "load_seconds": load_seconds, "vram_idle_gib": idle_vram,
        "vram_peak_gib": peak_vram, "generation_seconds": generation_seconds,
        "audio_seconds": audio_seconds, "rtf": float(np.mean(rtfs)),
        "rtf_std": float(np.std(rtfs, ddof=1)), "wer": float(np.mean(wers)),
        "cer": float(np.mean(cers)), "corpus_wer": float(jiwer.wer(
            " ".join(normalize(text) for text in TEXTS),
            " ".join(normalize(sample["transcript"]) for sample in samples),
        )),
        "wer_ci95": bootstrap_ci(wers), "cer_ci95": bootstrap_ci(cers),
        "rtf_ci95": bootstrap_ci(rtfs), "max_wer": MAX_WER,
        "passed": float(np.mean(wers)) <= MAX_WER,
        "protocol": PROTOCOL, "voice": str(VOICE.resolve()),
        "samples": samples,
    }
    return result


def write_reports(results: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    successful = [result for result in results if "error" not in result]
    summary = {
        "protocol": PROTOCOL,
        "models_total": len(results),
        "models_completed": len(successful),
        "models_passed": sum(bool(result.get("passed")) for result in successful),
        "models_quality_failed": sum(not result.get("passed", False) for result in successful),
        "wer_ranking": [result["model_id"] for result in sorted(successful, key=lambda item: item["wer"])],
        "rtf_ranking": [result["model_id"] for result in sorted(successful, key=lambda item: item["rtf"])],
        "vram_ranking": [result["model_id"] for result in sorted(successful, key=lambda item: item["vram_peak_gib"])],
        "best_wer": min(successful, key=lambda item: item["wer"])["model_id"] if successful else None,
        "best_rtf": min(successful, key=lambda item: item["rtf"])["model_id"] if successful else None,
        "best_vram": min(successful, key=lambda item: item["vram_peak_gib"])["model_id"] if successful else None,
        "unavailable_metrics": {
            "pesq": "Not computed: no paired reference waveform exists for the six canonical texts.",
            "mcd": "Not computed: no paired reference waveform exists for the six canonical texts.",
        },
    }
    baseline = next((result for result in successful if result["model_id"] == "fp16"), None)
    comparisons = {}
    if baseline is not None:
        try:
            from scipy.stats import shapiro, wilcoxon
        except ImportError:
            comparisons["scipy"] = "unavailable"
        else:
            for result in successful:
                if result["model_id"] == "fp16":
                    continue
                baseline_wers = [sample["wer"] for sample in baseline["samples"]]
                model_wers = [sample["wer"] for sample in result["samples"]]
                differences = np.asarray(model_wers) - np.asarray(baseline_wers)
                try:
                    wilcoxon_result = wilcoxon(differences, alternative="less", zero_method="wilcox")
                    wilcoxon_p = float(wilcoxon_result.pvalue)
                except ValueError:
                    wilcoxon_p = None
                try:
                    shapiro_p = float(shapiro(differences).pvalue)
                except ValueError:
                    shapiro_p = None
                comparisons[result["model_id"]] = {
                    "baseline": "fp16",
                    "mean_wer_difference_vs_fp16": float(np.mean(differences)),
                    "shapiro_p_difference": shapiro_p,
                    "wilcoxon_p_model_better": wilcoxon_p,
                    "n_pairs": len(differences),
                }
    summary["paired_statistics_vs_fp16"] = comparisons
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["model_id", "kind", "artifact_status", "artifact_gib", "load_seconds", "vram_idle_gib",
              "vram_peak_gib", "rtf", "rtf_std", "wer", "cer", "corpus_wer", "passed"]
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result.get(field) for field in fields} for result in results)
    lines = ["# Sprint 8: Benchmark Final", "", f"Protocol: `{PROTOCOL}`", "", "| Modelo | Estado | Disco GiB | VRAM pico | RTF | WER | CER | Pasa |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        if "error" in result:
            lines.append(f"| {result['model_id']} | ERROR | — | — | — | — | — | No |")
            continue
        lines.append(f"| {result['model_id']} | {result['artifact_status']} | {result['artifact_gib']:.2f} | {result['vram_peak_gib']:.2f} | {result['rtf']:.4f} | {result['wer']:.4f} | {result['cer']:.4f} | {'Sí' if result['passed'] else 'No'} |")
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=sorted(MODELS))
    parser.add_argument("--models", nargs="*", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--append", action="store_true", help="Merge selected results with an existing report")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args.worker), ensure_ascii=False))
        return
    results = []
    for model_id in args.models:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, __file__, "--worker", model_id],
            cwd=ROOT, capture_output=True, text=True,
        )
        if process.returncode != 0:
            results.append({"model_id": model_id, "error": process.stderr[-4000:]})
            continue
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1])
        result["worker_wall_seconds"] = time.perf_counter() - started
        results.append(result)
        print(f"{model_id}: WER={result['wer']:.4f}, RTF={result['rtf']:.4f}, VRAM={result['vram_peak_gib']:.2f} GiB")
    if args.append and (OUTPUT / "results.json").is_file():
        previous = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
        selected = set(args.models)
        results = [result for result in previous if result.get("model_id") not in selected] + results
        results.sort(key=lambda result: result.get("model_id", ""))
    write_reports(results)
    print(f"Benchmark written to {OUTPUT}")


if __name__ == "__main__":
    main()
