#!/usr/bin/env python3
# ruff: noqa: E402
"""Fresh-process reload and TTS smoke test for the selective GPTQ artifact."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import numpy as np
import librosa
import jiwer
import whisper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gptq_vibevoice import VibeVoiceProcessor, load_gptq_vibevoice, validate
from quantization_common import atomic_json
from validate_vibevoice_es_quality import TEXTS, normalize


VOICE = ROOT / "VibeVoice_repo" / "demo" / "voices" / "en-Alice_woman.wav"
TARGET_SR = 24000
PROTOCOL = "six-texts-en-Alice-large-v3-WER-0.30-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wer", type=float, default=0.30)
    args = parser.parse_args()
    if not VOICE.is_file():
        raise FileNotFoundError(f"Missing fixed voice sample: {VOICE}")
    metrics_path = args.output.with_suffix(".json")
    manifest_path = args.model / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metrics_path.is_file() and manifest.get("status") == "quality_failed":
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol") == PROTOCOL
            and existing.get("gptq_modules") == 196
            and existing.get("max_wer") == args.max_wer
            and existing.get("passed") is False
        ):
            validate(args.model)
            print("Existing GPTQ quality rejection validated; generation not repeated")
            print(json.dumps(existing, indent=2, ensure_ascii=False))
            return

    torch.cuda.empty_cache()
    load_start = time.perf_counter()
    model = load_gptq_vibevoice(args.model, device="cuda:0", backend="triton")
    load_seconds = time.perf_counter() - load_start
    processor = VibeVoiceProcessor.from_pretrained(args.model, local_files_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    packed = sum(
        hasattr(module, "qweight") for module in model.model.language_model.modules()
    )
    if packed != 196:
        raise RuntimeError(f"Expected 196 GPTQ modules, found {packed}")
    peak_vram = torch.cuda.memory_allocated() / 1024**3
    generation_seconds = []
    durations = []
    audio_stats = []
    wav_paths = []
    for index, text in enumerate(TEXTS, start=1):
        torch.manual_seed(42 + index - 1)
        inputs = processor(
            text=[f"Speaker 1: {text}"],
            voice_samples=[[str(VOICE)]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in inputs.items()
            if value is not None
        }
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=600,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=True,
            )
        elapsed = time.perf_counter() - started
        if not generated.speech_outputs or generated.speech_outputs[0] is None:
            raise RuntimeError(f"GPTQ produced no audio for sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError(f"GPTQ produced NaN or Inf audio for sample {index}")
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        peak = float(np.max(np.abs(audio)))
        if rms < 1e-5 or peak < 1e-4:
            raise RuntimeError(f"GPTQ produced silence for sample {index}: RMS={rms}, peak={peak}")
        duration = len(audio) / TARGET_SR
        wav = args.output if index == 1 else args.output.with_name(
            f"{args.output.stem}-{index:02d}{args.output.suffix}"
        )
        temporary = wav.with_suffix(wav.suffix + ".partial")
        sf.write(temporary, audio, TARGET_SR, subtype="PCM_16", format="WAV")
        temporary.replace(wav)
        generation_seconds.append(elapsed)
        durations.append(duration)
        audio_stats.append({"rms": rms, "peak": peak})
        wav_paths.append(wav)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)

    del model, generated, inputs, processor
    gc.collect()
    torch.cuda.empty_cache()
    asr = whisper.load_model(
        "large-v3", device="cpu", download_root=str(Path.home() / ".cache" / "whisper")
    )
    samples = []
    for text, wav, stats in zip(TEXTS, wav_paths, audio_stats):
        audio_16k, _ = librosa.load(wav, sr=16000, mono=True)
        transcript = asr.transcribe(
            audio_16k,
            language="es",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=1,
            sample_len=64,
        )["text"].strip()
        samples.append(
            {
                "reference": text,
                "transcript": transcript,
                "wer": float(jiwer.wer(normalize(text), normalize(transcript))),
                "cer": float(jiwer.cer(normalize(text), normalize(transcript))),
                **stats,
            }
        )
    del asr
    wer = float(np.mean([sample["wer"] for sample in samples]))

    metrics = {
        "model": str(args.model.resolve()),
        "audio": str(args.output.resolve()),
        "gptq_modules": packed,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "audio_seconds": durations,
        "wer": wer,
        "cer": float(np.mean([sample["cer"] for sample in samples])),
        "max_wer": args.max_wer,
        "passed": wer <= args.max_wer,
        "rtf": float(np.mean([elapsed / duration for elapsed, duration in zip(generation_seconds, durations)])),
        "vram_peak_gib": peak_vram,
        "voice": str(VOICE.resolve()),
        "protocol": PROTOCOL,
        "samples": samples,
    }
    atomic_json(metrics_path, metrics)
    manifest["smoke"] = metrics
    if not metrics["passed"]:
        manifest["status"] = "quality_failed"
        atomic_json(manifest_path, manifest)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"GPTQ rejected: WER {wer:.4f} exceeds {args.max_wer:.4f}")
        return
    if manifest.get("status") not in {"structure_validated", "validated"}:
        raise RuntimeError(f"Unexpected pre-smoke artifact status: {manifest.get('status')}")
    manifest["status"] = "validated"
    atomic_json(manifest_path, manifest)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
