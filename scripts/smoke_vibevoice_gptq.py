#!/usr/bin/env python3
"""Fresh-process reload and TTS smoke test for the selective GPTQ artifact."""

import argparse
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

from gptq_vibevoice import VibeVoiceProcessor, load_gptq_vibevoice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text",
        default="Hola, esta es una prueba de síntesis de voz en español.",
    )
    parser.add_argument("--max-wer", type=float, default=0.5)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_gptq_vibevoice(args.model, device="cuda:0", backend="triton")
    load_seconds = time.perf_counter() - load_start
    processor = VibeVoiceProcessor.from_pretrained(args.model)

    inputs = processor(
        text=[f"Speaker 1: {args.text}"],
        voice_samples=None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in inputs.items()
        if value is not None
    }
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )
    generation_seconds = time.perf_counter() - generation_start
    if not output.speech_outputs or output.speech_outputs[0] is None:
        raise RuntimeError("GPTQ smoke test produced no audio")

    audio = output.speech_outputs[0].squeeze().float().cpu().numpy()
    if not np.isfinite(audio).all():
        raise RuntimeError("GPTQ smoke test produced NaN or Inf audio")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-5 or peak < 1e-4:
        raise RuntimeError(f"GPTQ smoke test produced silence: RMS={rms}, peak={peak}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 24000)
    duration = len(audio) / 24000
    packed = sum(
        hasattr(module, "qweight") for module in model.model.language_model.modules()
    )
    tts_peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    del model, output
    torch.cuda.empty_cache()
    asr = whisper.load_model("large-v3", device="cuda")
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=24000, target_sr=16000)
    transcript = asr.transcribe(
        audio_16k,
        language="es",
        fp16=True,
        condition_on_previous_text=False,
        verbose=False,
    )["text"].strip()
    wer = float(jiwer.wer(args.text.lower(), transcript.lower()))
    if wer > args.max_wer:
        raise RuntimeError(
            f"GPTQ smoke WER {wer:.3f} exceeds {args.max_wer:.3f}: {transcript!r}"
        )

    metrics = {
        "model": str(args.model.resolve()),
        "audio": str(args.output.resolve()),
        "gptq_modules": packed,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "audio_seconds": duration,
        "audio_rms": rms,
        "audio_peak": peak,
        "transcript": transcript,
        "wer": wer,
        "rtf": generation_seconds / duration,
        "vram_peak_gib": tts_peak_vram,
    }
    if packed != 196:
        raise RuntimeError(f"Expected 196 GPTQ modules, found {packed}")
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.model / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"structure_validated", "validated"}:
        raise RuntimeError(f"Unexpected pre-smoke artifact status: {manifest.get('status')}")
    manifest["status"] = "validated"
    manifest["smoke"] = metrics
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
