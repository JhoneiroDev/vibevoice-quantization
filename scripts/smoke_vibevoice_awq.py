#!/usr/bin/env python3
# ruff: noqa: E402
"""Fresh-process reload and TTS smoke test for the selective AWQ artifact."""

import argparse
import json
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from awq.modules.linear.gemm import WQLinear_GEMM
from awq_vibevoice import VibeVoiceProcessor, load_awq_vibevoice
from quantization_common import atomic_json


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
    model = load_awq_vibevoice(args.model, device="cuda:0")
    load_seconds = time.perf_counter() - load_start
    processor = VibeVoiceProcessor.from_pretrained(
        args.model, local_files_only=True
    )

    hits = set()
    handles = []
    for name, module in model.model.language_model.named_modules():
        if isinstance(module, WQLinear_GEMM):
            handles.append(module.register_forward_hook(lambda _m, _i, _o, n=name: hits.add(n)))

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
    for handle in handles:
        handle.remove()
    if len(hits) != 196:
        raise RuntimeError(f"Only {len(hits)}/196 AWQ modules executed")
    if not output.speech_outputs or output.speech_outputs[0] is None:
        raise RuntimeError("AWQ smoke test produced no audio")

    audio = output.speech_outputs[0].squeeze().float().cpu().numpy()
    if not np.isfinite(audio).all():
        raise RuntimeError("AWQ smoke test produced NaN or Inf audio")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-5 or peak < 1e-4:
        raise RuntimeError(f"AWQ smoke test produced silence: RMS={rms}, peak={peak}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 24000)
    duration = len(audio) / 24000
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
            f"AWQ smoke WER {wer:.3f} exceeds {args.max_wer:.3f}: {transcript!r}"
        )

    metrics = {
        "model": str(args.model.resolve()),
        "audio": str(args.output.resolve()),
        "awq_modules": len(hits),
        "backend": "AutoAWQ GEMM/Triton",
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
    metrics_path = args.output.with_suffix(".json")
    atomic_json(metrics_path, metrics)
    manifest_path = args.model / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"structure_validated", "validated"}:
        raise RuntimeError(f"Unexpected pre-smoke artifact status: {manifest.get('status')}")
    manifest["status"] = "validated"
    manifest["smoke"] = metrics
    atomic_json(manifest_path, manifest)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
