#!/usr/bin/env python3
"""Persistent Spanish TTS quality gate for the corrected VibeVoice checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
import whisper


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "VibeVoice_repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if hasattr(transformers, "CONFIG_MAPPING"):
    transformers.CONFIG_MAPPING._extra_content.pop("vibevoice_acoustic_tokenizer", None)
    if hasattr(transformers.CONFIG_MAPPING, "_mapping"):
        transformers.CONFIG_MAPPING._mapping.pop("vibevoice_acoustic_tokenizer", None)

from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: E402
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: E402


TEXTS = [
    "El ping\u00fcino camina r\u00e1pidamente sobre el hielo.",
    "\u00bfCu\u00e1ndo llegar\u00e1 el pr\u00f3ximo tren a la estaci\u00f3n?",
    "La educaci\u00f3n p\u00fablica requiere atenci\u00f3n y planificaci\u00f3n.",
    "Mi hermana compr\u00f3 veintitr\u00e9s manzanas para la reuni\u00f3n.",
    "\u00a1Qu\u00e9 alegr\u00eda escuchar m\u00fasica espa\u00f1ola esta ma\u00f1ana!",
    "Los cient\u00edficos analizaron cuidadosamente los resultados.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f10-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wer", type=float, default=0.30)
    args = parser.parse_args()

    model_path = args.model.resolve()
    output = args.output.resolve()
    config = model_path / "config.json"
    weights = sorted(model_path.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError(f"Incomplete corrected checkpoint: {model_path}")
    source_hashes = {path.name: sha256(path) for path in [config, *weights]}
    output.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    processor = VibeVoiceProcessor.from_pretrained(model_path, local_files_only=True)
    rtfs = []
    peak_vram = torch.cuda.memory_allocated() / 1024**3
    wav_paths = []
    for index, text in enumerate(TEXTS, start=1):
        inputs = processor(
            text=[f"Speaker 1: {text}"],
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
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=600,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )
        elapsed = time.perf_counter() - started
        if not generated.speech_outputs or generated.speech_outputs[0] is None:
            raise RuntimeError(f"No audio produced for sample {index}")
        audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite audio for sample {index}")
        duration = len(audio) / 24000
        rtfs.append(elapsed / duration)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**3)
        partial = output / f"sample_{index:02d}.wav.partial"
        final = output / f"sample_{index:02d}.wav"
        sf.write(partial, audio, 24000, subtype="PCM_16", format="WAV")
        partial.replace(final)
        wav_paths.append(final)

    del model, processor, generated, inputs
    gc.collect()
    torch.cuda.empty_cache()

    whisper_cache = Path.home() / ".cache" / "whisper"
    asr = whisper.load_model("large-v3", device="cuda", download_root=str(whisper_cache))
    samples = []
    for text, wav_path in zip(TEXTS, wav_paths):
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        transcript = asr.transcribe(
            audio,
            language="es",
            fp16=True,
            verbose=False,
            condition_on_previous_text=False,
        )["text"].strip()
        wer = float(jiwer.wer(normalize(text), normalize(transcript)))
        cer = float(jiwer.cer(normalize(text), normalize(transcript)))
        samples.append({"reference": text, "transcript": transcript, "wer": wer, "cer": cer})
        print(f"WER={wer:.3f} | {transcript}")

    metrics = {
        "source": str(model_path),
        "source_hashes": source_hashes,
        "max_wer": args.max_wer,
        "wer": float(np.mean([sample["wer"] for sample in samples])),
        "cer": float(np.mean([sample["cer"] for sample in samples])),
        "rtf": float(np.mean(rtfs)),
        "vram_peak_gib": peak_vram,
        "samples": samples,
    }
    metrics["passed"] = metrics["wer"] <= args.max_wer
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "metrics.json")
    print(f"WER medio={metrics['wer']:.4f} | CER={metrics['cer']:.4f}")
    if not metrics["passed"]:
        raise RuntimeError(f"Quality gate failed: WER {metrics['wer']:.4f} > {args.max_wer:.4f}")


if __name__ == "__main__":
    main()
