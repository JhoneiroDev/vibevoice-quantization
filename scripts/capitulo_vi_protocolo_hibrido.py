#!/usr/bin/env python3
"""Protocolo hibrido real para los resultados del Capitulo VI.

Calidad: cinco bloques disjuntos de 20 frases ya evaluadas.
Hardware: cinco procesos limpios por modelo, con seis sintesis por proceso.

Desde Jupyter:
    %run scripts/capitulo_vi_protocolo_hibrido.py --run-hardware

La ejecucion es reanudable: cada repeticion fisica se guarda por separado.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REPO = ROOT / "VibeVoice_repo"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
OUTPUT = ROOT / "outputs" / "capitulo_vi_hibrido"
QUALITY_RESULTS = ROOT / "outputs" / "sprint8_benchmark_n100" / "results.json"
PPL_RESULTS = ROOT / "outputs" / "sprint8_perplexity" / "results.json"
HARDWARE_RUNS = OUTPUT / "hardware_runs"
N_MODELS = 7
N_SENTENCES = 100
N_BLOCKS = 5
BLOCK_SIZE = 20
N_REPETITIONS = 5
TARGET_SR = 24_000

MODEL_ORDER = ["fp16", "torchao_int4", "gptq_tts", "awq", "int8", "nf4", "smoothquant"]
DISPLAY_NAMES = {
    "fp16": "Base BF16",
    "torchao_int4": "TorchAO HQQ INT4",
    "gptq_tts": "GPTQ INT4",
    "awq": "AutoAWQ INT4",
    "int8": "LLM.int8()",
    "nf4": "NF4",
    "smoothquant": "SmoothQuant W8A8",
}
CONTROL_TEXTS = [
    "El tren llegó temprano.",
    "La reunión comenzará mañana a las nueve.",
    "Los científicos analizaron cuidadosamente los resultados.",
    "La educación pública requiere atención, planificación y recursos sostenibles.",
    "El desarrollo de sistemas inteligentes exige evaluar tanto su precisión como el consumo de memoria disponible.",
    "Durante la presentación final, el investigador explicó detalladamente cómo la cuantización permite ejecutar modelos de síntesis de voz en equipos con recursos computacionales limitados.",
]


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_and_index_quality() -> tuple[dict[str, dict], dict[str, dict]]:
    quality_rows = load_json(QUALITY_RESULTS)
    ppl_rows = load_json(PPL_RESULTS)
    quality = {row["model_id"]: row for row in quality_rows if "error" not in row}
    perplexity = {row["model_id"]: row for row in ppl_rows if "error" not in row}
    if set(quality) != set(MODEL_ORDER) or set(perplexity) != set(MODEL_ORDER):
        raise ValueError(
            f"Se esperaban {N_MODELS} modelos. Calidad={sorted(quality)}, PPL={sorted(perplexity)}"
        )

    canonical_references: list[str] | None = None
    canonical_indices: list[int] | None = None
    for model_id in MODEL_ORDER:
        samples = quality[model_id].get("samples", [])
        sentences = perplexity[model_id].get("sentences", [])
        if len(samples) != N_SENTENCES or len(sentences) != N_SENTENCES:
            raise ValueError(
                f"{model_id}: calidad={len(samples)}, PPL={len(sentences)}; se requieren 100"
            )
        references = [str(sample["reference"]).strip() for sample in samples]
        ppl_texts = [str(sentence["sentence"]).strip() for sentence in sentences]
        indices = [int(sample["index"]) for sample in samples]
        dataset_indices = [int(sentence["dataset_index"]) for sentence in sentences]
        if references != ppl_texts:
            mismatch = next(i for i, pair in enumerate(zip(references, ppl_texts)) if pair[0] != pair[1])
            raise ValueError(f"{model_id}: texto de calidad/PPL no coincide en posicion {mismatch + 1}")
        if indices != list(range(1, N_SENTENCES + 1)):
            raise ValueError(f"{model_id}: los indices de calidad no son consecutivos 1..100")
        if any(int(item["tokens"]) <= 0 for item in sentences):
            raise ValueError(f"{model_id}: existe una frase sin tokens puntuables")
        if canonical_references is None:
            canonical_references = references
            canonical_indices = dataset_indices
        elif references != canonical_references or dataset_indices != canonical_indices:
            raise ValueError(f"{model_id}: el corpus o su orden difiere del resto de modelos")
    return quality, perplexity


def compute_quality_blocks() -> tuple[list[dict], dict[str, dict[str, float]]]:
    quality, perplexity = validate_and_index_quality()
    blocks: list[dict] = []
    summary: dict[str, dict[str, float]] = {}
    for model_id in MODEL_ORDER:
        samples = quality[model_id]["samples"]
        ppl_sentences = perplexity[model_id]["sentences"]
        for block_index in range(N_BLOCKS):
            start = block_index * BLOCK_SIZE
            stop = start + BLOCK_SIZE
            quality_block = samples[start:stop]
            ppl_block = ppl_sentences[start:stop]
            tokens = sum(int(item["tokens"]) for item in ppl_block)
            total_nll = sum(float(item["mean_nll"]) * int(item["tokens"]) for item in ppl_block)
            blocks.append(
                {
                    "model_id": model_id,
                    "block": block_index + 1,
                    "first_sentence": start + 1,
                    "last_sentence": stop,
                    "n_sentences": len(quality_block),
                    "wer": float(np.mean([item["wer"] for item in quality_block])),
                    "cer": float(np.mean([item["cer"] for item in quality_block])),
                    "rtf": float(np.mean([item["rtf"] for item in quality_block])),
                    "tokens": tokens,
                    "total_nll": total_nll,
                    "ppl": math.exp(total_nll / tokens),
                }
            )
        model_blocks = [row for row in blocks if row["model_id"] == model_id]
        summary[model_id] = {}
        for metric in ("wer", "cer", "rtf", "ppl"):
            values = np.asarray([row[metric] for row in model_blocks], dtype=np.float64)
            summary[model_id][f"{metric}_mean"] = float(values.mean())
            summary[model_id][f"{metric}_std"] = float(values.std(ddof=1))
        summary[model_id]["corpus_wer"] = float(quality[model_id]["wer"])
        summary[model_id]["corpus_cer"] = float(quality[model_id]["cer"])
        summary[model_id]["corpus_ppl"] = float(perplexity[model_id]["perplexity"])
    if len(blocks) != N_MODELS * N_BLOCKS:
        raise AssertionError("Se esperaban exactamente 35 resultados de bloque")
    return blocks, summary


def artifact_size_gib(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1024**3


def hardware_worker(model_id: str, repetition: int, output_path: Path) -> dict:
    if model_id not in MODEL_ORDER:
        raise ValueError(f"Modelo desconocido: {model_id}")
    if len(CONTROL_TEXTS) != 6:
        raise AssertionError("El minicorpus debe contener exactamente seis frases")
    for path in (SCRIPTS, REPO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    from sprint8_benchmark import MODELS, load_variant
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible")
    model_path, _kind = MODELS[model_id]
    if not model_path.is_dir():
        raise FileNotFoundError(f"No existe el artefacto: {model_path}")

    processor = VibeVoiceProcessor.from_pretrained(model_path, local_files_only=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = load_variant(model_path, _kind)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=20)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    generation_seconds: list[float] = []
    audio_seconds: list[float] = []
    voice = REPO / "demo" / "voices" / "en-Alice_woman.wav"
    if not voice.is_file():
        raise FileNotFoundError(f"No existe la voz fija: {voice}")
    try:
        for index, text in enumerate(CONTROL_TEXTS, start=1):
            torch.manual_seed(10_000 + index)
            inputs = processor(
                text=[f"Speaker 1: {text}"],
                voice_samples=[[str(voice)]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {
                key: value.to("cuda:0") if torch.is_tensor(value) else value
                for key, value in inputs.items()
                if value is not None
            }
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not generated.speech_outputs or generated.speech_outputs[0] is None:
                raise RuntimeError(f"No se genero audio para la frase {index}")
            audio = generated.speech_outputs[0].squeeze().float().cpu().numpy()
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"Audio invalido para la frase {index}")
            generation_seconds.append(elapsed)
            audio_seconds.append(len(audio) / TARGET_SR)
            del generated, inputs, audio
        result = {
            "model_id": model_id,
            "repetition": repetition,
            "source": "measured",
            "sentences": len(CONTROL_TEXTS),
            "disk_gib": artifact_size_gib(model_path),
            "load_s": load_seconds,
            "vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "generation_s": sum(generation_seconds),
            "audio_s": sum(audio_seconds),
            "rtf": sum(generation_seconds) / sum(audio_seconds),
        }
        if not all(math.isfinite(value) and value >= 0 for key, value in result.items()
                   if key in {"disk_gib", "load_s", "vram_gib", "generation_s", "audio_s", "rtf"}):
            raise RuntimeError(f"El worker produjo metricas invalidas: {result}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
        return result
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def run_hardware_benchmark(resume: bool = True) -> list[dict]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible para el benchmark fisico")
    results: list[dict] = []
    for model_id in MODEL_ORDER:
        for repetition in range(1, N_REPETITIONS + 1):
            result_path = HARDWARE_RUNS / model_id / f"repetition_{repetition:02d}.json"
            if resume and result_path.is_file():
                result = load_json(result_path)
                if result.get("source") != "measured" or result.get("sentences") != 6:
                    raise ValueError(f"Resultado previo invalido: {result_path}")
                print(f"[REANUDADO] {DISPLAY_NAMES[model_id]} {repetition}/5")
                results.append(result)
                continue
            print(f"[EJECUTANDO] {DISPLAY_NAMES[model_id]} {repetition}/5", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                model_id,
                "--repetition",
                str(repetition),
                "--worker-output",
                str(result_path),
            ]
            process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Fallo {model_id} repeticion {repetition}:\n"
                    + (process.stderr or process.stdout)[-5000:]
                )
            result = load_json(result_path)
            results.append(result)
            print(
                f"  carga={result['load_s']:.3f}s VRAM={result['vram_gib']:.3f}GiB "
                f"RTF={result['rtf']:.4f}",
                flush=True,
            )
    return results


def load_complete_hardware_results() -> list[dict]:
    results = []
    for model_id in MODEL_ORDER:
        for repetition in range(1, N_REPETITIONS + 1):
            path = HARDWARE_RUNS / model_id / f"repetition_{repetition:02d}.json"
            result = load_json(path)
            if result.get("model_id") != model_id or int(result.get("repetition", 0)) != repetition:
                raise ValueError(f"Identidad incorrecta en {path}")
            if result.get("source") != "measured":
                raise ValueError(f"El resultado no es una medicion real: {path}")
            results.append(result)
    if len(results) != N_MODELS * N_REPETITIONS:
        raise AssertionError("Se esperaban exactamente 35 mediciones fisicas")
    return results


def summarize_hardware(results: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for model_id in MODEL_ORDER:
        rows = [row for row in results if row["model_id"] == model_id]
        if len(rows) != N_REPETITIONS:
            raise ValueError(f"{model_id}: se esperaban cinco repeticiones fisicas")
        summary[model_id] = {}
        for metric in ("load_s", "vram_gib", "rtf"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[model_id][f"{metric}_mean"] = float(values.mean())
            summary[model_id][f"{metric}_std"] = float(values.std(ddof=1))
        disk_values = np.asarray([row["disk_gib"] for row in rows])
        if not np.allclose(disk_values, disk_values[0], rtol=0, atol=1e-12):
            raise ValueError(f"El tamaño del artefacto {model_id} cambio durante el benchmark")
        summary[model_id]["disk_gib_mean"] = float(disk_values[0])
        summary[model_id]["disk_gib_std"] = 0.0
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(combined: dict[str, dict[str, float]]) -> None:
    print("| Modelo | Disco GiB | Carga s | VRAM GiB | RTF físico | RTF corpus | WER | CER | PPL |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_id in MODEL_ORDER:
        row = combined[model_id]
        formatted = [
            DISPLAY_NAMES[model_id],
            f"{row['disk_gib_mean']:.4f} +/- 0.0000",
            f"{row['load_s_mean']:.4f} +/- {row['load_s_std']:.4f}",
            f"{row['vram_gib_mean']:.4f} +/- {row['vram_gib_std']:.4f}",
            f"{row['hardware_rtf_mean']:.4f} +/- {row['hardware_rtf_std']:.4f}",
            f"{row['quality_rtf_mean']:.4f} +/- {row['quality_rtf_std']:.4f}",
            f"{row['wer_mean']:.4f} +/- {row['wer_std']:.4f}",
            f"{row['cer_mean']:.4f} +/- {row['cer_std']:.4f}",
            f"{row['ppl_mean']:.4f} +/- {row['ppl_std']:.4f}",
        ]
        print("| " + " | ".join(formatted) + " |")


def write_markdown_report(
    combined: dict[str, dict[str, float]],
    blocks: list[dict],
    hardware_rows: list[dict],
    pareto: list[str],
) -> None:
    baseline = combined["fp16"]
    best_wer = min(MODEL_ORDER, key=lambda item: combined[item]["wer_mean"])
    best_cer = min(MODEL_ORDER, key=lambda item: combined[item]["cer_mean"])
    best_ppl = min(MODEL_ORDER, key=lambda item: combined[item]["ppl_mean"])
    best_vram = min(MODEL_ORDER, key=lambda item: combined[item]["vram_gib_mean"])
    best_rtf = min(MODEL_ORDER, key=lambda item: combined[item]["hardware_rtf_mean"])
    lines = [
        "# Resultados del Protocolo Hibrido Real - Capitulo VI",
        "",
        "## Resumen metodologico",
        "",
        "- Modelos evaluados: 7 (BF16 y seis variantes cuantizadas).",
        "- Calidad linguistica: 100 frases held-out reales por modelo, divididas en cinco bloques disjuntos de 20.",
        "- Hardware: cinco procesos limpios por modelo y seis frases fijas por proceso (210 sintesis reales).",
        "- Dispersion: desviacion estandar muestral entre cinco observaciones (`ddof=1`).",
        "- PPL por bloque: `exp(sum(NLL_i) / sum(tokens_i))`; no se promediaron PPL individuales.",
        "- Tamano en disco: propiedad estatica del artefacto, reportada con desviacion `0.0000 GiB`.",
        "- Metricas sinteticas: ninguna.",
        "",
        "## Resultados consolidados",
        "",
        "| Modelo | Disco (GiB) | Carga (s) | VRAM (GiB) | RTF fisico | RTF corpus | WER | CER | PPL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        row = combined[model_id]
        lines.append(
            f"| {DISPLAY_NAMES[model_id]} | {row['disk_gib_mean']:.4f} +/- 0.0000 "
            f"| {row['load_s_mean']:.4f} +/- {row['load_s_std']:.4f} "
            f"| {row['vram_gib_mean']:.4f} +/- {row['vram_gib_std']:.4f} "
            f"| {row['hardware_rtf_mean']:.4f} +/- {row['hardware_rtf_std']:.4f} "
            f"| {row['quality_rtf_mean']:.4f} +/- {row['quality_rtf_std']:.4f} "
            f"| {row['wer_mean']:.4f} +/- {row['wer_std']:.4f} "
            f"| {row['cer_mean']:.4f} +/- {row['cer_std']:.4f} "
            f"| {row['ppl_mean']:.4f} +/- {row['ppl_std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Hallazgos principales",
            "",
            f"- Menor WER: **{DISPLAY_NAMES[best_wer]}** ({combined[best_wer]['wer_mean']:.4f}).",
            f"- Menor CER: **{DISPLAY_NAMES[best_cer]}** ({combined[best_cer]['cer_mean']:.4f}).",
            f"- Menor PPL: **{DISPLAY_NAMES[best_ppl]}** ({combined[best_ppl]['ppl_mean']:.4f}).",
            f"- Menor VRAM: **{DISPLAY_NAMES[best_vram]}** ({combined[best_vram]['vram_gib_mean']:.4f} GiB).",
            f"- Menor RTF fisico global: **{DISPLAY_NAMES[best_rtf]}** ({combined[best_rtf]['hardware_rtf_mean']:.4f}).",
            "- Frente de Pareto calculado al minimizar simultaneamente VRAM y WER: **"
            + ", ".join(DISPLAY_NAMES[item] for item in pareto)
            + "**.",
            "",
            "## Cambios respecto a BF16",
            "",
            "Los porcentajes positivos en disco y VRAM representan reduccion; en WER representan una disminucion del error.",
            "",
            "| Modelo | Reduccion disco | Reduccion VRAM | Cambio WER | Cambio RTF fisico |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_id in MODEL_ORDER[1:]:
        row = combined[model_id]
        disk_reduction = 100 * (1 - row["disk_gib_mean"] / baseline["disk_gib_mean"])
        vram_reduction = 100 * (1 - row["vram_gib_mean"] / baseline["vram_gib_mean"])
        wer_reduction = 100 * (1 - row["wer_mean"] / baseline["wer_mean"])
        rtf_change = 100 * (row["hardware_rtf_mean"] / baseline["hardware_rtf_mean"] - 1)
        lines.append(
            f"| {DISPLAY_NAMES[model_id]} | {disk_reduction:+.2f}% | {vram_reduction:+.2f}% "
            f"| {wer_reduction:+.2f}% | {rtf_change:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "SmoothQuant W8A8 obtuvo el menor WER, mientras que LLM.int8() obtuvo el menor CER. "
            "Ambos redujeron WER y CER frente a BF16, aunque LLM.int8() presento el mayor costo temporal "
            "del conjunto y SmoothQuant tambien tuvo un RTF fisico superior al baseline. "
            "NF4 alcanzo el menor consumo de VRAM y el menor tamano en disco, manteniendose en el frente de Pareto. "
            "TorchAO fue la variante INT4 con menor RTF fisico, pero no pertenece al frente definido exclusivamente por VRAM y WER.",
            "",
            "Las desviaciones de tiempo de carga son elevadas en varios modelos porque se conservaron las cinco "
            "repeticiones tal como fueron observadas. La primera lectura de algunos artefactos fue mas lenta por el "
            "estado de la cache del sistema de archivos; no se elimino como outlier ni se sustituyo por datos simulados.",
            "",
            "## Figuras",
            "",
            "![Recursos fisicos](figura_recursos_fisicos.png)",
            "",
            "![Error linguistico](figura_error_linguistico.png)",
            "",
            "![Frontera de Pareto](figura_frontera_pareto.png)",
            "",
            "## Resultados por bloque de calidad",
            "",
            "| Modelo | Bloque | Frases | WER | CER | RTF | Tokens | PPL agregada |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in blocks:
        lines.append(
            f"| {DISPLAY_NAMES[row['model_id']]} | {row['block']} "
            f"| {row['first_sentence']}-{row['last_sentence']} | {row['wer']:.6f} "
            f"| {row['cer']:.6f} | {row['rtf']:.6f} | {row['tokens']} | {row['ppl']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Repeticiones fisicas",
            "",
            "Esta tabla incorpora todos los campos de los 35 archivos `hardware_runs/*/*.json`.",
            "",
            "| Archivo JSON | Modelo | Repeticion | Origen | Frases | Disco (GiB) | Carga (s) | VRAM (GiB) | RTF | Audio (s) | Generacion (s) |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hardware_rows:
        raw_path = f"hardware_runs/{row['model_id']}/repetition_{int(row['repetition']):02d}.json"
        lines.append(
            f"| [`{raw_path}`]({raw_path}) | {DISPLAY_NAMES[row['model_id']]} "
            f"| {row['repetition']} | {row['source']} | {row['sentences']} "
            f"| {row['disk_gib']:.6f} | {row['load_s']:.6f} | {row['vram_gib']:.6f} "
            f"| {row['rtf']:.6f} | {row['audio_s']:.6f} | {row['generation_s']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Archivos y trazabilidad",
            "",
            "| Archivo | Uso |",
            "|---|---|",
            "| `resumen_hibrido.json` | Resumen numerico completo y metadatos del diseno. |",
            "| `calidad_5_bloques.csv` | Las 35 observaciones reales usadas para WER, CER, RTF corpus y PPL. |",
            "| `hardware_5_repeticiones.csv` | Las 35 mediciones fisicas usadas para carga, VRAM y RTF fisico. |",
            "| `hardware_runs/*/*.json` | Evidencia cruda y reanudable de cada repeticion fisica. |",
            "| `calibracion_telemetria_logs.txt` | Reconstruccion documental de calibracion; no es captura CUDA en vivo. |",
            "| `scripts/capitulo_vi_protocolo_hibrido.py` | Implementacion reproducible del protocolo. |",
            "| `outputs/sprint8_benchmark_n100/results.json` | Fuente de las 700 evaluaciones linguisticas reales. |",
            "| `outputs/sprint8_perplexity/results.json` | Fuente de NLL y tokens por frase. |",
            "",
            "## Nota sobre calibracion",
            "",
            "`calibracion_telemetria_logs.txt` esta rotulado como reconstruccion documental. Sus agregados fueron "
            "proporcionados por el autor, pero las variaciones por capa no proceden de una captura original del runtime. "
            "No debe citarse como telemetria GPU observada en vivo.",
        ]
    )
    (OUTPUT / "resumen_resultados_capitulo_vi.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def export_results(blocks: list[dict], quality: dict, hardware_rows: list[dict], hardware: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "calidad_5_bloques.csv", blocks)
    write_csv(OUTPUT / "hardware_5_repeticiones.csv", hardware_rows)
    combined = {}
    for model_id in MODEL_ORDER:
        combined[model_id] = {
            **quality[model_id],
            "quality_rtf_mean": quality[model_id]["rtf_mean"],
            "quality_rtf_std": quality[model_id]["rtf_std"],
            **hardware[model_id],
            "hardware_rtf_mean": hardware[model_id]["rtf_mean"],
            "hardware_rtf_std": hardware[model_id]["rtf_std"],
        }
    payload = {
        "design": {
            "quality": "5 disjoint blocks x 20 sentences from 100 real held-out observations",
            "hardware": "5 clean-process repetitions x 6 fixed control sentences",
            "standard_deviation": "sample SD (ddof=1)",
            "synthetic_metrics": False,
        },
        "models": combined,
    }
    (OUTPUT / "resumen_hibrido.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    import capitulo_vi_resultados as figures

    figures.OUTPUT_DIR = OUTPUT
    figures._academic_style()
    figures.plot_resources(combined)
    figures.plot_linguistic_error(combined)
    pareto = figures.plot_pareto(combined)
    figures.write_calibration_log()
    write_markdown_report(combined, blocks, hardware_rows, pareto)
    print_table(combined)
    print("Frente de Pareto calculado para VRAM y WER: " + ", ".join(DISPLAY_NAMES[x] for x in pareto))
    print(f"Resultados guardados en {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--worker", choices=MODEL_ORDER)
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.repetition not in range(1, N_REPETITIONS + 1) or args.worker_output is None:
            parser.error("El worker requiere --repetition 1..5 y --worker-output")
        result = hardware_worker(args.worker, args.repetition, args.worker_output.resolve())
        print(json.dumps(result))
        return

    blocks, quality = compute_quality_blocks()
    if args.run_hardware:
        hardware_rows = run_hardware_benchmark(resume=not args.no_resume)
    else:
        hardware_rows = load_complete_hardware_results()
    hardware = summarize_hardware(hardware_rows)
    export_results(blocks, quality, hardware_rows, hardware)


if __name__ == "__main__":
    main()
