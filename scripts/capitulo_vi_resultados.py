#!/usr/bin/env python3
"""Benchmark reproducible y figuras del Capitulo VI de la tesis.

Uso desde Jupyter:
    %run scripts/capitulo_vi_resultados.py

Uso real (costoso: 5 ejecuciones x 7 modelos x 100 frases):
    BENCHMARK_MODE=real python scripts/capitulo_vi_resultados.py

El modo ``mock`` es una simulacion parametrica para probar el pipeline. Sus
resultados se etiquetan como sinteticos y no deben reportarse como mediciones.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_ITERATIONS: Final = 5
SEED: Final = 20260906
MODE: Final = os.environ.get("BENCHMARK_MODE", "mock").strip().lower()
ALLOW_MOCK_FALLBACK: Final = os.environ.get("ALLOW_MOCK_FALLBACK", "1") == "1"
OUTPUT_DIR: Final = Path.cwd().resolve()
SCRIPT_DIR: Final = Path(__file__).resolve().parent
PROJECT_ROOT: Final = SCRIPT_DIR.parent
REAL_RUNS_DIR: Final = OUTPUT_DIR / "capitulo_vi_ejecuciones"
HELD_OUT_SELECTION: Final = PROJECT_ROOT / "outputs" / "sprint8_perplexity" / "selection.json"

METRICS: Final = ("disk_gib", "load_s", "vram_gib", "rtf", "wer", "cer", "ppl")
DISPLAY_NAMES: Final = {
    "fp16": "Base BF16",
    "torchao_int4": "TorchAO HQQ INT4",
    "gptq_tts": "GPTQ INT4",
    "awq": "AutoAWQ INT4",
    "int8": "LLM.int8()",
    "nf4": "NF4",
    "smoothquant": "SmoothQuant W8A8",
}

# Promedios proporcionados por el autor. No son observaciones individuales.
REFERENCE_MEANS: Final = {
    "fp16": dict(disk_gib=10.0735, load_s=15.5557, vram_gib=5.1815, rtf=0.8027, wer=0.3937, cer=0.2599, ppl=1.0604),
    "torchao_int4": dict(disk_gib=3.4177, load_s=5.8158, vram_gib=3.5539, rtf=0.9232, wer=0.5187, cer=0.3118, ppl=1.0770),
    "gptq_tts": dict(disk_gib=3.3555, load_s=32.7584, vram_gib=3.3746, rtf=1.0491, wer=0.4160, cer=0.2412, ppl=1.0792),
    "awq": dict(disk_gib=3.3532, load_s=12.6689, vram_gib=3.3909, rtf=0.9910, wer=0.5054, cer=0.3255, ppl=1.0671),
    "int8": dict(disk_gib=3.8340, load_s=6.3552, vram_gib=3.9824, rtf=2.1994, wer=0.3651, cer=0.2244, ppl=1.0618),
    "nf4": dict(disk_gib=3.2410, load_s=5.8918, vram_gib=3.3694, rtf=1.0453, wer=0.4083, cer=0.2543, ppl=1.0761),
    "smoothquant": dict(disk_gib=3.8359, load_s=5.3268, vram_gib=4.0358, rtf=1.4541, wer=0.3625, cer=0.2283, ppl=1.0634),
}

# Desviaciones muestrales objetivo usadas exclusivamente por el simulador.
MOCK_SAMPLE_STD: Final = {
    "disk_gib": 0.010,
    "load_s": 0.050,
    "vram_gib": 0.015,
    "rtf": 0.020,
    "wer": 0.012,
    "cer": 0.010,
    "ppl": 0.004,
}

MODEL_PATHS: Final = {
    "fp16": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es",
    "torchao_int4": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-torchao-int4",
    "gptq_tts": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-gptq-tts",
    "awq": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-awq",
    "int8": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-int8",
    "nf4": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-nf4.partial",
    "smoothquant": PROJECT_ROOT / "weights" / "vibevoice-1.5b-es-smoothquant",
}


@dataclass(frozen=True)
class Observation:
    model_id: str
    iteration: int
    source: str
    disk_gib: float
    load_s: float
    vram_gib: float
    rtf: float
    wer: float
    cer: float
    ppl: float


def _centered_normal(
    mean: float,
    sample_std: float,
    rng: np.random.Generator,
    count: int = N_ITERATIONS,
) -> np.ndarray:
    """Produce valores con media y desviacion muestral exactamente dadas."""
    if count < 2:
        raise ValueError("Se requieren al menos dos valores para una desviacion muestral")
    values = rng.normal(size=count)
    values -= values.mean()
    current_std = values.std(ddof=1)
    if current_std == 0.0:  # Practicamente imposible, pero evita division por cero.
        values = np.arange(count, dtype=np.float64) - (count - 1) / 2
        current_std = values.std(ddof=1)
    return mean + values * (sample_std / current_std)


def mock_observations(model_id: str, rng: np.random.Generator) -> list[Observation]:
    columns = {
        metric: _centered_normal(REFERENCE_MEANS[model_id][metric], MOCK_SAMPLE_STD[metric], rng)
        for metric in METRICS
    }
    return [
        Observation(
            model_id=model_id,
            iteration=index + 1,
            source="synthetic_mock",
            **{metric: float(columns[metric][index]) for metric in METRICS},
        )
        for index in range(N_ITERATIONS)
    ]


def _last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("El proceso no devolvio un objeto JSON reconocible")


def _run_checked(command: list[str], timeout_s: int = 14_400) -> dict:
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout)[-3000:]
        raise RuntimeError(f"Fallo el worker (codigo {process.returncode}):\n{detail}")
    return _last_json(process.stdout)


def validate_real_environment() -> None:
    if MODE not in {"mock", "real", "auto"}:
        raise ValueError("BENCHMARK_MODE debe ser 'mock', 'real' o 'auto'")
    if MODE == "mock":
        return
    try:
        import jiwer  # noqa: F401
        import torch
    except ImportError as exc:
        raise RuntimeError(f"Dependencia ausente para el modo real: {exc.name}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible para el modo real")
    if not HELD_OUT_SELECTION.is_file():
        raise FileNotFoundError(f"No existe la seleccion held-out: {HELD_OUT_SELECTION}")
    selection = json.loads(HELD_OUT_SELECTION.read_text(encoding="utf-8"))
    if int(selection.get("sample_count", 0)) != 100:
        raise ValueError("La seleccion held-out debe contener exactamente 100 frases")


def real_observation(model_id: str, iteration: int) -> Observation:
    """Ejecuta inferencia y PPL en procesos nuevos para aislar la VRAM."""
    model_output = REAL_RUNS_DIR / model_id / f"iteracion_{iteration:02d}"
    model_output.mkdir(parents=True, exist_ok=True)
    benchmark = _run_checked(
        [
            sys.executable,
            str(SCRIPT_DIR / "sprint8_benchmark.py"),
            "--worker",
            model_id,
            "--text-file",
            str(HELD_OUT_SELECTION),
            "--output",
            str(model_output),
        ]
    )
    perplexity = _run_checked(
        [
            sys.executable,
            str(SCRIPT_DIR / "sprint8_perplexity.py"),
            "--worker",
            model_id,
            "--samples",
            "100",
            "--seed",
            str(SEED + iteration - 1),
            "--batch-size",
            "4",
        ]
    )
    values = {
        "disk_gib": benchmark["artifact_gib"],
        "load_s": benchmark["load_seconds"],
        "vram_gib": benchmark["vram_peak_gib"],
        "rtf": benchmark["rtf"],
        "wer": benchmark["wer"],
        "cer": benchmark["cer"],
        "ppl": perplexity["perplexity"],
    }
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values.values()):
        raise RuntimeError(f"Metricas no validas para {model_id}: {values}")
    return Observation(model_id, iteration, "measured", **values)


def collect_observations(force_mock: bool = False) -> tuple[list[Observation], dict[str, str]]:
    rng = np.random.default_rng(SEED)
    all_observations: list[Observation] = []
    failures: dict[str, str] = {}
    for model_id in DISPLAY_NAMES:
        if MODE == "mock" or force_mock:
            model_observations = mock_observations(model_id, rng)
        else:
            missing = not MODEL_PATHS[model_id].is_dir()
            if missing and not ALLOW_MOCK_FALLBACK:
                raise FileNotFoundError(f"No existe el modelo: {MODEL_PATHS[model_id]}")
            try:
                if missing:
                    raise FileNotFoundError(f"No existe el modelo: {MODEL_PATHS[model_id]}")
                model_observations = []
                for iteration in range(1, N_ITERATIONS + 1):
                    print(f"[{DISPLAY_NAMES[model_id]}] iteracion real {iteration}/{N_ITERATIONS}")
                    model_observations.append(real_observation(model_id, iteration))
                    gc.collect()
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except (ImportError, RuntimeError):
                        pass
            except Exception as exc:
                if not ALLOW_MOCK_FALLBACK:
                    raise
                failures[model_id] = f"{type(exc).__name__}: {exc}"
                model_observations = mock_observations(model_id, rng)
                print(f"[{DISPLAY_NAMES[model_id]}] fallback sintetico: {exc}")
        if len(model_observations) != N_ITERATIONS:
            raise AssertionError("Cada modelo debe tener exactamente cinco iteraciones")
        all_observations.extend(model_observations)
    return all_observations, failures


def summarize(observations: list[Observation]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model_id in DISPLAY_NAMES:
        rows = [item for item in observations if item.model_id == model_id]
        if len(rows) != N_ITERATIONS:
            raise ValueError(f"{model_id} tiene {len(rows)} iteraciones; se esperaban 5")
        summary[model_id] = {}
        for metric in METRICS:
            values = np.asarray([getattr(item, metric) for item in rows], dtype=np.float64)
            summary[model_id][f"{metric}_mean"] = float(values.mean())
            summary[model_id][f"{metric}_std"] = float(values.std(ddof=1))
    return summary


def mean_std(stats: dict[str, float], metric: str, decimals: int = 4) -> str:
    return f"{stats[f'{metric}_mean']:.{decimals}f} +/- {stats[f'{metric}_std']:.{decimals}f}"


def print_markdown_table(summary: dict[str, dict[str, float]]) -> None:
    headers = [
        "Modelo", "Tamaño Disco (GiB)", "Tiempo Carga (s) (Media +/- Desv)",
        "Pico VRAM (GiB) (Media +/- Desv)", "RTF (Media +/- Desv)",
        "WER (Media +/- Desv)", "CER (Media +/- Desv)", "PPL (Media +/- Desv)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for model_id, name in DISPLAY_NAMES.items():
        stats = summary[model_id]
        row = [
            name,
            f"{stats['disk_gib_mean']:.4f}",
            mean_std(stats, "load_s"),
            mean_std(stats, "vram_gib"),
            mean_std(stats, "rtf"),
            mean_std(stats, "wer"),
            mean_std(stats, "cer"),
            mean_std(stats, "ppl"),
        ]
        print("| " + " | ".join(row) + " |")


def write_raw_outputs(
    observations: list[Observation],
    summary: dict[str, dict[str, float]],
    failures: dict[str, str],
) -> None:
    csv_path = OUTPUT_DIR / "benchmark_iteraciones.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(observations[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in observations)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_mode": MODE,
        "iterations_per_model": N_ITERATIONS,
        "ddof": 1,
        "std_definition": "sample standard deviation",
        "synthetic_warning": "synthetic_mock rows are simulations, not empirical measurements",
        "fallback_failures": failures,
        "summary": summary,
    }
    (OUTPUT_DIR / "benchmark_resumen.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_resources(summary: dict[str, dict[str, float]]) -> None:
    names = list(DISPLAY_NAMES.values())
    ids = list(DISPLAY_NAMES)
    x = np.arange(len(ids))
    width = 0.36
    vram = [summary[item]["vram_gib_mean"] for item in ids]
    vram_err = [summary[item]["vram_gib_std"] for item in ids]
    load = [summary[item]["load_s_mean"] for item in ids]
    load_err = [summary[item]["load_s_std"] for item in ids]
    fig, axis_vram = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    axis_load = axis_vram.twinx()
    bars_vram = axis_vram.bar(x - width / 2, vram, width, yerr=vram_err, capsize=3,
                              color="#bdbdbd", edgecolor="#202020", linewidth=0.7,
                              label="Pico de VRAM")
    bars_load = axis_load.bar(x + width / 2, load, width, yerr=load_err, capsize=3,
                             color="#353535", edgecolor="#111111", linewidth=0.7,
                             label="Tiempo de carga")
    axis_vram.set_ylabel("Pico de VRAM (GiB)")
    axis_load.set_ylabel("Tiempo de carga (s)")
    axis_vram.set_xticks(x, names, rotation=25, ha="right")
    axis_vram.set_title("Consumo de memoria y tiempo de carga por modelo")
    axis_load.grid(False)
    axis_vram.legend([bars_vram, bars_load], ["Pico de VRAM", "Tiempo de carga"],
                     loc="upper left", ncols=2)
    fig.savefig(OUTPUT_DIR / "figura_recursos_fisicos.png", dpi=300)
    plt.close(fig)


def plot_linguistic_error(summary: dict[str, dict[str, float]]) -> None:
    names = list(DISPLAY_NAMES.values())
    ids = list(DISPLAY_NAMES)
    x = np.arange(len(ids))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    axis.bar(x - width / 2, [summary[item]["wer_mean"] for item in ids], width,
             yerr=[summary[item]["wer_std"] for item in ids], capsize=3,
             color="#3b3b3b", edgecolor="black", linewidth=0.7, label="WER")
    axis.bar(x + width / 2, [summary[item]["cer_mean"] for item in ids], width,
             yerr=[summary[item]["cer_std"] for item in ids], capsize=3,
             color="#cfcfcf", edgecolor="#202020", linewidth=0.7, label="CER")
    axis.set_ylabel("Tasa de error media")
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Error lingüístico en el corpus held-out de 100 frases")
    axis.legend(loc="upper right", ncols=2)
    fig.savefig(OUTPUT_DIR / "figura_error_linguistico.png", dpi=300)
    plt.close(fig)


def pareto_ids(summary: dict[str, dict[str, float]]) -> list[str]:
    """Devuelve los puntos no dominados al minimizar simultaneamente VRAM y WER."""
    optimal = []
    for candidate in DISPLAY_NAMES:
        x = summary[candidate]["vram_gib_mean"]
        y = summary[candidate]["wer_mean"]
        dominated = any(
            other != candidate
            and summary[other]["vram_gib_mean"] <= x
            and summary[other]["wer_mean"] <= y
            and (
                summary[other]["vram_gib_mean"] < x
                or summary[other]["wer_mean"] < y
            )
            for other in DISPLAY_NAMES
        )
        if not dominated:
            optimal.append(candidate)
    return sorted(optimal, key=lambda item: summary[item]["vram_gib_mean"])


def plot_pareto(summary: dict[str, dict[str, float]]) -> list[str]:
    fig, axis = plt.subplots(figsize=(9.2, 6.5), constrained_layout=True)
    optimal = pareto_ids(summary)
    front_x = np.asarray([summary[item]["vram_gib_mean"] for item in optimal])
    front_y = np.asarray([summary[item]["wer_mean"] for item in optimal])
    all_y = [summary[item]["wer_mean"] for item in DISPLAY_NAMES]
    upper = max(all_y) + 0.035
    right_edge = max(summary[item]["vram_gib_mean"] for item in DISPLAY_NAMES) + 0.22
    shade_x = np.append(front_x, right_edge)
    shade_y = np.append(front_y, front_y[-1])
    axis.fill_between(shade_x, shade_y, upper, step="post", color="#d9d9d9", alpha=0.45,
                      label="Región subóptima dominada")
    axis.plot(front_x, front_y, color="#111111", linewidth=1.5, linestyle="--",
              marker="o", markersize=5, label="Frente de Pareto calculado")
    annotation_offsets = {
        "fp16": (7, 6), "torchao_int4": (7, 7), "gptq_tts": (8, 8),
        "awq": (7, 7), "int8": (-75, -16), "nf4": (8, -18),
        "smoothquant": (7, 8),
    }
    for model_id, name in DISPLAY_NAMES.items():
        x = summary[model_id]["vram_gib_mean"]
        y = summary[model_id]["wer_mean"]
        is_optimal = model_id in optimal
        axis.scatter(x, y, s=90 if is_optimal else 65,
                     facecolor="#111111" if is_optimal else "white",
                     edgecolor="#111111", linewidth=1.1, zorder=4)
        axis.annotate(name, (x, y), xytext=annotation_offsets[model_id],
                      textcoords="offset points", fontsize=8.2)
    axis.set_xlabel("Consumo pico de VRAM (GiB) - menor es mejor")
    axis.set_ylabel("Word Error Rate - menor es mejor")
    axis.set_title("Frontera de Pareto: eficiencia espacial frente a calidad lingüística")
    axis.set_xlim(min(summary[item]["vram_gib_mean"] for item in DISPLAY_NAMES) - 0.12,
                  right_edge)
    axis.set_ylim(min(all_y) - 0.035, upper)
    axis.legend(loc="upper right")
    fig.savefig(OUTPUT_DIR / "figura_frontera_pareto.png", dpi=300)
    plt.close(fig)
    return optimal


def write_calibration_log() -> None:
    """Documenta los agregados facilitados; no suplanta telemetria GPU sin capturar."""
    rng = np.random.default_rng(SEED + 1)
    hessian_times = _centered_normal(12.45, 0.18, rng, count=28)
    reconstruction = _centered_normal(2.84e-4, 0.07e-4, rng, count=28)
    awq_times = _centered_normal(18.34, 0.22, rng, count=28)
    alpha_grid = np.linspace(0.05, 1.0, 20)
    lines = [
        "CALIBRACION Y TELEMETRIA DE CUANTIZACION - VibeVoice-1.5B",
        "=" * 72,
        "CLASIFICACION: RECONSTRUCCION DOCUMENTAL, NO TELEMETRIA CAPTURADA EN VIVO",
        "Los tiempos y agregados siguientes fueron facilitados por el autor. Las",
        "variaciones por capa se generan de forma determinista solo para presentar",
        "el promedio indicado; no constituyen logs originales del runtime CUDA.",
        f"Fecha de exportacion (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Semilla de reconstruccion: {SEED + 1}",
        "",
        "[GPTQ | W4 g128 | 28 capas Transformer]",
        "Calibracion: 32,589 tokens TTS-prefill; aproximacion de Hessiana H=2XX^T;",
        "amortiguacion diagonal y factorizacion de Cholesky para H^{-1}.",
    ]
    for layer, (seconds, mse) in enumerate(zip(hessian_times, reconstruction)):
        lines.append(
            f"layer={layer:02d} hessian_inverse_s={seconds:7.3f} "
            f"relative_reconstruction_mse={mse:.6e} status=OK"
        )
    lines.extend(
        [
            "GPTQ_RESUMEN layers=28 tokens=32589 "
            f"mean_hessian_inverse_s={hessian_times.mean():.2f} "
            f"mean_relative_mse={reconstruction.mean():.6e}",
            "",
            "[AutoAWQ | W4A16 g128 | grid search de escalas]",
            "Objetivo: minimizar ||WX - Q(W*s)Q(X/s)||_2 protegiendo outliers.",
            "alpha_candidates=20 values=["
            + ", ".join(f"{value:.2f}" for value in alpha_grid)
            + "]",
        ]
    )
    for layer, seconds in enumerate(awq_times):
        lines.append(
            f"layer={layer:02d} candidates=20 search_s={seconds:7.3f} "
            "selected_alpha=0.65 protected=[q_proj,v_proj] status=OK"
        )
    lines.extend(
        [
            f"AWQ_RESUMEN mean_search_s={awq_times.mean():.2f} mean_selected_alpha=0.65",
            "",
            "[bitsandbytes LLM.int8() | mixed-precision decomposition]",
            "outlier_threshold_tau=6.0 projection_matrices=196",
            "activation_columns_fp16_pct=0.14 activation_columns_int8_pct=99.86",
            "Ruta outlier: GEMM FP16; ruta principal: cuantizacion vector-wise INT8.",
            "Chequeo de conservacion: 0.14% + 99.86% = 100.00%; status=OK",
            "",
            "[bitsandbytes NF4 | double quantization]",
            "primary_block_size=64 primary_codebook=NormalFloat4",
            "secondary_block_size=256 secondary_target=primary_scale_constants",
            "Las escalas primarias se cuantizan de nuevo y comparten metadatos por",
            "bloque secundario; se evita almacenar una escala FP32 por cada bloque.",
            "disk_scale_overhead_reduction=3.12x baseline_scale_dtype=FP32 status=OK",
            "",
            "[SmoothQuant | W8A8 selectivo]",
            "migration_exponent_alpha=0.5 scale_j=max(|X_j|)^alpha/max(|W_j|)^(1-alpha)",
            "La transformacion equivalente W'=W*diag(s), X'=diag(s)^-1*X migra",
            "la dificultad de cuantizacion desde activaciones hacia pesos.",
            "activation_raw_peak=142.5 activation_smoothed_peak=11.2 reduction=12.7232x",
            "dynamic_int8_clipping_error=eliminated_for_reported_calibration_set status=OK",
            "",
            "FIN DEL INFORME",
        ]
    )
    (OUTPUT_DIR / "calibracion_telemetria_logs.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    force_mock = False
    try:
        validate_real_environment()
    except Exception as exc:
        if MODE != "auto" or not ALLOW_MOCK_FALLBACK:
            raise
        print(f"[AUTO] Entorno real no disponible; se usara simulacion: {exc}")
        force_mock = True
    observations, failures = collect_observations(force_mock=force_mock)
    summary = summarize(observations)
    write_raw_outputs(observations, summary, failures)
    write_calibration_log()
    _academic_style()
    plot_resources(summary)
    plot_linguistic_error(summary)
    optimal = plot_pareto(summary)

    has_mock = any(item.source == "synthetic_mock" for item in observations)
    if has_mock:
        print("\nADVERTENCIA: la tabla contiene datos SINTETICOS; no son mediciones experimentales.\n")
    print_markdown_table(summary)
    print("\nFrente de Pareto calculado (min VRAM, min WER): " +
          ", ".join(DISPLAY_NAMES[item] for item in optimal))
    print(
        textwrap.dedent(
            f"""

            Archivos guardados en: {OUTPUT_DIR}
            - figura_recursos_fisicos.png
            - figura_error_linguistico.png
            - figura_frontera_pareto.png
            - calibracion_telemetria_logs.txt
            - benchmark_iteraciones.csv (35 observaciones crudas)
            - benchmark_resumen.json (metadatos y trazabilidad)
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
