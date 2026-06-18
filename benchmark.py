from __future__ import annotations

import statistics
import time

import numpy as np

from config import AppConfig, ProcessOptions
from rife_engine import load_rife


def run_benchmark(model_path: str, backend: str, width: int = 1920, height: int = 1080, rounds: int = 20) -> dict:
    options = ProcessOptions(target_fps=60, model_path=model_path, backend=backend)
    messages: list[str] = []
    model = load_rife(options, AppConfig(), messages.append)
    first = np.zeros((height, width, 3), dtype=np.uint8)
    second = np.full_like(first, 32)
    model.infer(first, second, 0.5)
    timings = []
    for _ in range(rounds):
        started = time.perf_counter()
        model.infer(first, second, 0.5)
        timings.append((time.perf_counter() - started) * 1000)
    median = statistics.median(timings)
    return {
        "backend": model.name,
        "resolution": f"{width}x{height}",
        "median_ms": round(median, 2),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95) - 1], 2),
        "inference_fps": round(1000 / median, 2),
        "rounds": rounds,
    }
