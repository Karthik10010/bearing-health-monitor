"""Latency / size / memory measurements.

All numbers are MEASURED ON THE DEVELOPMENT LAPTOP (Windows, CPU), NOT ON EDGE HARDWARE.
"""
from __future__ import annotations

import os
import platform
import time
import tracemalloc
from typing import Callable

import numpy as np

LABEL = "Measured on development laptop (Windows, CPU), not on edge hardware."


def time_fn(fn: Callable[[], object], repeats: int = 1000, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    ms = t * 1000
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "repeats": repeats}


def peak_python_memory_mb(fn: Callable[[], object]) -> float:
    """Peak memory allocated by Python/NumPy during one call (tracemalloc)."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def process_memory_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def machine_info() -> dict:
    import psutil

    return {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "logical_cpus": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "label": LABEL}


def file_size_kb(p) -> float:
    return os.path.getsize(p) / 1024
