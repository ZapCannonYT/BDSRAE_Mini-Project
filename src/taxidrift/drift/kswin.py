"""KSWIN: Kolmogorov-Smirnov windowing (Raab, Heusinger and Schleif, 2020).

Compares the newest `stat_size` values with a random sample of the same size from the older part
of the window. Drift when the KS distance D exceeds sqrt(-ln(alpha) / stat_size).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from taxidrift.drift.base import DriftDetector


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    both = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, both, side="right") / len(a)
    cdf_b = np.searchsorted(b, both, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


class KSWIN(DriftDetector):
    name = "kswin"

    def __init__(self, window_size: int = 100, stat_size: int = 30, alpha: float = 0.005,
                 seed: int | None = 42) -> None:
        super().__init__()
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.seed = seed
        self._threshold = math.sqrt(-math.log(alpha) / stat_size)
        self._rng = np.random.default_rng(seed)
        self._window: deque[float] = deque(maxlen=window_size)

    def params(self) -> dict:
        return {"window_size": self.window_size, "stat_size": self.stat_size, "alpha": self.alpha,
                "seed": self.seed}

    def update(self, x: float) -> bool:
        self._step()
        self._window.append(float(x))
        if len(self._window) < self.window_size:
            return False
        values = np.fromiter(self._window, dtype=float, count=self.window_size)
        recent = values[-self.stat_size:]
        older = values[self._rng.integers(0, self.window_size - self.stat_size, self.stat_size)]
        if _ks_distance(recent, older) > self._threshold:
            self.drift_detected = True
            self._window = deque(recent, maxlen=self.window_size)
        return self.drift_detected
