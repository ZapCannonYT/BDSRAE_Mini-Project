"""Population Stability Index (PSI) over a sliding window.

PSI = sum over bins of (a_i - e_i) * ln(a_i / e_i), where e_i and a_i are the shares of the
reference and of the current window falling in bin i. Common reading: < 0.1 stable,
0.1-0.25 moderate shift (warning), > 0.25 significant shift (drift).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from taxidrift.drift.base import DriftDetector

EPS = 1e-6  # floor for empty bins so ln() stays finite


def quantile_edges(reference: Sequence[float], bins: int) -> np.ndarray:
    """Bin edges from reference quantiles, with -inf/+inf at the ends."""
    inner = np.unique(np.quantile(np.asarray(reference, dtype=float), np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return np.maximum(counts / max(len(values), 1), EPS)


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """PSI of `current` against `reference` using quantile bins of the reference."""
    reference = np.asarray(reference, dtype=float)
    edges = quantile_edges(reference, bins)
    e, a = _shares(reference, edges), _shares(np.asarray(current, dtype=float), edges)
    return float(np.sum((a - e) * np.log(a / e)))


class PSIDetector(DriftDetector):
    """Compares the last `window` values with a fixed reference sample.

    Recomputes PSI every `step` values once the window is full. Fires on the upward crossing of
    `drift` and re-arms when PSI falls back below `warn`, so a long shift gives one alarm rather
    than one per hour. `warning_detected` is set while PSI is at or above `warn`.
    """

    name = "psi"

    def __init__(self, reference: Sequence[float], window: int = 168, bins: int = 10,
                 warn: float = 0.1, drift: float = 0.25, step: int = 1) -> None:
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.window = window
        self.bins = bins
        self.warn = warn
        self.drift = drift
        self.step = step
        self.last_psi: float | None = None
        self._edges = quantile_edges(self.reference, bins)
        self._ref_shares = _shares(self.reference, self._edges)
        self._buffer: deque[float] = deque(maxlen=window)
        self._armed = True

    def params(self) -> dict:
        return {"reference": self.reference, "window": self.window, "bins": self.bins,
                "warn": self.warn, "drift": self.drift, "step": self.step}

    def update(self, x: float) -> bool:
        self._step()
        self._buffer.append(float(x))
        if len(self._buffer) < self.window or self.n_seen % self.step:
            self.warning_detected = self.last_psi is not None and self.last_psi >= self.warn
            return False
        a = _shares(np.fromiter(self._buffer, dtype=float, count=self.window), self._edges)
        self.last_psi = float(np.sum((a - self._ref_shares) * np.log(a / self._ref_shares)))
        self.warning_detected = self.last_psi >= self.warn
        if self.last_psi < self.warn:
            self._armed = True
        if self.last_psi >= self.drift and self._armed:
            self._armed = False
            self.drift_detected = True
        return self.drift_detected
