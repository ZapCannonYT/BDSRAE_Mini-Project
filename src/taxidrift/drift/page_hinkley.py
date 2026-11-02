"""Page-Hinkley test for a change in the mean (Page, 1954), two-sided."""

from __future__ import annotations

from taxidrift.drift.base import DriftDetector


class PageHinkley(DriftDetector):
    name = "page_hinkley"

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 1.0,
                 min_instances: int = 30) -> None:
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances
        self._clear()

    def _clear(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._up = 0.0
        self._up_min = 0.0
        self._down = 0.0
        self._down_max = 0.0

    def params(self) -> dict:
        return {"delta": self.delta, "threshold": self.threshold, "alpha": self.alpha,
                "min_instances": self.min_instances}

    def update(self, x: float) -> bool:
        self._step()
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._up = self.alpha * self._up + (x - self._mean - self.delta)
        self._up_min = min(self._up_min, self._up)
        self._down = self.alpha * self._down + (x - self._mean + self.delta)
        self._down_max = max(self._down_max, self._down)
        if self._n >= self.min_instances and (self._up - self._up_min > self.threshold
                                              or self._down_max - self._down > self.threshold):
            self.drift_detected = True
            self._clear()
        return self.drift_detected
