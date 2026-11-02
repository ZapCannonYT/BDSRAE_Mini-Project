"""ADWIN (ADaptive WINdowing), Bifet and Gavalda, SDM 2007.

Watches a numeric stream (we use the model's normalised error). Keeps a variable-length window of
recent values compressed into exponential-histogram buckets, and drops the oldest part whenever
two sub-windows have significantly different means.
"""

from __future__ import annotations

import math

from taxidrift.drift.base import DriftDetector


class ADWIN(DriftDetector):
    name = "adwin"

    def __init__(self, delta: float = 0.002, clock: int = 32, max_buckets: int = 5,
                 min_window: int = 5, grace_period: int = 10) -> None:
        super().__init__()
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window = min_window
        self.grace_period = grace_period
        # rows[i] holds buckets of 2**i values as [total, variance], oldest first
        self.rows: list[list[list[float]]] = [[]]
        self._width = 0
        self._total = 0.0
        self._variance = 0.0

    def params(self) -> dict:
        return {"delta": self.delta, "clock": self.clock, "max_buckets": self.max_buckets,
                "min_window": self.min_window, "grace_period": self.grace_period}

    def update(self, x: float) -> bool:
        self._step()
        self._insert(float(x))
        if self.n_seen % self.clock == 0 and self._width > self.grace_period:
            self.drift_detected = self._shrink()
        return self.drift_detected

    def _insert(self, x: float) -> None:
        if self._width > 0:
            mean = self._total / self._width
            self._variance += self._width * (x - mean) ** 2 / (self._width + 1)
        self._width += 1
        self._total += x
        self.rows[0].append([x, 0.0])
        for i, row in enumerate(self.rows):
            if len(row) <= self.max_buckets:
                break
            (t1, v1), (t2, v2) = row[0], row[1]
            size = 2.0**i
            merged = [t1 + t2, v1 + v2 + size * size / (2 * size) * (t1 / size - t2 / size) ** 2]
            del row[:2]
            if i + 1 == len(self.rows):
                self.rows.append([])
            self.rows[i + 1].append(merged)

    def _shrink(self) -> bool:
        changed = False
        cut = True
        while cut and self._width > self.grace_period:
            cut = False
            n0, t0 = 0, 0.0
            n1, t1 = self._width, self._total
            variance = self._variance / self._width
            delta_prime = math.log(2 * math.log(self._width) / self.delta)
            for i in range(len(self.rows) - 1, -1, -1):
                size = 2**i
                for total, _ in self.rows[i]:
                    n0 += size
                    n1 -= size
                    t0 += total
                    t1 -= total
                    if n1 < self.min_window:
                        break
                    if n0 < self.min_window:
                        continue
                    m_recip = 1 / (n0 - self.min_window + 1) + 1 / (n1 - self.min_window + 1)
                    eps = math.sqrt(2 * m_recip * variance * delta_prime) + 2 / 3 * delta_prime * m_recip
                    if abs(t0 / n0 - t1 / n1) > eps:
                        cut = True
                        break
                if cut or n1 < self.min_window:
                    break
            if cut:
                changed = True
                self._drop_oldest()
        return changed

    def _drop_oldest(self) -> None:
        i = len(self.rows) - 1
        while not self.rows[i]:
            i -= 1
        total, var = self.rows[i].pop(0)
        size = 2**i
        rest = self._width - size
        if rest > 0:
            diff = total / size - (self._total - total) / rest
            self._variance -= var + size * rest / (size + rest) * diff * diff
            self._variance = max(self._variance, 0.0)
        else:
            self._variance = 0.0
        self._width = rest
        self._total -= total
        while len(self.rows) > 1 and not self.rows[-1]:
            self.rows.pop()

    @property
    def width(self) -> int:
        """Current window length."""
        return self._width

    @property
    def estimation(self) -> float:
        """Mean of the current window."""
        return self._total / self._width if self._width else 0.0

    @property
    def n_buckets(self) -> int:
        return sum(len(r) for r in self.rows)
