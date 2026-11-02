"""Common interface for all streaming drift detectors.

Every detector is fed one value at a time with update(). After each update, `drift_detected`
(and optionally `warning_detected`) say whether that value triggered an alarm. Keeping one
interface lets the experiments, the monitor and the retraining controller swap detectors freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class DriftDetector(ABC):
    name = "base"

    def __init__(self) -> None:
        self.n_seen = 0
        self.drift_detected = False
        self.warning_detected = False

    @abstractmethod
    def update(self, x: float) -> bool:
        """Consume one value. Set drift_detected / warning_detected and return drift_detected.

        Implementations must call `self._step()` first so the counters and flags are kept in sync.
        """

    def _step(self) -> None:
        self.n_seen += 1
        self.drift_detected = False
        self.warning_detected = False

    def reset(self) -> None:
        """Forget everything seen so far (called after a retrain)."""
        self.__init__(**self.params())

    def params(self) -> dict:
        """Constructor arguments, used by reset() and logged with every alarm."""
        return {}

    def run(self, stream: Iterable[float]) -> list[int]:
        """Feed a whole stream and return the 0-based indices where drift was detected."""
        alarms = []
        for i, x in enumerate(stream):
            if self.update(x):
                alarms.append(i)
        return alarms
