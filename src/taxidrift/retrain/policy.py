"""Retraining policies: decide, hour by hour, whether to retrain.

The engine calls should_retrain() once per stream hour with the latest detector alarms. Cooldown
handling is shared here so every policy is compared fairly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RetrainPolicy(ABC):
    name = "base"

    def __init__(self, cooldown_hours: int = 72) -> None:
        self.cooldown = pd.Timedelta(hours=cooldown_hours)
        self.last_retrain: pd.Timestamp | None = None
        self.reason = ""

    def should_retrain(self, now: pd.Timestamp, alarms: dict[str, bool]) -> bool:
        """alarms maps signal name -> fired this hour.

        Signals: adwin, psi, psi_warn (city-wide), adwin_borough, psi_borough (any borough).
        Signals are always passed to _decide so stateful policies do not miss any during cooldown.
        """
        decided = self._decide(now, alarms)
        if self.last_retrain is not None and now - self.last_retrain < self.cooldown:
            return False
        if decided:
            self.last_retrain = now
            return True
        return False

    @abstractmethod
    def _decide(self, now: pd.Timestamp, alarms: dict[str, bool]) -> bool: ...


class NeverRetrain(RetrainPolicy):
    name = "never"

    def _decide(self, now, alarms) -> bool:
        return False


class PeriodicRetrain(RetrainPolicy):
    """Retrain at the start of every month."""

    name = "periodic"

    def _decide(self, now, alarms) -> bool:
        self.reason = "monthly schedule"
        return now.day == 1 and now.hour == 0


class DetectorTriggered(RetrainPolicy):
    """Retrain when any of the named signals fires."""

    def __init__(self, *signals: str, cooldown_hours: int = 72, name: str | None = None) -> None:
        super().__init__(cooldown_hours)
        self.signals = signals
        self.name = name or "_or_".join(signals)

    def _decide(self, now, alarms) -> bool:
        fired = [s for s in self.signals if alarms.get(s)]
        self.reason = "alarm: " + ", ".join(fired)
        return bool(fired)


class CombinedPolicy(RetrainPolicy):
    """ADWIN alarm confirmed by a PSI warning within the previous `confirm_hours`."""

    name = "combined"

    def __init__(self, confirm_hours: int = 168, cooldown_hours: int = 72) -> None:
        super().__init__(cooldown_hours)
        self.confirm = pd.Timedelta(hours=confirm_hours)
        self.last_warn: pd.Timestamp | None = None

    def _decide(self, now, alarms) -> bool:
        if alarms.get("psi_warn") or alarms.get("psi"):
            self.last_warn = now
        self.reason = "adwin confirmed by psi warning"
        return bool(alarms.get("adwin")) and self.last_warn is not None and now - self.last_warn <= self.confirm


class OracleRetrain(RetrainPolicy):
    """Retrain at known real change dates (reference point for comparison only)."""

    name = "oracle"

    def __init__(self, change_dates: list[str], cooldown_hours: int = 0) -> None:
        super().__init__(cooldown_hours)
        self.change_dates = {pd.Timestamp(d) for d in change_dates}

    def _decide(self, now, alarms) -> bool:
        self.reason = "known change date"
        return now in self.change_dates


# Real events: NY PAUSE plus three weeks of post-crash data, NYC reopening phases 1 and 3,
# restrictions lifted, Omicron wave.
ORACLE_DATES = ["2020-03-22", "2020-04-12", "2020-06-08", "2020-07-06", "2021-06-15", "2021-12-15"]


def build_policies(cooldown_hours: int) -> list[RetrainPolicy]:
    return [
        NeverRetrain(cooldown_hours),
        PeriodicRetrain(0),
        DetectorTriggered("adwin", cooldown_hours=cooldown_hours, name="adwin"),
        DetectorTriggered("psi", cooldown_hours=cooldown_hours, name="psi"),
        CombinedPolicy(cooldown_hours=cooldown_hours),
        DetectorTriggered("adwin", "adwin_borough", cooldown_hours=cooldown_hours, name="adwin_any_borough"),
        OracleRetrain(ORACLE_DATES),
    ]
