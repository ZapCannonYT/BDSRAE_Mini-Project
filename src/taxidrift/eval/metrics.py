"""Shared evaluation metrics, so every experiment scores detectors and models the same way.

Detection scoring: for each true change point cp, the first alarm in [cp, min(cp + max_delay,
next cp)) is a true detection and its delay is alarm - cp. Later alarms in that same interval
belong to the same drift and are not counted. Every alarm outside these intervals is a false alarm.
A change point with no such alarm is missed. Positions are stream indices (hours).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def score_detections(alarms: Sequence[int], change_points: Sequence[int], n: int,
                     max_delay: int) -> dict:
    alarms = sorted(alarms)
    cps = sorted(change_points)
    delays: list[int | None] = []
    in_drift: set[int] = set()

    for i, cp in enumerate(cps):
        end = min(cp + max_delay, cps[i + 1] if i + 1 < len(cps) else n)
        inside = [a for a in alarms if cp <= a < end]
        delays.append(inside[0] - cp if inside else None)
        in_drift.update(inside)

    false_alarms = [a for a in alarms if a not in in_drift]
    detected = [d for d in delays if d is not None]
    return {
        "delays": delays,
        "mean_delay": float(np.mean(detected)) if detected else None,
        "missed": sum(d is None for d in delays),
        "miss_rate": (sum(d is None for d in delays) / len(cps)) if cps else None,
        "n_false_alarms": len(false_alarms),
        "false_alarms_per_1000": 1000 * len(false_alarms) / n if n else None,
        "mtfa": mean_time_between(false_alarms),
    }


def mean_time_between(events: Sequence[int]) -> float | None:
    """Mean gap between consecutive events (mean time between false alarms when given false alarms)."""
    if len(events) < 2:
        return None
    return float(np.mean(np.diff(sorted(events))))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred) -> float:
    """Total absolute error as a percentage of total demand; stable for quiet zones."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1.0))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; pairs where both values are 0 count as 0 error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    err = np.where(denom == 0, 0.0, 2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))
    return float(100 * np.mean(err))
