"""Demand matrix and features for forecasting pickups per zone.

The demand matrix Y has one row per hour and one column per zone. Features for a target hour are
built only from values at least `horizon` hours before it, so the same code serves training,
offline replay and the online monitor without leaking the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from taxidrift.config import load_config, resolve


@dataclass
class Demand:
    Y: np.ndarray            # hours x zones, float32 pickups
    hours: pd.DatetimeIndex  # one per row
    zones: np.ndarray        # zone id per column

    def index_of(self, when) -> int:
        return int(self.hours.searchsorted(pd.Timestamp(when)))


def load_demand(path=None) -> Demand:
    """Load the zone-hour Parquet dataset written by taxidrift.data.reference."""
    path = path or resolve(load_config()["data"]["processed_dir"]) / "zone_hour"
    df = pd.read_parquet(path, columns=["window_start", "zone", "pickups"])
    wide = df.pivot(index="window_start", columns="zone", values="pickups").sort_index()
    wide = wide.fillna(0)
    return Demand(wide.to_numpy(dtype=np.float32), pd.DatetimeIndex(wide.index), wide.columns.to_numpy())


@dataclass
class FeatureSpec:
    horizon: int
    lags: list[int]
    rolling: list[int]

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "FeatureSpec":
        m = (cfg or load_config())["model"]
        return cls(m["horizon"], m["lags_hours"], m["rolling_hours"])

    @property
    def max_history(self) -> int:
        return max(max(self.lags), self.horizon + max(self.rolling))

    @property
    def names(self) -> list[str]:
        return ([f"lag_{k}" for k in self.lags] + [f"roll_{r}" for r in self.rolling]
                + ["hour", "dayofweek", "is_holiday", "zone"])


_ny_holidays = holidays.US(subdiv="NY", years=range(2018, 2023))
_holiday_dates = np.array(sorted(np.datetime64(d) for d in _ny_holidays), dtype="datetime64[D]")


def calendar_features(hours: pd.DatetimeIndex) -> np.ndarray:
    """hour, dayofweek, is_holiday for each hour."""
    days = hours.values.astype("datetime64[D]")
    return np.column_stack([hours.hour, hours.dayofweek, np.isin(days, _holiday_dates)]).astype(np.float32)


def build_features(Y: np.ndarray, hours: pd.DatetimeIndex, targets: np.ndarray, spec: FeatureSpec,
                   zone_idx: np.ndarray | None = None) -> np.ndarray:
    """Feature rows for every (target hour, zone) pair, ordered by target then zone.

    Column order matches spec.names. Every target must be >= spec.max_history. The `zone` feature
    is the column index of the zone in Y.
    """
    targets = np.asarray(targets)
    zone_idx = np.arange(Y.shape[1]) if zone_idx is None else zone_idx
    n_t, n_z = len(targets), len(zone_idx)
    Y_z = Y[:, zone_idx]
    cols = [Y_z[targets - k].ravel() for k in spec.lags]
    origin = targets - spec.horizon  # latest hour we may look at
    for r in spec.rolling:
        cols.append(_rolling_mean(Y_z, origin, r).ravel())
    cal = calendar_features(hours[targets])
    for c in range(cal.shape[1]):
        cols.append(np.repeat(cal[:, c], n_z))
    cols.append(np.tile(zone_idx, n_t).astype(np.float32))
    return np.column_stack(cols).astype(np.float32)


def _rolling_mean(Y_z: np.ndarray, origin: np.ndarray, r: int) -> np.ndarray:
    # A cumulative sum over the whole matrix would be recomputed for every hour online, so only
    # the needed span is summed.
    if len(origin) == 1:
        return Y_z[origin[0] - r + 1: origin[0] + 1].mean(axis=0, keepdims=True)
    lo = int(origin.min()) - r + 1
    cs = np.vstack([np.zeros((1, Y_z.shape[1]), dtype=np.float64),
                    np.cumsum(Y_z[lo: int(origin.max()) + 1], axis=0, dtype=np.float64)])
    return ((cs[origin - lo + 1] - cs[origin - lo + 1 - r]) / r).astype(np.float32)
