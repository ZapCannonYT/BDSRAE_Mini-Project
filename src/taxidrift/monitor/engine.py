"""Hour-by-hour monitoring logic, shared by the offline replay and the Kafka monitor service.

For each observed hour: score the prediction made earlier, feed the detectors, ask the policy
whether to retrain, then predict the next hour. Detectors see scale-free values (see
taxidrift.model.train.make_calibration): error share and demand, each divided by its usual level
for that hour of the week.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from taxidrift.drift.adwin import ADWIN
from taxidrift.drift.psi import PSIDetector
from taxidrift.model.features import FeatureSpec
from taxidrift.model.train import hourly_wape, predict
from taxidrift.retrain.controller import RetrainController
from taxidrift.retrain.policy import RetrainPolicy

CLIP = 5.0  # ratio cap, so one wild hour cannot dominate a detector's mean


def error_ratio(y: np.ndarray, p: np.ndarray, how: int, profile: np.ndarray) -> float:
    return float(min(hourly_wape(y[None], p[None])[0] / profile[how], CLIP))


@dataclass
class Monitor:
    """Detectors for one scope (the city or one borough)."""

    scope: str
    mask: np.ndarray
    err_profile: np.ndarray
    dem_profile: np.ndarray
    adwin: ADWIN
    psi: PSIDetector
    recent_demand: list[float] = field(default_factory=list)

    def observe(self, y: np.ndarray, p: np.ndarray, how: int) -> dict:
        err = error_ratio(y[self.mask], p[self.mask], how, self.err_profile)
        dem = float(y[self.mask].sum() / self.dem_profile[how])
        self.recent_demand.append(dem)
        return {"err": err, "dem": dem, "adwin": self.adwin.update(err), "psi": self.psi.update(dem),
                "psi_warn": self.psi.warning_detected, "psi_value": self.psi.last_psi}

    def rebase(self, keep_hours: int) -> None:
        """After a promoted retrain: new error baseline, and PSI compares with the new training period."""
        self.adwin.reset()
        ref = np.array(self.recent_demand[-keep_hours:])
        if len(ref) >= 168:
            self.psi = PSIDetector(ref, self.psi.window, self.psi.bins, self.psi.warn, self.psi.drift, self.psi.step)
        else:
            self.psi.reset()


def build_monitors(cal: dict, masks: dict[str, np.ndarray], n_zones: int, drift_cfg: dict) -> dict[str, Monitor]:
    all_zones = np.ones(n_zones, dtype=bool)
    psi_cfg = drift_cfg["psi"]
    delta = drift_cfg["adwin"]["delta"][0]
    out = {}
    for scope, mask in {"city": all_zones, **masks}.items():
        out[scope] = Monitor(scope, mask, cal[f"err_{scope}"], cal[f"dem_{scope}"], ADWIN(delta=delta),
                             PSIDetector(cal[f"demref_{scope}"], psi_cfg["window_hours"], psi_cfg["bins"],
                                         psi_cfg["warn"], psi_cfg["drift"]))
    return out


class Engine:
    def __init__(self, Y: np.ndarray, hours: pd.DatetimeIndex, spec: FeatureSpec, monitors: dict[str, Monitor],
                 policy: RetrainPolicy, controller: RetrainController, audit=None, num_threads: int = 0,
                 on_alarm=None) -> None:
        self.Y, self.hours, self.spec = Y, hours, spec
        self.monitors = monitors
        self.policy = policy
        self.controller = controller
        self.audit = audit
        self.num_threads = num_threads
        self.on_alarm = on_alarm  # optional callback(record), e.g. publish to Kafka
        self.P = np.full(Y.shape, np.nan, dtype=np.float32)
        self.retrains: list[dict] = []
        self.alarms: list[dict] = []

    def prime(self, t: int) -> None:
        """Make the prediction for hour t before the stream reaches it."""
        self.P[t] = predict(self.controller.champion, self.Y, self.hours, np.array([t]), self.spec, self.num_threads)[0]

    def step(self, t: int) -> dict:
        """Hour t has just been observed in self.Y. Returns its per-scope readings."""
        now = self.hours[t]
        y, p = self.Y[t], self.P[t]
        how = now.dayofweek * 24 + now.hour
        readings = {s: m.observe(y, p, how) for s, m in self.monitors.items()}
        self._record_alarms(now, readings)

        city = readings["city"]
        boroughs = [r for s, r in readings.items() if s != "city"]
        signals = {"adwin": city["adwin"], "psi": city["psi"], "psi_warn": city["psi_warn"],
                   "adwin_borough": any(r["adwin"] for r in boroughs),
                   "psi_borough": any(r["psi"] for r in boroughs)}
        if self.policy.should_retrain(now, signals):
            self._retrain(t, now)

        if t + self.spec.horizon < len(self.hours):
            self.prime(t + self.spec.horizon)
        return readings

    def _record_alarms(self, now, readings) -> None:
        for scope, r in readings.items():
            for detector in ("adwin", "psi"):
                if r[detector]:
                    record = {"timestamp": str(now), "detector": detector, "scope": scope, "level": "drift",
                              "statistic": r["psi_value"] if detector == "psi" else r["err"]}
                    self.alarms.append(record)
                    if self.audit:
                        self.audit.write("drift_alarm", str(now), **{k: v for k, v in record.items() if k != "timestamp"})
                    if self.on_alarm:
                        self.on_alarm(record)

    def _retrain(self, t: int, now) -> None:
        reason = self.policy.reason
        result = self.controller.retrain(t, self.Y, self.hours, reason)
        record = {"timestamp": str(now), "reason": reason, "promoted": result.promoted,
                  "challenger_wape": result.challenger_wape, "champion_wape": result.champion_wape,
                  "seconds": result.seconds}
        self.retrains.append(record)
        if self.audit:
            event = "retrain_promoted" if result.promoted else "retrain_rejected"
            self.audit.write(event, str(now), **{k: v for k, v in record.items() if k != "timestamp"})
        if result.promoted:
            for m in self.monitors.values():
                m.rebase(self.controller.train_hours)
