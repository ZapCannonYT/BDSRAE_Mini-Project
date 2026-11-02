"""Train the demand model, score it against the seasonal-naive baseline, and save the champion.

Run:  python -m taxidrift.model.train
Output: models/champion.txt, models/calibration.npz, results/model/baseline_metrics.csv
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxidrift.config import load_config, resolve
from taxidrift.data.zones import zone_to_borough
from taxidrift.eval.metrics import mae, smape, wape
from taxidrift.model.features import Demand, FeatureSpec, build_features, load_demand

PARAMS = {"objective": "poisson", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.9, "verbosity": -1, "seed": 42}


def train_model(Y, hours, targets, spec: FeatureSpec, valid_targets=None, rounds: int = 300,
                num_threads: int = 0) -> lgb.Booster:
    """Fit LightGBM on the given target hours; early-stop on valid_targets when provided."""
    X = build_features(Y, hours, targets, spec)
    train = lgb.Dataset(X, Y[targets].ravel(), feature_name=spec.names, categorical_feature=["zone"],
                        free_raw_data=True)
    params = {**PARAMS, "num_threads": num_threads}
    if valid_targets is None:
        return lgb.train(params, train, num_boost_round=rounds)
    valid = lgb.Dataset(build_features(Y, hours, valid_targets, spec), Y[valid_targets].ravel(), reference=train)
    return lgb.train(params, train, num_boost_round=rounds, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(20, verbose=False)])


def predict(model: lgb.Booster, Y, hours, targets, spec: FeatureSpec, num_threads: int = 0,
            zone_idx=None) -> np.ndarray:
    """Predictions shaped (len(targets), number of zones)."""
    X = build_features(Y, hours, targets, spec, zone_idx)
    n_z = Y.shape[1] if zone_idx is None else len(zone_idx)
    return model.predict(X, num_threads=num_threads).reshape(len(targets), n_z)


def hour_of_week(hours: pd.DatetimeIndex) -> np.ndarray:
    return (hours.dayofweek * 24 + hours.hour).to_numpy()


def borough_masks(zones: np.ndarray) -> dict[str, np.ndarray]:
    lookup = zone_to_borough()
    names = np.array([lookup.get(int(z), "Unknown") for z in zones])
    return {b: names == b for b in sorted(set(names)) if b not in ("Unknown", "N/A", "EWR")}


def hourly_wape(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row absolute error as a share of demand, for rows of (hours, zones)."""
    return np.abs(y - p).sum(axis=1) / np.maximum(y.sum(axis=1), 1.0)


def _profile(values: np.ndarray, how: np.ndarray) -> np.ndarray:
    prof = np.ones(168)
    for h in range(168):
        sel = values[how == h]
        if len(sel):
            prof[h] = max(float(np.mean(sel)), 1e-6)
    return prof


def make_calibration(y: np.ndarray, p: np.ndarray, hours: pd.DatetimeIndex,
                     masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hour-of-week means of error share and of demand, for the city and each borough.

    Detectors see values divided by these profiles, which removes the daily and weekly cycle and
    the size differences between boroughs. A stable stream then hovers around 1.
    """
    how = hour_of_week(hours)
    cal = {}
    for scope, mask in {"city": np.ones(y.shape[1], dtype=bool), **masks}.items():
        cal[f"err_{scope}"] = _profile(hourly_wape(y[:, mask], p[:, mask]), how)
        cal[f"dem_{scope}"] = _profile(y[:, mask].sum(axis=1), how)
        # reference sample for PSI: calibration-period demand after normalising
        cal[f"demref_{scope}"] = y[:, mask].sum(axis=1) / cal[f"dem_{scope}"][how]
    return cal


def score(y, p) -> dict:
    return {"mae": mae(y, p), "smape": smape(y, p), "wape": wape(y, p)}


def main() -> None:
    cfg = load_config()
    m = cfg["model"]
    spec = FeatureSpec.from_config(cfg)
    d: Demand = load_demand()
    masks = borough_masks(d.zones)

    train_end, cal_end, stream_end = d.index_of(m["train_to"]), d.index_of(m["calibrate_to"]), d.index_of(m["stream_from_to"])
    train_t = np.arange(max(spec.max_history, d.index_of(m["train_from"])), train_end)
    cal_t = np.arange(train_end, cal_end)
    test_t = np.arange(cal_end, stream_end)

    t0 = time.perf_counter()
    model = train_model(d.Y, d.hours, train_t, spec, valid_targets=cal_t)
    secs = time.perf_counter() - t0
    print(f"trained on {len(train_t)} hours x {d.Y.shape[1]} zones in {secs:.1f}s, {model.best_iteration} rounds")

    p_cal = predict(model, d.Y, d.hours, cal_t, spec)
    p_test = predict(model, d.Y, d.hours, test_t, spec)
    naive_test = d.Y[test_t - 168]

    rows = [{"model": "lightgbm", "split": "Oct-Dec 2019", "scope": "city", **score(d.Y[test_t], p_test)},
            {"model": "seasonal_naive", "split": "Oct-Dec 2019", "scope": "city", **score(d.Y[test_t], naive_test)}]
    for b, mask in masks.items():
        rows.append({"model": "lightgbm", "split": "Oct-Dec 2019", "scope": b,
                     **score(d.Y[test_t][:, mask], p_test[:, mask])})
        rows.append({"model": "seasonal_naive", "split": "Oct-Dec 2019", "scope": b,
                     **score(d.Y[test_t][:, mask], naive_test[:, mask])})
    out = resolve(cfg["results_dir"]) / "model"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "baseline_metrics.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    models = resolve(cfg["models_dir"])
    models.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models / "champion.txt"))
    np.savez(models / "calibration.npz", **make_calibration(d.Y[cal_t], p_cal, d.hours[cal_t], masks))
    (models / "champion.json").write_text(json.dumps({"train_seconds": secs, "rounds": model.best_iteration,
                                                      "train_hours": len(train_t), "horizon": spec.horizon}))


if __name__ == "__main__":
    main()
