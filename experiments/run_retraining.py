"""Experiment 2: retraining policies over Oct 2019 - Dec 2021, with per-borough error.

Replays the shared monitor Engine hour by hour on the zone-hour data (no Kafka needed).
Output: results/retraining/<policy>_daily.csv, retrains.csv, alarms.csv, summary.csv.
Run:  python experiments/run_retraining.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxidrift.config import load_config, resolve
from taxidrift.model.features import FeatureSpec, load_demand
from taxidrift.model.train import borough_masks
from taxidrift.monitor.engine import Engine, build_monitors
from taxidrift.retrain.controller import RetrainController
from taxidrift.retrain.policy import build_policies

PERIODS = {"before (Oct 2019-Feb 2020)": ("2019-10-01", "2020-03-01"),
           "crash (Mar-Jun 2020)": ("2020-03-01", "2020-07-01"),
           "recovery (Jul 2020-Dec 2021)": ("2020-07-01", "2022-01-01")}


def run_policy(policy_name: str) -> dict:
    cfg = load_config()
    d = load_demand()
    spec = FeatureSpec.from_config(cfg)
    models = resolve(cfg["models_dir"])
    cal = dict(np.load(models / "calibration.npz"))
    masks = borough_masks(d.zones)
    rc = cfg["retrain"]
    policy = {p.name: p for p in build_policies(rc["cooldown_hours"])}[policy_name]
    controller = RetrainController(spec, rc["train_window_weeks"], rc["gate_holdout_hours"], num_threads=2)
    controller.champion = lgb.Booster(model_file=str(models / "champion.txt"))
    engine = Engine(d.Y, d.hours, spec, build_monitors(cal, masks, d.Y.shape[1], cfg["drift"]), policy, controller,
                    num_threads=1)

    start, end = d.index_of(cfg["model"]["calibrate_to"]), len(d.hours)
    engine.prime(start)
    t0 = time.time()
    for t in range(start, end):
        engine.step(t)
    seconds = time.time() - t0

    days = d.hours[start:end].floor("D")
    y, p = d.Y[start:end], engine.P[start:end]
    # sMAPE terms are summed per zone-hour; pairs that are both zero count as zero error
    denom = np.abs(y) + np.abs(p)
    smape_terms = np.where(denom == 0, 0.0, 2 * np.abs(y - p) / np.where(denom == 0, 1, denom))
    frames = []
    for scope, mask in {"city": np.ones(d.Y.shape[1], dtype=bool), **masks}.items():
        frames.append(pd.DataFrame({"day": days, "scope": scope, "abs_err": np.abs(y[:, mask] - p[:, mask]).sum(axis=1),
                                    "demand": y[:, mask].sum(axis=1), "smape_sum": smape_terms[:, mask].sum(axis=1),
                                    "n": int(mask.sum())}).groupby(["day", "scope"], as_index=False).sum())
    daily = pd.concat(frames)
    daily["policy"] = policy_name

    out = resolve(cfg["results_dir"]) / "retraining"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / f"{policy_name}_daily.csv", index=False)
    print(f"{policy_name}: {len(engine.retrains)} retrains, "
          f"{sum(r['promoted'] for r in engine.retrains)} promoted, {seconds:.0f}s", flush=True)
    return {"policy": policy_name, "replay_seconds": seconds,
            "retrains": [{**r, "policy": policy_name} for r in engine.retrains],
            "alarms": [{**r, "policy": policy_name} for r in engine.alarms]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cfg = load_config()
    names = [x.name for x in build_policies(cfg["retrain"]["cooldown_hours"])]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_policy, names, chunksize=1)
    out = resolve(cfg["results_dir"]) / "retraining"
    retrains = pd.DataFrame([r for res in results for r in res["retrains"]])
    retrains.to_csv(out / "retrains.csv", index=False)
    pd.DataFrame([r for res in results for r in res["alarms"]]).to_csv(out / "alarms.csv", index=False)

    rows = []
    for res in results:
        name = res["policy"]
        daily = pd.read_csv(out / f"{name}_daily.csv", parse_dates=["day"])
        sub = retrains[retrains.policy == name] if len(retrains) else retrains
        for period, (lo, hi) in PERIODS.items():
            part = daily[(daily.day >= lo) & (daily.day < hi)]
            for scope, g in part.groupby("scope"):
                # g.n is zones per hour summed over the hours of each day, i.e. zone-hours
                rows.append({"policy": name, "period": period, "scope": scope,
                             "wape": 100 * g.abs_err.sum() / g.demand.sum(),
                             "mae": g.abs_err.sum() / g.n.sum(),
                             "smape": 100 * g.smape_sum.sum() / g.n.sum()})
        rows.append({"policy": name, "period": "cost", "scope": "city", "retrains": len(sub),
                     "promoted": int(sub.promoted.sum()) if len(sub) else 0,
                     "retrain_seconds": float(sub.seconds.sum()) if len(sub) else 0.0})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[summary.scope == "city"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
