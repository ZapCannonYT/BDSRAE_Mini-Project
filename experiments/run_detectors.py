"""Experiment 1: detector comparison on the synthetic drift benchmark and on real COVID drift.

Output: results/detectors/benchmark.csv (one row per scenario x seed x detector x stream scope),
results/detectors/covid_alarms.csv. Run:  python experiments/run_detectors.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxidrift.config import load_config, resolve
from taxidrift.drift.adwin import ADWIN
from taxidrift.drift.inject import DriftSpec, inject
from taxidrift.drift.kswin import KSWIN
from taxidrift.drift.page_hinkley import PageHinkley
from taxidrift.drift.psi import PSIDetector
from taxidrift.eval.metrics import score_detections
from taxidrift.model.features import FeatureSpec, load_demand
from taxidrift.model.train import borough_masks, hour_of_week, hourly_wape, predict
from taxidrift.monitor.engine import CLIP

SCENARIOS = {  # name -> kind, magnitude
    "abrupt_down_50pct": ("abrupt", 0.5),
    "abrupt_up_50pct": ("abrupt", 1.5),
    "gradual_down_50pct": ("gradual", 0.5),
    "profile_shift_3h": ("profile_shift", 3),
}
TARGETS = ["city", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

G: dict = {}  # per-process state


def detector_factories(cfg: dict, cal: dict, scope: str) -> dict:
    """name -> (input signal, constructor)."""
    d = cfg["drift"]
    out = {}
    for delta in d["adwin"]["delta"]:
        out[f"adwin_d{delta}"] = ("err", lambda delta=delta: ADWIN(delta=delta))
    out["adwin_demand_d0.002"] = ("dem", lambda: ADWIN(delta=0.002))
    for thr in d["psi"]["drift_grid"]:
        out[f"psi_t{thr}"] = ("dem", lambda thr=thr: PSIDetector(cal[f"demref_{scope}"], d["psi"]["window_hours"],
                                                                d["psi"]["bins"], warn=0.4 * thr, drift=thr, step=6))
    for thr in d["page_hinkley"]["threshold_grid"]:
        out[f"page_hinkley_t{thr}"] = ("err", lambda thr=thr: PageHinkley(delta=d["page_hinkley"]["delta"], threshold=thr))
    for alpha in d["kswin"]["alpha_grid"]:
        out[f"kswin_a{alpha}"] = ("err", lambda alpha=alpha: KSWIN(d["kswin"]["window_size"], d["kswin"]["stat_size"], alpha))
    return out


def signals(y: np.ndarray, p: np.ndarray, how: np.ndarray, cal: dict, mask: np.ndarray, scope: str) -> dict:
    err = np.minimum(hourly_wape(y[:, mask], p[:, mask]) / cal[f"err_{scope}"][how], CLIP)
    dem = y[:, mask].sum(axis=1) / cal[f"dem_{scope}"][how]
    return {"err": err, "dem": dem}


def _init(cfg):
    d = load_demand()
    spec = FeatureSpec.from_config(cfg)
    models = resolve(cfg["models_dir"])
    cal = dict(np.load(models / "calibration.npz"))
    masks = borough_masks(d.zones)
    masks["city"] = np.ones(d.Y.shape[1], dtype=bool)
    t0, t1 = d.index_of(cfg["model"]["calibrate_to"]), d.index_of(cfg["model"]["stream_from_to"])
    G.update(cfg=cfg, d=d, spec=spec, cal=cal, masks=masks, stream=np.arange(t0, t1),
             model=lgb.Booster(model_file=str(models / "champion.txt")),
             base=np.load(models / "bench_base_pred.npy"))
    G["how"] = hour_of_week(d.hours[G["stream"]])


def hash_str(s: str) -> int:
    # stable across processes, unlike hash()
    return sum((i + 1) * ord(c) for i, c in enumerate(s))


def run_case(case: tuple) -> list[dict]:
    scenario, target, seed = case
    d, spec, cal, masks, stream = G["d"], G["spec"], G["cal"], G["masks"], G["stream"]
    rng = np.random.default_rng([seed, hash_str(scenario + target)])
    n = len(stream)
    max_delay = G["cfg"]["benchmark"]["max_delay_hours"]

    if scenario == "no_drift":
        # every zone gets a fresh Poisson realisation of the observed counts
        Y = d.Y.copy()
        Y[stream] = rng.poisson(Y[stream])
        P = predict(G["model"], Y, d.hours, stream, spec, num_threads=1)
        cp_rel = None
    else:
        kind, magnitude = SCENARIOS[scenario]
        cp_rel = int(rng.integers(24 * 30, 24 * 50))  # change point 30-50 days into the 92-day stream
        mask = masks[target]
        spec_d = DriftSpec(kind, int(stream[cp_rel]), magnitude, duration_hours=336)
        Y = inject(d.Y[: stream[-1] + 1], spec_d, mask, seed=int(rng.integers(2**31)))
        P = G["base"].copy()
        zones = np.flatnonzero(mask)
        P[np.ix_(np.arange(cp_rel, n), zones)] = predict(G["model"], Y, d.hours, stream[cp_rel:], spec,
                                                         num_threads=1, zone_idx=zones)

    y = Y[stream]
    rows = []
    scopes = TARGETS if scenario == "no_drift" else ["city"] if target == "city" else ["city", target]
    for scope in scopes:
        sig = signals(y, P, G["how"], cal, masks[scope], scope)
        for name, (signal, make) in detector_factories(G["cfg"], cal, scope).items():
            alarms = make().run(sig[signal])
            score = score_detections(alarms, [] if cp_rel is None else [cp_rel], n, max_delay)
            pre_change = len(alarms) if cp_rel is None else sum(a < cp_rel for a in alarms)
            rows.append({"scenario": scenario, "target": target, "seed": seed, "detector": name,
                         "family": name.split("_d0")[0].split("_t")[0].split("_a0")[0],
                         "stream_scope": scope, "change_point": cp_rel,
                         "delay": score["delays"][0] if score["delays"] else None,
                         "detected": bool(score["delays"] and score["delays"][0] is not None),
                         "false_alarms": score["n_false_alarms"], "pre_change_alarms": pre_change,
                         "hours_watched": n if cp_rel is None else cp_rel})
    return rows


def covid_replay(cfg) -> pd.DataFrame:
    """Fixed 2019 model on real data from Oct 2019 to Jun 2020; when does each detector first fire?"""
    d = load_demand()
    spec = FeatureSpec.from_config(cfg)
    models = resolve(cfg["models_dir"])
    cal = dict(np.load(models / "calibration.npz"))
    masks = borough_masks(d.zones)
    masks["city"] = np.ones(d.Y.shape[1], dtype=bool)
    model = lgb.Booster(model_file=str(models / "champion.txt"))
    t = np.arange(d.index_of(cfg["model"]["calibrate_to"]), d.index_of("2020-07-01"))
    P = predict(model, d.Y, d.hours, t, spec)
    how = hour_of_week(d.hours[t])
    onset = pd.Timestamp("2020-03-15")
    rows = []
    for scope in TARGETS:
        sig = signals(d.Y[t], P, how, cal, masks[scope], scope)
        for name, (signal, make) in detector_factories(cfg, cal, scope).items():
            alarms = make().run(sig[signal])
            times = d.hours[t][alarms] if alarms else pd.DatetimeIndex([])
            before = times[times < pd.Timestamp("2020-03-01")]
            after = times[times >= pd.Timestamp("2020-03-01")]
            first = after[0] if len(after) else None
            rows.append({"detector": name, "scope": scope, "alarms_before_march": len(before),
                         "first_alarm_after_mar1": first,
                         "hours_from_mar15": (first - onset) / pd.Timedelta(hours=1) if first is not None else None,
                         "total_alarms": len(alarms)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cfg = load_config()
    seeds = a.seeds or cfg["benchmark"]["n_seeds"]
    out = resolve(cfg["results_dir"]) / "detectors"
    out.mkdir(parents=True, exist_ok=True)

    covid = covid_replay(cfg)
    covid.to_csv(out / "covid_alarms.csv", index=False)
    print(covid[covid.scope == "city"].to_string(index=False), flush=True)

    # Baseline predictions are computed once, then workers re-predict only what each drift changes.
    d = load_demand()
    spec = FeatureSpec.from_config(cfg)
    models = resolve(cfg["models_dir"])
    stream = np.arange(d.index_of(cfg["model"]["calibrate_to"]), d.index_of(cfg["model"]["stream_from_to"]))
    base = predict(lgb.Booster(model_file=str(models / "champion.txt")), d.Y, d.hours, stream, spec)
    np.save(models / "bench_base_pred.npy", base)

    cases = [("no_drift", "none", s) for s in range(seeds)]
    cases += [(sc, t, s) for sc in SCENARIOS for t in TARGETS for s in range(seeds)]
    print(f"{len(cases)} runs on {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, res in enumerate(pool.imap_unordered(run_case, cases, chunksize=4), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"{i}/{len(cases)} in {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
