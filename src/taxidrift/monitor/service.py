"""Online monitor: consume `zone_demand`, predict, feed detectors, trigger retraining.

Run:  python -m taxidrift.monitor.service [--policy combined] [--start 2020-02-01] [--end 2020-05-01]

History before --start comes from the batch zone-hour dataset, standing in for what a production
system would already hold. From --start on, demand comes only from Kafka. Spark emits only zones
with at least one trip in an hour, so absent zones count as zero. The zone_demand topic has one
partition, sorted by hour, so an hour is complete as soon as a later hour appears.
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, Producer

from taxidrift.config import load_config, resolve
from taxidrift.model.features import FeatureSpec, load_demand
from taxidrift.model.train import borough_masks
from taxidrift.monitor.audit import AuditLog
from taxidrift.monitor.engine import Engine, build_monitors
from taxidrift.retrain.controller import RetrainController
from taxidrift.retrain.policy import build_policies


def mlflow_logger(cfg):
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    def log(model, result, reason):
        with mlflow.start_run(run_name="promoted" if result.promoted else "rejected"):
            mlflow.log_param("reason", reason)
            mlflow.log_metrics({"challenger_wape": result.challenger_wape, "champion_wape": result.champion_wape,
                                "promoted": float(result.promoted), "seconds": result.seconds})
            if result.promoted:
                mlflow.log_text(model.model_to_string(), "model.txt")

    return log


def run(policy_name: str, start: str, end: str, group: str, idle_seconds: int) -> None:
    cfg = load_config()
    spec = FeatureSpec.from_config(cfg)
    d = load_demand()
    start_t, end_t = d.index_of(start), d.index_of(end)
    Y = d.Y.copy()
    Y[start_t:] = 0  # everything from here on must arrive through Kafka
    zone_col = {int(z): i for i, z in enumerate(d.zones)}

    models = resolve(cfg["models_dir"])
    cal = dict(np.load(models / "calibration.npz"))
    masks = borough_masks(d.zones)
    policy = {p.name: p for p in build_policies(cfg["retrain"]["cooldown_hours"])}[policy_name]
    controller = RetrainController(spec, cfg["retrain"]["train_window_weeks"], cfg["retrain"]["gate_holdout_hours"],
                                   num_threads=4, log_model=mlflow_logger(cfg))
    controller.champion = lgb.Booster(model_file=str(models / "champion.txt"))
    out = resolve(cfg["results_dir"]) / "stream"
    out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "audit_log.jsonl")

    producer = Producer({"bootstrap.servers": cfg["kafka"]["bootstrap_servers"]})

    def publish(record):
        producer.produce(cfg["kafka"]["topics"]["drift_events"], json.dumps(record))
        producer.poll(0)
        print(f"ALARM {record['timestamp']} {record['detector']:5s} scope={record['scope']}", flush=True)

    engine = Engine(Y, d.hours, spec, build_monitors(cal, masks, Y.shape[1], cfg["drift"]), policy, controller,
                    audit=audit, on_alarm=publish)
    engine.prime(start_t)

    consumer = Consumer({"bootstrap.servers": cfg["kafka"]["bootstrap_servers"], "group.id": group,
                         "auto.offset.reset": "earliest"})
    consumer.subscribe([cfg["kafka"]["topics"]["zone_demand"]])

    next_t, latest, n_msgs = start_t, -1, 0
    rows, last_msg, t_begin = [], time.time(), time.time()
    print(f"monitor up: policy={policy_name}, stream from {start}", flush=True)
    while next_t < end_t and time.time() - last_msg < idle_seconds:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        last_msg = time.time()
        rec = json.loads(msg.value())
        hour = d.index_of(rec["window_start"])
        n_msgs += 1
        if hour < next_t or hour >= len(d.hours) or rec["zone"] not in zone_col:
            continue  # late for a finalised hour, or outside the dataset
        Y[hour, zone_col[rec["zone"]]] = rec["pickups"]
        latest = max(latest, hour)
        while next_t < latest and next_t < end_t:
            readings = engine.step(next_t)
            n_retrains = len(engine.retrains)
            rows.append({"hour": d.hours[next_t], "demand": float(Y[next_t].sum()),
                         **{f"err_{s}": r["err"] for s, r in readings.items()},
                         "psi_city": readings["city"]["psi_value"], "retrains": n_retrains})
            if next_t % 24 == 0:
                print(f"{d.hours[next_t]} demand={Y[next_t].sum():7.0f} err_ratio={readings['city']['err']:.2f} "
                      f"psi={readings['city']['psi_value'] or 0:.3f} retrains={n_retrains}", flush=True)
            if engine.retrains and len(rows) and engine.retrains[-1]["timestamp"] == str(d.hours[next_t]):
                r = engine.retrains[-1]
                print(f"RETRAIN {r['timestamp']} {'PROMOTED' if r['promoted'] else 'REJECTED'} "
                      f"challenger={r['challenger_wape']:.1f}% champion={r['champion_wape']:.1f}% ({r['reason']})", flush=True)
            next_t += 1
        last_msg = time.time()  # a retrain can take a while; that is not an idle stream

    producer.flush(5)
    consumer.close()
    pd.DataFrame(rows).to_csv(out / "monitor_hourly.csv", index=False)
    pd.DataFrame(engine.alarms).to_csv(out / "monitor_alarms.csv", index=False)
    pd.DataFrame(engine.retrains).to_csv(out / "monitor_retrains.csv", index=False)
    print(f"done: {len(rows)} hours, {n_msgs} messages, {len(engine.alarms)} alarms, "
          f"{len(engine.retrains)} retrains, {time.time() - t_begin:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="combined")
    p.add_argument("--start", default="2020-02-01")
    p.add_argument("--end", default="2020-05-01")
    p.add_argument("--group", default=f"monitor-{int(time.time())}")
    p.add_argument("--idle-seconds", type=int, default=300)
    a = p.parse_args()
    run(a.policy, a.start, a.end, a.group, a.idle_seconds)


if __name__ == "__main__":
    main()
