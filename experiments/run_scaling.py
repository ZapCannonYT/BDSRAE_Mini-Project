"""Experiment 3: big-data measurements. Needs Kafka running (scripts/pipeline.sh up) for `stream`.

  python experiments/run_scaling.py batch      Spark batch job on 1, 2, 4 standalone workers
  python experiments/run_scaling.py storage    raw vs aggregated size, partition pruning
  python experiments/run_scaling.py stream     streaming throughput vs workers and latency vs replay speed
  python experiments/run_scaling.py detectors  per-update time and memory of each detector

Output: results/scaling/*.csv
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from taxidrift.config import load_config, resolve
from taxidrift.data.reference import build_zone_hour
from taxidrift.data.zones import load_zone_lookup
from taxidrift.sparkutil import get_spark
from taxidrift.stream.aggregator import KAFKA_PACKAGE, build_query

CFG = load_config()
OUT = resolve(CFG["results_dir"]) / "scaling"
OUT.mkdir(parents=True, exist_ok=True)
WORKER_COUNTS = [1, 2, 4]
CORES_PER_WORKER = 2
MASTER = "spark://localhost:7077"
BROKER = "localhost:9092"


def spark_class(*args, log) -> subprocess.Popen:
    import pyspark

    home = os.path.dirname(pyspark.__file__)
    env = {**os.environ, "SPARK_HOME": home, "PYSPARK_PYTHON": sys.executable, "SPARK_LOCAL_IP": "127.0.0.1"}
    return subprocess.Popen([os.path.join(home, "bin", "spark-class"), *args], env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)


class Cluster:
    """A real standalone Spark cluster on this machine: one master JVM and N worker JVMs."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.procs: list[subprocess.Popen] = []

    def __enter__(self):
        logs = Path.home() / "taxidrift" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.procs.append(spark_class("org.apache.spark.deploy.master.Master", "--host", "localhost",
                                      "--port", "7077", "--webui-port", "8080", log=logs / "spark-master.log"))
        time.sleep(5)
        for i in range(self.workers):
            self.procs.append(spark_class("org.apache.spark.deploy.worker.Worker", MASTER,
                                          "--cores", str(CORES_PER_WORKER), "--memory", "1g",
                                          "--webui-port", str(8081 + i), log=logs / f"spark-worker-{i}.log"))
        time.sleep(5)
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait(timeout=30)
        time.sleep(2)


def batch() -> None:
    zones = load_zone_lookup()["zone"].astype(int).tolist()
    d = CFG["data"]
    tmp = Path.home() / "taxidrift" / "scaling_out"
    rows = []
    for workers in WORKER_COUNTS:
        with Cluster(workers):
            for rep in range(2):
                spark = get_spark(f"batch-{workers}w", master=MASTER, spark__executor__memory="1g",
                                  spark__executor__cores=str(CORES_PER_WORKER), spark__driver__memory="1g",
                                  spark__sql__shuffle__partitions=str(workers * CORES_PER_WORKER * 3))
                shutil.rmtree(tmp, ignore_errors=True)
                res = build_zone_hour(spark, str(resolve(d["raw_dir"])), str(tmp), d["valid_from"], d["valid_to"], zones)
                spark.stop()
                rows.append({"workers": workers, "cores": workers * CORES_PER_WORKER, "repeat": rep, **res})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / "batch_scaling.csv", index=False)
    shutil.rmtree(tmp, ignore_errors=True)


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def storage() -> None:
    from pyspark.sql import functions as F

    d = CFG["data"]
    raw, proc = resolve(d["raw_dir"]), resolve(d["processed_dir"]) / "zone_hour"
    spark = get_spark("storage", master="local[4]")
    rows = [{"item": "raw trip Parquet", "bytes": _dir_size(raw)}, {"item": "zone-hour Parquet", "bytes": _dir_size(proc)}]
    # A partition filter reads one month's files; the same dates on window_start with the partition
    # columns hidden has to open every file.
    no_parts = spark.read.parquet(*[str(p) for p in proc.glob("year=*/month=*")])
    queries = {"partition filter (year, month)":
               lambda: spark.read.parquet(str(proc)).filter("year = 2020 AND month = 3").agg(F.sum("pickups")).first(),
               "timestamp filter, no partition columns":
               lambda: no_parts.filter("window_start >= '2020-03-01' AND window_start < '2020-04-01'").agg(F.sum("pickups")).first()}
    q = []
    for name, fn in queries.items():
        fn()  # warm-up so both see the same caches
        for rep in range(5):
            t0 = time.perf_counter()
            fn()
            q.append({"query": name, "repeat": rep, "seconds": time.perf_counter() - t0})
    pd.DataFrame(rows).to_csv(OUT / "storage.csv", index=False)
    pd.DataFrame(q).to_csv(OUT / "pruning.csv", index=False)
    print(pd.DataFrame(rows), pd.DataFrame(q).groupby("query").seconds.median(), sep="\n")
    spark.stop()


def _topics(*args):
    kafka = next((Path.home() / "taxidrift").glob("kafka_*"))
    subprocess.run([str(kafka / "bin" / "kafka-topics.sh"), "--bootstrap-server", BROKER, *args],
                   check=False, capture_output=True)


def _fresh_topic(name: str, partitions: int) -> None:
    _topics("--delete", "--if-exists", "--topic", name)
    time.sleep(3)
    _topics("--create", "--topic", name, "--partitions", str(partitions), "--replication-factor", "1")
    time.sleep(1)


def _produce(topic: str, start: str, end: str, speedup: float, limit_hours=None) -> dict:
    from taxidrift.stream.producer import replay

    return replay(BROKER, topic, resolve(CFG["data"]["raw_dir"]), start, end, speedup, limit_hours=limit_hours)


def stream() -> None:
    from confluent_kafka import Consumer, TopicPartition

    tmp = Path.home() / "taxidrift" / "scaling_stream"
    rows = []
    # Throughput: a backlog of one busy month, drained by clusters of different sizes.
    for partitions in (1, 4):
        _fresh_topic("bench_trips", partitions)
        prod = _produce("bench_trips", "2020-01", "2020-01", 0)
        base = {"partitions": partitions, "producer_rate": prod["rate"], "trips": prod["trips"]}
        for workers in WORKER_COUNTS if partitions == 4 else [2]:
            _fresh_topic("bench_out", 4)
            with Cluster(workers):
                shutil.rmtree(tmp, ignore_errors=True)
                spark = get_spark(f"stream-{workers}w", master=MASTER, packages=KAFKA_PACKAGE,
                                  spark__executor__memory="1g", spark__executor__cores=str(CORES_PER_WORKER),
                                  spark__sql__shuffle__partitions=str(workers * CORES_PER_WORKER * 2))
                t0 = time.perf_counter()
                q = build_query(spark, BROKER, "bench_trips", "bench_out", "1 hour", "2 hours",
                                str(tmp / "pq"), str(tmp / "cp"), available_now=True)
                q.awaitTermination()
                secs = time.perf_counter() - t0
                spark.stop()
            rows.append({**base, "workers": workers, "cores": workers * CORES_PER_WORKER, "seconds": secs,
                         "rows_per_second": prod["trips"] / secs})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "stream_throughput.csv", index=False)

    # Latency: paced replay. Compare when a window closes in the stream with when its count reached Kafka.
    lat = []
    for speedup in (3600, 14400):
        _fresh_topic("bench_trips", 4)
        _fresh_topic("bench_out", 4)
        shutil.rmtree(tmp, ignore_errors=True)
        spark = get_spark("stream-latency", master="local[6]", packages=KAFKA_PACKAGE, spark__sql__shuffle__partitions="8")
        q = build_query(spark, BROKER, "bench_trips", "bench_out", "1 hour", "2 hours", str(tmp / "pq"), str(tmp / "cp"))
        time.sleep(15)
        begin = time.time()
        _produce("bench_trips", "2020-02", "2020-02", speedup, limit_hours=72)
        time.sleep(20)
        q.stop()
        spark.stop()
        c = Consumer({"bootstrap.servers": BROKER, "group.id": f"lat-{speedup}-{begin}", "auto.offset.reset": "earliest"})
        c.assign([TopicPartition("bench_out", p, 0) for p in range(4)])
        first_seen: dict[str, float] = {}
        idle = 0
        while idle < 10:
            m = c.poll(1.0)
            if m is None:
                idle += 1
                continue
            idle = 0
            if m.error():
                continue
            w = json.loads(m.value())["window_start"]
            ts = m.timestamp()[1] / 1000
            first_seen[w] = min(first_seen.get(w, ts), ts)
        c.close()
        stream_begin = pd.Timestamp("2020-02-01")
        wm_wait = 2 * 3600 / speedup  # the stream must pass window end + watermark first
        for w, ts in first_seen.items():
            closes = begin + (pd.Timestamp(w) + pd.Timedelta(hours=1) - stream_begin).total_seconds() / speedup
            lat.append({"speedup": speedup, "window": w, "seconds_after_close": ts - closes,
                        "processing_latency": ts - closes - wm_wait})
        print(pd.DataFrame(lat).groupby("speedup").processing_latency.describe(), flush=True)
        pd.DataFrame(lat).to_csv(OUT / "stream_latency.csv", index=False)


def detectors() -> None:
    from taxidrift.drift.adwin import ADWIN
    from taxidrift.drift.kswin import KSWIN
    from taxidrift.drift.page_hinkley import PageHinkley
    from taxidrift.drift.psi import PSIDetector

    rng = np.random.default_rng(0)
    data = rng.normal(1, 0.2, 200_000)
    ref = rng.normal(1, 0.2, 2000)
    makers = {"adwin": ADWIN, "page_hinkley": PageHinkley, "kswin": KSWIN,
              "psi (step=1)": lambda: PSIDetector(ref, window=168),
              "psi (step=6)": lambda: PSIDetector(ref, window=168, step=6)}
    rows = []
    for name, make in makers.items():
        n = len(data) if "kswin" not in name else 50_000
        d = make()
        t0 = time.perf_counter()
        d.run(data[:n])
        us = 1e6 * (time.perf_counter() - t0) / n
        tracemalloc.start()
        make().run(data[:20_000])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"detector": name, "microseconds_per_update": us, "peak_kib": peak / 1024})
    pd.DataFrame(rows).to_csv(OUT / "detector_cost.csv", index=False)
    print(pd.DataFrame(rows).round(1))


if __name__ == "__main__":
    {"batch": batch, "storage": storage, "stream": stream, "detectors": detectors}[sys.argv[1]]()
