"""Replay raw TLC trips into the Kafka `trips` topic in pickup-time order.

Run:  python -m taxidrift.stream.producer --start 2020-02 --end 2020-04 --speedup 3600
      (--speedup 0 sends as fast as possible, used for throughput tests)
Message format: taxidrift.schemas.TRIP_FIELDS, keyed by pickup zone.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from confluent_kafka import Producer

from taxidrift.config import load_config, resolve

BATCH_MINUTES = 5  # pacing granularity, in stream time


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def load_month(raw_dir, month: str):
    """Pickup times (seconds since epoch) and zones for one month, sorted, bad timestamps dropped."""
    table = pq.read_table(raw_dir / f"yellow_tripdata_{month}.parquet", columns=["tpep_pickup_datetime", "PULocationID"])
    ts = table.column(0).to_numpy().astype("datetime64[s]")
    zones = table.column(1).to_numpy()
    lo = np.datetime64(f"{month}-01")
    hi = lo.astype("datetime64[M]") + 1
    keep = (ts >= lo) & (ts < hi.astype("datetime64[s]"))
    ts, zones = ts[keep], zones[keep]
    order = np.argsort(ts, kind="stable")
    return ts[order], zones[order]


def replay(bootstrap_servers: str, topic: str, raw_dir, start_month: str, end_month: str, speedup: float,
           log_every: float = 5.0, limit_hours: float | None = None) -> dict:
    producer = Producer({"bootstrap.servers": bootstrap_servers, "linger.ms": 50, "batch.num.messages": 20000,
                         "queue.buffering.max.messages": 2000000, "compression.type": "lz4"})
    t_begin = time.time()
    stream_begin = None
    sent = 0
    last_log, last_sent = t_begin, 0
    for month in months(start_month, end_month):
        ts, zones = load_month(raw_dir, month)
        epochs = ts.astype("int64")
        stamps = np.datetime_as_string(ts, unit="s")
        bucket = epochs // (BATCH_MINUTES * 60)
        cuts = np.flatnonzero(np.diff(bucket)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(ts)]):
            if stream_begin is None:
                stream_begin = epochs[lo]
            if limit_hours and epochs[lo] - stream_begin >= limit_hours * 3600:
                break
            if speedup > 0:
                wait = t_begin + (epochs[lo] - stream_begin) / speedup - time.time()
                if wait > 0:
                    time.sleep(wait)
            for i in range(lo, hi):
                value = '{"pickup_datetime":"%s","pickup_zone":%d}' % (stamps[i], zones[i])
                while True:
                    try:
                        producer.produce(topic, value, key=str(zones[i]))
                        break
                    except BufferError:
                        producer.poll(0.05)
            sent += int(hi - lo)
            producer.poll(0)
            now = time.time()
            if now - last_log >= log_every:
                print(f"stream time {stamps[hi - 1]}  sent={sent}  rate={(sent - last_sent) / (now - last_log):.0f}/s", flush=True)
                last_log, last_sent = now, sent
    producer.flush()
    seconds = time.time() - t_begin
    print(f"finished: {sent} trips in {seconds:.1f}s ({sent / seconds:.0f}/s)", flush=True)
    return {"trips": sent, "seconds": seconds, "rate": sent / seconds}


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=os.environ.get("REPLAY_START", "2020-02"))
    p.add_argument("--end", default=os.environ.get("REPLAY_END", "2020-04"))
    p.add_argument("--speedup", type=float, default=float(os.environ.get("REPLAY_SPEEDUP", cfg["stream"]["replay_speedup"])))
    p.add_argument("--topic", default=cfg["kafka"]["topics"]["trips"])
    a = p.parse_args()
    res = replay(cfg["kafka"]["bootstrap_servers"], a.topic, resolve(cfg["data"]["raw_dir"]), a.start, a.end, a.speedup)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
