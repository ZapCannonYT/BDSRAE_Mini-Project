"""Offline Spark batch job: raw trip Parquet -> hourly pickups per zone.

Output is the training/reference dataset for the model and for PSI.

Run:  python -m taxidrift.data.reference
"""

from __future__ import annotations

import glob
import json
import os
import time

from taxidrift.config import load_config, resolve
from taxidrift.data.zones import UNKNOWN_ZONES, load_zone_lookup
from taxidrift.sparkutil import get_spark


def build_zone_hour(spark, raw_dir: str, out_dir: str, valid_from: str, valid_to: str,
                    zones: list[int], write: bool = True) -> dict:
    """Aggregate all monthly yellow_tripdata_*.parquet files into zone-hour counts.

    Output: Parquet at `out_dir`, partitioned by year and month, columns
    window_start (timestamp), zone (int), pickups (int). Every zone has a row for every hour.
    Returns row counts and the runtime.
    """
    from pyspark.sql import functions as F

    t0 = time.perf_counter()
    files = sorted(glob.glob(os.path.join(raw_dir, "yellow_tripdata_*.parquet")))
    # Column types differ slightly between months, so each file is read and cast on its own.
    trips = None
    for path in files:
        one = spark.read.parquet(path).select(F.col("tpep_pickup_datetime").cast("timestamp").alias("pickup"),
                                              F.col("PULocationID").cast("int").alias("zone"))
        trips = one if trips is None else trips.unionByName(one)

    valid = trips.filter((F.col("pickup") >= valid_from) & (F.col("pickup") < valid_to)
                         & ~F.col("zone").isin(list(UNKNOWN_ZONES)))
    counts = (valid.withColumn("window_start", F.date_trunc("hour", "pickup"))
              .groupBy("window_start", "zone").agg(F.count("*").cast("int").alias("pickups")))

    hours = spark.sql(f"SELECT explode(sequence(timestamp'{valid_from} 00:00:00', "
                      f"timestamp'{valid_to} 00:00:00' - interval 1 hour, interval 1 hour)) AS window_start")
    grid = hours.crossJoin(spark.createDataFrame([(z,) for z in zones], ["zone"]).withColumn("zone", F.col("zone").cast("int")))
    full = (grid.join(counts, ["window_start", "zone"], "left").fillna(0, ["pickups"])
            .withColumn("year", F.year("window_start")).withColumn("month", F.month("window_start")))

    result = {"files": len(files), "output_rows": None}
    if write:
        full.repartition("year", "month").write.mode("overwrite").partitionBy("year", "month").parquet(out_dir)
        result["output_rows"] = spark.read.parquet(out_dir).count()
        result["trips"] = spark.read.parquet(out_dir).agg(F.sum("pickups")).first()[0]
    else:
        result["output_rows"] = full.count()
    result["seconds"] = time.perf_counter() - t0
    return result


def main() -> None:
    cfg = load_config()["data"]
    spark = get_spark("zone-hour-batch")
    zones = load_zone_lookup()["zone"].astype(int).tolist()
    out = resolve(cfg["processed_dir"]) / "zone_hour"
    res = build_zone_hour(spark, str(resolve(cfg["raw_dir"])), str(out), cfg["valid_from"], cfg["valid_to"], zones)
    print(json.dumps(res, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
