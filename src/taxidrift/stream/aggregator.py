"""Spark Structured Streaming job: `trips` topic -> hourly pickups per zone.

Run:  python -m taxidrift.stream.aggregator
Writes each closed window to the `zone_demand` topic (taxidrift.schemas.ZONE_DEMAND_FIELDS)
and to Parquet partitioned by year and month. Progress metrics go to results/stream/.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time

from taxidrift.config import load_config, resolve
from taxidrift.sparkutil import get_spark

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"


def build_query(spark, bootstrap_servers: str, in_topic: str, out_topic: str, window: str, watermark: str,
                parquet_dir: str, checkpoint_dir: str, available_now: bool = False):
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    schema = StructType([StructField("pickup_datetime", StringType()), StructField("pickup_zone", IntegerType())])
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", bootstrap_servers)
           .option("subscribe", in_topic)
           .option("startingOffsets", "earliest")
           .option("failOnDataLoss", "false")
           .load())
    trips = (raw.select(F.from_json(F.col("value").cast("string"), schema).alias("t"))
             .select(F.to_timestamp("t.pickup_datetime").alias("pickup"), F.col("t.pickup_zone").alias("zone")))
    counts = (trips.withWatermark("pickup", watermark)
              .groupBy(F.window("pickup", window), "zone").count())

    def sink(batch, batch_id):
        rows = batch.select(F.col("window.start").alias("window_start"), "zone",
                            F.col("count").cast("int").alias("pickups")).cache()
        # One partition, sorted by hour: the monitor relies on this order to know when an hour is complete.
        (rows.repartition(1).sortWithinPartitions("window_start")
         .select(F.col("zone").cast("string").alias("key"),
                     F.to_json(F.struct(F.date_format("window_start", "yyyy-MM-dd'T'HH:mm:ss").alias("window_start"),
                                        "zone", "pickups")).alias("value"))
         .write.format("kafka").option("kafka.bootstrap.servers", bootstrap_servers)
         .option("topic", out_topic).save())
        (rows.withColumn("year", F.year("window_start")).withColumn("month", F.month("window_start"))
         .write.mode("append").partitionBy("year", "month").parquet(parquet_dir))
        rows.unpersist()

    writer = counts.writeStream.outputMode("append").foreachBatch(sink).option("checkpointLocation", checkpoint_dir)
    if available_now:
        writer = writer.trigger(availableNow=True)
    return writer.start()


def watch(query, progress_csv, idle_stop: float | None = None) -> None:
    """Follow the query, saving each micro-batch's throughput and duration."""
    fields = ["batch", "time", "input_rows", "input_rate", "processed_rate", "batch_ms", "watermark"]
    seen, last_rows = -1, time.time()
    with open(progress_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        while query.isActive:
            p = query.lastProgress
            if p and p["batchId"] != seen:
                seen = p["batchId"]
                n = p["numInputRows"]
                w.writerow([seen, p["timestamp"], n, p.get("inputRowsPerSecond"), p.get("processedRowsPerSecond"),
                            p["durationMs"].get("triggerExecution"), p.get("eventTime", {}).get("watermark")])
                f.flush()
                if n:
                    last_rows = time.time()
                    print(f"batch {seen}: {n} rows, {p.get('processedRowsPerSecond', 0):.0f} rows/s, "
                          f"watermark {p.get('eventTime', {}).get('watermark')}", flush=True)
            if idle_stop and time.time() - last_rows > idle_stop:
                query.stop()
                break
            time.sleep(1)


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--idle-stop", type=float, default=None, help="stop after this many seconds with no input")
    p.add_argument("--fresh", action="store_true", help="delete the checkpoint and streamed Parquet first")
    a = p.parse_args()
    topics = cfg["kafka"]["topics"]
    stream_dir = resolve(cfg["data"]["processed_dir"]).parent / "stream"
    checkpoint = stream_dir / "checkpoint"
    parquet = stream_dir / "zone_demand"
    if a.fresh:
        shutil.rmtree(stream_dir, ignore_errors=True)
    stream_dir.mkdir(parents=True, exist_ok=True)
    out = resolve(cfg["results_dir"]) / "stream"
    out.mkdir(parents=True, exist_ok=True)
    spark = get_spark("zone-demand-stream", packages=KAFKA_PACKAGE,
                      spark__sql__shuffle__partitions=os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
    query = build_query(spark, cfg["kafka"]["bootstrap_servers"], topics["trips"], topics["zone_demand"],
                        cfg["stream"]["window"], cfg["stream"]["watermark"], str(parquet), str(checkpoint))
    watch(query, out / "spark_progress.csv", a.idle_stop)
    spark.stop()


if __name__ == "__main__":
    main()
