#!/usr/bin/env bash
# Native pipeline runner (Linux or WSL, no Docker needed).
#   scripts/pipeline.sh setup     install the Python environment and Kafka into ~/taxidrift
#   scripts/pipeline.sh data      copy or download the raw data, build zone-hour with Spark, train the champion
#   scripts/pipeline.sh up        start Kafka and create topics
#   scripts/pipeline.sh demo      replay Mar-Apr 2020 through Kafka, Spark and the monitor
#   scripts/pipeline.sh down      stop Kafka and clear its data
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$HOME/taxidrift"
KAFKA_VERSION=4.1.2
KAFKA="$HOME_DIR/kafka_2.13-$KAFKA_VERSION"
VENV="$HOME/taxidrift-venv"
PY="$VENV/bin/python"
BOOTSTRAP=localhost:9092
export TAXIDRIFT_CONFIG="$ROOT/configs/wsl.yaml"
LOGS="$HOME_DIR/logs"
mkdir -p "$LOGS"

setup() {
  [ -x "$PY" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -e "$ROOT[dev]" nbconvert ipykernel reportlab
  if [ ! -d "$KAFKA" ]; then
    cd "$HOME_DIR"
    wget -q "https://dlcdn.apache.org/kafka/$KAFKA_VERSION/kafka_2.13-$KAFKA_VERSION.tgz"
    tar xzf "kafka_2.13-$KAFKA_VERSION.tgz" && rm "kafka_2.13-$KAFKA_VERSION.tgz"
  fi
}

data() {
  "$PY" "$ROOT/scripts/download_data.py"
  "$PY" -m taxidrift.data.reference
  "$PY" -m taxidrift.model.train
}

up() {
  local data_dir="$HOME_DIR/kafka-data"
  local conf="$HOME_DIR/kraft.properties"
  cp "$KAFKA/config/server.properties" "$conf"
  sed -i "s#^log.dirs=.*#log.dirs=$data_dir#; s#^num.partitions=.*#num.partitions=${KAFKA_PARTITIONS:-4}#" "$conf"
  if [ ! -f "$data_dir/meta.properties" ]; then
    "$KAFKA/bin/kafka-storage.sh" format -t "$("$KAFKA/bin/kafka-storage.sh" random-uuid)" -c "$conf" --standalone >/dev/null
  fi
  nohup "$KAFKA/bin/kafka-server-start.sh" "$conf" >"$LOGS/kafka.log" 2>&1 &
  echo $! >"$HOME_DIR/kafka.pid"
  for _ in $(seq 1 30); do
    "$KAFKA/bin/kafka-topics.sh" --bootstrap-server $BOOTSTRAP --list >/dev/null 2>&1 && break
    sleep 2
  done
  "$KAFKA/bin/kafka-topics.sh" --bootstrap-server $BOOTSTRAP --create --if-not-exists --topic trips     --partitions "${KAFKA_PARTITIONS:-4}" --replication-factor 1
  # Ordered: the monitor uses the order of this topic to know when an hour is complete.
  for topic in zone_demand drift_events; do
    "$KAFKA/bin/kafka-topics.sh" --bootstrap-server $BOOTSTRAP --create --if-not-exists --topic "$topic"       --partitions 1 --replication-factor 1
  done
}

down() {
  [ -f "$HOME_DIR/kafka.pid" ] && kill "$(cat "$HOME_DIR/kafka.pid")" 2>/dev/null || true
  sleep 5
  rm -rf "$HOME_DIR/kafka-data" "$HOME_DIR/kafka.pid"
}

demo() {
  cd "$ROOT"
  nohup "$PY" -m taxidrift.stream.aggregator --fresh --idle-stop 120 >"$LOGS/aggregator.log" 2>&1 &
  local agg=$!
  sleep 20
  nohup "$PY" -m taxidrift.monitor.service --policy "${POLICY:-combined}" --start 2020-03-01 --end 2020-05-01 \
    --idle-seconds 180 >"$LOGS/monitor.log" 2>&1 &
  local mon=$!
  "$PY" -m taxidrift.stream.producer --start 2020-03 --end 2020-04 --speedup "${SPEEDUP:-14400}"
  wait "$mon" || true
  wait "$agg" || true
  tail -5 "$LOGS/monitor.log"
}

"${1:-demo}"
