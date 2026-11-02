# 3-Minute Demo Video: Script and Recording Steps

## Before recording (2 minutes)

Open **three** Ubuntu (WSL) terminals side by side and run this in each:

```bash
cd /mnt/c/Users/Zap/Uni/SEM5/BD/MiniProject
export TAXIDRIFT_CONFIG=$PWD/configs/wsl.yaml
PY=~/taxidrift-venv/bin/python
```

Also open these, ready to switch to:
- `README.md` architecture diagram (preview on GitHub or in VS Code)
- `notebooks/results.ipynb` in VS Code, scrolled to section 3
- `docs/figures/` images

Reset Kafka once, in terminal 1:

```bash
bash scripts/pipeline.sh down; bash scripts/pipeline.sh up
```

## Live part: start in this order

Terminal 1, Spark. Wait until it stops printing startup lines (about 20 seconds):

```bash
$PY -m taxidrift.stream.aggregator --fresh --idle-stop 120 2>&1 | grep -E "^batch"
```

Terminal 2, the monitor. Wait for `monitor up`:

```bash
rm -f results/stream/audit_log.jsonl; $PY -m taxidrift.monitor.service --policy combined --start 2020-03-01 --end 2020-05-01 --idle-seconds 180
```

Terminal 3, start recording the screen, then run the producer:

```bash
$PY -m taxidrift.stream.producer --start 2020-03 --end 2020-04 --speedup 14400
```

4 stream hours pass every second, so:
- about 45 seconds in: PSI alarms (around 7-8 March)
- about 70 seconds in: ADWIN alarms and retrain decisions (around 12 March onward)

Record 2 minutes of this, then stop recording. You only use about 50 seconds in the final cut. To see the audit log afterwards:

```bash
tail -5 results/stream/audit_log.jsonl
```

## Script (about 400 words, 3:00 at a normal pace)

Split it between the three of you; each section names a speaker as a suggestion.

### 0:00-0:20 Problem. Show the title, then the README diagram. (Rahul Saran Sharma)
"We're team Rahul Saran Sharma, Rahul Prasanna and Raghav U. A demand model trained on normal travel breaks when the city changes. We use public NYC taxi data from 2019 to 2021, where COVID gives us real drift. We detect that drift on a live stream, retrain automatically, and check which boroughs get hurt."

### 0:20-0:50 Big data. Show the diagram, then the three terminals. (Rahul Saran Sharma)
"138.7 million trips go in. A producer replays them in time order into Kafka, and Spark Structured Streaming counts pickups per zone per hour. The offline Spark job builds 6.9 million zone-hour rows in 28 seconds. In the hours we spot-checked, the monitor's live counts match that batch job exactly. Here's the stream running at four hours of city time per second."

### 0:50-1:30 Drift, live. Show the monitor terminal with alarms appearing. (Rahul Prasanna)
"This monitor predicts every hour and feeds its errors to the detectors. We're replaying March 2020. Watch the alarms: PSI fires on demand around March 8, and ADWIN fires on model error on March 12, three days before the crash we expected on March 15. Staten Island doesn't fire until March 22, about nine days later."

### 1:30-2:05 Detector comparison. Show the detector chart. (Rahul Prasanna)
"We also injected drift with a known start time, across 8 seeds. Every error-based detector caught city-wide drift. ADWIN took a median of 26.5 hours with 4.4 false alarms per thousand hours. Page-Hinkley at its strict setting took 17 hours with 2.3. Its loose setting took 2 hours, but with 11 false alarms. PSI's numbers aren't comparable: its summer reference made it fire on normal seasonal change before our drift even started."

### 2:05-2:35 Responsible AI. Show the borough charts. (Raghav U)
"Is drift fair? During the crash, error doubled in Manhattan and more than doubled in Queens. A single city-wide detector caught drift confined to Brooklyn only 25 percent of the time; Brooklyn's own detector caught it every time. So we monitor per borough. Every alarm and retrain is also written to an audit log, and only zone-level counts are used, never individual trips."

### 2:35-3:00 Retraining policy. Show the policy chart and audit log. (Raghav U)
"Retraining surprised us. With a one-hour-ahead model, never retraining scored 28.9 percent error during the crash, and monthly retraining was worse at 36.3. The best policy, ADWIN on any borough, barely beat it at 28.6, with 58 retrains. Retraining also made Staten Island far worse, because our promotion check only looks at city-wide error. So our policy needs a per-borough check before promoting a model. Thanks for watching; the code and report are on GitHub."

## Editing and upload

1. Record with OBS, or Win+G for the Xbox Game Bar. Record the narration over the clips, or live.
2. Trim to under 3:00.
3. Upload to YouTube as unlisted, and put the link in the README.

## Numbers to check before you say them

Every number above comes from `results/`. The March alarm dates in the live section are from the fixed-model replay, so check them against what appears on screen in your recording, and say what you actually see if they differ.
