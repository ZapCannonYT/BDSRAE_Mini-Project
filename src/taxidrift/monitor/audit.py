"""Append-only audit log (JSON Lines) of every drift alarm and retraining decision.

This is the governance record for the Responsible AI part: each line says what happened,
when (stream time), why, and with which numbers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, stream_time: str, **details) -> dict:
        record = {
            "event": event,                 # "drift_alarm", "retrain_promoted", "retrain_rejected", ...
            "stream_time": stream_time,
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **details,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
