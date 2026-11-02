import pandas as pd

from taxidrift.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from taxidrift.monitor.audit import AuditLog
from taxidrift.retrain.policy import NeverRetrain, RetrainPolicy


def test_default_config_loads():
    cfg = load_config()
    assert cfg["kafka"]["topics"]["trips"] == "trips"
    assert cfg["drift"]["psi"]["drift"] == 0.25


def test_docker_config_extends_default():
    cfg = load_config(PROJECT_ROOT / "configs" / "docker.yaml")
    assert cfg["kafka"]["bootstrap_servers"] == "kafka:29092"
    assert cfg["drift"] == load_config(DEFAULT_CONFIG)["drift"]


def test_audit_log_roundtrip(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.write("drift_alarm", "2020-03-15T08:00:00", detector="adwin", scope="city")
    log.write("retrain_promoted", "2020-03-16T00:00:00", reason="adwin")
    records = log.read()
    assert [r["event"] for r in records] == ["drift_alarm", "retrain_promoted"]
    assert records[0]["detector"] == "adwin"


class _Always(RetrainPolicy):
    def _decide(self, now, alarms):
        return True


def test_cooldown_blocks_back_to_back_retrains():
    p = _Always(cooldown_hours=72)
    t0 = pd.Timestamp("2020-03-15")
    assert p.should_retrain(t0, {})
    assert not p.should_retrain(t0 + pd.Timedelta(hours=10), {})
    assert p.should_retrain(t0 + pd.Timedelta(hours=72), {})


def test_never_retrain():
    assert not NeverRetrain().should_retrain(pd.Timestamp("2020-01-01"), {"adwin": True})
