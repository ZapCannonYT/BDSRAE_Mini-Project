import pytest

from taxidrift.eval.metrics import mae, mean_time_between, score_detections, smape


def test_delay_and_false_alarms():
    # change at 100 and 300; alarms: false (50), hit at 110, repeat for the same drift (120),
    # false because too late (200), hit at 305
    r = score_detections([50, 110, 120, 200, 305], [100, 300], n=1000, max_delay=50)
    assert r["delays"] == [10, 5]
    assert r["mean_delay"] == 7.5
    assert r["missed"] == 0
    assert r["n_false_alarms"] == 2
    assert r["false_alarms_per_1000"] == 2.0


def test_missed_when_alarm_too_late():
    # 180 is past cp 100 + max_delay; 250 is exactly at the end of cp 200's window (exclusive)
    r = score_detections([180, 250], [100, 200], n=400, max_delay=50)
    assert r["delays"] == [None, None]
    assert r["missed"] == 2
    assert r["miss_rate"] == 1.0
    assert r["n_false_alarms"] == 2


def test_alarm_after_next_change_belongs_to_next_change():
    r = score_detections([210], [100, 200], n=400, max_delay=500)
    assert r["delays"] == [None, 10]


def test_no_change_points_all_alarms_false():
    r = score_detections([10, 20, 40], [], n=100, max_delay=10)
    assert r["n_false_alarms"] == 3
    assert r["miss_rate"] is None
    assert r["mtfa"] == 15.0


def test_mean_time_between_needs_two_events():
    assert mean_time_between([5]) is None


def test_mae_and_smape():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
    assert smape([0, 10], [0, 10]) == 0.0
    assert smape([10], [0]) == pytest.approx(200.0)
