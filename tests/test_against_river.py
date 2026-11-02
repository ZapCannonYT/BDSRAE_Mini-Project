"""Cross-check our detectors against the river library on shared streams."""

import numpy as np
import pytest

river = pytest.importorskip("river")
from river import drift as rdrift  # noqa: E402

from taxidrift.drift.adwin import ADWIN  # noqa: E402
from taxidrift.drift.page_hinkley import PageHinkley  # noqa: E402

rng = np.random.default_rng(1)
STREAMS = {
    "stable": rng.normal(1, 0.2, 4000),
    "abrupt": np.concatenate([rng.normal(1, 0.2, 2000), rng.normal(1.3, 0.2, 2000)]),
    "two_changes": np.concatenate([rng.normal(1, 0.2, 1500), rng.normal(1.5, 0.2, 1500), rng.normal(1, 0.2, 1500)]),
}


def _river_alarms(det, stream):
    alarms = []
    for i, x in enumerate(stream):
        det.update(float(x))
        if det.drift_detected:
            alarms.append(i)
    return alarms


CHANGES = {"stable": [], "abrupt": [2000], "two_changes": [1500, 3000]}


@pytest.mark.parametrize("name", STREAMS)
def test_adwin_detects_the_same_changes_as_river(name):
    # Known difference: our alarms can come earlier than river's and can repeat for one change,
    # so this compares which changes were found and how soon, not the exact alarm lists.
    ours = ADWIN(delta=0.002, clock=32, max_buckets=5, min_window=5, grace_period=10).run(STREAMS[name])
    theirs = _river_alarms(rdrift.ADWIN(delta=0.002, clock=32, max_buckets=5, min_window_length=5,
                                        grace_period=10), STREAMS[name])
    if not CHANGES[name]:
        assert ours == [] and theirs == []
    for cp in CHANGES[name]:
        first_ours = next(a for a in ours if a >= cp)
        first_theirs = next(a for a in theirs if a >= cp)
        assert first_ours - cp < 300 and first_theirs - cp < 300
        assert abs(first_ours - first_theirs) <= 128


@pytest.mark.parametrize("name", ["abrupt", "two_changes"])
def test_page_hinkley_alarms_close_to_river(name):
    ours = PageHinkley(delta=0.005, threshold=10, min_instances=30).run(STREAMS[name])
    theirs = _river_alarms(rdrift.PageHinkley(delta=0.005, threshold=10, min_instances=30, alpha=1.0, mode="both"),
                           STREAMS[name])
    assert ours and theirs
    assert abs(ours[0] - theirs[0]) <= 5
