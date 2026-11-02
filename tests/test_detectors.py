"""Behaviour every detector must have. These skip until the detector is implemented.

Owner: Rahul Prasanna. Add river cross-checks in test_against_river.py.
"""

import numpy as np
import pytest

from taxidrift.drift.adwin import ADWIN
from taxidrift.drift.kswin import KSWIN
from taxidrift.drift.page_hinkley import PageHinkley
from taxidrift.drift.psi import PSIDetector, psi

rng = np.random.default_rng(0)
STABLE = rng.normal(10, 1, 3000)
SHIFTED = np.concatenate([rng.normal(10, 1, 1500), rng.normal(14, 1, 1500)])

DETECTORS = [
    lambda: ADWIN(delta=0.002),
    lambda: PageHinkley(delta=0.005, threshold=50),
    lambda: KSWIN(alpha=0.0001, seed=0),
    lambda: PSIDetector(reference=STABLE[:500], window=200, step=50),
]


@pytest.mark.parametrize("make", DETECTORS)
def test_detects_mean_shift_soon_after_change(make):
    alarms = make().run(SHIFTED)
    after = [a for a in alarms if a >= 1500]
    assert after, "no alarm after the change"
    assert after[0] - 1500 < 300


@pytest.mark.parametrize("make", DETECTORS[:2])
def test_few_false_alarms_on_stable_stream(make):
    assert len(make().run(STABLE)) <= 1


def test_psi_is_zero_for_identical_and_large_for_shifted():
    assert psi(STABLE, STABLE) == pytest.approx(0.0, abs=1e-9)
    assert psi(STABLE[:1500], rng.normal(14, 1, 1500)) > 0.25


def test_reset_clears_state():
    d = ADWIN()
    d.run(SHIFTED)
    d.reset()
    assert d.n_seen == 0 and not d.drift_detected
