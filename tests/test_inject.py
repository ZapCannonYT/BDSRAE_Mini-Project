import numpy as np
import pytest

from taxidrift.drift.inject import DriftSpec, inject, multiplier


def _demand():
    return np.random.default_rng(0).poisson(100, size=(500, 4)).astype(float)


def test_nothing_changes_before_start_or_outside_mask():
    Y = _demand()
    mask = np.array([True, False, False, False])
    out = inject(Y, DriftSpec("abrupt", start=200, magnitude=0.5), mask, seed=1)
    assert np.array_equal(out[:200], Y[:200])
    assert np.array_equal(out[:, 1:], Y[:, 1:])
    assert out[200:, 0].mean() == pytest.approx(Y[200:, 0].mean() * 0.5, rel=0.05)


def test_gradual_ramps_then_holds():
    m = multiplier(DriftSpec("gradual", 0, magnitude=2.0, duration_hours=100), np.array([0, 50, 100, 500]))
    assert m.tolist() == [1.0, 1.5, 2.0, 2.0]


def test_recurring_alternates():
    m = multiplier(DriftSpec("recurring", 0, magnitude=2.0, duration_hours=10), np.array([0, 9, 10, 20]))
    assert m.tolist() == [2.0, 2.0, 1.0, 2.0]


def test_profile_shift_moves_series_later():
    Y = np.zeros((48, 1))
    Y[30, 0] = 1000.0
    out = inject(Y, DriftSpec("profile_shift", start=24, magnitude=3), seed=0)
    assert out[33, 0] > 800 and out[30, 0] == 0


def test_same_seed_is_reproducible():
    Y = _demand()
    spec = DriftSpec("abrupt", 100)
    assert np.array_equal(inject(Y, spec, seed=7), inject(Y, spec, seed=7))
