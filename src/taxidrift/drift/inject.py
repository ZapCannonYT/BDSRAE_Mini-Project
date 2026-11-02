"""Synthetic drift injection with known change points (ground truth for the benchmark).

Works on the demand matrix Y (hours x zones) from taxidrift.model.features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("abrupt", "gradual", "profile_shift", "recurring")


@dataclass
class DriftSpec:
    kind: str                  # one of KINDS
    start: int                 # true change point (hour index into Y)
    magnitude: float = 1.5     # demand multiplier, or hours for profile_shift
    duration_hours: int = 336  # ramp length for "gradual", half period for "recurring"


def multiplier(spec: DriftSpec, hours_after: np.ndarray) -> np.ndarray:
    """Demand multiplier for hours counted from the change point (0 = first drifted hour)."""
    if spec.kind == "abrupt":
        return np.full(hours_after.shape, spec.magnitude, dtype=float)
    if spec.kind == "gradual":
        return 1 + (spec.magnitude - 1) * np.minimum(hours_after / spec.duration_hours, 1.0)
    if spec.kind == "recurring":
        on = (hours_after // spec.duration_hours) % 2 == 0
        return np.where(on, spec.magnitude, 1.0)
    raise ValueError(f"no multiplier for kind {spec.kind!r}")


def inject(Y: np.ndarray, spec: DriftSpec, zone_mask: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Return a copy of Y with the drift applied from spec.start onwards in the masked zones.

    Multiplier kinds draw counts from Poisson(multiplier * count), which also gives each seed a
    different realisation. profile_shift moves each zone's series later by `magnitude` hours.
    """
    if spec.kind not in KINDS:
        raise ValueError(f"unknown kind {spec.kind!r}")
    zone_mask = np.ones(Y.shape[1], dtype=bool) if zone_mask is None else zone_mask
    rng = np.random.default_rng(seed)
    out = Y.copy()
    idx = np.arange(spec.start, Y.shape[0])
    if spec.kind == "profile_shift":
        lam = Y[idx - int(spec.magnitude)][:, zone_mask]
    else:
        lam = Y[idx][:, zone_mask] * multiplier(spec, idx - spec.start)[:, None]
    out[np.ix_(idx, zone_mask)] = rng.poisson(lam)
    return out
