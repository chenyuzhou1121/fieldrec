"""Convert prospective transition evidence to phase-dependent additive score responses."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import ResponseConfig


def centered_tie_neutral_percentile(values: Sequence[float]) -> np.ndarray:
    """Map a finite nonempty slate to [-1, 1] using ascending average tie ranks.

    A singleton or an all-equal slate maps to zero. Larger values receive larger
    percentiles, and tied candidates receive the same value.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must describe one one-dimensional slate.")
    if array.size == 0:
        raise ValueError("Cannot rank an empty candidate slate.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Mobility potentials must be finite.")
    if array.size == 1:
        return np.zeros(1, dtype=np.float64)
    (_, inverse, counts) = np.unique(array, return_inverse=True, return_counts=True)
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )
    # Midranks keep tied values neutral without imposing an arbitrary ordering.
    midranks = starts.astype(np.float64) + (counts.astype(np.float64) - 1.0) / 2.0
    rho = midranks[inverse] / float(array.size - 1)
    return 2.0 * rho - 1.0


def apply_response(
    base_scores: np.ndarray,
    potential: np.ndarray,
    gate: float,
    phase: str,
    config: ResponseConfig = ResponseConfig(),
) -> np.ndarray:
    """Return adjusted scores for one finite nonempty candidate slate.

    gate is binary; phase selects kappa_fast or kappa_slow case-insensitively.
    With an active response, potential must match base_scores and be finite.
    Add beta * gate * max(kappa * centered_percentile - tau, 0) per candidate.
    A zero gate, zero strength, or unrecognized/disabled phase returns a copy.
    """
    base = np.asarray(base_scores)
    if base.ndim != 1 or not np.isfinite(base).all() or base.size == 0:
        raise ValueError("Backbone scores must be one finite nonempty slate.")
    if not np.isfinite(gate) or gate not in (0.0, 1.0):
        raise ValueError("Support gate must be binary.")
    kappa = {"FAST": config.kappa_fast, "SLOW": config.kappa_slow}.get(
        str(phase).upper(), 0
    )
    if gate == 0 or config.beta == 0 or kappa == 0:
        return base.copy()
    values = np.asarray(potential, dtype=np.float64)
    if values.shape != base.shape:
        raise ValueError("Potential must cover the complete backbone slate.")
    x = centered_tie_neutral_percentile(values)
    delta = config.beta * gate * np.maximum(kappa * x - config.tau, 0.0)
    active = delta > 0.0
    result = base.astype(np.float64) if np.any(active) else base.copy()
    result[active] = base[active].astype(np.float64) + delta[active]
    return result
