"""Define validated settings and load state, field, and response sections from JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StateConfig:
    """Bound each normalized rating residual by a clip value in (0, 1]."""

    residual_clip: float = 1.0

    def __post_init__(self):
        if not 0 < self.residual_clip <= 1:
            raise ValueError("residual_clip must lie in (0, 1].")


@dataclass(frozen=True)
class ContextualFieldConfig:
    """Configure a square P/A grid, support thresholds, and Gaussian regularizers.

    Transition and user thresholds are separate minimum counts. Context-wide
    thresholds select named contexts; cell thresholds control local support.
    shrinkage_prior_ess sets the parent weight relative to effective sample size.
    max_fallback_cells measures Euclidean distance in grid-cell units.
    """

    grid_size: int = 20
    x_range: tuple[float, float] = (-1.0, 1.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    min_shared_cell_transitions: int = 30
    min_shared_cell_users: int = 5
    min_context_cell_transitions: int = 3
    min_context_cell_users: int = 2
    min_context_transitions: int = 500
    min_context_users: int = 200
    shrinkage_prior_ess: float = 25.0
    covariance_regularization: float = 0.0001
    covariance_shrinkage: float = 0.05
    min_variance: float = 1e-05
    max_fallback_cells: float = 2.0

    def __post_init__(self) -> None:
        if int(self.grid_size) < 4:
            raise ValueError("grid_size must be at least four.")
        if not (
            len(self.x_range) == 2
            and len(self.y_range) == 2
            and (float(self.x_range[0]) < float(self.x_range[1]))
            and (float(self.y_range[0]) < float(self.y_range[1]))
        ):
            raise ValueError("Field coordinate ranges must be increasing pairs.")
        integer_thresholds = (
            self.min_shared_cell_transitions,
            self.min_shared_cell_users,
            self.min_context_cell_transitions,
            self.min_context_cell_users,
            self.min_context_transitions,
            self.min_context_users,
        )
        if any((int(value) < 1 for value in integer_thresholds)):
            raise ValueError("All context-field support thresholds must be positive.")
        if float(self.shrinkage_prior_ess) <= 0:
            raise ValueError("shrinkage_prior_ess must be positive.")
        if float(self.covariance_regularization) <= 0 or float(self.min_variance) <= 0:
            raise ValueError("Covariance regularizers must be positive.")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1].")
        if float(self.max_fallback_cells) < 0:
            raise ValueError("max_fallback_cells must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return all field settings as a dictionary suitable for JSON encoding."""
        return asdict(self)


@dataclass(frozen=True)
class ResponseConfig:
    """Configure phase signs and the additive candidate-score response.

    kappa_fast and kappa_slow select -1, 0, or 1 for each phase. beta scales
    max(kappa * centered_percentile - tau, 0), with tau in [0, 1].
    """

    kappa_fast: int = -1
    kappa_slow: int = 1
    beta: float = 0.5
    tau: float = 0.0

    def __post_init__(self):
        if any(
            value not in (-1, 0, 1)
            for value in (self.kappa_fast, self.kappa_slow)
        ):
            raise ValueError("Phase responses must lie in {-1, 0, 1}.")
        if not 0 <= self.beta < float("inf") or not 0 <= self.tau <= 1:
            raise ValueError("Invalid response strength or threshold.")


def load_config(
    path: str | Path,
) -> tuple[StateConfig, ContextualFieldConfig, ResponseConfig]:
    """Read required state/field/response JSON objects and return their configs.

    The mobility section is read separately by the field-fitting script.
    """
    payload = json.loads(Path(path).read_text())
    return (
        StateConfig(**payload["state"]),
        ContextualFieldConfig(**payload["field"]),
        ResponseConfig(**payload["response"]),
    )
