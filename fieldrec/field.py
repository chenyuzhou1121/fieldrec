"""Fit and query user-balanced Gaussian transition grids with context shrinkage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ContextualFieldConfig
from .density import regularize_covariance

EPS = 1e-12
ALLOWED_SOURCE_SPLITS = frozenset({"train", "development"})


def _normalize_context(value: Any, other_label: str) -> str:
    if value is None:
        return str(other_label)
    if not isinstance(value, str):
        try:
            if bool(pd.isna(value)):
                return str(other_label)
        except (TypeError, ValueError):
            pass
    normalized = str(value).strip()
    return normalized if normalized else str(other_label)


def _user_balanced_weights(
    users: pd.Series, contexts: pd.Series | None = None
) -> np.ndarray:
    """Give each user unit total weight, separately within each supplied context."""
    if contexts is None:
        counts = users.groupby(users, sort=False).transform("count")
    else:
        keys = pd.DataFrame(
            {
                "user": users.astype(str).to_numpy(copy=False),
                "context": contexts.astype(str).to_numpy(copy=False),
            },
            index=users.index,
        )
        counts = keys.groupby(["context", "user"], sort=False)["user"].transform(
            "count"
        )
    return 1.0 / np.maximum(counts.to_numpy(dtype=np.float64), 1.0)


def _weighted_mean_covariance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("values must have shape (n, 2).")
    if weights.shape != (len(values),) or np.any(weights < 0):
        raise ValueError("weights must be a non-negative vector aligned with values.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weighted moments require positive total weight.")
    mean = np.sum(values * weights[:, None], axis=0) / total
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    weight2 = float(np.square(weights).sum())
    # Kish effective sample size determines the strength of context shrinkage.
    ess = total * total / max(weight2, EPS)
    return (mean, covariance, float(ess))


def blend_gaussian_moments(
    child_mean: Sequence[float],
    child_covariance: np.ndarray,
    parent_mean: Sequence[float],
    parent_covariance: np.ndarray,
    child_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return moments of a two-component mixture with child weight in [0, 1].

    Both means have shape (2,) and covariances (2, 2). The covariance includes
    the between-mean contribution as well as the two component covariances.
    """
    alpha = float(child_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("child_weight must be in [0, 1].")
    child_mu = np.asarray(child_mean, dtype=np.float64)
    parent_mu = np.asarray(parent_mean, dtype=np.float64)
    child_cov = np.asarray(child_covariance, dtype=np.float64)
    parent_cov = np.asarray(parent_covariance, dtype=np.float64)
    if child_mu.shape != (2,) or parent_mu.shape != (2,):
        raise ValueError("Gaussian means must be two-dimensional.")
    if child_cov.shape != (2, 2) or parent_cov.shape != (2, 2):
        raise ValueError("Gaussian covariances must be two-dimensional.")
    mean = alpha * child_mu + (1.0 - alpha) * parent_mu
    difference = child_mu - parent_mu
    covariance = (
        alpha * child_cov
        + (1.0 - alpha) * parent_cov
        + alpha * (1.0 - alpha) * np.outer(difference, difference)
    )
    return (mean, covariance)


@dataclass(frozen=True)
class ContextSchema:
    """Name the transition-table context column and the pooled fallback label."""

    column: str
    other_label: str = "__OTHER__"

    def __post_init__(self) -> None:
        if not str(self.column).strip():
            raise ValueError("Context column must be non-empty.")
        if not str(self.other_label).strip():
            raise ValueError("Context other_label must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return the context column and fallback label for JSON encoding."""
        return asdict(self)


@dataclass(frozen=True)
class GaussianDensity2D:
    """Store a two-dimensional transition Gaussian and its fitting support.

    mean has shape (2,) and covariance (2, 2). rows and unique_users count
    observations and users; effective_sample_size summarizes fitting weights.
    shrinkage_weight is the child contribution, and supported marks usability.
    """

    mean: np.ndarray
    covariance: np.ndarray
    rows: int
    unique_users: int
    effective_sample_size: float
    shrinkage_weight: float
    supported: bool = True

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("GaussianDensity2D has invalid dimensions.")
        if bool(self.supported) and (
            not (np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance)))
        ):
            raise ValueError("A supported global density must be finite.")
        if not 0.0 <= float(self.shrinkage_weight) <= 1.0:
            raise ValueError("shrinkage_weight must be in [0, 1].")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "covariance", covariance.copy())

    def to_dict(self) -> dict[str, Any]:
        """Return Gaussian arrays as lists together with fitting support metadata."""
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "rows": int(self.rows),
            "unique_users": int(self.unique_users),
            "effective_sample_size": float(self.effective_sample_size),
            "shrinkage_weight": float(self.shrinkage_weight),
            "supported": bool(self.supported),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GaussianDensity2D":
        """Restore Gaussian moments and support metadata from to_dict output."""
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
            rows=int(payload["rows"]),
            unique_users=int(payload["unique_users"]),
            effective_sample_size=float(payload["effective_sample_size"]),
            shrinkage_weight=float(payload["shrinkage_weight"]),
            supported=bool(payload.get("supported", True)),
        )


@dataclass(frozen=True)
class GaussianGrid2D:
    """Store transition Gaussians indexed by starting P/A grid cells.

    For grid size g, x_edges/y_edges have length g+1, means has shape (g,g,2),
    and covariances (g,g,2,2). Counts, user counts, effective sample sizes,
    shrinkage weights, and the supported mask each have shape (g,g).
    Unsupported cells may contain NaN moments and are excluded from queries.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    counts: np.ndarray
    unique_users: np.ndarray
    effective_sample_sizes: np.ndarray
    shrinkage_weights: np.ndarray
    supported: np.ndarray
    config: ContextualFieldConfig

    def __post_init__(self) -> None:
        x_edges = np.asarray(self.x_edges, dtype=np.float64)
        y_edges = np.asarray(self.y_edges, dtype=np.float64)
        grid = int(self.config.grid_size)
        expected = grid + 1
        if x_edges.shape != (expected,) or y_edges.shape != (expected,):
            raise ValueError("Field edges do not match grid_size.")
        if not (np.all(np.diff(x_edges) > 0) and np.all(np.diff(y_edges) > 0)):
            raise ValueError("Field edges must be strictly increasing.")
        shapes = {
            "means": (grid, grid, 2),
            "covariances": (grid, grid, 2, 2),
            "counts": (grid, grid),
            "unique_users": (grid, grid),
            "effective_sample_sizes": (grid, grid),
            "shrinkage_weights": (grid, grid),
            "supported": (grid, grid),
        }
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} does not match the configured grid.")
        supported = np.asarray(self.supported, dtype=bool)
        if np.any(supported):
            if not np.all(np.isfinite(np.asarray(self.means)[supported])):
                raise ValueError("Supported field means must be finite.")
            if not np.all(np.isfinite(np.asarray(self.covariances)[supported])):
                raise ValueError("Supported field covariances must be finite.")
        object.__setattr__(self, "x_edges", x_edges.copy())
        object.__setattr__(self, "y_edges", y_edges.copy())
        object.__setattr__(
            self, "means", np.asarray(self.means, dtype=np.float64).copy()
        )
        object.__setattr__(
            self, "covariances", np.asarray(self.covariances, dtype=np.float64).copy()
        )
        object.__setattr__(
            self, "counts", np.asarray(self.counts, dtype=np.int64).copy()
        )
        object.__setattr__(
            self, "unique_users", np.asarray(self.unique_users, dtype=np.int64).copy()
        )
        object.__setattr__(
            self,
            "effective_sample_sizes",
            np.asarray(self.effective_sample_sizes, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "shrinkage_weights",
            np.asarray(self.shrinkage_weights, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "supported", supported.copy())

    @staticmethod
    def _bracket(centers: np.ndarray, value: float) -> tuple[int, int, float]:
        if value <= centers[0]:
            return (0, 0, 0.0)
        if value >= centers[-1]:
            end = len(centers) - 1
            return (end, end, 0.0)
        high = int(np.searchsorted(centers, value, side="right"))
        low = high - 1
        fraction = float(
            (value - centers[low]) / max(centers[high] - centers[low], EPS)
        )
        return (low, high, fraction)

    def query(self, state: Sequence[float] | Mapping[str, float]) -> dict[str, Any]:
        """Query [P, A] or a mapping with P/A keys for a local transition density.

        Return status and a reason for invalid/out-of-range or unsupported
        states. Otherwise return mean, covariance, and interpolated support
        metadata. Bilinear weights are renormalized over supported neighbors;
        if none exist, use the nearest supported cell within max_fallback_cells.
        That fallback has status INTERPOLATED; neighbor queries are SUPPORTED.
        """
        if isinstance(state, Mapping):
            point = np.asarray([state["P"], state["A"]], dtype=np.float64)
        else:
            point = np.asarray(state, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            return {"status": "OUT_OF_DOMAIN", "reason": "nonfinite_state"}
        if not (
            self.x_edges[0] <= point[0] <= self.x_edges[-1]
            and self.y_edges[0] <= point[1] <= self.y_edges[-1]
        ):
            return {"status": "OUT_OF_DOMAIN", "reason": "outside_field_range"}
        if not np.any(self.supported):
            return {"status": "LOW_SUPPORT", "reason": "no_supported_context_cells"}
        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        (x0, x1, fx) = self._bracket(x_centers, float(point[0]))
        (y0, y1, fy) = self._bracket(y_centers, float(point[1]))
        candidates = [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x1, y0, fx * (1.0 - fy)),
            (x0, y1, (1.0 - fx) * fy),
            (x1, y1, fx * fy),
        ]
        merged: dict[tuple[int, int], float] = {}
        # At a grid edge, clamped brackets can refer to the same cell repeatedly.
        for x_bin, y_bin, weight in candidates:
            merged[x_bin, y_bin] = merged.get((x_bin, y_bin), 0.0) + float(weight)
        components = [
            (x_bin, y_bin, weight)
            for ((x_bin, y_bin), weight) in merged.items()
            if weight > 0 and bool(self.supported[x_bin, y_bin])
        ]
        interpolation = "bilinear_supported"
        distance_cells = 0.0
        if not components:
            supported_bins = np.argwhere(self.supported)
            x_width = float(self.x_edges[1] - self.x_edges[0])
            y_width = float(self.y_edges[1] - self.y_edges[0])
            x_position = (float(point[0]) - float(self.x_edges[0])) / x_width - 0.5
            y_position = (float(point[1]) - float(self.y_edges[0])) / y_width - 0.5
            distances = np.sqrt(
                np.square(supported_bins[:, 0] - x_position)
                + np.square(supported_bins[:, 1] - y_position)
            )
            nearest = int(np.argmin(distances))
            distance_cells = float(distances[nearest])
            if distance_cells > float(self.config.max_fallback_cells):
                return {
                    "status": "LOW_SUPPORT",
                    "reason": "nearest_context_cell_too_far",
                    "distance_cells": distance_cells,
                }
            (x_bin, y_bin) = supported_bins[nearest]
            components = [(int(x_bin), int(y_bin), 1.0)]
            interpolation = "nearest_context_supported"
        weights = np.asarray([item[2] for item in components], dtype=np.float64)
        weights /= max(float(weights.sum()), EPS)
        means = np.asarray(
            [self.means[item[0], item[1]] for item in components], dtype=np.float64
        )
        mean = np.sum(means * weights[:, None], axis=0)
        covariance = np.zeros((2, 2), dtype=np.float64)
        # Moment matching includes variance between neighboring cell means.
        for weight, component, component_mean in zip(weights, components, means):
            difference = component_mean - mean
            covariance += float(weight) * (
                self.covariances[component[0], component[1]]
                + np.outer(difference, difference)
            )
        covariance = regularize_covariance(
            covariance,
            diagonal_regularization=float(self.config.covariance_regularization),
            shrinkage=float(self.config.covariance_shrinkage),
            min_variance=float(self.config.min_variance),
        )

        def weighted(array: np.ndarray) -> float:
            values = np.asarray(
                [array[item[0], item[1]] for item in components], dtype=np.float64
            )
            return float(np.sum(values * weights))

        return {
            "status": "SUPPORTED"
            if interpolation == "bilinear_supported"
            else "INTERPOLATED",
            "interpolation": interpolation,
            "distance_cells": float(distance_cells),
            "supporting_cells": int(len(components)),
            "mean": mean,
            "covariance": covariance,
            "transition_count": weighted(self.counts),
            "unique_users": weighted(self.unique_users),
            "effective_sample_size": weighted(self.effective_sample_sizes),
            "context_shrinkage_weight": weighted(self.shrinkage_weights),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return grid arrays for JSON encoding; encode missing moments as null.

        Configuration is stored separately by the atlas. from_dict restores
        null entries in floating-point moment arrays as NaN.
        """
        return {
            "x_edges": self.x_edges.tolist(),
            "y_edges": self.y_edges.tolist(),
            "means": np.where(np.isnan(self.means), None, self.means).tolist(),
            "covariances": np.where(
                np.isnan(self.covariances), None, self.covariances
            ).tolist(),
            "counts": self.counts.tolist(),
            "unique_users": self.unique_users.tolist(),
            "effective_sample_sizes": self.effective_sample_sizes.tolist(),
            "shrinkage_weights": self.shrinkage_weights.tolist(),
            "supported": self.supported.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], config: ContextualFieldConfig
    ) -> "GaussianGrid2D":
        """Restore grid arrays using their matching field configuration."""
        return cls(
            x_edges=np.asarray(payload["x_edges"], dtype=np.float64),
            y_edges=np.asarray(payload["y_edges"], dtype=np.float64),
            means=np.asarray(payload["means"], dtype=np.float64),
            covariances=np.asarray(payload["covariances"], dtype=np.float64),
            counts=np.asarray(payload["counts"], dtype=np.int64),
            unique_users=np.asarray(payload["unique_users"], dtype=np.int64),
            effective_sample_sizes=np.asarray(
                payload["effective_sample_sizes"], dtype=np.float64
            ),
            shrinkage_weights=np.asarray(
                payload["shrinkage_weights"], dtype=np.float64
            ),
            supported=np.asarray(payload["supported"], dtype=bool),
            config=config,
        )


@dataclass
class _MomentGrid:
    means: np.ndarray
    covariances: np.ndarray
    counts: np.ndarray
    unique_users: np.ndarray
    effective_sample_sizes: np.ndarray


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Include the exact upper endpoint in the last cell without moving other values.
    adjusted = np.where(values == edges[-1], np.nextafter(edges[-1], edges[0]), values)
    return np.digitize(adjusted, edges) - 1


def _moment_grid(
    data: pd.DataFrame,
    weights: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid_size: int,
) -> _MomentGrid:
    means = np.full((grid_size, grid_size, 2), np.nan, dtype=np.float64)
    covariances = np.full((grid_size, grid_size, 2, 2), np.nan, dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    unique_users = np.zeros((grid_size, grid_size), dtype=np.int64)
    ess = np.zeros((grid_size, grid_size), dtype=np.float64)
    ix = _digitize(data["P"].to_numpy(np.float64), x_edges)
    iy = _digitize(data["A"].to_numpy(np.float64), y_edges)
    deltas = data[["delta_P", "delta_A"]].to_numpy(np.float64)
    users = data["user_id"].astype(str).to_numpy(copy=False)
    for x_bin in range(grid_size):
        for y_bin in range(grid_size):
            selected = (ix == x_bin) & (iy == y_bin)
            count = int(selected.sum())
            counts[x_bin, y_bin] = count
            if count == 0:
                continue
            (mean, covariance, local_ess) = _weighted_mean_covariance(
                deltas[selected], weights[selected]
            )
            means[x_bin, y_bin] = mean
            covariances[x_bin, y_bin] = covariance
            unique_users[x_bin, y_bin] = int(np.unique(users[selected]).size)
            ess[x_bin, y_bin] = local_ess
    return _MomentGrid(means, covariances, counts, unique_users, ess)


def _regularized_grid(
    moments: _MomentGrid,
    supported: np.ndarray,
    config: ContextualFieldConfig,
    *,
    shrinkage_weights: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> GaussianGrid2D:
    covariances = np.full_like(moments.covariances, np.nan)
    for x_bin, y_bin in np.argwhere(supported):
        covariances[x_bin, y_bin] = regularize_covariance(
            moments.covariances[x_bin, y_bin],
            diagonal_regularization=float(config.covariance_regularization),
            shrinkage=float(config.covariance_shrinkage),
            min_variance=float(config.min_variance),
        )
    return GaussianGrid2D(
        x_edges=x_edges,
        y_edges=y_edges,
        means=moments.means,
        covariances=covariances,
        counts=moments.counts,
        unique_users=moments.unique_users,
        effective_sample_sizes=moments.effective_sample_sizes,
        shrinkage_weights=shrinkage_weights,
        supported=supported,
        config=config,
    )


def _finite_field_frame(
    frame: pd.DataFrame, schema: ContextSchema, config: ContextualFieldConfig
) -> pd.DataFrame:
    required = ["user_id", "P", "A", "delta_P", "delta_A", schema.column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Context-field input is missing columns: {missing}")
    data = frame[required].copy()
    data["user_id"] = data["user_id"].astype(str)
    for column in ("P", "A", "delta_P", "delta_A"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[["P", "A", "delta_P", "delta_A"]]).all(axis=1)
    in_range = data["P"].between(*config.x_range, inclusive="both") & data["A"].between(
        *config.y_range, inclusive="both"
    )
    data = data.loc[finite & in_range].reset_index(drop=True)
    if data.empty:
        raise ValueError("No finite in-range transitions are available for the atlas.")
    data["_raw_context"] = [
        _normalize_context(value, schema.other_label)
        for value in data[schema.column].tolist()
    ]
    return data


def _fit_frequent_contexts(
    data: pd.DataFrame, schema: ContextSchema, config: ContextualFieldConfig
) -> tuple[str, ...]:
    summary = data.groupby("_raw_context", sort=True).agg(
        rows=("user_id", "size"), users=("user_id", "nunique")
    )
    frequent = [
        str(context)
        for (context, row) in summary.iterrows()
        if str(context) != schema.other_label
        and int(row["rows"]) >= int(config.min_context_transitions)
        and (int(row["users"]) >= int(config.min_context_users))
    ]
    return tuple(sorted(frequent))


def _fit_global_density(
    data: pd.DataFrame, weights: np.ndarray, config: ContextualFieldConfig
) -> GaussianDensity2D:
    (mean, covariance, ess) = _weighted_mean_covariance(
        data[["delta_P", "delta_A"]].to_numpy(np.float64), weights
    )
    covariance = regularize_covariance(
        covariance,
        diagonal_regularization=float(config.covariance_regularization),
        shrinkage=float(config.covariance_shrinkage),
        min_variance=float(config.min_variance),
    )
    return GaussianDensity2D(
        mean=mean,
        covariance=covariance,
        rows=len(data),
        unique_users=data["user_id"].nunique(),
        effective_sample_size=ess,
        shrinkage_weight=1.0,
    )


@dataclass(frozen=True)
class ContextualFieldAtlas:
    """Store shared and context-specific local and state-independent Gaussians.

    frequent_contexts keep their names; remaining contexts share other_label.
    context_grids and context_globals contain every registered_context. The
    shared models supply parent moments for shrinkage; query_state resolves a
    context and queries its local grid. source_split records train/development.
    """

    schema: ContextSchema
    config: ContextualFieldConfig
    source_split: str
    frequent_contexts: tuple[str, ...]
    registered_contexts: tuple[str, ...]
    shared_grid: GaussianGrid2D
    shared_global: GaussianDensity2D
    context_grids: Mapping[str, GaussianGrid2D]
    context_globals: Mapping[str, GaussianDensity2D]
    fit_rows: int
    fit_users: int

    def __post_init__(self) -> None:
        if str(self.source_split) not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(
                f"source_split must be one of {sorted(ALLOWED_SOURCE_SPLITS)}."
            )
        registered = tuple((str(value) for value in self.registered_contexts))
        if set(registered) != set(self.context_grids) or set(registered) != set(
            self.context_globals
        ):
            raise ValueError("Registered context and density inventories disagree.")
        if not np.any(self.shared_grid.supported):
            raise ValueError("The shared parent field has no supported cells.")
        object.__setattr__(self, "registered_contexts", registered)
        object.__setattr__(
            self,
            "frequent_contexts",
            tuple((str(value) for value in self.frequent_contexts)),
        )

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        schema: ContextSchema,
        source_split: str,
        config: ContextualFieldConfig | None = None,
    ) -> "ContextualFieldAtlas":
        """Fit an atlas from one domain's source-visible transition rows.

        frame requires user_id, P, A, delta_P, delta_A, and schema.column. Drop
        nonfinite transitions and starting coordinates outside config ranges.
        The caller filters rows to source_split; this method records the split.
        Give each user unit weight in the shared model and separately in each
        context. Both shared and context cell thresholds must pass for a local
        context density to be supported. Raise if no shared cell is supported.
        """
        config = config or ContextualFieldConfig()
        if str(source_split) not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(
                f"source_split must be one of {sorted(ALLOWED_SOURCE_SPLITS)}."
            )
        data = _finite_field_frame(frame, schema, config)
        frequent = _fit_frequent_contexts(data, schema, config)
        # Pool infrequent and missing labels before computing context weights.
        has_other = any(value not in frequent for value in data["_raw_context"])
        registered = frequent + ((schema.other_label,) if has_other else ())
        data["_context"] = [
            value if value in frequent else schema.other_label
            for value in data["_raw_context"]
        ]
        grid = int(config.grid_size)
        x_edges = np.linspace(*config.x_range, grid + 1)
        y_edges = np.linspace(*config.y_range, grid + 1)
        shared_weights = _user_balanced_weights(data["user_id"])
        shared_moments = _moment_grid(data, shared_weights, x_edges, y_edges, grid)
        shared_supported = (
            shared_moments.counts >= int(config.min_shared_cell_transitions)
        ) & (shared_moments.unique_users >= int(config.min_shared_cell_users))
        if not np.any(shared_supported):
            raise ValueError("No shared state cell passed the support thresholds.")
        shared_grid = _regularized_grid(
            shared_moments,
            shared_supported,
            config,
            shrinkage_weights=shared_supported.astype(np.float64),
            x_edges=x_edges,
            y_edges=y_edges,
        )
        shared_global = _fit_global_density(data, shared_weights, config)
        context_grids: dict[str, GaussianGrid2D] = {}
        context_globals: dict[str, GaussianDensity2D] = {}
        context_weights_all = _user_balanced_weights(data["user_id"], data["_context"])
        for context in registered:
            selected = data["_context"].astype(str).eq(context).to_numpy()
            context_data = data.loc[selected].reset_index(drop=True)
            context_weights = context_weights_all[selected]
            moments = _moment_grid(
                context_data, context_weights, x_edges, y_edges, grid
            )
            own_support = (
                moments.counts >= int(config.min_context_cell_transitions)
            ) & (moments.unique_users >= int(config.min_context_cell_users))
            supported = own_support & shared_supported
            hierarchical_means = np.full_like(moments.means, np.nan)
            hierarchical_covariances = np.full_like(moments.covariances, np.nan)
            shrinkage_weights = np.zeros((grid, grid), dtype=np.float64)
            for x_bin, y_bin in np.argwhere(supported):
                # More effective observations give the context greater weight.
                alpha = float(
                    moments.effective_sample_sizes[x_bin, y_bin]
                    / (
                        moments.effective_sample_sizes[x_bin, y_bin]
                        + float(config.shrinkage_prior_ess)
                    )
                )
                (mean, covariance) = blend_gaussian_moments(
                    moments.means[x_bin, y_bin],
                    moments.covariances[x_bin, y_bin],
                    shared_moments.means[x_bin, y_bin],
                    shared_moments.covariances[x_bin, y_bin],
                    alpha,
                )
                hierarchical_means[x_bin, y_bin] = mean
                hierarchical_covariances[x_bin, y_bin] = regularize_covariance(
                    covariance,
                    diagonal_regularization=float(config.covariance_regularization),
                    shrinkage=float(config.covariance_shrinkage),
                    min_variance=float(config.min_variance),
                )
                shrinkage_weights[x_bin, y_bin] = alpha
            context_grids[context] = GaussianGrid2D(
                x_edges=x_edges,
                y_edges=y_edges,
                means=hierarchical_means,
                covariances=hierarchical_covariances,
                counts=moments.counts,
                unique_users=moments.unique_users,
                effective_sample_sizes=moments.effective_sample_sizes,
                shrinkage_weights=shrinkage_weights,
                supported=supported,
                config=config,
            )
            (child_mean, child_covariance, child_ess) = _weighted_mean_covariance(
                context_data[["delta_P", "delta_A"]].to_numpy(np.float64),
                context_weights,
            )
            alpha = float(child_ess / (child_ess + float(config.shrinkage_prior_ess)))
            (mean, covariance) = blend_gaussian_moments(
                child_mean,
                child_covariance,
                shared_global.mean,
                shared_global.covariance,
                alpha,
            )
            covariance = regularize_covariance(
                covariance,
                diagonal_regularization=float(config.covariance_regularization),
                shrinkage=float(config.covariance_shrinkage),
                min_variance=float(config.min_variance),
            )
            context_globals[context] = GaussianDensity2D(
                mean=mean,
                covariance=covariance,
                rows=len(context_data),
                unique_users=context_data["user_id"].nunique(),
                effective_sample_size=child_ess,
                shrinkage_weight=alpha,
                supported=True,
            )
        return cls(
            schema=schema,
            config=config,
            source_split=str(source_split),
            frequent_contexts=tuple(frequent),
            registered_contexts=tuple(registered),
            shared_grid=shared_grid,
            shared_global=shared_global,
            context_grids=context_grids,
            context_globals=context_globals,
            fit_rows=int(len(data)),
            fit_users=int(data["user_id"].nunique()),
        )

    def _resolve_context(self, value: Any) -> tuple[str | None, str]:
        raw = _normalize_context(value, self.schema.other_label)
        if raw in set(self.frequent_contexts):
            return (raw, "EXACT")
        if self.schema.other_label in self.context_grids:
            return (self.schema.other_label, "OTHER")
        return (None, "UNKNOWN")

    def query_state(
        self, state: Sequence[float] | Mapping[str, float], context: Any
    ) -> dict[str, Any]:
        """Resolve context and return a local query for [P, A] or a P/A mapping.

        Results include source_split, context_resolution, resolved_context, and
        local_density. Unknown contexts use the fitted other pool when present;
        without it, return UNKNOWN_CONTEXT and a LOW_SUPPORT local density.
        """
        (resolved, resolution) = self._resolve_context(context)
        if resolved is None:
            return {
                "status": "UNKNOWN_CONTEXT",
                "source_split": self.source_split,
                "context_resolution": resolution,
                "resolved_context": None,
                "local_density": {
                    "status": "LOW_SUPPORT",
                    "reason": "unregistered_context_without_other_pool",
                },
            }
        local = self.context_grids[resolved].query(state)
        return {
            "status": local["status"],
            "source_split": self.source_split,
            "context_resolution": resolution,
            "resolved_context": resolved,
            "local_density": local,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned atlas payload with configuration and all fitted models."""
        return {
            "schema_version": 1,
            "density": "hierarchical_context_local_gaussian",
            "null_density": "hierarchical_context_state_independent_gaussian",
            "source_split": self.source_split,
            "context_schema": self.schema.to_dict(),
            "config": self.config.to_dict(),
            "frequent_contexts": list(self.frequent_contexts),
            "registered_contexts": list(self.registered_contexts),
            "fit_rows": int(self.fit_rows),
            "fit_users": int(self.fit_users),
            "shared_grid": self.shared_grid.to_dict(),
            "shared_global": self.shared_global.to_dict(),
            "context_grids": {
                context: self.context_grids[context].to_dict()
                for context in sorted(self.context_grids)
            },
            "context_globals": {
                context: self.context_globals[context].to_dict()
                for context in sorted(self.context_globals)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextualFieldAtlas":
        """Restore an atlas from a schema-version-1 payload returned by to_dict."""
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Contextual field atlas schema_version must be 1.")
        schema = ContextSchema(**dict(payload["context_schema"]))
        raw_config = dict(payload["config"])
        raw_config["x_range"] = tuple(raw_config["x_range"])
        raw_config["y_range"] = tuple(raw_config["y_range"])
        config = ContextualFieldConfig(**raw_config)
        return cls(
            schema=schema,
            config=config,
            source_split=str(payload["source_split"]),
            frequent_contexts=tuple(payload["frequent_contexts"]),
            registered_contexts=tuple(payload["registered_contexts"]),
            shared_grid=GaussianGrid2D.from_dict(payload["shared_grid"], config),
            shared_global=GaussianDensity2D.from_dict(payload["shared_global"]),
            context_grids={
                str(context): GaussianGrid2D.from_dict(grid, config)
                for (context, grid) in dict(payload["context_grids"]).items()
            },
            context_globals={
                str(context): GaussianDensity2D.from_dict(density)
                for (context, density) in dict(payload["context_globals"]).items()
            },
            fit_rows=int(payload["fit_rows"]),
            fit_users=int(payload["fit_users"]),
        )
