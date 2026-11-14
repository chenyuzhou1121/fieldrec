"""Label history-only mobility using a user-balanced threshold on prior state changes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .field import _user_balanced_weights

EPS = 1e-12
ALLOWED_SOURCE_SPLITS = frozenset({"train", "development"})


def causal_history_mobility_scores(
    frame: pd.DataFrame,
    *,
    window: int = 5,
    min_prior: int = 2,
) -> pd.DataFrame:
    """Return mobility_score and mobility_prior_count aligned to frame.index.

    Required columns are user_id, current_event_index, delta_P, and delta_A.
    Within each user, sort by event index and then input order. Average the last
    window finite delta norms strictly before each row. Until min_prior norms
    exist, return NaN and a zero count. Nonfinite deltas do not enter history.
    """
    if int(window) < 1 or int(min_prior) < 1 or int(min_prior) > int(window):
        raise ValueError("Mobility requires 1 <= min_prior <= window.")
    user_column, order_column = "user_id", "current_event_index"
    delta_columns = ("delta_P", "delta_A")
    required = [user_column, order_column, *delta_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Mobility input is missing columns: {missing}")
    work = frame[required].copy()
    work["_original_position"] = np.arange(len(work), dtype=np.int64)
    work[order_column] = pd.to_numeric(work[order_column], errors="raise")
    for column in delta_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.sort_values(
        [user_column, order_column, "_original_position"], kind="mergesort"
    )
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    prior_counts = np.zeros(len(frame), dtype=np.int64)
    for _, group in work.groupby(user_column, sort=False):
        history: list[float] = []
        positions = group["_original_position"].to_numpy(np.int64)
        dx_values = group[delta_columns[0]].to_numpy(np.float64)
        dy_values = group[delta_columns[1]].to_numpy(np.float64)
        for position, dx, dy in zip(positions, dx_values, dy_values):
            # Label the starting state before appending this transition's change.
            if len(history) >= int(min_prior):
                scores[position] = float(np.mean(history[-int(window) :]))
                prior_counts[position] = int(min(len(history), int(window)))
            if np.isfinite(dx) and np.isfinite(dy):
                history.append(float(np.hypot(dx, dy)))
                if len(history) > int(window):
                    history = history[-int(window) :]
    return pd.DataFrame(
        {"mobility_score": scores, "mobility_prior_count": prior_counts},
        index=frame.index,
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    (values, weights) = (values[finite], weights[finite])
    if values.size == 0:
        raise ValueError("No finite mobility scores are available to fit a threshold.")
    order = np.argsort(values, kind="mergesort")
    (values, weights) = (values[order], weights[order])
    # Place each observation at the midpoint of its cumulative weight interval.
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= max(float(weights.sum()), EPS)
    return float(np.interp(float(quantile), cumulative, values))


@dataclass(frozen=True)
class HistoryOnlyMobilityLabeler:
    """Store a fitted mobility threshold, source split, and prior-history settings.

    Scores at or below threshold are SLOW; larger finite scores are FAST.
    source_split is train or development, and 1 <= min_prior <= window.
    """

    threshold: float
    source_split: str
    window: int = 5
    min_prior: int = 2
    threshold_quantile: float = 0.5

    def __post_init__(self) -> None:
        if str(self.source_split) not in ALLOWED_SOURCE_SPLITS:
            raise ValueError("Mobility labeler source_split must be train/development.")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("Mobility threshold must be finite.")
        if not 1 <= int(self.min_prior) <= int(self.window):
            raise ValueError("Mobility labeler requires 1 <= min_prior <= window.")
        if not 0.0 < float(self.threshold_quantile) < 1.0:
            raise ValueError("Mobility threshold_quantile must lie in (0, 1).")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HistoryOnlyMobilityLabeler":
        """Restore a labeler from the fields returned by to_dict."""
        return cls(**payload)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        source_split: str,
        window: int = 5,
        min_prior: int = 2,
        threshold_quantile: float = 0.5,
    ) -> "HistoryOnlyMobilityLabeler":
        """Fit a user-balanced quantile of causal mobility scores for one domain.

        frame has the mobility-score columns plus target_split. Threshold
        fitting selects train rows, or train and validation for development.
        Histories are computed from the supplied frame before that selection;
        callers supply chronological transitions appropriate to the source.
        """
        if str(source_split) not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(
                f"source_split must be one of {sorted(ALLOWED_SOURCE_SPLITS)}."
            )
        if not 0.0 < float(threshold_quantile) < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1).")
        if "target_split" not in frame.columns:
            raise ValueError("Mobility input lacks target_split.")
        scores = causal_history_mobility_scores(
            frame,
            window=window,
            min_prior=min_prior,
        )
        allowed = {"train"} if source_split == "train" else {"train", "validation"}
        fit_mask = frame["target_split"].astype(str).isin(allowed).to_numpy()
        finite = fit_mask & np.isfinite(scores["mobility_score"].to_numpy(np.float64))
        if not np.any(finite):
            raise ValueError(
                "No source-visible mobility histories can fit the labeler."
            )
        source_users = frame.loc[finite, "user_id"].astype(str)
        user_weights = _user_balanced_weights(source_users)
        threshold = _weighted_quantile(
            scores.loc[finite, "mobility_score"].to_numpy(np.float64),
            user_weights,
            float(threshold_quantile),
        )
        return cls(
            threshold=threshold,
            source_split=str(source_split),
            window=int(window),
            min_prior=int(min_prior),
            threshold_quantile=float(threshold_quantile),
        )

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        """Return history_mobility_mode labels aligned to a transition frame's index."""
        scores = causal_history_mobility_scores(
            frame,
            window=self.window,
            min_prior=self.min_prior,
        )["mobility_score"].to_numpy(np.float64)
        return pd.Series(
            self.label_scores(scores),
            index=frame.index,
            name="history_mobility_mode",
        )

    def label_scores(self, scores: np.ndarray) -> np.ndarray:
        """Map scores to same-shaped SLOW/FAST labels; nonfinite scores become __OTHER__."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.full(scores.shape, "__OTHER__", dtype="U9")
        finite = np.isfinite(scores)
        labels[finite & (scores <= self.threshold)] = "SLOW"
        labels[finite & (scores > self.threshold)] = "FAST"
        return labels

    def to_dict(self) -> dict[str, Any]:
        """Return threshold, source split, and history settings for JSON encoding."""
        return asdict(self)
