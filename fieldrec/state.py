"""Replay rating feedback into preference coordinates and signed embedding memories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .config import StateConfig

EPS = 1e-12


def cosine_similarity(vector: np.ndarray, memory: np.ndarray) -> float:
    """Return clipped cosine similarity for aligned vectors, or zero at zero norm."""
    vector = np.asarray(vector, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(memory))
    if denominator <= EPS:
        return 0.0
    return float(np.clip(np.dot(vector, memory) / denominator, -1.0, 1.0))


def rating_to_residual(
    rating: float, rating_baseline: float, residual_scale: float, residual_clip: float
) -> float:
    """Scale rating minus baseline by a positive scale and clip to +/-residual_clip."""
    if not all((np.isfinite(x) for x in (rating, rating_baseline, residual_scale))):
        raise ValueError("rating, rating_baseline, and residual_scale must be finite.")
    if float(residual_scale) <= 0:
        raise ValueError("residual_scale must be positive.")
    return float(
        np.clip(
            (float(rating) - float(rating_baseline)) / float(residual_scale),
            -float(residual_clip),
            float(residual_clip),
        )
    )


@dataclass(frozen=True)
class PreferenceSufficientState:
    """Store cumulative feedback statistics and equal-length embedding memories.

    S sums signed residuals; E sums their absolute values; G sums absolute
    residuals times pre-update alignment. positive and negative sum embeddings
    weighted by positive and absolute negative residuals, respectively.
    """

    S: float
    E: float
    G: float
    positive: np.ndarray
    negative: np.ndarray

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive, dtype=np.float32).copy()
        negative = np.asarray(self.negative, dtype=np.float32).copy()
        if positive.ndim != 1 or negative.ndim != 1 or positive.shape != negative.shape:
            raise ValueError(
                "positive and negative memories must be same-shaped vectors."
            )
        scalar_values = (self.S, self.E, self.G)
        if not all((np.isfinite(float(value)) for value in scalar_values)):
            raise ValueError("Preference sufficient statistics must be finite.")
        if not (np.isfinite(positive).all() and np.isfinite(negative).all()):
            raise ValueError("Semantic memories must be finite.")
        if float(self.E) < 0:
            raise ValueError("State masses cannot be negative.")
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)

    @classmethod
    def empty(cls, dimension: int) -> "PreferenceSufficientState":
        """Return zero statistics and memories for a positive embedding dimension."""
        if int(dimension) <= 0:
            raise ValueError("dimension must be positive.")
        zero = np.zeros(int(dimension), dtype=np.float32)
        return cls(0.0, 0.0, 0.0, zero, zero)


@dataclass(frozen=True)
class FeedbackUpdate:
    """Return updated P/A coordinates and an optional materialized sufficient state."""

    sufficient_state: Optional[PreferenceSufficientState]
    P: float
    A: float

    @property
    def state_vector(self) -> np.ndarray:
        """Return [P, A] as a two-element float64 array."""
        return np.asarray([self.P, self.A], dtype=np.float64)


def simulate_feedback_update(
    state: PreferenceSufficientState,
    *,
    item_embedding: np.ndarray,
    rating_residual: float,
    state_config: StateConfig,
    alignment_pre_update: Optional[float] = None,
    materialize_sufficient_state: bool = True,
) -> FeedbackUpdate:
    """Compute one feedback branch without mutating the input state.

    item_embedding must match the memory dimension; rating_residual is clipped
    by state_config. Alignment is half the positive-minus-negative cosine
    similarity before feedback. A supplied alignment must be in [-1, 1] and is
    required when materialize_sufficient_state is False; that mode returns
    coordinates with sufficient_state=None and skips memory updates.
    """
    item = np.asarray(item_embedding, dtype=np.float32)
    if item.shape != state.positive.shape:
        raise ValueError("item_embedding dimension does not match state memories.")
    if not np.isfinite(float(rating_residual)) or not np.isfinite(item).all():
        raise ValueError("Residual and embedding must be finite.")
    if not materialize_sufficient_state and alignment_pre_update is None:
        raise ValueError(
            "A precomputed alignment is required when the next sufficient state is not materialized."
        )
    if alignment_pre_update is not None and (
        not np.isfinite(float(alignment_pre_update))
        or not -1.0 <= float(alignment_pre_update) <= 1.0
    ):
        raise ValueError("alignment_pre_update must be finite and in [-1, 1].")
    S = float(state.S)
    E = float(state.E)
    G = float(state.G)
    positive: Optional[np.ndarray] = None
    negative: Optional[np.ndarray] = None
    if materialize_sufficient_state or alignment_pre_update is None:
        positive = state.positive.copy()
        negative = state.negative.copy()
    alignment = (
        0.5 * (cosine_similarity(item, positive) - cosine_similarity(item, negative))
        if alignment_pre_update is None
        else float(alignment_pre_update)
    )
    residual = float(
        np.clip(
            float(rating_residual),
            -float(state_config.residual_clip),
            float(state_config.residual_clip),
        )
    )
    mass = abs(residual)
    # Accumulate alignment against the old memories, then update those memories.
    S += residual
    E += mass
    G += mass * alignment
    P = float(np.clip(S / max(E, EPS), -1.0, 1.0)) if E > EPS else 0.0
    A = float(np.clip(G / max(E, EPS), -1.0, 1.0)) if E > EPS else 0.0
    next_state: Optional[PreferenceSufficientState] = None
    if materialize_sufficient_state:
        if positive is None or negative is None:
            raise RuntimeError("Semantic memories were not materialized.")
        memory = item
        if residual > 0:
            positive = positive + np.float32(residual) * memory
        elif residual < 0:
            negative = negative + np.float32(-residual) * memory
        next_state = PreferenceSufficientState(
            S=S,
            E=E,
            G=G,
            positive=positive,
            negative=negative,
        )
    return FeedbackUpdate(
        sufficient_state=next_state,
        P=P,
        A=A,
    )


def state_coordinates(state: PreferenceSufficientState) -> np.ndarray:
    """Return float32 [S/E, G/E] clipped to [-1, 1], or zeros for negligible E."""
    p = np.clip(state.S / max(state.E, EPS), -1.0, 1.0) if state.E > EPS else 0.0
    a = np.clip(state.G / max(state.E, EPS), -1.0, 1.0) if state.E > EPS else 0.0
    return np.asarray([p, a], dtype=np.float32)


def observe(
    state: PreferenceSufficientState,
    item_embedding: np.ndarray,
    rating: float,
    rating_baseline: float,
    residual_scale: float,
    config: StateConfig = StateConfig(),
) -> FeedbackUpdate:
    """Replay an observed rating and return its state and P/A coordinates.

    item_embedding matches the state dimension; rating_baseline and positive
    residual_scale define the normalized residual. Stored memories are rounded
    through float16 and returned as float32 after each observation.
    """
    result = simulate_feedback_update(
        state,
        item_embedding=item_embedding,
        rating_residual=rating_to_residual(
            rating, rating_baseline, residual_scale, config.residual_clip
        ),
        state_config=config,
    )
    updated = result.sufficient_state
    # Preserve the memory precision convention used by observed-state replay.
    updated = replace(
        updated,
        positive=updated.positive.astype(np.float16).astype(np.float32),
        negative=updated.negative.astype(np.float16).astype(np.float32),
    )
    return replace(result, sufficient_state=updated)
