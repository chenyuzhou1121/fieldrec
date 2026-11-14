"""Score hypothetical rating transitions against local and context-global densities."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .config import StateConfig
from .density import gaussian_log_prob
from .field import ContextualFieldAtlas
from .state import (
    PreferenceSufficientState,
    cosine_similarity,
    rating_to_residual,
    simulate_feedback_update,
    state_coordinates,
)


def enumerate_transitions(
    state: PreferenceSufficientState,
    embeddings: np.ndarray,
    baselines: np.ndarray,
    residual_scale: float,
    config: StateConfig = StateConfig(),
) -> np.ndarray:
    """Return (candidates, 5, 2) P/A changes for ratings 1 through 5.

    embeddings has shape (candidates, memory_dimension); baselines has one
    rating baseline per candidate. Each branch starts from the same state and
    uses the positive residual_scale without materializing updated memories.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    baselines = np.asarray(baselines, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[1:] != state.positive.shape:
        raise ValueError("Candidate embeddings must match the state dimension.")
    if baselines.shape != (len(embeddings),) or not np.isfinite(embeddings).all():
        raise ValueError("Candidate embeddings and baselines must align.")
    current = state_coordinates(state).astype(np.float64)
    deltas = np.empty((len(embeddings), 5, 2), dtype=np.float64)
    for c, (item, baseline) in enumerate(zip(embeddings, baselines)):
        alignment = 0.5 * (
            cosine_similarity(item, state.positive)
            - cosine_similarity(item, state.negative)
        )
        for branch, rating in enumerate(range(1, 6)):
            update = simulate_feedback_update(
                state,
                item_embedding=item,
                rating_residual=rating_to_residual(
                    rating, baseline, residual_scale, config.residual_clip
                ),
                state_config=config,
                alignment_pre_update=alignment,
                materialize_sufficient_state=False,
            )
            deltas[c, branch] = update.state_vector - current
    return deltas


def score_candidates(
    state: PreferenceSufficientState,
    embeddings: np.ndarray,
    baselines: np.ndarray,
    residual_scale: float,
    atlas: ContextualFieldAtlas,
    phase: str,
    query_split: str,
    config: StateConfig = StateConfig(),
) -> tuple[np.ndarray, float]:
    """Return per-candidate log density-ratio potentials and a binary support gate.

    Candidate arrays and residual_scale follow enumerate_transitions. phase is
    the context queried in atlas. train/validation queries require a train
    atlas; test requires development. Unsupported queries return zeros and gate
    zero. Supported scores sum density over the five possible rating branches
    separately for local and context-global models, then take the log ratio.
    """
    if query_split not in {"train", "validation", "test"}:
        raise ValueError("Unknown query split.")
    source = "development" if query_split == "test" else "train"
    if atlas.source_split != source:
        raise ValueError("Field source does not match the query split.")
    query = atlas.query_state(state_coordinates(state), phase)
    local = query["local_density"]
    resolved = query["resolved_context"]
    null = atlas.context_globals.get(resolved)
    gate = float(
        local["status"] in {"SUPPORTED", "INTERPOLATED"}
        and null is not None
        and null.supported
    )
    if gate == 0.0:
        return np.zeros(len(embeddings), dtype=np.float64), gate
    deltas = enumerate_transitions(state, embeddings, baselines, residual_scale, config)
    local_log = np.asarray(
        [
            [
                gaussian_log_prob(delta, local["mean"], local["covariance"])
                for delta in branches
            ]
            for branches in deltas
        ]
    )
    null_log = np.asarray(
        [
            [gaussian_log_prob(delta, null.mean, null.covariance) for delta in branches]
            for branches in deltas
        ]
    )
    # Equal branch weights cancel in the ratio; sum in log space for stability.
    potential = logsumexp(local_log, axis=1) - logsumexp(null_log, axis=1)
    if not np.isfinite(potential).all():
        raise ValueError("Non-finite prospective evidence.")
    return potential, gate
