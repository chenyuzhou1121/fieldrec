"""Evaluate target rankings and aggregate paired user differences across domains."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_ranks(scores: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    """Return one-based average tie ranks for zero-based targets in finite (Q, C) scores.

    Larger scores rank first. target_positions is an integer vector of length Q.
    """
    values = np.asarray(scores)
    positions = np.asarray(target_positions)
    if (
        values.ndim != 2
        or positions.shape != (len(values),)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Scores and targets must describe aligned finite slates.")
    if positions.dtype.kind not in "iu" or np.any(
        (positions < 0) | (positions >= values.shape[1])
    ):
        raise ValueError("Invalid target position.")
    target = values[np.arange(len(values)), positions]
    greater = np.count_nonzero(values > target[:, None], axis=1)
    tied = np.count_nonzero(values == target[:, None], axis=1) - 1
    return 1.0 + greater + 0.5 * tied


def metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-query MRR plus HR and NDCG at 5, 10, and 20 for one target per query."""
    ranks = np.asarray(ranks, dtype=np.float64)
    result = {"mrr": 1.0 / ranks}
    for k in (5, 10, 20):
        result[f"hr_{k}"] = (ranks <= k).astype(np.float64)
        result[f"ndcg_{k}"] = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return result


def evaluate(
    metadata: pd.DataFrame,
    base_scores: np.ndarray,
    fieldrec_scores: np.ndarray,
    target_positions: np.ndarray,
) -> pd.DataFrame:
    """Return per-query metrics for aligned base and adjusted score matrices.

    metadata supplies domain and user_id for each row. target_positions gives
    the zero-based target slot in both score matrices; output columns prefix
    each metric with base_ or fieldrec_.
    """
    if len(metadata) != len(base_scores) or np.shape(base_scores) != np.shape(
        fieldrec_scores
    ):
        raise ValueError("Metadata and score matrices must align.")
    result = metadata[["domain", "user_id"]].reset_index(drop=True).copy()
    for arm, scores in (("base", base_scores), ("fieldrec", fieldrec_scores)):
        for name, values in metrics(target_ranks(scores, target_positions)).items():
            result[f"{arm}_{name}"] = values
    return result


def aggregate(
    per_query: pd.DataFrame, bootstrap_reps: int = 10000, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Return domain metrics and an equally weighted macro summary from evaluate.

    Average queries within users, then users within domains. Bootstrap paired
    user-level NDCG@5 differences independently within each domain using seed.
    Report percentile 95% intervals and two-sided bootstrap sign proportions;
    relative gains are computed within each domain before macro averaging.
    """
    if bootstrap_reps < 1 or per_query.empty:
        raise ValueError("Bootstrap requires positive replicates and eligible queries.")
    users = per_query.groupby(["domain", "user_id"], sort=True).mean(numeric_only=True)
    domains = users.groupby(level="domain").mean()
    for name in ("mrr", "ndcg_5", "ndcg_10", "ndcg_20", "hr_5", "hr_10", "hr_20"):
        base, field = domains[f"base_{name}"], domains[f"fieldrec_{name}"]
        domains[f"gain_{name}"] = field - base
        domains[f"relative_gain_{name}"] = np.where(
            base > 0, field / base - 1.0, np.nan
        )
    macro = domains.mean().to_dict()
    rng = np.random.default_rng(seed)
    macro_bootstrap = np.zeros(bootstrap_reps)
    for domain, group in users.groupby(level="domain", sort=True):
        differences = (group["fieldrec_ndcg_5"] - group["base_ndcg_5"]).to_numpy()
        values, counts = np.unique(differences, return_counts=True)
        draws = np.empty(bootstrap_reps)
        # Multinomial frequencies resample users while grouping identical gains.
        for start in range(0, bootstrap_reps, 256):
            size = min(256, bootstrap_reps - start)
            frequencies = rng.multinomial(len(differences), counts / counts.sum(), size)
            draws[start : start + size] = frequencies @ values / len(differences)
        domains.loc[domain, "users"] = len(group)
        domains.loc[domain, "ci95_low"], domains.loc[domain, "ci95_high"] = np.quantile(
            draws, [0.025, 0.975]
        )
        domains.loc[domain, "p_two_sided"] = min(
            1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0))
        )
        macro_bootstrap += draws / len(domains)
    macro["ci95_low"], macro["ci95_high"] = np.quantile(macro_bootstrap, [0.025, 0.975])
    macro["p_two_sided"] = min(
        1.0, 2.0 * min(np.mean(macro_bootstrap <= 0), np.mean(macro_bootstrap >= 0))
    )
    macro["bootstrap_reps"] = bootstrap_reps
    macro["bootstrap_seed"] = seed
    macro["p_resolution"] = 2.0 / bootstrap_reps
    return domains, macro
