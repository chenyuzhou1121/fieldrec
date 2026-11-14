"""Report user-balanced ranking metrics and paired bootstrap summaries from scored NPZ."""

import argparse
import json
from pathlib import Path
import sys


def main():
    """Read scored CLI inputs, filter history counts, and print domain CSV and macro JSON."""
    parser = argparse.ArgumentParser(
        description="Compare frozen and FieldRec rankings with user-balanced domain averages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Read one or more NPZ files produced by score_candidates.py. Keep the\n"
            "evaluation population separate from users used to choose response settings.\n"
            "Use one target, 10 retriever-mined hard negatives and 89 uniform negatives per slate.\n"
            "target_position is zero-based; --min-history filters observed history counts.\n"
            "Each domain/query key must occur once across all files.\n"
            "Score ties use average ranks. Metrics are MRR plus HR and NDCG at 5, 10 and 20;\n"
            "average queries within users, then users within domains, then domains equally.\n"
            "Relative gains are averaged across domains after division within each domain.\n"
            "NDCG@5 intervals resample paired user differences separately within domains.\n"
            "Print a per-domain NDCG@5 CSV table followed by a macro JSON summary.\n\n"
            "Example from the repository root (Python 3.10 or later):\n"
            "  python -m scripts.evaluate --scores /path/to/scores.npz --min-history 5"
        ),
    )
    parser.add_argument(
        "--scores", type=Path, nargs="+", required=True,
        help="Scored NPZ files; include each domain/query only once.",
    )
    parser.add_argument(
        "--min-history", type=int, default=5,
        help="Minimum number of observed events per evaluated query (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-reps", type=int, default=10000,
        help="Number of paired user bootstrap replicates (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap sampling (default: 42).",
    )
    args = parser.parse_args()
    if args.min_history < 0 or args.bootstrap_reps < 1:
        parser.error("min-history must be nonnegative and bootstrap-reps must be positive.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import pandas as pd

    from fieldrec.evaluation import aggregate, evaluate

    frames, identities = [], []
    for path in args.scores:
        with np.load(path, allow_pickle=False) as scores:
            metadata = pd.DataFrame(
                {
                    key: scores[key]
                    for key in ("domain", "user_id", "query_id", "history_count")
                }
            )
            selected = metadata["history_count"].to_numpy() >= args.min_history
            identities.append(metadata.loc[selected, ["domain", "query_id"]])
            frames.append(
                evaluate(
                    metadata.loc[selected],
                    scores["base_scores"][selected],
                    scores["fieldrec_scores"][selected],
                    scores["target_position"][selected],
                )
            )
    if pd.concat(identities).duplicated(["domain", "query_id"]).any():
        raise ValueError("Each domain/query must be evaluated once.")
    per_query = pd.concat(frames, ignore_index=True)
    if per_query.empty:
        raise ValueError("No history-eligible queries.")
    domains, macro = aggregate(per_query, args.bootstrap_reps, args.seed)
    domains["p_two_sided"] = domains["p_two_sided"].map(
        lambda p: f"<{2 / args.bootstrap_reps:g}" if p == 0 else f"{p:g}"
    )
    if macro["p_two_sided"] == 0:
        macro["p_two_sided"] = f"<{2 / args.bootstrap_reps:g}"
    print(
        domains[
            ["users", "base_ndcg_5", "fieldrec_ndcg_5", "gain_ndcg_5",
             "relative_gain_ndcg_5", "ci95_low", "ci95_high", "p_two_sided"]
        ].to_csv()
    )
    print(
        json.dumps(
            {
                "queries": len(per_query),
                "domains": len(domains),
                # A zero base metric leaves its relative gain undefined.
                "macro": {
                    name: None if pd.isna(value) else value
                    for name, value in macro.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
