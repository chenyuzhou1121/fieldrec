"""Score frozen candidate slates with a fitted transition field and phase response."""

import argparse
import json
import logging
from pathlib import Path
import sys


def main():
    """Read CLI inputs/model/config, then write scored NPZ arrays and a log."""
    parser = argparse.ArgumentParser(
        description="Adjust frozen candidate scores using transitions for five possible ratings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input NPZ arrays (Q queries, N items, D embedding dimensions, W mobility window):\n"
            "  item_embeddings   (N,D) fixed item vectors used during state replay.\n"
            "  rating_baselines  (N,) item rating baselines frozen at the source cutoff.\n"
            "  residual_scale    Scalar positive rating-residual scale fitted on train.\n"
            "  candidate_indices (Q,100) unique zero-based indices into the item arrays.\n"
            "  base_scores       (Q,100) frozen scores in the same candidate order.\n"
            "  sufficient        (Q,3) S residual sum, E absolute residual sum, G weighted alignment sum.\n"
            "  positive, negative (Q,D) memories from fieldrec.observe; float16 round-tripped.\n"
            "  history_deltas    (Q,W,2) last W P/A changes between observed states, oldest first, NaN padded.\n"
            "  domain, user_id, query_id, history_count, target_position: each (Q,).\n"
            "  target_position   Zero-based target slot, used only by evaluation.\n"
            "Use numeric or Unicode arrays without pickled objects. W comes from the field.\n"
            "Exclude target feedback and future transitions from states and history_deltas.\n"
            "Build observed states with strictly earlier rating baselines and use the same\n"
            "embedding representation for replay and scoring. Train/validation use a train\n"
            "field; test uses a development field. Freeze response settings before test.\n"
            "Output NPZ keeps query metadata, candidate order, base_scores, fieldrec_scores,\n"
            "potential, support_gate and phase.\n\n"
            "Example from the repository root:\n"
            "  python -m scripts.score_candidates --inputs /path/to/candidates.npz "
            "--field /path/to/field.json --split validation --output /path/to/scores.npz"
        ),
    )
    parser.add_argument(
        "--inputs", type=Path, required=True,
        help="NPZ bundle of frozen scores, candidate vectors and observed query states.",
    )
    parser.add_argument(
        "--field", type=Path, required=True,
        help="JSON model produced by fit_field.py for the same domain and source split.",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True,
        help="Query split; determines the required field source.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.json",
        help="JSON state and response configuration (default: bundled configs/default.json).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination scored NPZ for evaluate.py; also writes a .log file.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np

    from fieldrec.config import load_config
    from fieldrec.field import ContextualFieldAtlas
    from fieldrec.mobility import HistoryOnlyMobilityLabeler
    from fieldrec.response import apply_response
    from fieldrec.scoring import score_candidates
    from fieldrec.state import PreferenceSufficientState

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output.with_suffix(".log")),
            logging.StreamHandler(),
        ],
    )
    state_config, _, response = load_config(args.config)
    model = json.loads(args.field.read_text())
    atlas = ContextualFieldAtlas.from_dict(model["atlas"])
    labeler = HistoryOnlyMobilityLabeler.from_dict(model["mobility"])
    if atlas.source_split != labeler.source_split:
        raise ValueError("Field and mobility sources differ.")
    with np.load(args.inputs, allow_pickle=False) as inputs:
        base, candidates = inputs["base_scores"], inputs["candidate_indices"]
        embeddings, baselines = (
            inputs["item_embeddings"].astype(np.float32),
            inputs["rating_baselines"],
        )
        sufficient = inputs["sufficient"]
        history = inputs["history_deltas"].astype(np.float64)
        positive, negative = inputs["positive"], inputs["negative"]
        q = len(base)
        if (
            base.shape != (q, 100)
            or candidates.shape != base.shape
            or candidates.dtype.kind not in "iu"
        ):
            raise ValueError("Expected 100 ordered candidates per query.")
        if embeddings.ndim != 2:
            raise ValueError("item_embeddings must have shape (items, dimensions).")
        if np.any((candidates < 0) | (candidates >= len(embeddings))):
            raise ValueError("Candidate index outside embedding catalogue.")
        if any(len(np.unique(row)) != 100 for row in candidates):
            raise ValueError("Candidates must be unique within each slate.")
        if (
            sufficient.shape != (q, 3)
            or positive.shape != negative.shape
            or positive.shape != (q, embeddings.shape[1])
        ):
            raise ValueError("State arrays do not align with query embeddings.")
        if baselines.shape != (len(embeddings),) or history.shape != (
            q, labeler.window, 2,
        ):
            raise ValueError("Baseline or history shape mismatch.")
        metadata = {
            key: inputs[key]
            for key in (
                "domain", "user_id", "query_id", "history_count", "target_position",
            )
        }
        if any(value.shape != (q,) for value in metadata.values()):
            raise ValueError("Query metadata does not align with score rows.")
        scale = float(inputs["residual_scale"].item())
        phases = np.full(q, "__OTHER__", dtype="U16")
        potentials, gates = np.zeros(base.shape, dtype=np.float64), np.zeros(q)
        adjusted = base.astype(np.float64)
        logging.info("Scoring queries=%d split=%s response=%s", q, args.split, response)
        for i in range(q):
            # Padding does not contribute to the mean of observed prior delta norms.
            valid = np.isfinite(history[i]).all(axis=1)
            if valid.sum() >= labeler.min_prior:
                speed = np.hypot(history[i, valid, 0], history[i, valid, 1]).mean()
                phases[i] = labeler.label_scores(np.asarray([speed]))[0]
            state = PreferenceSufficientState(*sufficient[i], positive[i], negative[i])
            potentials[i], gates[i] = score_candidates(
                state, embeddings[candidates[i]], baselines[candidates[i]],
                scale, atlas, str(phases[i]), args.split, state_config,
            )
            adjusted[i] = apply_response(
                base[i], potentials[i], gates[i], str(phases[i]), response
            )
            if (i + 1) % 1000 == 0:
                logging.info("Scored %d/%d queries", i + 1, q)
        with args.output.open("wb") as stream:
            np.savez_compressed(
                stream,
                **metadata,
                candidate_indices=candidates,
                base_scores=base,
                fieldrec_scores=adjusted,
                potential=potentials,
                support_gate=gates,
                phase=phases,
            )
    logging.info("Saved %d scored queries", q)


if __name__ == "__main__":
    main()
