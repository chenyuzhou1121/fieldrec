"""Fit a transition atlas and mobility labeler from a single-domain Parquet table."""

import argparse
import json
import logging
from pathlib import Path
import sys


def main():
    """Parse CLI paths/settings, filter source rows, and write model JSON and a log."""
    parser = argparse.ArgumentParser(
        description="Fit a user-balanced transition field and a history mobility labeler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Transition table columns:\n"
            "  user_id             User identifier within one domain.\n"
            "  current_event_index Chronological index of the starting state.\n"
            "  target_split        Split of the destination event.\n"
            "  P, A                Starting preference and alignment coordinates.\n"
            "  delta_P, delta_A    Next coordinates minus starting coordinates.\n"
            "Supply chronological source transitions between observed states, including short histories.\n"
            "Replay events with fieldrec.observe using causal rating baselines and\n"
            "fixed item embeddings. Store P/A coordinates and their differences as\n"
            "float32. The mobility window excludes the transition being labeled.\n"
            "Use a train field for validation and a development field for test.\n\n"
            "Example from the repository root:\n"
            "  python -m scripts.fit_field --transitions /path/to/transitions.parquet "
            "--source train --output /path/to/field.json"
        ),
    )
    parser.add_argument(
        "--transitions", type=Path, required=True,
        help="Parquet table of observed state transitions for one domain.",
    )
    parser.add_argument(
        "--source", choices=("train", "development"), default="train",
        help="Fit on train, or train plus validation for development (default: train).",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.json",
        help="JSON method configuration (default: bundled configs/default.json).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination JSON containing the field and mobility labeler; also writes a .log file.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd

    from fieldrec.config import load_config
    from fieldrec.field import ContextSchema, ContextualFieldAtlas
    from fieldrec.mobility import HistoryOnlyMobilityLabeler

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output.with_suffix(".log")),
            logging.StreamHandler(),
        ],
    )
    _, config, _ = load_config(args.config)
    frame = pd.read_parquet(args.transitions)
    # Filter before mobility replay so only source-visible deltas enter histories.
    allowed = {"train"} if args.source == "train" else {"train", "validation"}
    frame = frame[frame["target_split"].isin(allowed)].copy()
    logging.info("Fitting source=%s rows=%d config=%s", args.source, len(frame), config)
    labeler = HistoryOnlyMobilityLabeler.fit(
        frame,
        source_split=args.source,
        **json.loads(args.config.read_text())["mobility"],
    )
    frame["history_mobility_mode"] = labeler.transform(frame)
    atlas = ContextualFieldAtlas.fit(
        frame,
        schema=ContextSchema(column="history_mobility_mode"),
        source_split=args.source,
        config=config,
    )
    args.output.write_text(
        json.dumps({"atlas": atlas.to_dict(), "mobility": labeler.to_dict()})
    )
    logging.info("Saved field rows=%d users=%d", atlas.fit_rows, atlas.fit_users)


if __name__ == "__main__":
    main()
