#!/usr/bin/env python
"""CLI for adaptive retraining driven by the feedback loop (paper Step 7).

Checks :meth:`selfheal.monitoring.feedback.FeedbackLoop.should_retrain`
against the accumulated JSONL feedback events and, when retraining is due
(or ``--force`` is given), reruns the full LSTM training pipeline via
:func:`selfheal.monitoring.feedback.retrain_from_feedback` and prints the
held-out test metrics.

Implements the "adaptive retraining" part of Step 7 of *Self-Healing Cloud
Platforms Using LSTM-based Predictive Models for Early Fault Detection*
(IJCET 16(5) 2025).

Usage (from the project root)::

    python scripts/retrain_feedback.py [--config config.yaml] [--force]
"""

from __future__ import annotations

import os

# Silence tensorflow C++ logging before anything can import it (SPEC rule).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from pathlib import Path

# Make `import selfheal` work when run as `python scripts/retrain_feedback.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfheal.config import load_config, resolve_path, set_global_seeds
from selfheal.monitoring.feedback import FeedbackLoop, retrain_from_feedback


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, check the retraining condition, and retrain if due."""
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive retraining from the feedback loop (paper Step 7): "
            "retrains the LSTM fault predictor when enough feedback events "
            "have accumulated or the false-positive rate has degraded."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration (default: config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the should_retrain check and retrain unconditionally.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_rel = cfg.get("paths", {}).get("feedback_log", "feedback/events.jsonl")
    loop = FeedbackLoop(resolve_path(cfg, log_rel))
    n_events = len(loop.load_events())

    if not args.force and not loop.should_retrain(cfg):
        print(
            f"Retraining not due: {n_events} feedback event(s) in "
            f"{loop.path} do not meet the retraining criteria "
            "(use --force to retrain anyway)."
        )
        return 0

    reason = "--force" if args.force else "feedback criteria met"
    print(f"Retraining LSTM fault predictor ({reason}; {n_events} feedback event(s))...")
    set_global_seeds(cfg)
    result = retrain_from_feedback(cfg)

    test_metrics = result.get("test", {})
    print("Retraining complete. Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2, default=str))
    print(f"Artifacts directory: {result.get('artifacts_dir', '')}")
    print(f"Feedback log marked with a 'retrain' event: {loop.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
