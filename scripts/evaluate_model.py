#!/usr/bin/env python3
"""CLI: evaluate the trained LSTM model and write the report files (paper §4).

Usage (from the project root)::

    python scripts/evaluate_model.py [--config config.yaml]

Runs :func:`selfheal.models.evaluate.evaluate_model`, which reloads the saved
artifacts, replays the exact training pipeline on the chronological test
split, and writes ``metrics.json`` plus the evaluation figures into
``cfg["paths"]["reports"]``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPORT_FILES = (
    "metrics.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "training_history.png",
    "probability_timeline.png",
)


def main() -> None:
    """Parse CLI args, run evaluation, print metrics and report paths."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained LSTM fault-prediction model (paper §4)."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config file."
    )
    args = parser.parse_args()

    from selfheal.config import load_config, resolve_path
    from selfheal.models.evaluate import evaluate_model

    cfg = load_config(args.config)
    metrics = evaluate_model(cfg)

    print("\n=== Evaluation metrics (chronological test split) ===")
    for key in ("accuracy", "precision", "recall", "f1", "auc", "threshold"):
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"  {key:<10} {value:.4f}")
        else:
            print(f"  {key:<10} {value}")
    print(
        f"  confusion  tn={metrics['tn']}  fp={metrics['fp']}  "
        f"fn={metrics['fn']}  tp={metrics['tp']}"
    )

    reports_dir = resolve_path(cfg, cfg["paths"]["reports"])
    print("\nReports written:")
    for name in _REPORT_FILES:
        print(f"  {reports_dir / name}")


if __name__ == "__main__":
    main()
