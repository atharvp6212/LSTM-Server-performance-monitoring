#!/usr/bin/env python3
"""CLI: train the LSTM early-fault-detection model (paper Step 4).

Usage (from the project root)::

    python scripts/train_model.py [--config config.yaml] [--source {synthetic,kaggle}]

Runs :func:`selfheal.models.train.train_model` — the full paper pipeline
(Steps 1–4 of Patil & Patil 2025, IJCET 16(5)) — and prints the held-out
chronological test metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """Parse CLI args, run training, print the test metrics."""
    parser = argparse.ArgumentParser(
        description="Train the LSTM fault-prediction model (paper Step 4)."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config file."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "kaggle"],
        default=None,
        help="Override cfg['dataset']['source'] for this run.",
    )
    args = parser.parse_args()

    from selfheal.config import load_config
    from selfheal.models.train import train_model

    cfg = load_config(args.config)
    if args.source is not None:
        cfg.setdefault("dataset", {})["source"] = args.source

    summary = train_model(cfg)
    test = summary["test"]
    print("\n=== Test metrics (chronological hold-out) ===")
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        value = test.get(key)
        if isinstance(value, float):
            print(f"  {key:<10} {value:.4f}")
        else:
            print(f"  {key:<10} {value}")
    print(
        f"  confusion  tn={test['tn']}  fp={test['fp']}  "
        f"fn={test['fn']}  tp={test['tp']}"
    )
    print(f"\nArtifacts written to: {summary['artifacts_dir']}")


if __name__ == "__main__":
    main()
