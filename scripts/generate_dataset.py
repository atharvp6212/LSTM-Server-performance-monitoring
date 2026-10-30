#!/usr/bin/env python3
"""Generate the synthetic Kubernetes telemetry dataset (paper §3.1/§3.2).

Thin argparse CLI over :func:`selfheal.data.dataset.generate_k8s_telemetry`.
Writes ``k8s_logs.csv`` (Table 1 log schema, ~n_records rows) and
``k8s_metrics.csv`` (canonical cluster metrics with gradually building §3.2
faults) and prints a generation summary.

Usage (from the project root):
    python scripts/generate_dataset.py [--config config.yaml] [--records N] [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from selfheal.config import load_config, resolve_path, set_global_seeds
from selfheal.data.dataset import generate_k8s_telemetry


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, generate the telemetry, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Kubernetes telemetry (paper §3.1 Table 1, §3.2 faults)."
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument(
        "--records", type=int, default=None, help="override dataset.synthetic.n_records (~log rows)"
    )
    parser.add_argument(
        "--out", default=None, help="output directory (default: cfg paths.data_raw)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seeds(cfg)
    if args.records is not None:
        cfg.setdefault("dataset", {}).setdefault("synthetic", {})["n_records"] = int(args.records)

    logs_df, metrics_df = generate_k8s_telemetry(cfg, out_dir=args.out)

    out_dir = resolve_path(cfg, args.out if args.out is not None else cfg["paths"]["data_raw"])
    tick_seconds = cfg.get("dataset", {}).get("synthetic", {}).get("tick_seconds", 15)
    counts = logs_df["event_type"].value_counts()
    anomaly = metrics_df["anomaly"].to_numpy()
    onsets = int(((anomaly[1:] == 1) & (anomaly[:-1] == 0)).sum()) + int(anomaly[0] == 1)

    print("Synthetic Kubernetes telemetry generated (paper §3.1 Table 1, §3.2 fault buildup)")
    print(f"  logs   : {len(logs_df):,} rows -> {out_dir / 'k8s_logs.csv'}")
    for event_type in ("Normal", "Warning", "Error"):
        print(f"           {event_type:<8}= {int(counts.get(event_type, 0)):,}")
    print(f"  metrics: {len(metrics_df):,} ticks @ {tick_seconds}s -> {out_dir / 'k8s_metrics.csv'}")
    print(
        f"  anomaly: {int(anomaly.sum()):,} ticks ({100.0 * float(anomaly.mean()):.2f}%) "
        f"across {onsets} fault window(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
