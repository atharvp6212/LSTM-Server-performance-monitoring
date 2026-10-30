"""Shared pytest fixtures for the self-healing platform test suite.

Supports the test plan in SPEC.md ("tests — test-agent") for the paper
*Self-Healing Cloud Platforms Using LSTM-based Predictive Models for Early
Fault Detection* (IJCET 16(5) 2025, DOI: 10.34218/IJCET_16_05_012).

Provides:

* ``cfg`` — a tiny, fully self-contained config dict mirroring
  ``config.yaml`` (paper Steps 1–7 parameters) but scaled down so every test
  runs in well under 20 seconds: window_size 10, horizon 3, lags 2,
  n_records 1500, an 8-unit LSTM trained for 1 epoch.  All paths live under
  ``tmp_path`` so no test ever writes into the real project tree.
* ``rng`` — a seeded ``numpy.random.Generator`` for deterministic test data.
* ``make_frame`` — factory building small canonical telemetry frames
  (SPEC "Canonical data contract": ``timestamp`` + numeric features +
  ``anomaly``).
* an autouse fixture reseeding the global RNGs before every test.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Silence TF before any test module imports it (SPEC cross-cutting rule).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # make `import selfheal` work everywhere
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42

#: Canonical synthetic metric names (mirrors selfheal.data.dataset.CANONICAL_METRICS;
#: duplicated here so conftest never imports concurrently-built modules at collection).
METRIC_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "network_io",
    "pod_restarts",
    "error_event_rate",
    "warning_event_rate",
]


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Reseed the global RNGs before every test (determinism, SPEC rule)."""
    random.seed(SEED)
    np.random.seed(SEED)


@pytest.fixture()
def rng() -> np.random.Generator:
    """A per-test seeded numpy Generator."""
    return np.random.default_rng(SEED)


@pytest.fixture()
def project_root() -> Path:
    """Absolute path of the real project root (read-only use in tests)."""
    return PROJECT_ROOT


@pytest.fixture()
def kaggle_csv_path() -> Path:
    """Path to the cited Kaggle traffic CSV shipped in data/raw (paper [17])."""
    return PROJECT_ROOT / "data" / "raw" / "balanced_shuffled_traffic.csv"


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    """Tiny config dict shaped exactly like config.yaml, rooted at tmp_path.

    Small enough that any pipeline stage (including a 1-epoch model fit)
    finishes in seconds; every path resolves inside ``tmp_path`` so tests are
    hermetic.  A fresh dict is built per test, so tests may mutate it freely.
    """
    return {
        "project": {
            "name": "selfheal-test",
            "seed": SEED,
            "root": str(tmp_path),
        },
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "artifacts": "models/artifacts",
            "reports": "reports",
            "feedback_log": "feedback/events.jsonl",
        },
        "dataset": {
            "source": "synthetic",
            "kaggle_csv": str(PROJECT_ROOT / "data" / "raw" / "balanced_shuffled_traffic.csv"),
            "max_raw_features": 12,
            "synthetic": {
                "n_records": 1500,
                "n_pods": 4,
                "n_nodes": 2,
                "namespaces": ["default", "production"],
                "tick_seconds": 15,
                "fault_rate": 0.02,
            },
        },
        "preprocessing": {
            "impute_strategy": "ffill",
            "scaler": "minmax",
        },
        "windowing": {
            "window_size": 10,
            "stride": 1,
            "horizon": 3,
        },
        "features": {
            "lags": 2,
            "rolling": True,
            "delta": True,
            "utilization_ratio": True,
            "threshold_flag": True,
            "tau": 0.9,
            "limits": {
                "cpu_usage": 100.0,
                "memory_usage": 100.0,
                "network_io": 1000.0,
                "pod_restarts": 5.0,
                "error_event_rate": 20.0,
                "warning_event_rate": 40.0,
            },
        },
        "model": {
            "lstm_units": 8,
            "dropout": 0.1,
            "dense_units": 8,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 1,
            "early_stopping_patience": 2,
            "class_weight": "balanced",
        },
        "decision": {
            "fault_threshold": 0.7,
            "severity_bands": {
                "alert": 0.7,
                "scale": 0.8,
                "restart": 0.9,
                "drain": 0.97,
            },
        },
        "healing": {
            "mode": "simulate",
            "cooldown_seconds": 120,
            "max_actions_per_hour": 5,
            "kubernetes": {
                "kubeconfig": None,
                "namespace": "default",
                "max_replicas": 10,
                "alert_webhook": None,
            },
        },
        "monitoring": {
            "fp_rate_retrain_threshold": 0.2,
            "retrain_every_events": 10,
        },
    }


@pytest.fixture()
def make_frame():
    """Factory for small canonical telemetry frames (SPEC data contract).

    ``make_frame(n=..., seed=..., anomaly_idx=[...])`` returns a frame with a
    sorted datetime64 ``timestamp`` column, the six canonical float metrics
    and an int 0/1 ``anomaly`` column (1 at the given row indices).
    """

    def _make(
        n: int = 120,
        seed: int = SEED,
        anomaly_idx: list[int] | None = None,
        tick_seconds: int = 15,
    ) -> pd.DataFrame:
        gen = np.random.default_rng(seed)
        anomaly = np.zeros(n, dtype=np.int64)
        for i in anomaly_idx or []:
            anomaly[i] = 1
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01", periods=n, freq=f"{tick_seconds}s"
                ),
                "cpu_usage": gen.uniform(10.0, 90.0, n),
                "memory_usage": gen.uniform(20.0, 80.0, n),
                "network_io": gen.uniform(0.0, 800.0, n),
                "pod_restarts": gen.integers(0, 4, n).astype(np.float64),
                "error_event_rate": gen.uniform(0.0, 10.0, n),
                "warning_event_rate": gen.uniform(0.0, 20.0, n),
                "anomaly": anomaly,
            }
        )

    return _make
