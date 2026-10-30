"""Tests for selfheal.data.dataset (paper §3.1 dataset + Table 1 schema).

SPEC test list: synthetic generator (schema == LOG_SCHEMA, canonical metrics
frame, anomalies present, gradual fault buildup per paper §3.2) and the
Kaggle traffic loader (canonical, sorted, no NaN/inf, feature cap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selfheal.config import resolve_path
from selfheal.data.dataset import (
    CANONICAL_METRICS,
    LOG_SCHEMA,
    feature_columns,
    generate_k8s_telemetry,
    load_dataset,
    load_kaggle_traffic,
    load_synthetic,
)


def _assert_canonical(df: pd.DataFrame) -> list[str]:
    """Assert the SPEC canonical-telemetry-frame contract; return feature cols."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns and "anomaly" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    ts = df["timestamp"].to_numpy()
    assert (ts[1:] >= ts[:-1]).all(), "timestamps must be sorted ascending"
    assert set(np.unique(df["anomaly"].to_numpy())) <= {0, 1}
    feats = feature_columns(df)
    assert feats, "canonical frame needs at least one feature column"
    values = df[feats].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "feature columns must be finite"
    return feats


# --------------------------------------------------------------------------- #
# synthetic generator (paper §3.1/§3.2, Table 1)                              #
# --------------------------------------------------------------------------- #
def test_generate_k8s_telemetry_schema_and_files(cfg: dict) -> None:
    """logs_df matches Table 1 exactly; metrics_df is canonical; CSVs written."""
    logs_df, metrics_df = generate_k8s_telemetry(cfg)

    # Table 1 log schema — exact columns, in order.
    assert list(logs_df.columns) == LOG_SCHEMA
    n_records = cfg["dataset"]["synthetic"]["n_records"]
    assert 0.5 * n_records <= len(logs_df) <= 2.0 * n_records

    # Canonical cluster-aggregated metrics frame with exactly the six metrics.
    feats = _assert_canonical(metrics_df)
    assert set(feats) == set(CANONICAL_METRICS)
    assert metrics_df["timestamp"].is_unique, "one row per tick"

    out_dir = resolve_path(cfg, cfg["paths"]["data_raw"])
    assert (Path(out_dir) / "k8s_logs.csv").exists()
    assert (Path(out_dir) / "k8s_metrics.csv").exists()


def test_generate_k8s_telemetry_has_anomalies_with_gradual_buildup(cfg: dict) -> None:
    """Faults exist and build up gradually before the labeled anomaly period.

    Paper §3.2: "faults build up gradually but do not emerge suddenly" — in
    the 10 ticks before an anomaly onset at least one canonical metric must
    already exceed its global normal-period mean.
    """
    _, metrics_df = generate_k8s_telemetry(cfg)
    anomaly = metrics_df["anomaly"].to_numpy(dtype=np.int64)
    assert anomaly.sum() > 0, "generator must produce labeled anomaly periods"

    onsets = [
        i for i in range(1, len(anomaly))
        if anomaly[i] == 1 and anomaly[i - 1] == 0 and i >= 10
    ]
    assert onsets, "expected at least one anomaly onset with 10 ticks of lead-in"

    normal = metrics_df.loc[metrics_df["anomaly"] == 0]
    buildup_seen = False
    for onset in onsets:
        pre = metrics_df.iloc[onset - 10 : onset]
        for metric in CANONICAL_METRICS:
            if float(pre[metric].mean()) > float(normal[metric].mean()):
                buildup_seen = True
                break
        if buildup_seen:
            break
    assert buildup_seen, (
        "no metric elevated in the 10 ticks before any anomaly onset — "
        "faults must ramp up gradually (paper §3.2)"
    )


def test_load_synthetic_roundtrip(cfg: dict) -> None:
    """load_synthetic reads k8s_metrics.csv back into a canonical frame."""
    _, metrics_df = generate_k8s_telemetry(cfg)
    path = resolve_path(cfg, cfg["paths"]["data_raw"]) / "k8s_metrics.csv"
    loaded = load_synthetic(path)
    feats = _assert_canonical(loaded)
    assert set(feats) == set(CANONICAL_METRICS)
    assert len(loaded) == len(metrics_df)


def test_load_dataset_synthetic_generates_when_absent(cfg: dict) -> None:
    """Dispatch: source == synthetic generates the CSV on demand, then loads."""
    csv_path = resolve_path(cfg, cfg["paths"]["data_raw"]) / "k8s_metrics.csv"
    assert not csv_path.exists()
    df = load_dataset(cfg)
    _assert_canonical(df)
    assert csv_path.exists(), "load_dataset must generate the missing synthetic CSV"


def test_feature_columns_excludes_timestamp_and_anomaly(make_frame) -> None:
    """feature_columns == every column except timestamp and anomaly (SPEC)."""
    df = make_frame(n=10)
    feats = feature_columns(df)
    assert "timestamp" not in feats and "anomaly" not in feats
    assert set(feats) == set(df.columns) - {"timestamp", "anomaly"}


# --------------------------------------------------------------------------- #
# Kaggle traffic loader (paper §3.1, cited dataset [17])                      #
# --------------------------------------------------------------------------- #
def test_load_kaggle_traffic_canonical(kaggle_csv_path: Path) -> None:
    """Kaggle CSV -> canonical frame: sorted, finite, feature cap respected."""
    if not kaggle_csv_path.exists():
        pytest.skip(f"Kaggle CSV not present at {kaggle_csv_path}")
    max_features = 8
    df = load_kaggle_traffic(kaggle_csv_path, max_raw_features=max_features)

    feats = _assert_canonical(df)
    assert len(feats) <= max_features

    # Ports / protocol identifiers are excluded from the selected features.
    for col in feats:
        name = str(col).lower()
        assert "port" not in name, f"port-like column selected: {col}"
        assert name != "protocol", "protocol id must not be a feature"

    # The balanced dataset carries both classes; labels come from "Label".
    labels = df["anomaly"].to_numpy(dtype=np.int64)
    assert 0.0 < labels.mean() < 1.0, "expected both benign and attack labels"
