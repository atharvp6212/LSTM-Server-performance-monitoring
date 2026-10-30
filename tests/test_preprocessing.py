"""Tests for selfheal.data.preprocessing (paper Step 1: cleaning + Min-Max).

SPEC test list: clean handles inf/NaN/duplicate timestamps; Scaler roundtrip
(save/load, transform range [0, 1]).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selfheal.data.preprocessing import Scaler, clean, preprocess

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "network_io",
    "pod_restarts",
    "error_event_rate",
    "warning_event_rate",
]


def _messy_frame() -> pd.DataFrame:
    """A deliberately dirty frame: unsorted, duplicate ts, inf and NaN values."""
    ts = pd.to_datetime(
        [
            "2025-01-01 00:01:00",  # out of order
            "2025-01-01 00:00:00",
            "2025-01-01 00:02:00",
            "2025-01-01 00:02:00",  # duplicate timestamp
            "2025-01-01 00:03:00",
            "2025-01-01 00:04:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "cpu_usage": [np.nan, 10.0, np.inf, 99.0, 40.0, 50.0],
            "memory_usage": [30.0, 20.0, 35.0, 1.0, -np.inf, 45.0],
            "network_io": [100.0, 90.0, 110.0, 2.0, 120.0, np.nan],
            "pod_restarts": [0.0, 0.0, 1.0, 9.0, 1.0, 2.0],
            "error_event_rate": [1.0, 0.5, 1.5, 8.0, 2.0, 2.5],
            "warning_event_rate": [3.0, 2.0, 4.0, 7.0, 5.0, 6.0],
            "anomaly": [0, 0, 0, 1, 1, 0],
        }
    )


# --------------------------------------------------------------------------- #
# clean (Step 1 cleaning half)                                                #
# --------------------------------------------------------------------------- #
def test_clean_sorts_dedups_and_imputes(cfg: dict) -> None:
    """ffill strategy: sorted unique timestamps, all features finite."""
    out = clean(_messy_frame(), cfg)

    ts = out["timestamp"].to_numpy()
    assert (ts[1:] > ts[:-1]).all(), "timestamps must be strictly increasing"
    assert len(out) == 5, "duplicate timestamp must be dropped (keep first)"

    values = out[FEATURES].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "inf/NaN must be imputed away"

    # keep=first on the duplicated 00:02:00 tick -> cpu stays the imputed
    # value of the first duplicate row, not 99.0 from the second.
    row_02 = out.loc[out["timestamp"] == pd.Timestamp("2025-01-01 00:02:00")]
    assert float(row_02["memory_usage"].iloc[0]) == 35.0

    # anomaly column survives cleaning as int 0/1.
    assert "anomaly" in out.columns
    assert set(np.unique(out["anomaly"].to_numpy())) <= {0, 1}


def test_clean_ffill_covers_leading_nan(cfg: dict) -> None:
    """"ffill" is ffill().bfill(): a NaN in the first row is back-filled."""
    out = clean(_messy_frame(), cfg)
    # After sorting, the first row is 00:00:00 (cpu 10.0); the original NaN
    # lands at 00:01:00 and must forward-fill from it.
    first = out.iloc[0]
    second = out.iloc[1]
    assert float(second["cpu_usage"]) == float(first["cpu_usage"]) == 10.0


def test_clean_drop_strategy_removes_incomplete_rows(cfg: dict) -> None:
    """"drop" removes rows with missing feature values instead of imputing."""
    cfg_drop = copy.deepcopy(cfg)
    cfg_drop["preprocessing"]["impute_strategy"] = "drop"
    out = clean(_messy_frame(), cfg_drop)
    values = out[FEATURES].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert len(out) < 5, "rows with NaN/inf features must be dropped"


# --------------------------------------------------------------------------- #
# Scaler (Step 1 Min-Max half)                                                #
# --------------------------------------------------------------------------- #
def test_scaler_transform_range_and_inverse(cfg: dict, make_frame) -> None:
    """Transformed features span [0, 1]; inverse_transform restores raw units."""
    df = clean(make_frame(n=80), cfg)
    scaler = Scaler().fit(df, FEATURES)
    assert scaler.feature_cols == FEATURES

    scaled = scaler.transform(df)
    values = scaled[FEATURES].to_numpy(dtype=np.float64)
    assert values.min() >= 0.0 and values.max() <= 1.0
    np.testing.assert_allclose(values.min(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(values.max(axis=0), 1.0, atol=1e-9)

    # Non-feature columns are untouched.
    assert scaled["timestamp"].equals(df["timestamp"])
    assert scaled["anomaly"].equals(df["anomaly"])
    # transform returns a copy — the input frame keeps its raw values.
    assert float(df[FEATURES[0]].max()) > 1.0

    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(
        restored[FEATURES].to_numpy(dtype=np.float64),
        df[FEATURES].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )


def test_scaler_save_load_roundtrip(cfg: dict, make_frame, tmp_path: Path) -> None:
    """save/load persists both the fitted scaler and its feature columns."""
    df = clean(make_frame(n=60), cfg)
    scaler = Scaler().fit(df, FEATURES)
    path = tmp_path / "artifacts" / "scaler.joblib"
    scaler.save(path)
    assert path.exists()

    loaded = Scaler.load(path)
    assert loaded.feature_cols == FEATURES
    np.testing.assert_allclose(
        loaded.transform(df)[FEATURES].to_numpy(dtype=np.float64),
        scaler.transform(df)[FEATURES].to_numpy(dtype=np.float64),
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# preprocess (full Step 1)                                                    #
# --------------------------------------------------------------------------- #
def test_preprocess_fit_and_reuse(cfg: dict, make_frame) -> None:
    """fit=True fits a fresh scaler; fit=False reuses the given one."""
    df = make_frame(n=80)
    scaled, scaler = preprocess(df, cfg, fit=True)
    assert isinstance(scaler, Scaler)
    values = scaled[FEATURES].to_numpy(dtype=np.float64)
    assert values.min() >= 0.0 and values.max() <= 1.0

    rescaled, same_scaler = preprocess(df, cfg, scaler=scaler, fit=False)
    assert same_scaler is scaler
    np.testing.assert_allclose(
        rescaled[FEATURES].to_numpy(dtype=np.float64), values, atol=1e-12
    )


def test_preprocess_requires_scaler_when_not_fitting(cfg: dict, make_frame) -> None:
    """fit=False without a scaler is a contract violation (SPEC)."""
    with pytest.raises(ValueError):
        preprocess(make_frame(n=20), cfg, scaler=None, fit=False)
