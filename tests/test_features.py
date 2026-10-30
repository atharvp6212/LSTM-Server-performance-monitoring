"""Tests for selfheal.data.features (paper Step 3: feature engineering).

SPEC test list: engineered_dim matches the output; lag/delta/flag values
verified on a tiny hand-built window.  Channel order per SPEC: raw, lags
(l = 1..L), rolling mean, rolling std, delta, utilization ratio, flags.
"""

from __future__ import annotations

import copy

import numpy as np

from selfheal.data.features import engineer_features, engineered_dim, limits_vector
from selfheal.data.preprocessing import Scaler, clean


def _tiny_window() -> np.ndarray:
    """One window (M=1, w=4, d=2) with hand-checkable values in [0, 1]."""
    return np.array(
        [[[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.8, 0.2]]], dtype=np.float32
    )


def test_engineered_dim_matches_output_shape(cfg: dict, rng: np.random.Generator) -> None:
    """d' = engineered_dim(d, cfg) always equals the produced last axis."""
    m, w, d = 3, 10, 6
    windows = rng.random((m, w, d)).astype(np.float32)

    engineered = engineer_features(windows, cfg)
    # Fixture cfg: lags=2, rolling, delta, ratio, flags all on ->
    # blocks = 1 + 2 + 2 + 1 + 1 + 1 = 8.
    assert engineered_dim(d, cfg) == 8 * d
    assert engineered.shape == (m, w, 8 * d)
    assert engineered.dtype == np.float32


def test_engineered_dim_respects_switches(cfg: dict, rng: np.random.Generator) -> None:
    """Every cfg["features"] switch adds/removes exactly its channels."""
    windows = rng.random((2, 6, 3)).astype(np.float32)

    combos = [
        {"lags": 0, "rolling": False, "delta": False,
         "utilization_ratio": False, "threshold_flag": False},  # raw only -> d
        {"lags": 3, "rolling": False, "delta": True,
         "utilization_ratio": False, "threshold_flag": False},  # (1+3+1)d
        {"lags": 1, "rolling": True, "delta": False,
         "utilization_ratio": True, "threshold_flag": True},    # (1+1+2+1+1)d
    ]
    expected_blocks = [1, 5, 6]
    for switches, blocks in zip(combos, expected_blocks):
        cfg_i = copy.deepcopy(cfg)
        cfg_i["features"].update(switches)
        out = engineer_features(windows, cfg_i)
        assert engineered_dim(3, cfg_i) == blocks * 3
        assert out.shape[2] == blocks * 3


def test_engineered_values_on_tiny_window(cfg: dict) -> None:
    """Verify lag padding, rolling stats, delta, ratio and flag values (Step 3)."""
    windows = _tiny_window()
    limits = np.array([0.5, 1.0], dtype=np.float32)
    out = engineer_features(windows, cfg, limits=limits)  # (1, 4, 16)
    assert out.shape == (1, 4, 16)
    got = out[0]  # (w=4, d'=16)
    x = windows[0]  # (4, 2)

    # Channel layout for d=2, L=2: [0:2] raw, [2:4] lag1, [4:6] lag2,
    # [6:8] mean, [8:10] std, [10:12] delta, [12:14] ratio, [14:16] flags.
    np.testing.assert_allclose(got[:, 0:2], x, atol=1e-6)

    # a. lags shift within the window, padded with the window's first value.
    lag1 = np.array([[0.1, 0.9], [0.1, 0.9], [0.2, 0.8], [0.4, 0.6]])
    lag2 = np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.2, 0.8]])
    np.testing.assert_allclose(got[:, 2:4], lag1, atol=1e-6)
    np.testing.assert_allclose(got[:, 4:6], lag2, atol=1e-6)

    # b. rolling mean/std broadcast across the window.
    np.testing.assert_allclose(
        got[:, 6:8], np.broadcast_to(x.mean(axis=0), (4, 2)), atol=1e-5
    )
    np.testing.assert_allclose(
        got[:, 8:10], np.broadcast_to(x.std(axis=0), (4, 2)), atol=1e-5
    )

    # c. rate of change: x_t - x_{t-1}, first step 0.
    delta = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.4, -0.4]])
    np.testing.assert_allclose(got[:, 10:12], delta, atol=1e-6)

    # d. utilization ratio r = x / limit with limits (0.5, 1.0).
    ratio = x / limits
    np.testing.assert_allclose(got[:, 12:14], ratio, atol=1e-6)

    # e. threshold flags (r > tau=0.9) as float — strict inequality, so the
    # ratio exactly at 0.9 stays 0.
    flags = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(got[:, 14:16], flags, atol=1e-6)


def test_default_limits_make_ratio_equal_raw(cfg: dict) -> None:
    """limits=None means limit 1.0 per feature: ratio block == raw block."""
    windows = _tiny_window()
    out = engineer_features(windows, cfg, limits=None)
    np.testing.assert_allclose(out[0][:, 12:14], windows[0], atol=1e-6)


def test_limits_vector_raw_and_disabled(cfg: dict) -> None:
    """Configured limits align with feature order; missing features get 1.0."""
    feature_cols = ["cpu_usage", "custom_metric", "network_io"]
    limits = limits_vector(feature_cols, cfg, scaler=None)
    assert limits is not None
    np.testing.assert_allclose(limits, np.array([100.0, 1.0, 1000.0]), atol=1e-6)

    cfg_off = copy.deepcopy(cfg)
    cfg_off["features"]["utilization_ratio"] = False
    assert limits_vector(feature_cols, cfg_off, scaler=None) is None


def test_limits_vector_scaled_space(cfg: dict, make_frame) -> None:
    """With a scaler: limit_scaled = (limit - data_min) / (data_max - data_min)."""
    df = clean(make_frame(n=50), cfg)
    # Overwrite two columns with exactly known ranges.
    df["cpu_usage"] = np.linspace(0.0, 50.0, len(df))  # limit 100 -> scaled 2.0
    df["memory_usage"] = np.linspace(0.0, 100.0, len(df))  # limit 100 -> 1.0
    feature_cols = ["cpu_usage", "memory_usage"]
    scaler = Scaler().fit(df, feature_cols)

    limits = limits_vector(feature_cols, cfg, scaler=scaler)
    assert limits is not None
    np.testing.assert_allclose(limits, np.array([2.0, 1.0]), rtol=1e-6)
