"""Tests for selfheal.data.windowing (paper Step 2: sliding time windows).

SPEC test list: shapes, horizon labeling correctness on a hand-built y,
stride handling.
"""

from __future__ import annotations

import numpy as np

from selfheal.data.windowing import make_windows, window_dataframe


def test_make_windows_shapes_and_dtype(rng: np.random.Generator) -> None:
    """(N, d) input -> (M, w, d) float32 windows with M = N - w - h + 1."""
    n, d, w, h = 50, 4, 10, 3
    X = rng.random((n, d))
    y = np.zeros(n, dtype=np.int64)

    windows, labels = make_windows(X, y, window_size=w, stride=1, horizon=h)

    expected_m = n - w - h + 1
    assert windows.shape == (expected_m, w, d)
    assert windows.dtype == np.float32
    assert labels.shape == (expected_m,)
    assert (labels == 0).all()
    # Window k holds rows k .. k+w-1.
    np.testing.assert_allclose(windows[0], X[0:w].astype(np.float32))
    np.testing.assert_allclose(windows[5], X[5 : 5 + w].astype(np.float32))


def test_horizon_labeling_on_hand_built_y() -> None:
    """label_k = 1 iff max(y[k+w : k+w+horizon]) == 1 (early prediction).

    N=12, w=4, horizon=3 and a single fault at step 8: the fault lies within
    the post-window horizon exactly for window starts k with
    k+4 <= 8 <= k+6, i.e. k in {2, 3, 4}.
    """
    n, w, h = 12, 4, 3
    X = np.arange(n, dtype=np.float64).reshape(n, 1)
    y = np.zeros(n, dtype=np.int64)
    y[8] = 1

    windows, labels = make_windows(X, y, window_size=w, stride=1, horizon=h)

    assert windows.shape == (n - w - h + 1, w, 1)  # M = 6
    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 1, 0]))


def test_label_excludes_faults_inside_the_window() -> None:
    """A fault *inside* the window does not label it — only the horizon does."""
    n, w, h = 10, 4, 2
    X = np.zeros((n, 1))
    y = np.zeros(n, dtype=np.int64)
    y[1] = 1  # inside window 0 (rows 0..3), before every horizon of k >= 0

    _, labels = make_windows(X, y, window_size=w, stride=1, horizon=h)
    assert labels[0] == 0, "fault within the window itself must not set the label"


def test_stride_handling() -> None:
    """Window m starts at row m * stride; only complete horizons survive."""
    n, w, h, stride = 20, 5, 2, 3
    X = np.arange(n, dtype=np.float64).reshape(n, 1)
    y = np.zeros(n, dtype=np.int64)
    y[9] = 1  # labels windows with k+5 <= 9 <= k+6 -> k in {3, 4} -> only k=3

    windows, labels = make_windows(X, y, window_size=w, stride=stride, horizon=h)

    last_start = n - w - h  # 13
    expected_starts = list(range(0, last_start + 1, stride))  # 0,3,6,9,12
    assert windows.shape == (len(expected_starts), w, 1)
    for m, k in enumerate(expected_starts):
        np.testing.assert_allclose(windows[m, :, 0], np.arange(k, k + w))
    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 0, 0]))


def test_too_short_series_yields_empty(rng: np.random.Generator) -> None:
    """N < w + horizon -> zero windows, both outputs empty but well-shaped."""
    X = rng.random((6, 3))
    y = np.zeros(6, dtype=np.int64)
    windows, labels = make_windows(X, y, window_size=5, stride=1, horizon=2)
    assert windows.shape == (0, 5, 3)
    assert labels.shape == (0,)


def test_window_dataframe_uses_cfg(cfg: dict, make_frame) -> None:
    """Convenience wrapper pulls arrays and cfg["windowing"] parameters."""
    df = make_frame(n=40, anomaly_idx=[30])
    feature_cols = [c for c in df.columns if c not in ("timestamp", "anomaly")]

    windows, labels = window_dataframe(df, cfg, feature_cols)

    w = cfg["windowing"]["window_size"]  # 10
    h = cfg["windowing"]["horizon"]  # 3
    expected_m = len(df) - w - h + 1
    assert windows.shape == (expected_m, w, len(feature_cols))
    assert windows.dtype == np.float32
    assert labels.shape == (expected_m,)
    # Fault at row 30: labeled windows are starts k with k+10 <= 30 <= k+12.
    expected = np.zeros(expected_m, dtype=np.int64)
    for k in range(expected_m):
        if k + w <= 30 <= k + w + h - 1:
            expected[k] = 1
    np.testing.assert_array_equal(labels, expected)
