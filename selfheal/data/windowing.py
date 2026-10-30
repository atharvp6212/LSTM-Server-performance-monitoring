"""Paper Step 2 — sliding time windows over scaled telemetry.

Implements the sliding-window stage (methodology Step 2) of: Atharv S. Patil,
Sanjay S. Patil (2025). "Self-Healing Cloud Platforms Using LSTM-based
Predictive Models for Early Fault Detection." IJCET 16(5), 177-187,
DOI: 10.34218/IJCET_16_05_012.

The paper forms input sequences ``X_k = {x_k, ..., x_{k+w-1}}`` of window
size ``w`` from the normalised metric stream (Step 1,
:mod:`selfheal.data.preprocessing`) and labels each window by whether a fault
occurs within the prediction ``horizon`` immediately AFTER the window — this
is what makes the model *predictive* (early fault detection) rather than
merely detective:

    ``label_k = max(y[k+w : k+w+horizon])``

Windows whose full horizon would extend past the end of the series are
dropped, so every label is well defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int = 1,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a scaled feature matrix into labelled sliding windows (Step 2).

    Args:
        X: Scaled features, shape ``(N, d)`` (a 1-D array is treated as
           ``(N, 1)``).
        y: Per-step anomaly labels, shape ``(N,)`` of int 0/1.
        window_size: ``w`` — number of consecutive steps per window.
        stride: Step between consecutive window starts (>= 1).
        horizon: Look-ahead length; ``label_k = max(y[k+w : k+w+horizon])``,
            i.e. 1 iff a fault occurs within ``horizon`` steps after the
            window (paper Step 2 labelling for early prediction).

    Returns:
        ``(windows, labels)`` — ``windows`` of shape ``(M, w, d)`` float32
        and ``labels`` of shape ``(M,)`` int64.  Window ``m`` starts at row
        ``m * stride``.  Windows without a complete post-window horizon are
        dropped; if ``N < window_size + horizon`` both arrays are empty
        (``M = 0``).

    Raises:
        ValueError: If ``window_size``/``stride``/``horizon`` are not
            positive, ``X`` is not 1-D/2-D, or ``len(y) != len(X)``.
    """
    if window_size <= 0:
        raise ValueError(f"make_windows: window_size must be >= 1, got {window_size}")
    if stride <= 0:
        raise ValueError(f"make_windows: stride must be >= 1, got {stride}")
    if horizon <= 0:
        raise ValueError(f"make_windows: horizon must be >= 1, got {horizon}")

    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if X.ndim != 2:
        raise ValueError(f"make_windows: X must be 2-D (N, d), got shape {X.shape}")
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(
            f"make_windows: X has {n} rows but y has {y.shape[0]} entries"
        )

    last_start = n - window_size - horizon  # last k with a full window AND horizon
    if last_start < 0:
        return (
            np.empty((0, window_size, d), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    starts = np.arange(0, last_start + 1, stride)

    # (n - w + 1, d, w) view -> select starts -> (M, w, d); fancy indexing copies.
    win_view = sliding_window_view(X, window_size, axis=0)
    windows = np.ascontiguousarray(
        win_view[starts].transpose(0, 2, 1), dtype=np.float32
    )

    # label_k = max(y[k+w : k+w+horizon]) — vectorised via a horizon-wide view.
    lab_view = sliding_window_view(y, horizon)  # (n - horizon + 1, horizon)
    labels = lab_view[starts + window_size].max(axis=1).astype(np.int64)

    return windows, labels


def window_dataframe(
    df: pd.DataFrame,
    cfg: dict,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Window a canonical telemetry frame using ``cfg["windowing"]`` (Step 2).

    Convenience wrapper: pulls ``df[feature_cols]`` (float32) and the
    ``anomaly`` column, then calls :func:`make_windows` with
    ``window_size``/``stride``/``horizon`` from ``cfg["windowing"]``.

    Args:
        df: Canonical telemetry frame — normally already scaled by
            :func:`selfheal.data.preprocessing.preprocess`.
        cfg: Full project config; uses ``cfg["windowing"]``.
        feature_cols: Feature columns to window, in canonical order.

    Returns:
        ``(windows, labels)`` as from :func:`make_windows`.

    Raises:
        KeyError: If any of ``feature_cols`` or the ``anomaly`` column is
            missing from ``df``.
    """
    wcfg = cfg.get("windowing", {})
    window_size = int(wcfg.get("window_size", 30))
    stride = int(wcfg.get("stride", 1))
    horizon = int(wcfg.get("horizon", 1))

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"window_dataframe: frame is missing feature columns {missing}")
    if "anomaly" not in df.columns:
        raise KeyError("window_dataframe: frame has no 'anomaly' column")

    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return make_windows(X, y, window_size=window_size, stride=stride, horizon=horizon)
