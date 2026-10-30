"""Paper Step 3 — feature engineering on sliding windows.

Implements the feature-engineering stage (methodology Step 3) of: Atharv S.
Patil, Sanjay S. Patil (2025). "Self-Healing Cloud Platforms Using LSTM-based
Predictive Models for Early Fault Detection." IJCET 16(5), 177-187,
DOI: 10.34218/IJCET_16_05_012.

Step 3 derives, per raw metric: lag features, rolling statistics (mean/std),
rate of change (per-step delta), a resource-utilisation ratio
``r = x / limit`` and a safety-threshold flag ``f = (r > tau)``.  The
engineered output keeps the *sequence* form ``F_k ∈ R^(w × d')`` so it can
feed the LSTM directly (paper Step 4 / Figure 1).

Channel order along the last axis (SPEC): raw, lags (l = 1..L), rolling mean,
rolling std, delta, utilisation ratio, threshold flags — each block present
only when its switch in ``cfg["features"]`` is on (for lags the switch is
``lags > 0``).  All operations are vectorised over the M windows.
"""

from __future__ import annotations

import numpy as np

from selfheal.data.preprocessing import Scaler

_MIN_LIMIT: float = 1e-6  # lower clip for limits, avoids division by zero


def engineered_dim(d: int, cfg: dict) -> int:
    """Engineered feature dimension ``d'`` for raw dimension ``d`` (Step 3).

    ``d' = d * (1 + L + 2*rolling + delta + utilization_ratio +
    threshold_flag)`` with each term counted only when the corresponding
    switch in ``cfg["features"]`` is enabled.

    Args:
        d: Raw per-step feature dimension.
        cfg: Full project config; uses ``cfg["features"]``.

    Returns:
        ``d'`` — matches the last axis of :func:`engineer_features` output.

    Raises:
        ValueError: If ``d`` or the configured lag count is negative.
    """
    if d < 0:
        raise ValueError(f"engineered_dim: d must be >= 0, got {d}")
    fcfg = cfg.get("features", {})
    lags = int(fcfg.get("lags", 0) or 0)
    if lags < 0:
        raise ValueError(f"engineered_dim: cfg features.lags must be >= 0, got {lags}")
    blocks = 1 + lags  # raw + lag channels
    if fcfg.get("rolling", False):
        blocks += 2  # rolling mean + rolling std
    if fcfg.get("delta", False):
        blocks += 1  # rate of change
    if fcfg.get("utilization_ratio", False):
        blocks += 1  # utilisation ratio
    if fcfg.get("threshold_flag", False):
        blocks += 1  # threshold flags
    return d * blocks


def engineer_features(
    windows: np.ndarray,
    cfg: dict,
    limits: np.ndarray | None = None,
) -> np.ndarray:
    """Append Step-3 engineered channels to each window, fully vectorised.

    Args:
        windows: Scaled windows ``(M, w, d)`` in ``[0, 1]`` (output of paper
            Step 2, :func:`selfheal.data.windowing.make_windows`).
        cfg: Full project config; switches and ``tau`` from
            ``cfg["features"]``.
        limits: Per-feature limits vector ``(d,)`` in the same (scaled) space
            as ``windows`` — normally built by :func:`limits_vector`.  When
            ``None``, every limit defaults to 1.0 (i.e. the full scaled
            range), so the ratio equals the raw scaled value.

    Returns:
        ``F`` of shape ``(M, w, d')`` float32 with ``d' =``
        :func:`engineered_dim`, channels ordered: raw, lags ``l=1..L`` (each
        lag shifts the series by ``l`` within the window, padding with the
        window's first value), per-window mean and std broadcast across ``w``,
        per-step delta ``x_t - x_{t-1}`` (first step 0), utilisation ratio
        ``r = x / limit`` and threshold flags ``(r > tau)`` as float.

    Raises:
        ValueError: If ``windows`` is not 3-D, has zero window length, or
            ``limits`` does not have ``d`` entries.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(
            f"engineer_features: windows must be (M, w, d), got shape {windows.shape}"
        )
    m, w, d = windows.shape
    if w < 1:
        raise ValueError("engineer_features: window length w must be >= 1")

    fcfg = cfg.get("features", {})
    lags = int(fcfg.get("lags", 0) or 0)
    if lags < 0:
        raise ValueError(f"engineer_features: cfg features.lags must be >= 0, got {lags}")
    use_rolling = bool(fcfg.get("rolling", False))
    use_delta = bool(fcfg.get("delta", False))
    use_ratio = bool(fcfg.get("utilization_ratio", False))
    use_flag = bool(fcfg.get("threshold_flag", False))
    tau = float(fcfg.get("tau", 0.9))

    blocks: list[np.ndarray] = [windows]  # a. raw channels first (SPEC order)

    # a. lag channels: series shifted by l within the window, padded with the
    #    window's first value -> index max(t - l, 0) along the time axis.
    if lags > 0:
        t_idx = np.arange(w)
        for lag in range(1, lags + 1):
            blocks.append(windows[:, np.maximum(t_idx - lag, 0), :])

    # b. rolling stats: per-window mean/std per feature, broadcast across w.
    if use_rolling:
        mean = windows.mean(axis=1, keepdims=True, dtype=np.float32)
        std = windows.std(axis=1, keepdims=True).astype(np.float32)
        blocks.append(np.broadcast_to(mean, windows.shape))
        blocks.append(np.broadcast_to(std, windows.shape))

    # c. rate of change: per-step delta, first step 0.
    if use_delta:
        delta = np.zeros_like(windows)
        delta[:, 1:, :] = windows[:, 1:, :] - windows[:, :-1, :]
        blocks.append(delta)

    # d./e. utilisation ratio and threshold flags share r = x / limit.
    if use_ratio or use_flag:
        if limits is None:
            lim = np.ones(d, dtype=np.float32)
        else:
            lim = np.asarray(limits, dtype=np.float32).reshape(-1)
            if lim.shape[0] != d:
                raise ValueError(
                    f"engineer_features: limits has {lim.shape[0]} entries, "
                    f"expected d={d}"
                )
        lim = np.maximum(lim, np.float32(_MIN_LIMIT))  # guard div-by-zero
        ratio = (windows / lim).astype(np.float32)
        if use_ratio:
            blocks.append(ratio)
        if use_flag:
            blocks.append((ratio > tau).astype(np.float32))

    if len(blocks) == 1:
        engineered = windows.copy()
    else:
        engineered = np.concatenate(blocks, axis=2, dtype=np.float32)

    expected = engineered_dim(d, cfg)
    if engineered.shape[2] != expected:  # pragma: no cover - internal invariant
        raise RuntimeError(
            f"engineer_features: produced d'={engineered.shape[2]} but "
            f"engineered_dim says {expected}"
        )
    return engineered


def limits_vector(
    feature_cols: list[str],
    cfg: dict,
    scaler: "Scaler | None" = None,
) -> np.ndarray | None:
    """Per-feature limits aligned with ``feature_cols`` (Step 3, channel d).

    Builds the limits array from ``cfg["features"]["limits"]`` (raw resource
    limits, e.g. ``cpu_usage: 100.0``).  When a fitted
    :class:`~selfheal.data.preprocessing.Scaler` is given, each configured
    limit is mapped into the scaled space the windows live in:

        ``limit_scaled = (limit - data_min) / (data_max - data_min)``

    clipped to ``[1e-6, inf)`` (a zero data range falls back to a divisor of
    1.0, mirroring sklearn's MinMaxScaler handling).  Features without a
    configured limit get limit 1.0 — the full scaled range.

    Args:
        feature_cols: Canonical feature order of the window channels.
        cfg: Full project config; uses ``cfg["features"]``.
        scaler: Optional fitted scaler used to translate raw limits into the
            scaled space.  Must have been fitted on (at least) every feature
            that has a configured limit.

    Returns:
        Float32 array of shape ``(len(feature_cols),)``, or ``None`` when
        ``cfg["features"]["utilization_ratio"]`` is off.

    Raises:
        ValueError: If a feature with a configured limit is missing from the
            fitted ``scaler``.
        RuntimeError: If ``scaler`` is provided but not fitted.
    """
    fcfg = cfg.get("features", {})
    if not fcfg.get("utilization_ratio", False):
        return None
    limits_cfg: dict = fcfg.get("limits") or {}

    out = np.ones(len(feature_cols), dtype=np.float32)
    if scaler is not None:
        data_min = scaler.data_min_  # raises RuntimeError if not fitted
        data_max = scaler.data_max_
        scaler_index = {col: i for i, col in enumerate(scaler.feature_cols)}

    for i, col in enumerate(feature_cols):
        if col not in limits_cfg:
            continue  # unconfigured feature -> limit 1.0 (scaled data)
        raw_limit = float(limits_cfg[col])
        if scaler is None:
            value = raw_limit
        else:
            if col not in scaler_index:
                raise ValueError(
                    f"limits_vector: feature {col!r} has a configured limit but "
                    "the scaler was not fitted on it"
                )
            j = scaler_index[col]
            data_range = data_max[j] - data_min[j]
            if data_range <= 0.0:
                data_range = 1.0  # constant feature: mirror sklearn's fallback
            value = (raw_limit - data_min[j]) / data_range
        out[i] = max(value, _MIN_LIMIT)
    return out
