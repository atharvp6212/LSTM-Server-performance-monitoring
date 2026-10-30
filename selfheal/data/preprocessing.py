"""Paper Step 1 — data cleaning and Min-Max normalisation.

Implements the "Data Collection and Preprocessing" stage (methodology Step 1)
of: Atharv S. Patil, Sanjay S. Patil (2025). "Self-Healing Cloud Platforms
Using LSTM-based Predictive Models for Early Fault Detection." IJCET 16(5),
177-187, DOI: 10.34218/IJCET_16_05_012.

The paper normalises all telemetry metrics with Min-Max scaling so that every
feature lies in ``[0, 1]`` before the sliding-window step (Step 2,
:mod:`selfheal.data.windowing`).  This module provides:

* :class:`Scaler` — a persistable wrapper around
  :class:`sklearn.preprocessing.MinMaxScaler` restricted to the feature
  columns of a canonical telemetry frame.
* :func:`clean` — deterministic cleaning (sorting, de-duplication, inf/NaN
  imputation, clipping) per ``cfg["preprocessing"]``.
* :func:`preprocess` — the full Step-1 pipeline: ``clean`` then scale.

A *canonical telemetry frame* (SPEC "Canonical data contract") is a
``pd.DataFrame`` with a ``timestamp`` column, N numeric feature columns and an
int 0/1 ``anomaly`` column; the feature columns are every column except
``timestamp`` and ``anomaly``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#: Columns of a canonical telemetry frame that are never treated as features.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("timestamp", "anomaly")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns of a canonical frame (everything but timestamp/anomaly)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


class Scaler:
    """Min-Max scaler over the feature columns only (paper Step 1).

    Wraps :class:`sklearn.preprocessing.MinMaxScaler` and remembers which
    columns it was fitted on, so the exact same transformation can be applied
    at inference time (paper Step 5, the live predictor) after being persisted
    with :meth:`save` / :meth:`load`.

    Attributes:
        feature_cols: Ordered list of the feature columns the scaler was
            fitted on.  Empty until :meth:`fit` is called.
    """

    def __init__(self) -> None:
        self.feature_cols: list[str] = []
        self._scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Scaler":
        """Fit the Min-Max scaler on ``df[feature_cols]``.

        Args:
            df: Canonical telemetry frame (typically the output of
                :func:`clean`).
            feature_cols: Columns to scale; stored on the instance.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: If ``df`` has no rows or ``feature_cols`` is empty.
            KeyError: If any of ``feature_cols`` is missing from ``df``.
        """
        if not feature_cols:
            raise ValueError("Scaler.fit requires at least one feature column")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Scaler.fit: frame is missing feature columns {missing}")
        if len(df) == 0:
            raise ValueError("Scaler.fit: cannot fit on an empty frame")
        values = df[list(feature_cols)].to_numpy(dtype=np.float64)
        self._scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(values)
        self.feature_cols = list(feature_cols)
        return self

    # ------------------------------------------------------------ transforms
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the feature columns scaled to [0, 1].

        Values outside the fitted data range are clipped to ``[0, 1]`` so the
        downstream window contract (windows ``(M, w, d)`` in ``[0, 1]``,
        SPEC/paper Step 2-3) always holds, even for live telemetry that
        exceeds the training range.  Non-feature columns are left untouched.
        """
        scaler = self._require_fitted()
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Scaler.transform: frame is missing feature columns {missing}")
        out = df.copy()
        if len(out) == 0:
            return out
        values = out[self.feature_cols].to_numpy(dtype=np.float64)
        scaled = np.clip(scaler.transform(values), 0.0, 1.0)
        out[self.feature_cols] = scaled
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the feature columns mapped back to raw units."""
        scaler = self._require_fitted()
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Scaler.inverse_transform: frame is missing feature columns {missing}"
            )
        out = df.copy()
        if len(out) == 0:
            return out
        values = out[self.feature_cols].to_numpy(dtype=np.float64)
        out[self.feature_cols] = scaler.inverse_transform(values)
        return out

    # -------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler and its feature columns with joblib."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"feature_cols": self.feature_cols, "scaler": self._scaler, "format": 1},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        """Load a scaler previously written by :meth:`save`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not a valid Scaler payload.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler.load: no such file: {path}")
        payload = joblib.load(path)
        if (
            not isinstance(payload, dict)
            or "scaler" not in payload
            or "feature_cols" not in payload
        ):
            raise ValueError(f"Scaler.load: {path} is not a saved Scaler payload")
        inst = cls()
        inst._scaler = payload["scaler"]
        inst.feature_cols = list(payload["feature_cols"])
        return inst

    # -------------------------------------------------------- introspection
    @property
    def data_min_(self) -> np.ndarray:
        """Per-feature minimum seen at fit time, aligned with ``feature_cols``."""
        return np.asarray(self._require_fitted().data_min_, dtype=np.float64)

    @property
    def data_max_(self) -> np.ndarray:
        """Per-feature maximum seen at fit time, aligned with ``feature_cols``."""
        return np.asarray(self._require_fitted().data_max_, dtype=np.float64)

    def _require_fitted(self) -> MinMaxScaler:
        if self._scaler is None:
            raise RuntimeError("Scaler is not fitted; call fit() or load() first")
        return self._scaler


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean a canonical telemetry frame (paper Step 1, cleaning half).

    Performs, in order:

    1. sort ascending by ``timestamp`` (stable), dropping rows whose
       timestamp is missing (NaT cannot be ordered);
    2. drop duplicate timestamps, keeping the first occurrence;
    3. coerce feature columns to numeric and replace ``±inf`` with ``NaN``;
    4. impute per ``cfg["preprocessing"]["impute_strategy"]``:
       ``"ffill"`` → ``ffill().bfill()``, ``"mean"`` → column-mean fill,
       ``"drop"`` → drop rows with any missing feature value; any feature
       value still missing afterwards (e.g. an all-NaN column) is set to 0.0;
    5. clip features to the finite ``float32`` range (model inputs are
       float32 per SPEC).

    The ``anomaly`` column is never dropped; if present it is coerced to an
    int 0/1 column (missing labels become 0).

    Args:
        df: Input frame (any row order, may contain inf/NaN/duplicates).
        cfg: Full project config; uses ``cfg["preprocessing"]``.

    Returns:
        A cleaned copy with a fresh ``RangeIndex``.

    Raises:
        ValueError: For an unknown ``impute_strategy``.
    """
    strategy = str(cfg.get("preprocessing", {}).get("impute_strategy", "ffill")).lower()
    if strategy not in ("ffill", "mean", "drop"):
        raise ValueError(
            f"clean: unknown impute_strategy {strategy!r} (expected ffill|mean|drop)"
        )

    out = df.copy()
    if "timestamp" in out.columns:
        out = out[out["timestamp"].notna()]
        out = out.sort_values("timestamp", kind="stable")
        out = out.drop_duplicates(subset="timestamp", keep="first")
    out = out.reset_index(drop=True)

    feat_cols = _feature_columns(out)
    if feat_cols:
        out[feat_cols] = out[feat_cols].apply(pd.to_numeric, errors="coerce")
        out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)

        if strategy == "ffill":
            out[feat_cols] = out[feat_cols].ffill().bfill()
        elif strategy == "mean":
            out[feat_cols] = out[feat_cols].fillna(out[feat_cols].mean())
        else:  # "drop"
            out = out.dropna(subset=feat_cols).reset_index(drop=True)

        # Safety net for columns that are entirely NaN (ffill/bfill and mean
        # cannot recover those): fall back to a neutral 0.0.
        out[feat_cols] = out[feat_cols].fillna(0.0)

        finfo = np.finfo(np.float32)
        out[feat_cols] = (
            out[feat_cols]
            .clip(lower=float(finfo.min), upper=float(finfo.max))
            .astype(np.float64)
        )

    if "anomaly" in out.columns:
        anomaly = pd.to_numeric(out["anomaly"], errors="coerce").fillna(0)
        out["anomaly"] = anomaly.astype(np.int64)

    return out


def preprocess(
    df: pd.DataFrame,
    cfg: dict,
    scaler: Scaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, Scaler]:
    """Full paper Step 1: :func:`clean` then Min-Max scale the features.

    Args:
        df: Raw canonical telemetry frame.
        cfg: Full project config (``cfg["preprocessing"]``).
        scaler: A previously fitted :class:`Scaler`; required when
            ``fit=False`` (inference path with the persisted training scaler).
        fit: ``True`` fits a fresh :class:`Scaler` on the cleaned frame
            (training path); ``False`` reuses ``scaler``.

    Returns:
        ``(scaled_df, scaler)`` — the cleaned frame with feature columns in
        ``[0, 1]`` and the scaler that produced it.

    Raises:
        ValueError: If ``fit=False`` and ``scaler`` is ``None``, or the
            cleaned frame is empty when fitting.
    """
    cleaned = clean(df, cfg)
    if fit:
        scaler = Scaler().fit(cleaned, _feature_columns(cleaned))
    elif scaler is None:
        raise ValueError("preprocess: fit=False requires a fitted `scaler`")
    return scaler.transform(cleaned), scaler
