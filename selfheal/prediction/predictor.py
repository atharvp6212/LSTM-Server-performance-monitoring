"""Fault prediction: thresholding and the decision rule A_k.

Implements paper Step 5 ("Prediction and Decision Making") of *Self-Healing
Cloud Platforms Using LSTM-based Predictive Models for Early Fault Detection*
(IJCET 16(5) 2025): the trained LSTM emits a fault probability ŷ ∈ [0, 1] for
the most recent telemetry window; the decision rule compares ŷ against the
fault threshold T (``cfg["decision"]["fault_threshold"]``) and maps it onto an
escalating severity band (``cfg["decision"]["severity_bands"]``) that Step 6
translates into a healing action.

Only the public names from SPEC.md are exported: :class:`Decision` and
:class:`FaultPredictor`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selfheal.config import resolve_path
from selfheal.data.features import engineer_features, limits_vector
from selfheal.data.preprocessing import Scaler

__all__ = ["Decision", "FaultPredictor"]


@dataclass
class Decision:
    """Outcome of the paper's Step 5 decision rule for one prediction.

    Attributes:
        probability: model output ŷ ∈ [0, 1] (fault likelihood within horizon).
        trigger: ``probability >= cfg["decision"]["fault_threshold"]`` (T).
        severity: ``"none" | "alert" | "scale" | "restart" | "drain"`` — the
            highest configured band whose threshold is <= ``probability``.
    """

    probability: float
    trigger: bool
    severity: str


class FaultPredictor:
    """Runs the trained LSTM on raw telemetry windows and applies the rule A_k.

    Paper Step 5: loads the persisted artifacts of Step 4 (``model.keras``,
    ``scaler.joblib``, ``meta.json``), replays the exact Step 1–3 pipeline
    (Min-Max scaling, sliding window, feature engineering) on an incoming raw
    window, and converts the resulting probability into a :class:`Decision`.

    Attributes:
        window_size: sliding-window length w the model was trained with.
        feature_cols: canonical feature column order the model expects.
    """

    def __init__(
        self,
        model: Any,
        scaler: Scaler,
        cfg: dict,
        feature_cols: list[str],
        window_size: int,
        limits: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        """Build a predictor from already-loaded components.

        ``model`` is duck-typed (needs only ``predict``) so that unit tests can
        exercise :meth:`decide` without loading a real Keras model.
        """
        self.model = model
        self.scaler = scaler
        self.cfg = cfg
        self.feature_cols: list[str] = list(feature_cols)
        self.window_size: int = int(window_size)
        self.limits = limits
        self.meta: dict = dict(meta or {})

    @classmethod
    def from_artifacts(cls, cfg: dict) -> "FaultPredictor":
        """Load ``model.keras``, ``scaler.joblib`` and ``meta.json``.

        Artifacts are read from ``cfg["paths"]["artifacts"]`` (written by the
        Step 4 training pipeline, see SPEC selfheal/models/train.py). The
        utilization-ratio limits (paper Step 3d) are rebuilt in scaled space
        with the *saved* scaler so inference matches training exactly.

        Raises:
            FileNotFoundError: if any of the three artifact files is missing.
        """
        artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
        model_path = artifacts_dir / "model.keras"
        scaler_path = artifacts_dir / "scaler.joblib"
        meta_path = artifacts_dir / "meta.json"
        for path in (model_path, scaler_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing artifact {path}; run scripts/train_model.py first."
                )

        # Lazy TF import: keeps module import light and silences TF chatter.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras  # noqa: PLC0415  (heavyweight, loaded on demand)

        model = keras.saving.load_model(model_path)
        scaler = Scaler.load(scaler_path)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        feature_cols: list[str] = list(meta["feature_cols"])
        window_size = int(meta["window_size"])
        limits = limits_vector(feature_cols, cfg, scaler)
        return cls(
            model=model,
            scaler=scaler,
            cfg=cfg,
            feature_cols=feature_cols,
            window_size=window_size,
            limits=limits,
            meta=meta,
        )

    def predict_proba(self, raw_window: pd.DataFrame) -> float:
        """Predict the fault probability ŷ for the latest raw telemetry.

        Paper Steps 1–5 applied at inference time: ``raw_window`` is a
        canonical (unscaled) frame slice with at least ``window_size`` rows.
        The saved Min-Max scaler is applied (Step 1), the LAST ``window_size``
        rows form the sliding window X_k (Step 2), features are engineered
        with the saved limits (Step 3), and the LSTM produces ŷ (Steps 4–5).

        Returns:
            float in [0, 1].

        Raises:
            ValueError: if ``raw_window`` is too short or lacks feature columns.
        """
        if raw_window is None or len(raw_window) < self.window_size:
            got = 0 if raw_window is None else len(raw_window)
            raise ValueError(
                f"raw_window needs >= window_size ({self.window_size}) rows, got {got}"
            )
        missing = [c for c in self.feature_cols if c not in raw_window.columns]
        if missing:
            raise ValueError(f"raw_window is missing feature columns: {missing}")

        scaled = self.scaler.transform(raw_window)
        tail = scaled.iloc[-self.window_size :]
        window = tail[self.feature_cols].to_numpy(dtype=np.float32)  # (w, d)
        windows = window[np.newaxis, :, :]  # (1, w, d)
        engineered = engineer_features(windows, self.cfg, limits=self.limits)
        raw_out = self.model.predict(engineered, verbose=0)
        prob = float(np.asarray(raw_out, dtype=np.float64).reshape(-1)[-1])
        if not math.isfinite(prob):
            return 0.0
        return float(min(1.0, max(0.0, prob)))

    def decide(self, probability: float) -> Decision:
        """Apply the paper's Step 5 decision rule A_k to a probability.

        ``trigger`` is ``probability >= cfg["decision"]["fault_threshold"]``
        (inclusive at the exact threshold). ``severity`` is the *highest* band
        in ``cfg["decision"]["severity_bands"]`` whose threshold is <=
        ``probability`` (again inclusive), or ``"none"`` when the probability
        falls below every band. A non-finite probability never triggers.
        """
        p = float(probability)
        decision_cfg = self.cfg.get("decision", {}) or {}
        if not math.isfinite(p):
            return Decision(probability=p, trigger=False, severity="none")

        fault_threshold = float(decision_cfg.get("fault_threshold", 0.7))
        bands: dict[str, float] = decision_cfg.get("severity_bands", {}) or {}
        severity = "none"
        # Sorted ascending by threshold: the last qualifying band is the
        # highest one — deterministic even if cfg ordering changes.
        for name, threshold in sorted(bands.items(), key=lambda kv: float(kv[1])):
            if float(threshold) <= p:
                severity = str(name)
        return Decision(probability=p, trigger=p >= fault_threshold, severity=severity)
