"""Training pipeline for the LSTM early-fault-detection model.

Implements paper Step 4 (model training) of Patil & Patil (2025), IJCET 16(5),
on top of Steps 1–3: load the telemetry dataset (§3.1), clean and Min-Max
normalise it (Step 1), cut sliding windows with a look-ahead fault horizon
(Step 2), engineer sequence features (Step 3), then train the
``Input -> LSTM -> Dropout -> Dense -> Dense(sigmoid)`` network with binary
cross-entropy (Step 4 / Figure 1).

The split is strictly chronological (no shuffling across split boundaries) so
that the evaluation reflects genuine early prediction rather than look-ahead
leakage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from selfheal.config import resolve_path, set_global_seeds


def chronological_split(
    F: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple:
    """Time-ordered train/validation/test split (paper Step 4 setup).

    The window arrays are already in time order (Step 2 sliding windows), so
    slicing by position preserves chronology: no window from the future leaks
    into an earlier split.

    Parameters
    ----------
    F: ``(M, w, d')`` engineered windows.  y: ``(M,)`` labels.
    train_frac / val_frac: fractions of ``M`` (the remainder is the test set).

    Returns
    -------
    ``(F_tr, y_tr, F_va, y_va, F_te, y_te)``
    """
    if len(F) != len(y):
        raise ValueError(f"F and y disagree in length: {len(F)} vs {len(y)}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if val_frac < 0.0 or train_frac + val_frac > 1.0:
        raise ValueError(
            f"val_frac must be >= 0 with train_frac + val_frac <= 1, "
            f"got train_frac={train_frac}, val_frac={val_frac}"
        )
    n = len(F)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return (
        F[:n_tr],
        y[:n_tr],
        F[n_tr : n_tr + n_va],
        y[n_tr : n_tr + n_va],
        F[n_tr + n_va :],
        y[n_tr + n_va :],
    )


def _binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Binary classification metrics at the paper's decision threshold T.

    Covers the paper §4 evaluation metrics (accuracy, precision, recall, F1,
    AUC) plus confusion-matrix counts.  All ratios guard against division by
    zero (empty class -> 0.0); AUC is ``None`` when the test labels contain a
    single class (undefined).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    auc: float | None = None
    if np.unique(y_true).size == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        ),
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_model(cfg: dict) -> dict:
    """Run the full paper pipeline (Steps 1–4) and persist the artifacts.

    Pipeline: ``load_dataset -> preprocess(fit=True) -> window_dataframe ->
    limits_vector -> engineer_features -> chronological_split ->
    build_lstm_model -> fit`` with early stopping (monitor ``val_auc``, mode
    max, best weights restored) and, when ``cfg["model"]["class_weight"] ==
    "balanced"``, class weights that counter the fault/normal imbalance.

    Saves into ``cfg["paths"]["artifacts"]``: ``model.keras``,
    ``scaler.joblib``, ``meta.json`` and ``history.json``.

    Returns
    -------
    ``{"history": <fit history lists>, "test": <metrics dict>,
    "artifacts_dir": str}`` where the test metrics (accuracy, precision,
    recall, f1, auc, tn/fp/fn/tp) come from the held-out chronological test
    split.
    """
    set_global_seeds(cfg)

    # Heavy / concurrently-owned imports are local so that lightweight users
    # of chronological_split never pull in TensorFlow or the data stack.
    import keras

    from selfheal.data.dataset import feature_columns, load_dataset
    from selfheal.data.features import engineer_features, limits_vector
    from selfheal.data.preprocessing import preprocess
    from selfheal.data.windowing import window_dataframe
    from selfheal.models.lstm_model import build_lstm_model

    df = load_dataset(cfg)
    if df is None or len(df) == 0:
        raise ValueError("load_dataset returned an empty frame; nothing to train on.")

    # Paper Step 1: cleaning + Min-Max normalisation (scaler fitted on this run).
    df_scaled, scaler = preprocess(df, cfg, fit=True)
    feature_cols = feature_columns(df_scaled)
    if not feature_cols:
        raise ValueError("Dataset has no feature columns besides timestamp/anomaly.")

    # Paper Step 2: sliding windows with look-ahead horizon labels.
    X, y = window_dataframe(df_scaled, cfg, feature_cols)
    if len(X) == 0:
        raise ValueError(
            "Windowing produced 0 windows — dataset shorter than "
            "window_size + horizon; generate more records or shrink the window."
        )

    # Paper Step 3: engineered sequence features F_k in R^(w x d').
    limits = limits_vector(feature_cols, cfg, scaler)
    F = np.asarray(engineer_features(X, cfg, limits), dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)

    F_tr, y_tr, F_va, y_va, F_te, y_te = chronological_split(F, y)
    if len(F_tr) == 0:
        raise ValueError("Chronological train split is empty; not enough windows.")
    if len(F_te) == 0:
        raise ValueError(
            "Chronological test split is empty; lower train/val fractions or "
            "provide more data."
        )

    # Paper Step 4: build and fit the LSTM classifier.
    model_cfg = cfg.get("model", {})
    model = build_lstm_model((int(F.shape[1]), int(F.shape[2])), cfg)

    has_val = len(F_va) > 0
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc" if has_val else "auc",
        mode="max",
        patience=int(model_cfg.get("early_stopping_patience", 5)),
        restore_best_weights=True,
    )

    class_weight: dict[int, float] | None = None
    if str(model_cfg.get("class_weight", "")) == "balanced":
        classes = np.unique(y_tr)
        if classes.size > 1:
            from sklearn.utils.class_weight import compute_class_weight

            weights = compute_class_weight("balanced", classes=classes, y=y_tr)
            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    fit_history = model.fit(
        F_tr,
        y_tr.astype(np.float32),
        validation_data=(F_va, y_va.astype(np.float32)) if has_val else None,
        epochs=int(model_cfg.get("epochs", 30)),
        batch_size=int(model_cfg.get("batch_size", 64)),
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )
    history = {
        key: [float(v) for v in values]
        for key, values in fit_history.history.items()
    }

    # Held-out test metrics at the paper's decision threshold T (Step 5).
    threshold = float(cfg.get("decision", {}).get("fault_threshold", 0.5))
    y_prob = model.predict(
        F_te, batch_size=int(model_cfg.get("batch_size", 64)), verbose=0
    ).reshape(-1)
    test_metrics = _binary_metrics(y_te, y_prob, threshold)

    # Persist artifacts for the predictor (Step 5) and evaluator.
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifacts_dir / "model.keras")
    scaler.save(artifacts_dir / "scaler.joblib")
    with open(artifacts_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    meta = {
        "feature_cols": list(feature_cols),
        "window_size": int(F.shape[1]),
        "horizon": int(cfg.get("windowing", {}).get("horizon", 1)),
        "stride": int(cfg.get("windowing", {}).get("stride", 1)),
        "engineered_dim": int(F.shape[2]),
        "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(cfg.get("dataset", {}).get("source", "synthetic")),
        "class_balance": {
            "0": int(np.sum(y == 0)),
            "1": int(np.sum(y == 1)),
        },
    }
    with open(artifacts_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "history": history,
        "test": test_metrics,
        "artifacts_dir": str(artifacts_dir),
    }
