"""Evaluation of the trained LSTM fault-prediction model.

Implements the paper's §4 evaluation (Patil & Patil 2025, IJCET 16(5)):
accuracy, precision, recall, F1 and AUC on a held-out chronological test
split, plus the diagnostic figures — confusion matrix, ROC curve, training
history and a predicted-probability vs true-anomaly timeline that visualises
the "early fault detection" behaviour of paper Steps 4–5.

The pipeline is rebuilt exactly as trained: the SAVED Min-Max scaler is
re-applied (Step 1, ``fit=False``), windows and engineered features use the
window size / horizon / feature order stored in ``meta.json`` (Steps 2–3),
and the split boundaries match :func:`selfheal.models.train.chronological_split`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from selfheal.config import resolve_path, set_global_seeds
from selfheal.models.train import chronological_split

_DPI = 150


def _binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Paper §4 metrics (accuracy/precision/recall/F1/AUC + confusion counts).

    Division-by-zero cases return 0.0; AUC is ``None`` when the test labels
    contain only one class.
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


def _plot_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    """Confusion-matrix heatmap for the test split (paper §4)."""
    cm = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    high = cm.max() / 2 if cm.max() > 0 else 0.5
    for (row, col), value in np.ndenumerate(cm):
        ax.text(
            col,
            row,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=13,
            color="white" if value > high else "black",
        )
    ax.set_xticks([0, 1], labels=["Normal (0)", "Fault (1)"])
    ax.set_yticks([0, 1], labels=["Normal (0)", "Fault (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Chronological Test Split")
    fig.colorbar(image, ax=ax, label="Window count")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auc: float | None, path: Path
) -> None:
    """ROC curve on the test split (paper §4 AUC metric)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    if auc is not None:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color="tab:blue", label=f"LSTM (AUC = {auc:.3f})")
    else:
        ax.text(
            0.5,
            0.55,
            "ROC undefined:\nsingle class in test split",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve — Chronological Test Split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    """Loss and AUC learning curves from ``history.json`` (paper Step 4)."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    if not history:
        for ax in (ax_loss, ax_auc):
            ax.text(
                0.5,
                0.5,
                "history.json not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    for key, label, style in (
        ("loss", "train", "-"),
        ("val_loss", "validation", "--"),
    ):
        values = history.get(key)
        if values:
            ax_loss.plot(range(1, len(values) + 1), values, style, label=label)
    for key, label, style in (("auc", "train", "-"), ("val_auc", "validation", "--")):
        values = history.get(key)
        if values:
            ax_auc.plot(range(1, len(values) + 1), values, style, label=label)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary cross-entropy loss")
    ax_loss.set_title("Training Loss")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_title("Training AUC")
    for ax in (ax_loss, ax_auc):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle("LSTM Training History")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_probability_timeline(
    x: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    """Predicted fault probability vs true anomaly over the test span.

    Visualises the paper's early-detection claim (Steps 4–5): the predicted
    probability should rise above the decision threshold T while the fault is
    still building up, ahead of the labelled anomaly windows.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(
        x,
        0.0,
        np.asarray(y_true, dtype=float),
        step="post",
        color="tab:red",
        alpha=0.30,
        linewidth=0,
        label="True anomaly (0/1)",
    )
    ax.plot(
        x,
        np.asarray(y_prob, dtype=float),
        color="tab:blue",
        linewidth=1.2,
        label="Predicted fault probability",
    )
    ax.axhline(
        threshold,
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label=f"Decision threshold T = {threshold:.2f}",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (chronological test split)")
    ax.set_ylabel("Fault probability / anomaly label")
    ax.set_title("Predicted Fault Probability vs True Anomaly — Test Split")
    ax.legend(loc="upper left")
    if not np.issubdtype(np.asarray(x).dtype, np.number):
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _test_time_axis(
    df: "Any", cfg: dict, first_test_window: int, n_test: int
) -> np.ndarray:
    """Timestamps of each test window's last observation (prediction time).

    Window ``k`` (Step 2, stride ``s``) covers rows ``[k*s, k*s + w - 1]``, so
    its prediction is anchored at the timestamp of row ``k*s + w - 1``.  Falls
    back to a plain window index when timestamps cannot be aligned.
    """
    windowing = cfg.get("windowing", {})
    window_size = int(windowing.get("window_size", 30))
    stride = int(windowing.get("stride", 1))
    indices = (first_test_window + np.arange(n_test)) * stride + window_size - 1
    try:
        timestamps = df["timestamp"].to_numpy()
        if n_test > 0 and indices[-1] < len(timestamps):
            return timestamps[indices]
    except (KeyError, TypeError):
        pass
    return np.arange(n_test)


def evaluate_model(cfg: dict) -> dict:
    """Evaluate the persisted model on the chronological test split (paper §4).

    Loads ``model.keras``, ``scaler.joblib`` and ``meta.json`` from
    ``cfg["paths"]["artifacts"]``, rebuilds the exact training pipeline with
    the saved scaler (``fit=False``) and the saved window/feature settings,
    and evaluates on the same held-out chronological test split used by
    :func:`selfheal.models.train.train_model`.

    Writes into ``cfg["paths"]["reports"]``: ``metrics.json``,
    ``confusion_matrix.png``, ``roc_curve.png``, ``training_history.png`` and
    ``probability_timeline.png`` (predicted probability and true anomaly on
    the same time axis).

    Returns the metrics dict (accuracy, precision, recall, f1, auc,
    tn/fp/fn/tp, threshold).
    """
    set_global_seeds(cfg)

    # Heavy / concurrently-owned imports stay local (see SPEC cross-cutting rules).
    import keras

    from selfheal.data.dataset import load_dataset
    from selfheal.data.features import engineer_features, limits_vector
    from selfheal.data.preprocessing import Scaler, preprocess
    from selfheal.data.windowing import window_dataframe

    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    model_path = artifacts_dir / "model.keras"
    scaler_path = artifacts_dir / "scaler.joblib"
    meta_path = artifacts_dir / "meta.json"
    for required in (model_path, scaler_path, meta_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing training artifact: {required}. "
                "Run `python scripts/train_model.py` first."
            )

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    scaler = Scaler.load(scaler_path)
    model = keras.saving.load_model(model_path)

    # Rebuild the SAME pipeline the model was trained with (meta-driven).
    eval_cfg = copy.deepcopy(cfg)
    if meta.get("dataset_source"):
        eval_cfg.setdefault("dataset", {})["source"] = meta["dataset_source"]
    windowing = eval_cfg.setdefault("windowing", {})
    windowing["window_size"] = int(meta["window_size"])
    windowing["horizon"] = int(meta["horizon"])
    if "stride" in meta:
        windowing["stride"] = int(meta["stride"])

    df = load_dataset(eval_cfg)
    if df is None or len(df) == 0:
        raise ValueError("load_dataset returned an empty frame; nothing to evaluate.")

    # Step 1 with the SAVED scaler (fit=False) — identical scaling to training.
    df_scaled, _ = preprocess(df, eval_cfg, scaler=scaler, fit=False)
    feature_cols = list(meta.get("feature_cols", []))
    missing = [c for c in feature_cols if c not in df_scaled.columns]
    if not feature_cols or missing:
        raise ValueError(
            f"Dataset is missing trained feature columns {missing or feature_cols}; "
            "was it regenerated with a different schema?"
        )

    # Steps 2–3 with the saved window size / horizon / limits.
    X, y = window_dataframe(df_scaled, eval_cfg, feature_cols)
    if len(X) == 0:
        raise ValueError(
            "Windowing produced 0 windows — dataset shorter than "
            "window_size + horizon."
        )
    limits = limits_vector(feature_cols, eval_cfg, scaler)
    F = np.asarray(engineer_features(X, eval_cfg, limits), dtype=np.float32)
    if int(F.shape[2]) != int(meta["engineered_dim"]):
        raise ValueError(
            f"Engineered feature dim {F.shape[2]} != trained dim "
            f"{meta['engineered_dim']}; the features config drifted since training."
        )
    y = np.asarray(y).reshape(-1).astype(int)

    F_tr, y_tr, F_va, y_va, F_te, y_te = chronological_split(F, y)
    if len(F_te) == 0:
        raise ValueError("Chronological test split is empty; not enough windows.")

    y_prob = model.predict(
        F_te,
        batch_size=int(cfg.get("model", {}).get("batch_size", 64)),
        verbose=0,
    ).reshape(-1)
    threshold = float(
        meta.get("threshold", cfg.get("decision", {}).get("fault_threshold", 0.5))
    )
    metrics = _binary_metrics(y_te, y_prob, threshold)
    metrics["threshold"] = threshold

    reports_dir = resolve_path(cfg, cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _plot_confusion_matrix(metrics, reports_dir / "confusion_matrix.png")
    _plot_roc_curve(y_te, y_prob, metrics["auc"], reports_dir / "roc_curve.png")

    history: dict[str, list[float]] = {}
    history_path = artifacts_dir / "history.json"
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError):
            history = {}
    _plot_training_history(history, reports_dir / "training_history.png")

    x = _test_time_axis(df_scaled, eval_cfg, len(F_tr) + len(F_va), len(F_te))
    _plot_probability_timeline(
        x, y_prob, y_te, threshold, reports_dir / "probability_timeline.png"
    )

    return metrics
