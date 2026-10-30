"""LSTM fault-prediction network architecture.

Implements paper Step 4 / Figure 1 of Patil & Patil (2025), "Self-Healing
Cloud Platforms Using LSTM-based Predictive Models for Early Fault Detection"
(IJCET 16(5)): ``Input -> LSTM -> Dropout -> Dense(ReLU) -> Dense(1, sigmoid)``
trained with binary cross-entropy loss to output the fault probability
``y_hat in [0, 1]`` for an engineered window ``F_k``.
"""

from __future__ import annotations

import keras


def build_lstm_model(input_shape: tuple[int, int], cfg: dict) -> keras.Model:
    """Build and compile the paper's LSTM classifier (Step 4 / Figure 1).

    Parameters
    ----------
    input_shape:
        ``(window_size, engineered_dim)`` — the shape of one engineered
        window ``F_k`` in ``R^(w x d')`` (paper Steps 2–3).
    cfg:
        Loaded ``config.yaml`` dict; hyper-parameters are read from
        ``cfg["model"]`` (``lstm_units``, ``dropout``, ``dense_units``,
        ``learning_rate``).

    Returns
    -------
    keras.Model
        A compiled ``keras.Sequential`` model: ``Input(input_shape) ->
        LSTM(lstm_units) -> Dropout(dropout) -> Dense(dense_units, relu) ->
        Dense(1, sigmoid)``, compiled with Adam and binary cross-entropy,
        tracking accuracy, precision, recall and AUC (paper §4 metrics).
    """
    if len(input_shape) != 2 or any(int(s) <= 0 for s in input_shape):
        raise ValueError(
            f"input_shape must be (window_size, n_features) with positive "
            f"entries, got {input_shape!r}"
        )
    model_cfg = cfg.get("model", {})
    model = keras.Sequential(
        [
            keras.Input(shape=(int(input_shape[0]), int(input_shape[1]))),
            keras.layers.LSTM(int(model_cfg.get("lstm_units", 64))),
            keras.layers.Dropout(float(model_cfg.get("dropout", 0.3))),
            keras.layers.Dense(int(model_cfg.get("dense_units", 32)), activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_fault_predictor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(model_cfg.get("learning_rate", 1e-3))
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
