"""Tests for selfheal.models.lstm_model (paper Step 4 / Figure 1 architecture).

SPEC test list: build_lstm_model output shape (None, 1), compiles, and one
training step on random data.  No full training runs.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from selfheal.models.lstm_model import build_lstm_model  # noqa: E402


def test_build_lstm_model_architecture_and_output_shape(cfg: dict) -> None:
    """Figure 1: Input -> LSTM -> Dropout -> Dense(ReLU) -> Dense(1, sigmoid)."""
    w, d = cfg["windowing"]["window_size"], 16
    model = build_lstm_model((w, d), cfg)

    assert model.output_shape == (None, 1)

    layer_types = [type(layer).__name__ for layer in model.layers]
    core = [t for t in layer_types if t in ("LSTM", "Dropout", "Dense")]
    assert core == ["LSTM", "Dropout", "Dense", "Dense"], (
        f"expected LSTM -> Dropout -> Dense -> Dense, got {layer_types}"
    )

    lstm = next(l for l in model.layers if type(l).__name__ == "LSTM")
    assert lstm.units == cfg["model"]["lstm_units"]
    head = model.layers[-1]
    assert head.units == 1
    assert getattr(head.activation, "__name__", "") == "sigmoid"

    # Compiled with BCE loss (paper Step 4) — ready to fit without recompiling.
    assert model.optimizer is not None
    assert "binary_crossentropy" in str(model.loss).lower()


def test_one_training_step_and_probability_output(
    cfg: dict, rng: np.random.Generator
) -> None:
    """A single fit epoch on random data runs; predictions are probabilities."""
    w, d = cfg["windowing"]["window_size"], 8
    model = build_lstm_model((w, d), cfg)

    X = rng.random((24, w, d)).astype(np.float32)
    y = rng.integers(0, 2, 24).astype(np.float32)

    history = model.fit(X, y, epochs=1, batch_size=8, verbose=0)
    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][0])

    probs = np.asarray(model.predict(X, verbose=0))
    assert probs.shape == (24, 1)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0
