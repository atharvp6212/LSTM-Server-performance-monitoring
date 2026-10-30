"""Tests for selfheal.prediction.predictor (paper Step 5: decision rule A_k).

SPEC test list: decide() band mapping including exact-threshold edges.
The predictor is constructed via ``__new__`` with only the attributes
``decide()`` needs, so no trained artifacts are required.
"""

from __future__ import annotations

import pytest

from selfheal.prediction.predictor import Decision, FaultPredictor


def _predictor(cfg: dict) -> FaultPredictor:
    """A FaultPredictor with only what decide() reads (no artifact loading)."""
    predictor = FaultPredictor.__new__(FaultPredictor)
    predictor.cfg = cfg
    predictor.window_size = cfg["windowing"]["window_size"]
    predictor.feature_cols = ["cpu_usage", "memory_usage"]
    return predictor


@pytest.mark.parametrize(
    ("probability", "trigger", "severity"),
    [
        # Below every band (fixture: T=0.7; alert .7 / scale .8 / restart .9 / drain .97)
        (0.0, False, "none"),
        (0.5, False, "none"),
        (0.6999, False, "none"),
        # Exact threshold edges are INCLUSIVE (>=), per SPEC Step 5.
        (0.7, True, "alert"),
        (0.75, True, "alert"),
        (0.8, True, "scale"),
        (0.85, True, "scale"),
        (0.9, True, "restart"),
        (0.95, True, "restart"),
        (0.97, True, "drain"),
        (1.0, True, "drain"),
    ],
)
def test_decide_band_mapping(
    cfg: dict, probability: float, trigger: bool, severity: str
) -> None:
    """severity = highest band with threshold <= p; trigger = p >= T."""
    decision = _predictor(cfg).decide(probability)
    assert isinstance(decision, Decision)
    assert decision.probability == pytest.approx(probability)
    assert decision.trigger is trigger
    assert decision.severity == severity


def test_decide_trigger_independent_of_bands(cfg: dict) -> None:
    """trigger follows fault_threshold even when bands start higher."""
    cfg["decision"]["fault_threshold"] = 0.5
    decision = _predictor(cfg).decide(0.6)
    assert decision.trigger is True
    assert decision.severity == "none"  # below the lowest band (alert @ 0.7)


def test_decision_dataclass_fields(cfg: dict) -> None:
    """Decision carries exactly the SPEC fields with the right types."""
    decision = _predictor(cfg).decide(0.91)
    assert isinstance(decision.probability, float)
    assert isinstance(decision.trigger, bool)
    assert decision.severity in ("none", "alert", "scale", "restart", "drain")
