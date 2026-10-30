"""Tests for selfheal.monitoring.metrics (paper Step 7 + §1 criteria).

SPEC test list: MTTR / uptime / FP-rate math verified on scripted event
sequences (the evaluation criteria the paper names in §1: MTTR, uptime
percentage, false positive rates).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfheal.healing.actions import HealingAction, HealingResult
from selfheal.monitoring.metrics import MetricsTracker


def _result(action: HealingAction, success: bool = True) -> HealingResult:
    return HealingResult(
        action=action,
        target={"pod": "pod-1"},
        success=success,
        started_at=0.0,
        completed_at=1.0,
        detail="test",
    )


def test_empty_tracker_defaults(cfg: dict) -> None:
    """No data: MTTR None, uptime 100.0, nothing counted."""
    tracker = MetricsTracker(cfg)
    assert tracker.mttr_seconds is None
    assert tracker.uptime_percent == 100.0
    assert tracker.predictions_total == 0
    assert tracker.actions_taken == {}


def test_mttr_mean_over_closed_faults(cfg: dict) -> None:
    """Each fault_start/recovery pair yields one MTTR sample; MTTR is the mean."""
    tracker = MetricsTracker(cfg)
    tracker.record_fault_start(100.0)
    tracker.record_recovery(160.0)  # 60 s
    tracker.record_fault_start(500.0)
    tracker.record_recovery(620.0)  # 120 s
    assert tracker.mttr_seconds == pytest.approx(90.0)


def test_recovery_without_open_fault_is_safe(cfg: dict) -> None:
    """A stray recovery must not crash or fabricate an MTTR sample."""
    tracker = MetricsTracker(cfg)
    tracker.record_recovery(50.0)
    assert tracker.mttr_seconds is None
    assert tracker.uptime_percent == 100.0


def test_uptime_percent_scripted(cfg: dict) -> None:
    """Observed span 1000 s with 100 s of downtime -> 90% uptime."""
    tracker = MetricsTracker(cfg)
    tracker.record_prediction(0.1, trigger=False, actual_fault=False, ts=0.0)
    tracker.record_fault_start(100.0)
    tracker.record_recovery(200.0)
    tracker.record_prediction(0.1, trigger=False, actual_fault=False, ts=1000.0)
    assert tracker.uptime_percent == pytest.approx(90.0)


def test_uptime_is_100_before_any_fault(cfg: dict) -> None:
    """Predictions alone never reduce uptime (SPEC: 100.0 before any fault)."""
    tracker = MetricsTracker(cfg)
    tracker.record_prediction(0.9, trigger=True, actual_fault=False, ts=0.0)
    tracker.record_prediction(0.9, trigger=True, actual_fault=False, ts=500.0)
    assert tracker.uptime_percent == 100.0


def test_confusion_rates_scripted(cfg: dict) -> None:
    """FP rate FP/(FP+TN), precision TP/(TP+FP), recall TP/(TP+FN)."""
    tracker = MetricsTracker(cfg)
    script = [
        (True, True),    # TP
        (True, True),    # TP
        (True, False),   # FP
        (False, True),   # FN
        (False, False),  # TN
        (False, False),  # TN
        (False, False),  # TN
    ]
    for i, (trigger, actual) in enumerate(script):
        tracker.record_prediction(0.5, trigger=trigger, actual_fault=actual,
                                  ts=float(i))
    # One unlabeled prediction: counts toward totals, not toward the rates.
    tracker.record_prediction(0.5, trigger=True, actual_fault=None, ts=99.0)

    assert tracker.false_positive_rate == pytest.approx(1.0 / 4.0)
    assert tracker.precision == pytest.approx(2.0 / 3.0)
    assert tracker.recall == pytest.approx(2.0 / 3.0)
    assert tracker.predictions_total == 8


def test_actions_taken_tally(cfg: dict) -> None:
    """record_action counts executed healing actions per action name."""
    tracker = MetricsTracker(cfg)
    tracker.record_action(_result(HealingAction.RESTART_POD))
    tracker.record_action(_result(HealingAction.RESTART_POD, success=False))
    tracker.record_action(_result(HealingAction.ALERT_ADMIN))

    taken = tracker.actions_taken
    assert sum(taken.values()) == 3
    restart_key = next(k for k in taken if "restart" in str(k).lower())
    alert_key = next(k for k in taken if "alert" in str(k).lower())
    assert taken[restart_key] == 2
    assert taken[alert_key] == 1


def test_report_is_json_serializable_and_saved(cfg: dict, tmp_path: Path) -> None:
    """report() carries every SPEC metric and save() writes it as JSON."""
    tracker = MetricsTracker(cfg)
    tracker.record_prediction(0.9, trigger=True, actual_fault=True, ts=0.0)
    tracker.record_fault_start(10.0)
    tracker.record_recovery(70.0)
    tracker.record_prediction(0.1, trigger=False, actual_fault=False, ts=100.0)
    tracker.record_action(_result(HealingAction.RESTART_POD))

    report = tracker.report()
    for key in (
        "mttr_seconds",
        "uptime_percent",
        "false_positive_rate",
        "precision",
        "recall",
        "actions_taken",
        "predictions_total",
    ):
        assert key in report, f"report() is missing {key!r}"
    assert report["mttr_seconds"] == pytest.approx(60.0)
    json.dumps(report)  # must not raise

    out = tmp_path / "reports" / "metrics.json"
    tracker.save(out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
