"""Tests for selfheal.monitoring.feedback (paper Step 7: feedback loop).

SPEC test list: JSONL roundtrip; should_retrain thresholds (event count since
the last retrain, and the false-positive-rate trigger with its >= 20 outcome
event floor).  retrain_from_feedback is not exercised here — it runs full
model training, which the test suite must avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

from selfheal.monitoring.feedback import FeedbackLoop


def _loop(tmp_path: Path) -> FeedbackLoop:
    return FeedbackLoop(tmp_path / "feedback" / "events.jsonl")


def _outcome(trigger: bool, actual: bool, ts: float) -> dict:
    """One outcome-bearing event per the SPEC feedback JSONL schema."""
    return {
        "type": "outcome",
        "ts": ts,
        "trigger": trigger,
        "actual_fault": actual,
    }


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """log_event appends JSON lines (with logged_at); load_events reads them back."""
    loop = _loop(tmp_path)
    assert loop.load_events() == []  # missing file -> empty list

    events = [
        {"type": "prediction", "ts": 1.0, "probability": 0.42, "trigger": False},
        {"type": "healing", "ts": 2.0, "action": "restart_pod", "success": True,
         "target": {"pod": "pod-1"}},
        {"type": "outcome", "ts": 3.0, "trigger": True, "actual_fault": True},
    ]
    for event in events:
        loop.log_event(event)

    loaded = loop.load_events()
    assert len(loaded) == 3
    for original, read_back in zip(events, loaded):
        for key, value in original.items():
            assert read_back[key] == value
        assert "logged_at" in read_back, "log_event must stamp logged_at"

    # The file really is line-delimited JSON.
    lines = [
        line
        for line in Path(loop.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    for line in lines:
        json.loads(line)


def test_should_retrain_event_count_threshold(tmp_path: Path, cfg: dict) -> None:
    """True once events since the last retrain reach retrain_every_events."""
    cfg["monitoring"]["retrain_every_events"] = 5
    loop = _loop(tmp_path)

    for i in range(4):
        loop.log_event({"type": "prediction", "ts": float(i), "trigger": False})
    assert loop.should_retrain(cfg) is False

    loop.log_event({"type": "prediction", "ts": 4.0, "trigger": False})
    assert loop.should_retrain(cfg) is True


def test_mark_retrained_resets_the_event_window(tmp_path: Path, cfg: dict) -> None:
    """After a retrain marker only newer events count toward the threshold."""
    cfg["monitoring"]["retrain_every_events"] = 3
    loop = _loop(tmp_path)
    for i in range(3):
        loop.log_event({"type": "prediction", "ts": float(i), "trigger": False})
    assert loop.should_retrain(cfg) is True

    loop.mark_retrained({"ts": 100.0, "detail": "test retrain"})
    assert loop.should_retrain(cfg) is False

    retrains = [e for e in loop.load_events() if e.get("type") == "retrain"]
    assert len(retrains) == 1, "mark_retrained must log a retrain event"

    # The window builds up again from zero after the marker.
    for i in range(3):
        loop.log_event({"type": "prediction", "ts": 200.0 + i, "trigger": False})
    assert loop.should_retrain(cfg) is True


def test_should_retrain_fp_rate_trigger(tmp_path: Path, cfg: dict) -> None:
    """FP rate above the threshold (with >= 20 outcome events) forces retraining."""
    cfg["monitoring"]["retrain_every_events"] = 1000  # keep the count path quiet
    cfg["monitoring"]["fp_rate_retrain_threshold"] = 0.2
    loop = _loop(tmp_path)

    # 10 FP (trigger, no fault) + 10 TN (no trigger, no fault): FP rate 0.5.
    for i in range(10):
        loop.log_event(_outcome(trigger=True, actual=False, ts=float(i)))
    for i in range(10):
        loop.log_event(_outcome(trigger=False, actual=False, ts=10.0 + i))
    assert loop.should_retrain(cfg) is True


def test_should_retrain_fp_rate_below_threshold(tmp_path: Path, cfg: dict) -> None:
    """A healthy FP rate over enough outcome events does not retrain."""
    cfg["monitoring"]["retrain_every_events"] = 1000
    cfg["monitoring"]["fp_rate_retrain_threshold"] = 0.2
    loop = _loop(tmp_path)

    # 1 FP + 19 TN: FP rate 0.05 < 0.2 -> no retraining.
    loop.log_event(_outcome(trigger=True, actual=False, ts=0.0))
    for i in range(19):
        loop.log_event(_outcome(trigger=False, actual=False, ts=1.0 + i))
    assert loop.should_retrain(cfg) is False


def test_fp_rate_needs_minimum_outcome_events(tmp_path: Path, cfg: dict) -> None:
    """Fewer than 20 outcome events never trip the FP-rate trigger."""
    cfg["monitoring"]["retrain_every_events"] = 1000
    cfg["monitoring"]["fp_rate_retrain_threshold"] = 0.2
    loop = _loop(tmp_path)

    # 10 outcome events, every one a false positive: still below the floor.
    for i in range(10):
        loop.log_event(_outcome(trigger=True, actual=False, ts=float(i)))
    assert loop.should_retrain(cfg) is False
