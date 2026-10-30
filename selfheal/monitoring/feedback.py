"""Feedback loop and adaptive retraining (paper Step 7).

Implements the "Performance Monitoring and Feedback Loop" step of
*Self-Healing Cloud Platforms Using LSTM-based Predictive Models for Early
Fault Detection* (IJCET 16(5) 2025): every prediction, healing action and
observed outcome is appended to a JSONL event log, and the accumulated
evidence drives *adaptive retraining* of the LSTM — either after enough new
events have been collected or when the observed false-positive rate degrades
past a configured threshold.

Feedback JSONL event schema (see SPEC.md):

``{"type": "prediction"|"healing"|"outcome"|"retrain", "ts": float_epoch,
"probability": float?, "trigger": bool?, "severity": str?, "action": str?,
"target": dict?, "success": bool?, "actual_fault": bool?,
"mttr_seconds": float?, "detail": str?}``

This module must stay cheap to import: tensorflow is only imported lazily
inside :func:`retrain_from_feedback`.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfheal.config import resolve_path

#: Minimum number of outcome-bearing events before the false-positive-rate
#: retraining trigger is considered statistically meaningful (SPEC §feedback).
MIN_OUTCOME_EVENTS: int = 20


def _json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into plain JSON-serializable Python types.

    Handles numpy scalars (via ``.item()``); anything else unrecognized is
    stringified so a retrain summary can never fail to serialize.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    item = getattr(obj, "item", None)  # numpy scalar -> python scalar
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(obj)


class FeedbackLoop:
    """Append-only JSONL event log driving adaptive retraining (paper Step 7).

    Parameters
    ----------
    path:
        Path of the JSONL file.  Parent directories are created immediately;
        the file itself is created on first :meth:`log_event`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: dict) -> None:
        """Append one event as a JSON line, stamping a ``logged_at`` ISO time.

        The caller's dict is not mutated.  Values that are not natively
        JSON-serializable (e.g. enums, numpy scalars) are coerced via ``str``.
        """
        record = dict(event)
        record.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
        line = json.dumps(_json_safe(record), default=str)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def load_events(self) -> list[dict]:
        """Read every event back; missing file -> ``[]``.

        Blank or corrupt lines (e.g. from an interrupted write) are skipped
        rather than raised, so a damaged log never breaks the control loop.
        """
        if not self.path.exists():
            return []
        events: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        return events

    def should_retrain(self, cfg: dict) -> bool:
        """Decide whether adaptive retraining is due (paper Step 7).

        Returns ``True`` when, over the events logged *since the last
        "retrain" event* (all events if none):

        * their count ``>= cfg["monitoring"]["retrain_every_events"]``, OR
        * the observed false-positive rate ``FP / (FP + TN)`` among
          outcome-bearing events (those carrying ``actual_fault``) exceeds
          ``cfg["monitoring"]["fp_rate_retrain_threshold"]``, provided at
          least :data:`MIN_OUTCOME_EVENTS` outcome events exist.

        Both windows reset at the last retrain marker so a single bad episode
        cannot re-trigger retraining forever after it has been acted upon.
        """
        monitoring = cfg.get("monitoring", {}) or {}
        every = int(monitoring.get("retrain_every_events", 500))
        fp_threshold = float(monitoring.get("fp_rate_retrain_threshold", 0.2))

        events = self.load_events()
        last_retrain = -1
        for i, event in enumerate(events):
            if event.get("type") == "retrain":
                last_retrain = i
        recent = events[last_retrain + 1 :]

        if len(recent) >= every:
            return True

        outcome_events = [e for e in recent if e.get("actual_fault") is not None]
        if len(outcome_events) >= MIN_OUTCOME_EVENTS:
            fp = sum(
                1
                for e in outcome_events
                if bool(e.get("trigger")) and not bool(e.get("actual_fault"))
            )
            tn = sum(
                1
                for e in outcome_events
                if e.get("trigger") is not None
                and not bool(e.get("trigger"))
                and not bool(e.get("actual_fault"))
            )
            denom = fp + tn
            fp_rate = (fp / denom) if denom > 0 else 0.0
            if fp_rate > fp_threshold:
                return True
        return False

    def mark_retrained(self, summary: dict) -> None:
        """Log a ``{"type": "retrain", ...summary}`` marker event.

        Subsequent :meth:`should_retrain` calls only consider events logged
        after this marker.  A ``ts`` is added when the summary lacks one.
        """
        event = dict(summary)
        event["type"] = "retrain"
        event.setdefault("ts", time.time())
        self.log_event(event)


def retrain_from_feedback(cfg: dict) -> dict:
    """Rerun the full training pipeline and mark the feedback log (paper Step 7).

    Calls :func:`selfheal.models.train.train_model` on fresh data (including
    any regenerated telemetry), appends a ``retrain`` marker carrying the
    held-out test metrics to the feedback log, and returns the training
    summary dict (``{"history": ..., "test": ..., "artifacts_dir": ...}``).

    The import of the training module is deliberately deferred so importing
    this module never drags in tensorflow.
    """
    # Lazy import: keeps tensorflow out of module-level imports (SPEC rule).
    from selfheal.models.train import train_model

    result = train_model(cfg)

    log_rel = cfg.get("paths", {}).get("feedback_log", "feedback/events.jsonl")
    loop = FeedbackLoop(resolve_path(cfg, log_rel))
    summary = {
        "ts": time.time(),
        "detail": "adaptive retraining via feedback loop (paper Step 7)",
        "test": _json_safe(result.get("test", {})),
        "artifacts_dir": str(result.get("artifacts_dir", "")),
    }
    loop.mark_retrained(summary)
    return result
