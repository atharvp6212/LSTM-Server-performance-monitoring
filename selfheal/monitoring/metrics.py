"""Runtime performance metrics for the self-healing control loop.

Implements paper Step 7 ("Performance Monitoring and Feedback Loop") of
*Self-Healing Cloud Platforms Using LSTM-based Predictive Models for Early
Fault Detection* (IJCET 16(5) 2025), together with the evaluation criteria
named in §1 of the paper: **MTTR** (mean time to recovery), **uptime
percentage**, and **false positive rate**.

The :class:`MetricsTracker` is fed by the closed loop
(``scripts/run_selfhealing.py`` / :class:`selfheal.healing.engine.SelfHealingEngine`):

* every model prediction is recorded via :meth:`MetricsTracker.record_prediction`
  (optionally with the ground-truth fault state, enabling the confusion counts
  behind precision / recall / false-positive rate),
* outage entry/exit is recorded via :meth:`MetricsTracker.record_fault_start`
  and :meth:`MetricsTracker.record_recovery` — each closed fault yields one
  MTTR sample and its duration counts as downtime,
* every executed healing action is tallied via
  :meth:`MetricsTracker.record_action`.

This module deliberately imports **no** heavy third-party packages
(no tensorflow / pandas / numpy) so it stays cheap to import from the
dashboard and the simulation loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsTracker:
    """Accumulates MTTR, uptime and prediction-quality statistics (paper Step 7).

    All timestamps are epoch seconds (simulated or real).  The *observed span*
    used for the uptime computation is ``max(ts) - min(ts)`` over every
    timestamp the tracker has seen (predictions, fault starts, recoveries).

    Uptime is defined as ``100 * (1 - downtime / observed_span)`` where
    downtime is the summed duration of closed faults plus the elapsed time of
    a still-open fault (paper §1: "uptime percentage").
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """Create an empty tracker.  ``cfg`` is optional and only retained for
        forward compatibility (nothing in the metric math is configurable)."""
        self.cfg: dict = cfg or {}
        # prediction confusion counts (only predictions with known actual_fault)
        self._tp: int = 0
        self._fp: int = 0
        self._tn: int = 0
        self._fn: int = 0
        self._predictions_total: int = 0
        # fault / recovery bookkeeping
        self._mttr_samples: list[float] = []
        self._open_fault_start: float | None = None
        self._faults_total: int = 0
        # healing action tallies
        self._actions: dict[str, int] = {}
        self._actions_succeeded: dict[str, int] = {}
        # observed time span
        self._min_ts: float | None = None
        self._max_ts: float | None = None

    # ------------------------------------------------------------------ #
    # recording                                                          #
    # ------------------------------------------------------------------ #
    def _observe_ts(self, ts: float) -> None:
        """Extend the observed time span with ``ts``."""
        ts = float(ts)
        if self._min_ts is None or ts < self._min_ts:
            self._min_ts = ts
        if self._max_ts is None or ts > self._max_ts:
            self._max_ts = ts

    def record_prediction(
        self,
        probability: float,
        trigger: bool,
        actual_fault: bool | None = None,
        ts: float | None = None,
    ) -> None:
        """Record one model prediction (paper Step 5 output feeding Step 7).

        Parameters
        ----------
        probability:
            The LSTM fault probability ``ŷ ∈ [0, 1]``.
        trigger:
            Whether the decision rule fired (``probability >= T``).
        actual_fault:
            Ground-truth fault state at prediction time, when known.  Only
            predictions with a known actual contribute to the confusion counts
            (false-positive rate, precision, recall).
        ts:
            Epoch-seconds timestamp; defaults to ``time.time()``.
        """
        self._observe_ts(time.time() if ts is None else ts)
        self._predictions_total += 1
        del probability  # part of the interface; not needed for the aggregates
        if actual_fault is None:
            return
        actual = bool(actual_fault)
        fired = bool(trigger)
        if fired and actual:
            self._tp += 1
        elif fired and not actual:
            self._fp += 1
        elif not fired and actual:
            self._fn += 1
        else:
            self._tn += 1

    def record_fault_start(self, ts: float) -> None:
        """Mark the start of an outage (ground-truth ``anomaly`` going 0→1).

        A duplicate start while a fault is already open is ignored (the
        earliest start defines the downtime window).
        """
        self._observe_ts(ts)
        if self._open_fault_start is not None:
            return
        self._open_fault_start = float(ts)
        self._faults_total += 1

    def record_recovery(self, ts: float) -> None:
        """Close the open fault: one MTTR sample + its downtime (paper §1 MTTR).

        A recovery with no open fault is a no-op.  Non-monotonic timestamps
        are clamped so a sample is never negative.
        """
        self._observe_ts(ts)
        if self._open_fault_start is None:
            return
        duration = max(0.0, float(ts) - self._open_fault_start)
        self._mttr_samples.append(duration)
        self._open_fault_start = None

    def record_action(self, result: "HealingResult") -> None:
        """Tally one executed healing action (paper Step 6 feeding Step 7).

        ``result`` is duck-typed: only ``.action`` (a ``str``-valued enum or
        string) and ``.success`` are read, so this module never needs to import
        ``selfheal.healing.actions``.
        """
        action = getattr(result, "action", None)
        key = getattr(action, "value", None)
        if not isinstance(key, str):
            key = str(action)
        self._actions[key] = self._actions.get(key, 0) + 1
        if bool(getattr(result, "success", False)):
            self._actions_succeeded[key] = self._actions_succeeded.get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # derived metrics (paper §1: MTTR, uptime percentage, FP rate)       #
    # ------------------------------------------------------------------ #
    @property
    def mttr_seconds(self) -> float | None:
        """Mean time to recovery over closed faults; ``None`` with no samples."""
        if not self._mttr_samples:
            return None
        return sum(self._mttr_samples) / len(self._mttr_samples)

    @property
    def downtime_seconds(self) -> float:
        """Total downtime: closed-fault durations plus any open fault so far."""
        down = sum(self._mttr_samples)
        if self._open_fault_start is not None and self._max_ts is not None:
            down += max(0.0, self._max_ts - self._open_fault_start)
        return down

    @property
    def observed_span_seconds(self) -> float:
        """Length of the observed time window (0.0 before two timestamps)."""
        if self._min_ts is None or self._max_ts is None:
            return 0.0
        return max(0.0, self._max_ts - self._min_ts)

    @property
    def uptime_percent(self) -> float:
        """``100 * (1 - downtime / observed_span)``, clamped to [0, 100].

        Returns 100.0 before any fault has been recorded, and also when the
        observed span is zero (no elapsed time observed — nothing to divide by).
        """
        if self._faults_total == 0:
            return 100.0
        span = self.observed_span_seconds
        if span <= 0.0:
            return 100.0
        uptime = 100.0 * (1.0 - self.downtime_seconds / span)
        return min(100.0, max(0.0, uptime))

    @property
    def false_positive_rate(self) -> float | None:
        """``FP / (FP + TN)`` over predictions with known actual; ``None`` if undefined."""
        denom = self._fp + self._tn
        if denom == 0:
            return None
        return self._fp / denom

    @property
    def precision(self) -> float | None:
        """``TP / (TP + FP)``; ``None`` if no positive predictions were labeled."""
        denom = self._tp + self._fp
        if denom == 0:
            return None
        return self._tp / denom

    @property
    def recall(self) -> float | None:
        """``TP / (TP + FN)``; ``None`` if no actual faults were labeled."""
        denom = self._tp + self._fn
        if denom == 0:
            return None
        return self._tp / denom

    @property
    def actions_taken(self) -> dict[str, int]:
        """Count of executed healing actions keyed by action name."""
        return dict(self._actions)

    @property
    def predictions_total(self) -> int:
        """Total number of predictions recorded (labeled or not)."""
        return self._predictions_total

    # ------------------------------------------------------------------ #
    # reporting                                                          #
    # ------------------------------------------------------------------ #
    def report(self) -> dict[str, Any]:
        """Return every metric as a JSON-serializable dict (paper Step 7)."""
        return {
            "mttr_seconds": self.mttr_seconds,
            "uptime_percent": self.uptime_percent,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "predictions_total": self.predictions_total,
            "actions_taken": self.actions_taken,
            "actions_succeeded": dict(self._actions_succeeded),
            "confusion": {"tp": self._tp, "fp": self._fp, "tn": self._tn, "fn": self._fn},
            "faults_total": self._faults_total,
            "faults_recovered": len(self._mttr_samples),
            "fault_open": self._open_fault_start is not None,
            "downtime_seconds": self.downtime_seconds,
            "observed_span_seconds": self.observed_span_seconds,
        }

    def save(self, path: str | Path) -> None:
        """Write :meth:`report` as JSON to ``path`` (creating parent dirs)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.report(), fh, indent=2)
            fh.write("\n")
