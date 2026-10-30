"""Self-healing engine: routes Step 5 decisions to Step 6 actions.

Implements paper Step 6 ("Self-Healing Integration") with the Step 7 logging
hooks of *Self-Healing Cloud Platforms Using LSTM-based Predictive Models for
Early Fault Detection* (IJCET 16(5) 2025): each :class:`Decision` from the
predictor is mapped onto a healing action by severity band, guarded by a
per-target cooldown and an hourly action budget (so a noisy model cannot storm
the cluster), executed through a :class:`BaseHealer`, and logged to the
feedback loop (Step 7 JSONL schema) and metrics tracker.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from selfheal.healing.actions import BaseHealer, HealingAction, HealingResult

if TYPE_CHECKING:  # concurrently-owned modules: annotate only, never import at runtime
    from selfheal.monitoring.feedback import FeedbackLoop
    from selfheal.monitoring.metrics import MetricsTracker
    from selfheal.prediction.predictor import Decision

__all__ = ["SelfHealingEngine"]

logger = logging.getLogger("selfheal.healing.engine")

_SEVERITY_TO_ACTION: dict[str, HealingAction] = {
    "alert": HealingAction.ALERT_ADMIN,
    "scale": HealingAction.SCALE_UP,
    "restart": HealingAction.RESTART_POD,
    "drain": HealingAction.DRAIN_NODE,
}


class SelfHealingEngine:
    """Decision-to-action router with throttling and Step 7 logging.

    Division of responsibility (per SPEC): the engine logs every prediction
    and healing action to the feedback JSONL and records executed actions into
    the metrics tracker via ``record_action``. Prediction outcomes against
    ground truth (``record_prediction`` with ``actual_fault``) and fault
    start/recovery timestamps are recorded by the closed-loop driver
    (``scripts/run_selfhealing.py``), which alone knows the ground truth.
    """

    def __init__(
        self,
        healer: BaseHealer,
        cfg: dict,
        feedback: "FeedbackLoop | None" = None,
        metrics: "MetricsTracker | None" = None,
    ) -> None:
        """Wire the engine to a healer and optional Step 7 sinks.

        Throttling parameters come from ``cfg["healing"]``:
        ``cooldown_seconds`` (per-target) and ``max_actions_per_hour``
        (global, non-alert actions only).
        """
        self.healer = healer
        self.cfg = cfg
        self.feedback = feedback
        self.metrics = metrics
        healing_cfg = cfg.get("healing", {}) or {}
        self.cooldown_seconds: float = float(healing_cfg.get("cooldown_seconds", 120))
        self.max_actions_per_hour: int = int(healing_cfg.get("max_actions_per_hour", 30))
        self._last_action_ts: dict[str, float] = {}
        self._action_times: deque[float] = deque()

    def handle(self, decision: "Decision", context: dict) -> HealingResult | None:
        """Handle one prediction: maybe execute a healing action (Step 6).

        Args:
            decision: Step 5 output (``probability``, ``trigger``, ``severity``).
            context: ``{"target": {...}, "top_metric": str | None,
                "timestamp": float}`` — target carries pod/node names,
                ``top_metric`` is the highest-pressure canonical metric, and
                ``timestamp`` is the (possibly simulated) epoch time used for
                cooldown accounting.

        Returns:
            The :class:`HealingResult`, or ``None`` when ``decision.trigger``
            is false (no action, prediction still logged to feedback).

        Action choice: "alert" severity always just notifies (ALERT_ADMIN) —
        the model isn't confident enough yet to act. Once severity reaches
        "scale"/"restart"/"drain", the concrete action is chosen from
        ``top_metric`` so it actually matches the underlying fault signature
        (paper §3.3: "restarting unstable pods, reallocation of resources,
        scaling of replicas, or isolation of problematic nodes" — the right
        one of these depends on *what* is failing, not just *how urgently*):
        memory/restart/error pressure -> RESTART_POD; cpu pressure ->
        SCALE_UP; network pressure -> SCALE_UP (moderate) or RESCHEDULE_NODE
        / DRAIN_NODE (severe) since that is placement-related; unrecognized
        or missing ``top_metric`` falls back to the severity ladder
        (alert/scale/restart/drain -> ALERT_ADMIN/SCALE_UP/RESTART_POD/
        DRAIN_NODE) so a target without metric context still gets a sane
        default. Non-alert actions respect the per-target cooldown and the
        hourly budget — when throttled the engine falls back to ALERT_ADMIN,
        which itself is never throttled.
        """
        context = dict(context or {})
        target = dict(context.get("target") or {})
        top_metric = context.get("top_metric")
        timestamp = context.get("timestamp")
        ts = float(timestamp) if timestamp is not None else time.time()
        probability = float(decision.probability)

        self._log_feedback(
            {
                "type": "prediction",
                "ts": ts,
                "probability": probability,
                "trigger": bool(decision.trigger),
                "severity": decision.severity,
                "target": target,
            }
        )
        if not decision.trigger:
            return None

        action = self._choose_action(decision.severity, top_metric)
        target_key = self._target_key(target)
        if action is not HealingAction.ALERT_ADMIN:
            throttle_reason = self._throttle_reason(target_key, ts)
            if throttle_reason is not None:
                logger.info(
                    "throttled %s for target=%s (%s); falling back to alert",
                    action.value,
                    target,
                    throttle_reason,
                )
                target["detail"] = (
                    f"throttled {action.value} ({throttle_reason}); "
                    f"fault probability {probability:.3f}"
                )
                action = HealingAction.ALERT_ADMIN
        if action is HealingAction.ALERT_ADMIN:
            target.setdefault(
                "detail",
                f"fault probability {probability:.3f} (severity {decision.severity})",
            )

        result = self.healer.execute(action, target)
        if action is not HealingAction.ALERT_ADMIN:
            self._register_action(target_key, ts)

        self._log_feedback(
            {
                "type": "healing",
                "ts": ts,
                "probability": probability,
                "trigger": True,
                "severity": decision.severity,
                "action": result.action.value,
                "target": result.target,
                "success": result.success,
                "detail": result.detail,
            }
        )
        if self.metrics is not None:
            self.metrics.record_action(result)
        return result

    def _choose_action(self, severity: str, top_metric: str | None) -> HealingAction:
        """Map (severity, top_metric) to an action (paper Step 6 routing rule).

        ``severity`` gates *whether* to act non-trivially ("alert" never
        does more than notify); ``top_metric`` then picks *which* action can
        actually clear that fault signature. Routing a fault to an action
        that cannot address its root cause (e.g. draining a node for a
        memory leak on one pod) would trigger forever without ever
        recovering, so metric match takes priority over the raw severity
        word whenever a metric is known.
        """
        severity = str(severity)
        if severity == "alert" or severity not in _SEVERITY_TO_ACTION:
            # Low confidence (or an unrecognized/"none" severity on a
            # triggered decision) degrades to the safest action.
            return HealingAction.ALERT_ADMIN
        if top_metric in ("memory_usage", "pod_restarts", "error_event_rate"):
            return HealingAction.RESTART_POD
        if top_metric == "cpu_usage":
            return HealingAction.SCALE_UP
        if top_metric == "network_io":
            if severity == "drain":
                return HealingAction.DRAIN_NODE
            if severity == "restart":
                return HealingAction.RESCHEDULE_NODE
            return HealingAction.SCALE_UP
        if top_metric == "warning_event_rate":
            return HealingAction.SCALE_UP
        # No/unrecognized metric context: fall back to the severity ladder.
        return _SEVERITY_TO_ACTION[severity]

    def _target_key(self, target: dict) -> str:
        """Stable identity for per-target cooldown bookkeeping."""
        identity = {k: v for k, v in target.items() if k != "detail"}
        return json.dumps(identity, sort_keys=True, default=str)

    def _throttle_reason(self, target_key: str, ts: float) -> str | None:
        """Return why a non-alert action is throttled now, or None if allowed."""
        last = self._last_action_ts.get(target_key)
        if last is not None and (ts - last) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (ts - last)
            return f"cooldown, {remaining:.0f}s remaining"
        while self._action_times and self._action_times[0] <= ts - 3600.0:
            self._action_times.popleft()
        if len(self._action_times) >= self.max_actions_per_hour:
            return f"max_actions_per_hour ({self.max_actions_per_hour}) reached"
        return None

    def _register_action(self, target_key: str, ts: float) -> None:
        """Record a non-alert action for cooldown / hourly-budget accounting."""
        self._last_action_ts[target_key] = ts
        self._action_times.append(ts)

    def _log_feedback(self, event: dict) -> None:
        """Append one Step 7 JSONL event to the feedback loop, if attached."""
        if self.feedback is not None:
            self.feedback.log_event(event)
