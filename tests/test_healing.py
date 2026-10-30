"""Tests for selfheal.healing (paper Step 6: self-healing integration).

SPEC test list: SimulatedHealer + engine — a triggered decision routes to the
expected action; the per-target cooldown is enforced; no action when
``trigger`` is False.  The cluster is a duck-typed stub implementing exactly
the SPEC SimulatedCluster healing hooks, so this file stays independent of
the concurrently built simulation module.
"""

from __future__ import annotations

from selfheal.healing.actions import HealingAction, HealingResult, SimulatedHealer
from selfheal.healing.engine import SelfHealingEngine
from selfheal.prediction.predictor import Decision


class ClusterStub:
    """Records every SPEC healing hook call; all hooks succeed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def restart_pod(self, pod: str | None = None) -> bool:
        self.calls.append(("restart_pod", pod))
        return True

    def reschedule_pod(self, pod: str | None = None) -> bool:
        self.calls.append(("reschedule_pod", pod))
        return True

    def scale_up(self) -> bool:
        self.calls.append(("scale_up", None))
        return True

    def drain_node(self, node: str | None = None) -> bool:
        self.calls.append(("drain_node", node))
        return True

    def alert_admin(self, detail: str = "") -> bool:
        self.calls.append(("alert_admin", detail))
        return True


class FeedbackStub:
    """Collects feedback events without touching the filesystem."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def log_event(self, event: dict) -> None:
        self.events.append(dict(event))


def _engine(cfg: dict, feedback: FeedbackStub | None = None):
    cluster = ClusterStub()
    engine = SelfHealingEngine(SimulatedHealer(cluster), cfg, feedback=feedback)
    return engine, cluster


def _decision(probability: float, severity: str, trigger: bool = True) -> Decision:
    return Decision(probability=probability, trigger=trigger, severity=severity)


def _ctx(ts: float, pod: str = "pod-1", top_metric: str | None = None) -> dict:
    return {"target": {"pod": pod}, "top_metric": top_metric, "timestamp": ts}


def test_no_action_when_trigger_false(cfg: dict) -> None:
    """decision.trigger False -> handle() is a no-op returning None."""
    engine, cluster = _engine(cfg)
    result = engine.handle(_decision(0.2, "none", trigger=False), _ctx(1000.0))
    assert result is None
    assert cluster.calls == [], "no healing hook may run without a trigger"


def test_severity_routes_to_expected_actions(cfg: dict) -> None:
    """alert/scale/restart/drain map to the paper Step 6 actions."""
    expectations = [
        ("alert", HealingAction.ALERT_ADMIN, "alert_admin"),
        ("scale", HealingAction.SCALE_UP, "scale_up"),
        ("restart", HealingAction.RESTART_POD, "restart_pod"),
        ("drain", HealingAction.DRAIN_NODE, "drain_node"),
    ]
    ts = 1000.0
    for i, (severity, action, hook) in enumerate(expectations):
        engine, cluster = _engine(cfg)  # fresh engine: no cooldown carry-over
        result = engine.handle(
            _decision(0.99, severity), _ctx(ts, pod=f"pod-{i}")
        )
        assert isinstance(result, HealingResult)
        assert result.action is action
        assert result.success is True
        assert result.started_at <= result.completed_at
        assert [name for name, _ in cluster.calls] == [hook]


def test_restart_with_network_pressure_becomes_reschedule(cfg: dict) -> None:
    """severity "restart" + top_metric network_io -> RESCHEDULE_NODE (SPEC)."""
    engine, cluster = _engine(cfg)
    result = engine.handle(
        _decision(0.92, "restart"), _ctx(1000.0, top_metric="network_io")
    )
    assert result is not None
    assert result.action is HealingAction.RESCHEDULE_NODE
    assert [name for name, _ in cluster.calls] == ["reschedule_pod"]


def test_cooldown_enforced_per_target(cfg: dict) -> None:
    """Within cooldown the same target degrades to ALERT_ADMIN; after it, acts."""
    cfg["healing"]["cooldown_seconds"] = 120
    engine, cluster = _engine(cfg)

    first = engine.handle(_decision(0.92, "restart"), _ctx(1000.0))
    assert first is not None and first.action is HealingAction.RESTART_POD

    # 10 s later — still cooling down: falls back to alert, never restarts.
    second = engine.handle(_decision(0.93, "restart"), _ctx(1010.0))
    assert second is not None and second.action is HealingAction.ALERT_ADMIN

    # After the cooldown expires the real action is allowed again.
    third = engine.handle(_decision(0.94, "restart"), _ctx(1000.0 + 130.0))
    assert third is not None and third.action is HealingAction.RESTART_POD

    assert [name for name, _ in cluster.calls] == [
        "restart_pod",
        "alert_admin",
        "restart_pod",
    ]


def test_cooldown_is_per_target(cfg: dict) -> None:
    """A different target is not blocked by another target's cooldown."""
    engine, cluster = _engine(cfg)
    engine.handle(_decision(0.92, "restart"), _ctx(1000.0, pod="pod-a"))
    other = engine.handle(_decision(0.92, "restart"), _ctx(1001.0, pod="pod-b"))
    assert other is not None and other.action is HealingAction.RESTART_POD
    assert [name for name, _ in cluster.calls] == ["restart_pod", "restart_pod"]


def test_max_actions_per_hour_throttles_to_alert(cfg: dict) -> None:
    """Beyond the hourly budget non-alert actions fall back to ALERT_ADMIN."""
    cfg["healing"]["max_actions_per_hour"] = 2
    engine, cluster = _engine(cfg)

    actions = []
    for i in range(3):  # distinct targets so the cooldown never interferes
        result = engine.handle(
            _decision(0.92, "restart"), _ctx(1000.0 + i, pod=f"pod-{i}")
        )
        assert result is not None
        actions.append(result.action)
    assert actions == [
        HealingAction.RESTART_POD,
        HealingAction.RESTART_POD,
        HealingAction.ALERT_ADMIN,
    ]

    # An hour later the budget window has passed: real actions resume.
    later = engine.handle(
        _decision(0.92, "restart"), _ctx(1000.0 + 3700.0, pod="pod-9")
    )
    assert later is not None and later.action is HealingAction.RESTART_POD


def test_alert_admin_is_never_throttled(cfg: dict) -> None:
    """ALERT_ADMIN executes every time, even back-to-back on one target."""
    engine, cluster = _engine(cfg)
    for i in range(4):
        result = engine.handle(_decision(0.75, "alert"), _ctx(1000.0 + i))
        assert result is not None
        assert result.action is HealingAction.ALERT_ADMIN
        assert result.success is True
    assert [name for name, _ in cluster.calls] == ["alert_admin"] * 4


def test_engine_logs_predictions_and_actions_to_feedback(cfg: dict) -> None:
    """Step 7 hook: every prediction is logged; executed actions add an event."""
    feedback = FeedbackStub()
    engine, _ = _engine(cfg, feedback=feedback)

    engine.handle(_decision(0.2, "none", trigger=False), _ctx(1000.0))
    engine.handle(_decision(0.92, "restart"), _ctx(2000.0))

    types = [event.get("type") for event in feedback.events]
    assert types.count("prediction") == 2, "every prediction must be logged"
    assert "healing" in types, "an executed action must be logged"
    healing = next(e for e in feedback.events if e.get("type") == "healing")
    assert healing.get("action") == HealingAction.RESTART_POD.value
    assert healing.get("success") is True
