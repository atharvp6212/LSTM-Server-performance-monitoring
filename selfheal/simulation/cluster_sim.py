"""Simulated Kubernetes cluster: the closed-loop self-healing testbed.

Stands in for a live cluster so the paper's Step 6 (self-healing actions via
the Kubernetes API) and Step 7 (performance monitoring and feedback loop) can
run end to end without real infrastructure (SPEC.md ``selfheal/simulation/``).

Fault dynamics implement paper §3.2: *"faults build up gradually but do not
emerge suddenly"* — every injected fault ramps up over 15–40 ticks before it
becomes an active outage (ground-truth ``anomaly = 1``). Healing hooks map the
paper's action set (RestartPod / RescheduleNode / ScaleUp / DrainNode /
AlertAdmin, Step 6) onto the simulated state:

* ``restart_pod``      clears ``crashloop`` / ``memory_leak`` on that pod;
* ``scale_up``         damps ``cpu_runaway`` / ``network_saturation`` pressure;
* ``reschedule_pod`` / ``drain_node`` fix node-level faults
  (``network_saturation``);
* a wrong action, or the right action on the wrong pod/node, does **not**
  clear the fault.

Only depends on numpy / pandas (and stdlib); the canonical metric list is
duplicated here on purpose so this module never imports the dataset module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# NOTE: mirrors ``selfheal.data.dataset.CANONICAL_METRICS`` (SPEC.md canonical
# data contract / paper Table 1). Kept as a local copy so the simulator's only
# third-party imports are numpy and pandas.
CANONICAL_METRICS: list[str] = [
    "cpu_usage",
    "memory_usage",
    "network_io",
    "pod_restarts",
    "error_event_rate",
    "warning_event_rate",
]

#: Injectable fault modes (SPEC.md ``SimulatedCluster``).
FAULT_KINDS: tuple[str, ...] = (
    "memory_leak",
    "cpu_runaway",
    "crashloop",
    "network_saturation",
)

#: Fault kinds cleared by restarting the anchor pod (paper Step 6, RestartPod).
RESTART_CLEARABLE: frozenset[str] = frozenset({"crashloop", "memory_leak"})
#: Fault kinds damped by adding replicas (paper Step 6, ScaleUp).
SCALE_DAMPABLE: frozenset[str] = frozenset({"cpu_runaway", "network_saturation"})
#: Node-level fault kinds fixed by drain / reschedule (paper Step 6).
NODE_LEVEL_FAULTS: frozenset[str] = frozenset({"network_saturation"})

# Raw-unit resource limits used for utilization/pressure scoring; overridden by
# cfg["features"]["limits"] when present (paper Step 3 utilization ratio).
_DEFAULT_LIMITS: dict[str, float] = {
    "cpu_usage": 100.0,
    "memory_usage": 100.0,
    "network_io": 1000.0,
    "pod_restarts": 5.0,
    "error_event_rate": 20.0,
    "warning_event_rate": 40.0,
}

_SIM_EPOCH_START = 1_735_689_600.0  # 2025-01-01T00:00:00Z, sim epoch seconds
_RESTART_WINDOW_TICKS = 20          # pod_restarts = cluster restarts in this window
_RAMP_TICKS_MIN, _RAMP_TICKS_MAX = 15, 40   # gradual buildup span (paper §3.2)
_OUTAGE_SEVERITY = 0.6              # effective severity needed to be an outage
_CLEAR_SEVERITY = 0.25              # below this an active fault resolves
_SCALE_DAMP = 0.45                  # multiplier applied per successful scale_up
_DAMPED_DECAY = 0.98                # slow per-tick dissipation of damped faults
_AUTO_RECOVER_TICKS = 240           # unhealed outage length until "manual ops"
_MAX_CONCURRENT_FAULTS = 2
_BASE_RESTART_P = 0.0015            # baseline restart probability per pod/tick
_BASE_ERR_LAM = 0.4                 # baseline error events per tick (Poisson)
_BASE_WARN_LAM = 1.1                # baseline warning events per tick (Poisson)


@dataclass
class _Pod:
    """Internal per-pod state (name, placement and metric baselines)."""

    name: str
    node: str
    base_cpu: float
    base_mem: float
    base_net: float
    restart_ticks: list[int] = field(default_factory=list)


@dataclass
class _Fault:
    """Internal fault lifecycle: ramping -> outage -> recovering -> removed."""

    kind: str
    pod: str
    ramp_ticks: int
    age: int = 0
    phase: str = "ramping"  # "ramping" | "outage" | "recovering"
    multiplier: float = 1.0  # damped (< 1.0) by scale_up
    outage_age: int = 0
    residual: float = 0.0   # decaying severity while "recovering"
    decay: float = 0.55     # per-tick residual decay factor while recovering


class SimulatedCluster:
    """Closed-loop testbed simulating ``n_pods`` pods across ``n_nodes`` nodes.

    Implements the SPEC.md ``selfheal/simulation/cluster_sim.py`` contract.
    Per-pod cpu/mem/net/restart dynamics = baseline + diurnal-style waves +
    Gaussian noise + additive fault pressure. Faults ramp gradually
    (paper §3.2) and, if unhealed, become an active outage (``anomaly = 1``,
    the ground truth the LSTM predictor is meant to pre-empt, paper Step 5).

    Attributes:
        current_time: Simulation time in epoch seconds; advanced by
            ``cfg["dataset"]["synthetic"]["tick_seconds"]`` per :meth:`tick`.
        nodes: Node names (``node-0`` ...).
        alerts: ``(sim_time, detail)`` tuples recorded by :meth:`alert_admin`.
    """

    def __init__(self, cfg: dict, seed: int | None = None) -> None:
        """Build the cluster from ``cfg`` (see ``config.yaml`` ``dataset.synthetic``).

        Args:
            cfg: Full configuration dict (``selfheal.config.load_config``).
            seed: RNG seed override; defaults to ``cfg["project"]["seed"]``.
        """
        syn = cfg.get("dataset", {}).get("synthetic", {}) or {}
        self.tick_seconds: float = float(syn.get("tick_seconds", 15))
        n_pods = max(1, int(syn.get("n_pods", 12)))
        n_nodes = max(1, int(syn.get("n_nodes", 3)))
        if seed is None:
            seed = int(cfg.get("project", {}).get("seed", 42))
        self._rng = np.random.default_rng(seed)

        limits_cfg = cfg.get("features", {}).get("limits", {}) or {}
        self._limits: dict[str, float] = {
            m: float(limits_cfg.get(m, _DEFAULT_LIMITS[m])) for m in CANONICAL_METRICS
        }

        self.nodes: list[str] = [f"node-{i}" for i in range(n_nodes)]
        self._pods: dict[str, _Pod] = {}
        self._pod_counter = 0
        for i in range(n_pods):
            self._spawn_pod(self.nodes[i % n_nodes])
        self._max_pods = max(n_pods * 2, n_pods + 1)

        self.current_time: float = _SIM_EPOCH_START
        self._tick_index = 0
        self._faults: list[_Fault] = []
        self.alerts: list[tuple[float, str]] = []
        # EMA state for cluster event rates (events per minute).
        self._err_rate = _BASE_ERR_LAM * 60.0 / self.tick_seconds
        self._warn_rate = _BASE_WARN_LAM * 60.0 / self.tick_seconds
        self._last_pod_metrics: dict[str, dict[str, float]] = {}
        self._last_aggregates: dict[str, float] = {}

    # ------------------------------------------------------------------ ticks

    def tick(self) -> pd.DataFrame:
        """Advance one tick and return ONE canonical telemetry row.

        The row has columns ``timestamp`` + ``CANONICAL_METRICS`` (aggregated
        over pods: mean cpu/mem/net, cluster restart count over the last
        ``20`` ticks, EMA-smoothed error/warning event rates per minute) +
        ``anomaly`` (ground truth, 1 only while a fault is an active outage).
        Implements the telemetry side of paper §3.2 with gradual fault buildup.
        """
        self._tick_index += 1
        self.current_time += self.tick_seconds
        self._advance_faults()

        adds, err_lam_add, warn_lam_add, restart_p = self._fault_effects()
        cpu_wave, mem_wave, net_wave = self._waves()

        pod_metrics: dict[str, dict[str, float]] = {}
        for name, pod in self._pods.items():
            cpu = float(np.clip(
                pod.base_cpu + cpu_wave + self._rng.normal(0.0, 2.5) + adds[name]["cpu"],
                0.0, 100.0))
            mem = float(np.clip(
                pod.base_mem + mem_wave + self._rng.normal(0.0, 1.2) + adds[name]["mem"],
                0.0, 100.0))
            net = float(np.clip(
                pod.base_net + net_wave + self._rng.normal(0.0, 25.0) + adds[name]["net"],
                0.0, 1200.0))
            if self._rng.random() < restart_p[name]:
                pod.restart_ticks.append(self._tick_index)
            self._prune_restarts(pod)
            pod_metrics[name] = {
                "cpu": cpu,
                "mem": mem,
                "net": net,
                "restarts": float(len(pod.restart_ticks)),
            }

        per_min = 60.0 / self.tick_seconds
        err_count = int(self._rng.poisson(_BASE_ERR_LAM + err_lam_add))
        warn_count = int(self._rng.poisson(_BASE_WARN_LAM + warn_lam_add))
        self._err_rate = 0.7 * self._err_rate + 0.3 * (err_count * per_min)
        self._warn_rate = 0.7 * self._warn_rate + 0.3 * (warn_count * per_min)

        n = len(pod_metrics)
        agg: dict[str, float] = {
            "cpu_usage": sum(m["cpu"] for m in pod_metrics.values()) / n,
            "memory_usage": sum(m["mem"] for m in pod_metrics.values()) / n,
            "network_io": sum(m["net"] for m in pod_metrics.values()) / n,
            "pod_restarts": float(sum(len(p.restart_ticks) for p in self._pods.values())),
            "error_event_rate": self._err_rate,
            "warning_event_rate": self._warn_rate,
        }
        self._last_pod_metrics = pod_metrics
        self._last_aggregates = agg

        return pd.DataFrame({
            "timestamp": [pd.Timestamp(self.current_time, unit="s")],
            **{m: [float(agg[m])] for m in CANONICAL_METRICS},
            "anomaly": [int(self.in_outage())],
        })

    # ------------------------------------------------------------ fault entry

    def inject_fault(self, kind: str | None = None, pod: str | None = None) -> str:
        """Start a fault ramping on ``pod`` (paper §3.2 gradual buildup).

        Args:
            kind: One of :data:`FAULT_KINDS`; random when ``None``.
            pod: Anchor pod; when ``None`` a pod without an active fault is
                chosen at random.

        Returns:
            The anchor pod name.

        Raises:
            ValueError: On an unknown fault kind or pod name.
        """
        if kind is None:
            kind = str(self._rng.choice(FAULT_KINDS))
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {kind!r}; expected one of {FAULT_KINDS}")
        if pod is None:
            anchored = {f.pod for f in self._faults if f.phase != "recovering"}
            candidates = [name for name in self._pods if name not in anchored]
            if not candidates:
                candidates = list(self._pods)
            pod = candidates[int(self._rng.integers(len(candidates)))]
        elif pod not in self._pods:
            raise ValueError(f"unknown pod {pod!r}")
        ramp = int(self._rng.integers(_RAMP_TICKS_MIN, _RAMP_TICKS_MAX + 1))
        self._faults.append(_Fault(kind=kind, pod=pod, ramp_ticks=ramp))
        return pod

    def maybe_inject_random_fault(self, probability: float) -> str | None:
        """With ``probability`` (per tick), inject a random fault; else ``None``.

        Never exceeds ``2`` concurrently active faults so the closed loop stays
        interpretable.
        """
        if probability <= 0.0:
            return None
        active = sum(1 for f in self._faults if f.phase != "recovering")
        if active >= _MAX_CONCURRENT_FAULTS:
            return None
        if self._rng.random() >= probability:
            return None
        return self.inject_fault()

    # ---------------------------------------------------------- healing hooks
    # Paper Step 6: Action in {RestartPod, RescheduleNode, ScaleUp, DrainNode,
    # AlertAdmin}. Each hook returns bool success; it clears/damps a fault ONLY
    # when it actually addresses that fault's pod/node — a wrong action or a
    # wrong target leaves the fault running.

    def restart_pod(self, pod: str | None = None) -> bool:
        """Restart ``pod`` (``None`` -> unhealthiest pod).

        Clears ``crashloop`` / ``memory_leak`` faults anchored on that pod (a
        fresh container releases leaked memory / resets the crash backoff);
        residual metric elevation then decays over a few ticks. The restart
        itself counts as a restart event. Does NOT clear other fault kinds or
        faults anchored elsewhere.
        """
        if pod is None:
            pod = self.unhealthiest_pod()
        state = self._pods.get(pod)
        if state is None:
            return False
        state.restart_ticks.append(self._tick_index)
        self._prune_restarts(state)
        for fault in self._faults:
            if (fault.pod == pod and fault.phase in ("ramping", "outage")
                    and fault.kind in RESTART_CLEARABLE):
                self._clear(fault, decay=0.45)
        return True

    def reschedule_pod(self, pod: str | None = None) -> bool:
        """Move ``pod`` (``None`` -> unhealthiest) to the least-loaded other node.

        Fixes node-level faults (``network_saturation``) anchored on that pod —
        the noisy neighbour is placed away from the saturated node (engine
        routes network pressure here, SPEC ``SelfHealingEngine``). Fails
        (``False``) when the pod is unknown or there is no other node.
        """
        if pod is None:
            pod = self.unhealthiest_pod()
        state = self._pods.get(pod)
        if state is None:
            return False
        others = [n for n in self.nodes if n != state.node]
        if not others:
            return False
        state.node = min(others, key=self._node_load)
        for fault in self._faults:
            if (fault.pod == pod and fault.phase in ("ramping", "outage")
                    and fault.kind in NODE_LEVEL_FAULTS):
                self._clear(fault, decay=0.55)
        return True

    def scale_up(self) -> bool:
        """Add a replica pod on the least-loaded node (capped at 2x initial pods).

        Damps ``cpu_runaway`` / ``network_saturation`` pressure (severity
        multiplier x ``0.45``): extra capacity absorbs load, the fault drops
        out of the outage band and dissipates over the following ticks. Has no
        effect on ``memory_leak`` / ``crashloop`` faults. ``False`` at the cap.
        """
        if len(self._pods) >= self._max_pods:
            return False
        self._spawn_pod(min(self.nodes, key=self._node_load))
        for fault in self._faults:
            if fault.phase in ("ramping", "outage") and fault.kind in SCALE_DAMPABLE:
                fault.multiplier *= _SCALE_DAMP
                if self._severity(fault) < _CLEAR_SEVERITY:
                    self._clear(fault, decay=0.7)
        return True

    def drain_node(self, node: str | None = None) -> bool:
        """Cordon ``node`` (``None`` -> node of the unhealthiest pod) and move
        its pods to the remaining nodes.

        Fixes node-level faults (``network_saturation``) whose anchor pod runs
        on that node. ``False`` for an unknown node or a single-node cluster.
        """
        if node is None:
            node = self.node_of(self.unhealthiest_pod())
        if node not in self.nodes:
            return False
        others = [n for n in self.nodes if n != node]
        if not others:
            return False
        for fault in self._faults:
            anchor = self._pods.get(fault.pod)
            if (anchor is not None and anchor.node == node
                    and fault.phase in ("ramping", "outage")
                    and fault.kind in NODE_LEVEL_FAULTS):
                self._clear(fault, decay=0.55)
        for state in self._pods.values():
            if state.node == node:
                state.node = min(others, key=self._node_load)
        return True

    def alert_admin(self, detail: str = "") -> bool:
        """Record an administrator alert (paper Step 6, AlertAdmin). Always True."""
        self.alerts.append((self.current_time, detail))
        return True

    # --------------------------------------------------------- introspection

    def unhealthiest_pod(self) -> str:
        """Pod with the highest resource pressure (max utilization ratio across
        cpu/mem/net/restarts, paper Step 3's utilization ratio applied per pod)."""
        metrics = self._last_pod_metrics or self._estimate_pod_metrics()

        def pressure(item: tuple[str, dict[str, float]]) -> float:
            _, m = item
            return max(
                m["cpu"] / max(self._limits["cpu_usage"], 1e-9),
                m["mem"] / max(self._limits["memory_usage"], 1e-9),
                m["net"] / max(self._limits["network_io"], 1e-9),
                m["restarts"] / max(self._limits["pod_restarts"], 1e-9),
            )

        return max(metrics.items(), key=pressure)[0]

    def top_pressure_metric(self) -> str:
        """Canonical metric with the highest utilization (value / limit).

        For cpu/mem/net the peak per-pod utilization is used (the pressure
        point is the worst pod, not the cluster mean); restart and event rates
        are cluster-level. Feeds the engine's context ``top_metric`` so network
        pressure routes to RESCHEDULE_NODE (SPEC ``SelfHealingEngine``).
        """
        pods = self._last_pod_metrics or self._estimate_pod_metrics()
        agg = self._last_aggregates or self._estimate_aggregates()
        pressure = {
            "cpu_usage": max(m["cpu"] for m in pods.values()),
            "memory_usage": max(m["mem"] for m in pods.values()),
            "network_io": max(m["net"] for m in pods.values()),
            "pod_restarts": agg["pod_restarts"],
            "error_event_rate": agg["error_event_rate"],
            "warning_event_rate": agg["warning_event_rate"],
        }
        return max(CANONICAL_METRICS,
                   key=lambda m: pressure[m] / max(self._limits[m], 1e-9))

    def has_active_fault(self) -> bool:
        """Ground truth: a fault is ramping or in outage (paper §3.2 buildup)."""
        return any(f.phase in ("ramping", "outage") for f in self._faults)

    def in_outage(self) -> bool:
        """Ground truth outage: ``anomaly == 1`` right now."""
        return any(f.phase == "outage" for f in self._faults)

    def node_of(self, pod: str) -> str | None:
        """Node currently hosting ``pod`` (``None`` for an unknown pod)."""
        state = self._pods.get(pod)
        return None if state is None else state.node

    # ------------------------------------------------------------- internals

    def _spawn_pod(self, node: str) -> str:
        """Create a pod with randomized baselines on ``node``."""
        name = f"pod-{self._pod_counter:02d}"
        self._pod_counter += 1
        self._pods[name] = _Pod(
            name=name,
            node=node,
            base_cpu=float(self._rng.uniform(18.0, 35.0)),
            base_mem=float(self._rng.uniform(30.0, 45.0)),
            base_net=float(self._rng.uniform(120.0, 320.0)),
        )
        return name

    def _node_load(self, node: str) -> int:
        return sum(1 for p in self._pods.values() if p.node == node)

    def _prune_restarts(self, pod: _Pod) -> None:
        cutoff = self._tick_index - _RESTART_WINDOW_TICKS
        pod.restart_ticks = [t for t in pod.restart_ticks if t > cutoff]

    def _severity(self, fault: _Fault) -> float:
        """Effective severity in [0, 1]: ramp progress x scale damping."""
        if fault.phase == "recovering":
            return fault.residual
        return min(1.0, fault.age / max(1, fault.ramp_ticks)) * fault.multiplier

    def _clear(self, fault: _Fault, decay: float) -> None:
        """Move a fault into the recovering phase (outage/buildup ends now;
        residual metric elevation decays by ``decay`` per tick)."""
        fault.residual = self._severity(fault)
        fault.phase = "recovering"
        fault.decay = decay
        fault.multiplier = 1.0

    def _advance_faults(self) -> None:
        """One tick of fault lifecycle (gradual ramp -> outage -> recovery)."""
        remaining: list[_Fault] = []
        for fault in self._faults:
            if fault.phase == "recovering":
                fault.residual *= fault.decay
                if fault.residual >= 0.02:
                    remaining.append(fault)
                continue
            fault.age += 1
            # Faults damped by scale_up dissipate slowly once fully ramped.
            if fault.multiplier < 1.0 and fault.age >= fault.ramp_ticks:
                fault.multiplier *= _DAMPED_DECAY
            severity = self._severity(fault)
            if fault.phase == "outage":
                fault.outage_age += 1
                if severity < _OUTAGE_SEVERITY:
                    fault.phase = "ramping"  # damped below the outage band
                elif fault.outage_age >= _AUTO_RECOVER_TICKS:
                    # Unhealed long enough that (simulated) operators step in —
                    # gives the --no-healing ablation a finite, painful MTTR.
                    self._clear(fault, decay=0.6)
                    remaining.append(fault)
                    continue
            if fault.phase == "ramping" and fault.age >= fault.ramp_ticks:
                if severity < _CLEAR_SEVERITY:
                    self._clear(fault, decay=0.7)
                elif severity >= _OUTAGE_SEVERITY:
                    fault.phase = "outage"
            remaining.append(fault)
        self._faults = remaining

    def _fault_effects(
        self,
    ) -> tuple[dict[str, dict[str, float]], float, float, dict[str, float]]:
        """Additive per-pod metric pressure + event-rate/restart effects of all
        faults at their current severity (paper §3.2 buildup shapes)."""
        adds = {name: {"cpu": 0.0, "mem": 0.0, "net": 0.0} for name in self._pods}
        err_lam_add = 0.0
        warn_lam_add = 0.0
        restart_p = {name: _BASE_RESTART_P for name in self._pods}
        for fault in self._faults:
            severity = self._severity(fault)
            anchor = self._pods.get(fault.pod)
            if severity <= 0.0 or anchor is None:
                continue
            if fault.kind == "memory_leak":
                adds[fault.pod]["mem"] += severity * (97.0 - anchor.base_mem)
                adds[fault.pod]["cpu"] += severity * 8.0
                err_lam_add += severity * 3.0
                if fault.phase == "outage":  # OOM kills once memory is exhausted
                    restart_p[fault.pod] += 0.10
                for name, pod in self._pods.items():
                    if name != fault.pod:
                        adds[name]["mem"] += severity * (
                            12.0 if pod.node == anchor.node else 4.0)
            elif fault.kind == "cpu_runaway":
                adds[fault.pod]["cpu"] += severity * (99.0 - anchor.base_cpu)
                warn_lam_add += severity * 4.0
                for name, pod in self._pods.items():
                    if name != fault.pod:
                        adds[name]["cpu"] += severity * (
                            10.0 if pod.node == anchor.node else 3.0)
            elif fault.kind == "crashloop":
                restart_p[fault.pod] += severity * 0.5
                err_lam_add += severity * 6.0
                adds[fault.pod]["cpu"] += severity * 8.0
            elif fault.kind == "network_saturation":
                adds[fault.pod]["net"] += severity * (950.0 - anchor.base_net)
                warn_lam_add += severity * 5.0
                for name, pod in self._pods.items():
                    if name != fault.pod:
                        adds[name]["net"] += severity * (
                            350.0 if pod.node == anchor.node else 60.0)
        return adds, err_lam_add, warn_lam_add, restart_p

    def _waves(self) -> tuple[float, float, float]:
        """Deterministic diurnal-style load variation (cpu, mem, net)."""
        day = 2.0 * math.pi * (self.current_time % 86400.0) / 86400.0
        busy = 2.0 * math.pi * (self.current_time % 900.0) / 900.0
        cpu_wave = 6.0 * math.sin(day) + 2.5 * math.sin(busy)
        mem_wave = 1.5 * math.sin(day)
        net_wave = 45.0 * math.sin(day) + 15.0 * math.sin(busy)
        return cpu_wave, mem_wave, net_wave

    def _estimate_pod_metrics(self) -> dict[str, dict[str, float]]:
        """Noise-free per-pod metric estimate (used before the first tick so
        introspection never consumes RNG state)."""
        adds, _, _, _ = self._fault_effects()
        cpu_wave, mem_wave, net_wave = self._waves()
        estimate: dict[str, dict[str, float]] = {}
        for name, pod in self._pods.items():
            estimate[name] = {
                "cpu": float(np.clip(pod.base_cpu + cpu_wave + adds[name]["cpu"], 0.0, 100.0)),
                "mem": float(np.clip(pod.base_mem + mem_wave + adds[name]["mem"], 0.0, 100.0)),
                "net": float(np.clip(pod.base_net + net_wave + adds[name]["net"], 0.0, 1200.0)),
                "restarts": float(len(pod.restart_ticks)),
            }
        return estimate

    def _estimate_aggregates(self) -> dict[str, float]:
        """Noise-free cluster aggregate estimate (pre-first-tick fallback)."""
        pods = self._estimate_pod_metrics()
        n = len(pods)
        return {
            "cpu_usage": sum(m["cpu"] for m in pods.values()) / n,
            "memory_usage": sum(m["mem"] for m in pods.values()) / n,
            "network_io": sum(m["net"] for m in pods.values()) / n,
            "pod_restarts": float(sum(len(p.restart_ticks) for p in self._pods.values())),
            "error_event_rate": self._err_rate,
            "warning_event_rate": self._warn_rate,
        }
