"""Dataset generation and loading for the self-healing cloud platform.

Implements the data layer of the paper *"Self-Healing Cloud Platforms Using
LSTM-based Predictive Models for Early Fault Detection"* (IJCET 16(5) 2025):

* **§3.1 (Dataset description, Table 1)** — a synthetic Kubernetes telemetry
  generator producing ~``n_records`` structured log entries with the Table 1
  schema (timestamp, namespace, pod name, event type, message, reason, status,
  anomaly label) plus a cluster-aggregated metrics time series (CPU, memory,
  network I/O, pod restarts, error/warning event rates) used for training.
* **§3.2 (Fault scenarios)** — injected faults (memory leaks, CPU runaway,
  crash loops, network saturation) *"build up gradually but do not emerge
  suddenly"*: each fault ramps over many ticks before the labeled anomaly
  (outage) window, then decays over a recovery phase.
* **§3.1 [17]** — a loader for the cited Kaggle microservice traffic dataset
  (``balanced_shuffled_traffic.csv``) mapped onto the canonical frame.

A **canonical telemetry frame** (see SPEC.md) is a :class:`pandas.DataFrame`
with a sorted ``timestamp`` column (datetime64), float64 feature columns, and
an int 0/1 ``anomaly`` ground-truth column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from selfheal.config import resolve_path

__all__ = [
    "KAGGLE_LABEL_COL",
    "LOG_SCHEMA",
    "CANONICAL_METRICS",
    "generate_k8s_telemetry",
    "load_kaggle_traffic",
    "load_synthetic",
    "load_dataset",
    "feature_columns",
]

#: Label column of the cited Kaggle traffic dataset [17].
KAGGLE_LABEL_COL = "Label"

#: Paper Table 1 — schema of the structured Kubernetes log entries.
LOG_SCHEMA = [
    "timestamp",
    "namespace",
    "pod_name",
    "event_type",
    "message",
    "reason",
    "status",
    "anomaly_label",
]

#: Canonical cluster-aggregated metric columns (paper §3.1).
CANONICAL_METRICS = [
    "cpu_usage",
    "memory_usage",
    "network_io",
    "pod_restarts",
    "error_event_rate",
    "warning_event_rate",
]

# --------------------------------------------------------------------------- #
# Synthetic generator internals (paper §3.1 Table 1 + §3.2 fault scenarios)   #
# --------------------------------------------------------------------------- #

_SIM_EPOCH = pd.Timestamp("2025-01-06 00:00:00")
_MIN_TICKS = 60

# Baseline per-tick Poisson rates of emitted log events in a healthy cluster.
_BASE_NORMAL_RATE = 3.0
_BASE_WARN_RATE = 1.2
_BASE_ERR_RATE = 0.35
_BASE_RESTART_RATE = 0.04

# Rough calibration constants used to size the tick count so that the total
# number of emitted log rows approximates cfg n_records (see §3.1, ~85,000).
_FAULT_EXTRA_ROWS = 800.0  # extra Warning/Error rows one fault cycle emits
_FAULT_OVERHEAD_TICKS = 160.0  # ramp+outage+recovery+gap ticks per fault cycle

_FAULT_KINDS = ("memory_leak", "cpu_runaway", "crashloop", "network_saturation")

# Per-fault additive effects on the cluster metrics (paper §3.2).  ``ramp``
# values are the peak elevation reached at the *end* of the gradual buildup
# (scaled by progress p in [0, 1]); ``outage`` values apply while anomaly=1.
# Keys: cpu/mem (%), net (MB/s), warn/err (events per tick), restart (per tick).
_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "memory_leak": {
        "ramp": {"cpu": 9.0, "mem": 46.0, "net": 90.0, "warn": 5.5, "err": 5.0, "restart": 0.9},
        "outage": {"cpu": 12.0, "mem": 50.0, "net": 110.0, "warn": 13.0, "err": 9.0, "restart": 2.4},
    },
    "cpu_runaway": {
        "ramp": {"cpu": 56.0, "mem": 6.0, "net": 110.0, "warn": 5.0, "err": 4.0, "restart": 0.8},
        "outage": {"cpu": 61.0, "mem": 8.0, "net": 140.0, "warn": 11.0, "err": 7.0, "restart": 1.8},
    },
    "crashloop": {
        "ramp": {"cpu": 9.0, "mem": 5.0, "net": 85.0, "warn": 7.0, "err": 5.5, "restart": 3.2},
        "outage": {"cpu": 11.0, "mem": 7.0, "net": 100.0, "warn": 15.0, "err": 10.0, "restart": 4.3},
    },
    "network_saturation": {
        "ramp": {"cpu": 11.0, "mem": 5.0, "net": 680.0, "warn": 6.0, "err": 4.5, "restart": 0.8},
        "outage": {"cpu": 14.0, "mem": 7.0, "net": 760.0, "warn": 13.0, "err": 8.0, "restart": 1.5},
    },
}

# Event pools: (reason, message template, pod status).  Reasons follow real
# Kubernetes event vocabulary (OOMKilled, BackOff, FailedScheduling,
# NodeNotReady, Unhealthy, ...) as sketched in paper Table 1.
_NORMAL_POOL: list[tuple[str, str, str]] = [
    ("Scheduled", "Successfully assigned {ns}/{pod} to {node}", "Pending"),
    ("Pulling", "Pulling image for container app in pod {pod}", "Pending"),
    ("Pulled", "Container image already present on machine for pod {pod}", "Running"),
    ("Created", "Created container app in pod {pod}", "Running"),
    ("Started", "Started container app in pod {pod}", "Running"),
    ("SuccessfulCreate", "Created pod {pod} via ReplicaSet", "Running"),
    ("Killing", "Stopping container app in pod {pod}", "Running"),
]
_NORMAL_WEIGHTS: dict[str | None, list[float]] = {
    None: [0.8, 0.7, 1.0, 1.0, 1.2, 0.5, 0.4],
}

_WARNING_POOL: list[tuple[str, str, str]] = [
    ("BackOff", "Back-off restarting failed container app in pod {pod}", "CrashLoopBackOff"),
    ("Unhealthy", "Liveness probe failed for pod {pod}: HTTP probe failed with statuscode: 500", "Running"),
    ("FailedScheduling", "0/{n_nodes} nodes are available: insufficient cpu/memory for pod {pod}", "Pending"),
    ("NodeNotReady", "Node {node} status is now: NodeNotReady", "Unknown"),
    ("FailedMount", "Unable to attach or mount volumes for pod {pod}: timed out waiting for the condition", "Pending"),
    ("ProbeWarning", "Readiness probe warning for pod {pod}: response took longer than expected", "Running"),
    ("NetworkNotReady", "network is not ready: container runtime network not ready on node {node}", "Pending"),
]
_WARNING_WEIGHTS: dict[str | None, list[float]] = {
    None: [1.0, 1.2, 0.6, 0.25, 0.4, 0.8, 0.25],
    "memory_leak": [1.2, 3.0, 1.2, 0.2, 0.3, 1.5, 0.1],
    "cpu_runaway": [1.0, 3.0, 2.0, 0.4, 0.2, 1.6, 0.2],
    "crashloop": [5.0, 2.0, 0.8, 0.2, 0.4, 1.0, 0.1],
    "network_saturation": [1.0, 1.5, 0.8, 2.5, 0.3, 1.0, 3.5],
}

_ERROR_POOL: list[tuple[str, str, str]] = [
    ("OOMKilled", "Container app in pod {pod} was OOMKilled: memory limit exceeded", "Failed"),
    ("BackOff", "Back-off restarting failed container app in pod {pod} (CrashLoopBackOff)", "CrashLoopBackOff"),
    ("FailedScheduling", "0/{n_nodes} nodes are available: insufficient resources for pod {pod}", "Pending"),
    ("NodeNotReady", "Node {node} is NotReady; evicting pods", "Unknown"),
    ("Unhealthy", "Liveness probe failed repeatedly for pod {pod}; container will be restarted", "Running"),
    ("Evicted", "Pod {pod} evicted from node {node} due to resource pressure", "Failed"),
    ("Failed", "Error: failed to start container app in pod {pod}", "Failed"),
]
_ERROR_WEIGHTS: dict[str | None, list[float]] = {
    None: [0.3, 0.8, 0.5, 0.2, 0.8, 0.3, 1.0],
    "memory_leak": [5.0, 1.0, 0.8, 0.2, 1.0, 1.8, 0.7],
    "cpu_runaway": [0.4, 1.0, 1.8, 0.5, 2.5, 2.0, 1.0],
    "crashloop": [0.8, 5.0, 0.5, 0.2, 1.5, 0.4, 2.0],
    "network_saturation": [0.3, 0.8, 1.0, 3.5, 1.8, 1.2, 1.0],
}

_SERVICES = (
    "web", "api", "auth", "cache", "worker", "frontend", "payments", "search",
    "ingest", "queue", "scheduler", "billing", "orders", "catalog", "gateway", "notify",
)


@dataclass(frozen=True)
class _FaultWindow:
    """One §3.2 fault occurrence: gradual ramp -> labeled outage -> recovery."""

    kind: str
    start: int  # first ramp tick
    ramp: int  # ticks of gradual buildup (anomaly stays 0)
    outage: int  # ticks with anomaly == 1
    recovery: int  # ticks decaying back to baseline
    target_pod: int  # index of the pod the fault is attributed to

    @property
    def end(self) -> int:
        return self.start + self.ramp + self.outage + self.recovery


def _normalized(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    return arr / arr.sum()


def _make_pods(
    rng: np.random.Generator, n_pods: int, n_nodes: int, namespaces: list[str]
) -> list[dict[str, str]]:
    """Build stable, realistic pod identities (name/namespace/node)."""
    hex_chars = "0123456789abcdef"
    suf_chars = "bcdfghjklmnpqrstvwxz2456789"
    pods: list[dict[str, str]] = []
    for i in range(n_pods):
        svc = _SERVICES[i % len(_SERVICES)]
        rs_hash = "".join(hex_chars[int(c)] for c in rng.integers(0, 16, size=9))
        suffix = "".join(suf_chars[int(c)] for c in rng.integers(0, len(suf_chars), size=5))
        pods.append(
            {
                "name": f"{svc}-{rs_hash}-{suffix}",
                "namespace": namespaces[int(rng.integers(len(namespaces)))],
                "node": f"node-{(i % n_nodes) + 1}",
            }
        )
    return pods


def _draw_event(
    rng: np.random.Generator, kind: str, start: int, limit: int, n_pods: int
) -> _FaultWindow | None:
    """Draw one fault window starting at ``start`` that fits before ``limit``.

    Durations follow §3.2: a long gradual ramp (tens of ticks) before a much
    shorter labeled anomaly window.  Returns None when there is no room.
    """
    ramp = int(rng.integers(40, 91))
    outage = int(rng.integers(10, 31))
    recovery = int(rng.integers(8, 21))
    avail = limit - start - 1
    total = ramp + outage + recovery
    if total > avail:
        if avail < 15:
            return None
        factor = avail / total
        ramp = max(8, int(ramp * factor))
        outage = max(3, int(outage * factor))
        recovery = max(2, int(recovery * factor))
        if ramp + outage + recovery > avail:
            return None
    return _FaultWindow(kind, start, ramp, outage, recovery, int(rng.integers(n_pods)))


def _largest_gap(events: list[_FaultWindow], n_ticks: int) -> tuple[int, int] | None:
    """Largest fault-free interval (with 10-tick margins) on the timeline."""
    gaps: list[tuple[int, int]] = []
    cursor = 10
    for ev in sorted(events, key=lambda e: e.start):
        if ev.start - 10 - cursor >= 1:
            gaps.append((cursor, ev.start - 10))
        cursor = max(cursor, ev.end + 10)
    if n_ticks - 10 - cursor >= 1:
        gaps.append((cursor, n_ticks - 10))
    if not gaps:
        return None
    return max(gaps, key=lambda g: g[1] - g[0])


def _schedule_faults(
    rng: np.random.Generator, n_ticks: int, fault_rate: float, n_pods: int
) -> list[_FaultWindow]:
    """Schedule non-overlapping §3.2 fault windows across the timeline.

    Fault kinds rotate round-robin (random initial offset) so every scenario
    class appears in long simulations.  At least one fault (up to four,
    depending on the timeline length) is guaranteed even when the stochastic
    pass yields none, so anomalies are always present in the dataset.
    """
    events: list[_FaultWindow] = []
    kind_i = int(rng.integers(len(_FAULT_KINDS)))
    t = min(40, max(5, n_ticks // 10))
    while t < n_ticks - 20:
        if rng.random() < fault_rate:
            ev = _draw_event(rng, _FAULT_KINDS[kind_i % len(_FAULT_KINDS)], t, n_ticks, n_pods)
            if ev is not None:
                events.append(ev)
                kind_i += 1
                t = ev.end + int(rng.integers(30, 90))
                continue
        t += 1

    desired = max(1, min(4, n_ticks // 900))
    attempts = 0
    while len(events) < desired and attempts < 8:
        attempts += 1
        gap = _largest_gap(events, n_ticks)
        if gap is None or gap[1] - gap[0] < 30:
            break
        start = gap[0] + (gap[1] - gap[0]) // 4
        ev = _draw_event(rng, _FAULT_KINDS[kind_i % len(_FAULT_KINDS)], start, gap[1], n_pods)
        if ev is None:
            break
        events.append(ev)
        kind_i += 1
    events.sort(key=lambda e: e.start)
    return events


def _fault_timeline(
    events: list[_FaultWindow], n_ticks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand fault windows into per-tick arrays.

    Returns (phase, kind_id, progress, target_pod):
      phase    0=normal, 1=ramp (gradual buildup), 2=outage (anomaly=1), 3=recovery
      kind_id  index into _FAULT_KINDS, -1 when no fault is active
      progress ramp: buildup fraction (0..1]; outage: 1; recovery: decays 1->0
    """
    phase = np.zeros(n_ticks, dtype=np.int8)
    kind_id = np.full(n_ticks, -1, dtype=np.int16)
    progress = np.zeros(n_ticks, dtype=np.float64)
    target = np.full(n_ticks, -1, dtype=np.int32)
    for ev in events:
        k = _FAULT_KINDS.index(ev.kind)
        r0, o0, c0 = ev.start, ev.start + ev.ramp, ev.start + ev.ramp + ev.outage
        end = ev.end
        phase[r0:o0] = 1
        progress[r0:o0] = np.arange(1, ev.ramp + 1, dtype=np.float64) / ev.ramp
        phase[o0:c0] = 2
        progress[o0:c0] = 1.0
        phase[c0:end] = 3
        progress[c0:end] = 1.0 - np.arange(1, ev.recovery + 1, dtype=np.float64) / ev.recovery
        kind_id[r0:end] = k
        target[r0:end] = ev.target_pod
    return phase, kind_id, progress, target


def generate_k8s_telemetry(
    cfg: dict, out_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic Kubernetes telemetry (paper §3.1 Table 1 + §3.2).

    Simulates a small cluster (``cfg["dataset"]["synthetic"]``: n_pods pods on
    n_nodes nodes) at a fixed tick interval and produces two frames, both also
    written as CSV into ``out_dir`` (default ``cfg["paths"]["data_raw"]``):

    * ``logs_df`` -> ``k8s_logs.csv``: ~``n_records`` structured log events
      with **exactly** the Table 1 :data:`LOG_SCHEMA` columns.  Per tick, the
      number of Warning and Error rows is Poisson-drawn from the current fault
      state and **equals** the ``warning_event_rate`` / ``error_event_rate``
      metric values of that tick, so logs and metrics are statistically
      consistent by construction; reasons (OOMKilled, BackOff,
      FailedScheduling, NodeNotReady, Unhealthy, ...) are weighted by the
      active fault kind.
    * ``metrics_df`` -> ``k8s_metrics.csv``: a canonical telemetry frame with
      one CLUSTER-AGGREGATED row per tick: ``timestamp`` +
      :data:`CANONICAL_METRICS` (float64) + ``anomaly`` (int 0/1).

    Faults (§3.2: memory leak, CPU runaway, crash loop, network saturation)
    **build up gradually**: each ramps over many ticks (progress ``p`` with a
    ``p**1.5`` metric ramp, warnings ``p**1.2`` appearing before errors
    ``p**2``) before the labeled outage window where ``anomaly == 1``, then
    decays through a recovery phase.

    All randomness derives from ``cfg["project"]["seed"]`` — the output is
    fully deterministic for a given config.

    Raises:
        ValueError: on non-positive ``n_records`` or ``tick_seconds``.
    """
    ds_cfg = cfg.get("dataset", {}).get("synthetic", {}) or {}
    n_records = int(ds_cfg.get("n_records", 85000))
    if n_records <= 0:
        raise ValueError(f"dataset.synthetic.n_records must be positive, got {n_records}")
    n_pods = int(ds_cfg.get("n_pods", 12))
    n_nodes = int(ds_cfg.get("n_nodes", 3))
    if n_pods < 1 or n_nodes < 1:
        raise ValueError("dataset.synthetic.n_pods and n_nodes must be >= 1")
    tick_seconds = float(ds_cfg.get("tick_seconds", 15))
    if tick_seconds <= 0:
        raise ValueError(f"dataset.synthetic.tick_seconds must be positive, got {tick_seconds}")
    fault_rate = float(np.clip(float(ds_cfg.get("fault_rate", 0.004)), 0.0, 1.0))
    namespaces = list(ds_cfg.get("namespaces") or ["default"])

    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    # Size the timeline so total emitted log rows approximate n_records:
    # baseline event rate plus the expected extra rows contributed per fault
    # cycle (ramp + outage + recovery + cooldown gap).
    base_rate = _BASE_NORMAL_RATE + _BASE_WARN_RATE + _BASE_ERR_RATE
    cycle_ticks = (1.0 / max(fault_rate, 1e-9)) + _FAULT_OVERHEAD_TICKS
    events_per_tick = base_rate + _FAULT_EXTRA_ROWS / cycle_ticks
    n_ticks = max(_MIN_TICKS, int(round(n_records / events_per_tick)))

    pods = _make_pods(rng, n_pods, n_nodes, namespaces)
    events = _schedule_faults(rng, n_ticks, fault_rate, n_pods)
    phase, kind_id, progress, target = _fault_timeline(events, n_ticks)

    normal_w = _normalized(_NORMAL_WEIGHTS[None])
    warn_w = {k: _normalized(v) for k, v in _WARNING_WEIGHTS.items()}
    err_w = {k: _normalized(v) for k, v in _ERROR_WEIGHTS.items()}

    # Metric arrays (float64) — canonical order per CANONICAL_METRICS.
    cpu_arr = np.empty(n_ticks, dtype=np.float64)
    mem_arr = np.empty(n_ticks, dtype=np.float64)
    net_arr = np.empty(n_ticks, dtype=np.float64)
    restart_arr = np.empty(n_ticks, dtype=np.float64)
    err_arr = np.empty(n_ticks, dtype=np.float64)
    warn_arr = np.empty(n_ticks, dtype=np.float64)
    anomaly_arr = (phase == 2).astype(np.int64)

    # Log row accumulators (Table 1 columns).
    l_off: list[float] = []
    l_ns: list[str] = []
    l_pod: list[str] = []
    l_type: list[str] = []
    l_msg: list[str] = []
    l_reason: list[str] = []
    l_status: list[str] = []
    l_label: list[int] = []

    def _emit(
        tick: int, severity: str, count: int, pool: list[tuple[str, str, str]],
        weights: np.ndarray, tgt: int, label: int,
    ) -> None:
        """Emit ``count`` log rows of one severity for one tick."""
        if count <= 0:
            return
        choices = rng.choice(len(pool), size=count, p=weights)
        offsets = rng.uniform(0.0, tick_seconds, size=count)
        base_off = tick * tick_seconds
        for j in range(count):
            if severity != "Normal" and tgt >= 0 and rng.random() < 0.7:
                pod = pods[tgt]  # fault-related events cluster on the faulty pod
            else:
                pod = pods[int(rng.integers(n_pods))]
            reason, template, status = pool[int(choices[j])]
            l_off.append(base_off + float(offsets[j]))
            l_ns.append(pod["namespace"])
            l_pod.append(pod["name"])
            l_type.append(severity)
            l_msg.append(
                template.format(pod=pod["name"], ns=pod["namespace"], node=pod["node"], n_nodes=n_nodes)
            )
            l_reason.append(reason)
            l_status.append(status)
            l_label.append(label)

    # AR(1) baseline noise states (cluster-level slow wander).
    cpu_ar = mem_ar = net_ar = 0.0

    for t in range(n_ticks):
        tod = ((t * tick_seconds) % 86400.0) / 86400.0  # time of day fraction
        cpu_ar = 0.9 * cpu_ar + rng.normal(0.0, 1.1)
        mem_ar = 0.9 * mem_ar + rng.normal(0.0, 0.8)
        net_ar = 0.9 * net_ar + rng.normal(0.0, 12.0)

        cpu = 34.0 + 4.0 * np.sin(2.0 * np.pi * (tod - 0.30)) + cpu_ar
        mem = 47.0 + 3.0 * np.sin(2.0 * np.pi * (tod - 0.40)) + mem_ar
        net = 210.0 + 40.0 * np.sin(2.0 * np.pi * (tod - 0.25)) + net_ar
        lam_warn = _BASE_WARN_RATE
        lam_err = _BASE_ERR_RATE
        lam_restart = _BASE_RESTART_RATE

        ph = int(phase[t])
        kind = _FAULT_KINDS[int(kind_id[t])] if kind_id[t] >= 0 else None
        if kind is not None:
            prof = _PROFILES[kind]
            p = float(progress[t])
            if ph == 1:  # gradual buildup (§3.2) — warnings lead, errors lag
                ramp = prof["ramp"]
                cpu += ramp["cpu"] * p**1.5
                mem += ramp["mem"] * p**1.5
                net += ramp["net"] * p**1.5
                lam_warn += ramp["warn"] * p**1.2
                lam_err += ramp["err"] * p**2.0
                lam_restart += ramp["restart"] * p**1.5
            elif ph == 2:  # labeled anomaly window (outage)
                out = prof["outage"]
                jitter = 0.92 + 0.08 * rng.random()
                cpu += out["cpu"] * jitter
                mem += out["mem"] * jitter
                net += out["net"] * jitter
                lam_warn += out["warn"] * jitter
                lam_err += out["err"] * jitter
                lam_restart += out["restart"] * jitter
            else:  # recovery: decay back toward baseline
                out = prof["outage"]
                cpu += out["cpu"] * 0.7 * p
                mem += out["mem"] * 0.7 * p
                net += out["net"] * 0.7 * p
                lam_warn += out["warn"] * 0.7 * p
                lam_err += out["err"] * 0.7 * p
                lam_restart += out["restart"] * 0.7 * p

        cpu = float(np.clip(cpu + rng.normal(0.0, 0.8), 0.0, 100.0))
        mem = float(np.clip(mem + rng.normal(0.0, 0.6), 0.0, 100.0))
        net = float(np.clip(net + rng.normal(0.0, 5.0), 0.0, 1050.0))
        warn_n = int(min(rng.poisson(float(np.clip(lam_warn, 0.05, 45.0))), 60))
        err_n = int(min(rng.poisson(float(np.clip(lam_err, 0.02, 25.0))), 40))
        restart_n = int(min(rng.poisson(float(np.clip(lam_restart, 0.0, 6.0))), 10))

        cpu_arr[t] = cpu
        mem_arr[t] = mem
        net_arr[t] = net
        restart_arr[t] = float(restart_n)
        err_arr[t] = float(err_n)
        warn_arr[t] = float(warn_n)

        # Log emission — Warning/Error counts equal the metric values above.
        label = int(anomaly_arr[t])
        tgt = int(target[t])
        normal_n = int(min(rng.poisson(_BASE_NORMAL_RATE + 0.8 * restart_n), 20))
        _emit(t, "Normal", normal_n, _NORMAL_POOL, normal_w, tgt, label)
        _emit(t, "Warning", warn_n, _WARNING_POOL, warn_w.get(kind, warn_w[None]), tgt, label)
        _emit(t, "Error", err_n, _ERROR_POOL, err_w.get(kind, err_w[None]), tgt, label)

    tick_index = _SIM_EPOCH + pd.to_timedelta(np.arange(n_ticks, dtype=np.float64) * tick_seconds, unit="s")
    metrics_df = pd.DataFrame(
        {
            "timestamp": tick_index,
            "cpu_usage": cpu_arr,
            "memory_usage": mem_arr,
            "network_io": net_arr,
            "pod_restarts": restart_arr,
            "error_event_rate": err_arr,
            "warning_event_rate": warn_arr,
            "anomaly": anomaly_arr,
        }
    )

    logs_df = pd.DataFrame(
        {
            "timestamp": _SIM_EPOCH + pd.to_timedelta(np.asarray(l_off, dtype=np.float64), unit="s"),
            "namespace": l_ns,
            "pod_name": l_pod,
            "event_type": l_type,
            "message": l_msg,
            "reason": l_reason,
            "status": l_status,
            "anomaly_label": np.asarray(l_label, dtype=np.int64),
        }
    )
    logs_df = logs_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    logs_df = logs_df[LOG_SCHEMA]

    out_path = resolve_path(cfg, out_dir) if out_dir is not None else resolve_path(cfg, cfg["paths"]["data_raw"])
    out_path.mkdir(parents=True, exist_ok=True)
    logs_df.to_csv(out_path / "k8s_logs.csv", index=False)
    metrics_df.to_csv(out_path / "k8s_metrics.csv", index=False)
    return logs_df, metrics_df


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #


def load_kaggle_traffic(path: str | Path, max_raw_features: int = 12) -> pd.DataFrame:
    """Parse the cited Kaggle microservice traffic CSV (paper §3.1, ref [17]).

    Steps (per SPEC): sort by ``Timestamp``; keep numeric candidate features,
    excluding ports/protocol identifier columns and any column whose inf/NaN
    share exceeds 0.5; replace remaining ±inf with NaN then forward/backward
    fill; select the top ``max_raw_features`` features by variance
    (descending-variance column order); map ``Label`` to the ``anomaly``
    column.  Returns a canonical telemetry frame.

    Raises:
        FileNotFoundError: when the CSV does not exist.
        ValueError: on an empty/invalid CSV or non-positive max_raw_features.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle traffic CSV not found: {csv_path}")
    if max_raw_features < 1:
        raise ValueError(f"max_raw_features must be >= 1, got {max_raw_features}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"Kaggle traffic CSV is empty: {csv_path}")
    if "Timestamp" not in df.columns or KAGGLE_LABEL_COL not in df.columns:
        raise ValueError(
            f"Kaggle traffic CSV must contain 'Timestamp' and '{KAGGLE_LABEL_COL}' columns"
        )

    ts = pd.to_datetime(df["Timestamp"], format="mixed", errors="coerce")
    valid = ts.notna()
    if not bool(valid.any()):
        raise ValueError(f"No parseable 'Timestamp' values in {csv_path}")
    df = df.loc[valid].copy()
    df["__ts"] = ts.loc[valid]
    df = df.sort_values("__ts", kind="stable").reset_index(drop=True)

    # anomaly = Label (numeric 0/1, or attack-name strings mapped to 1).
    lab = df[KAGGLE_LABEL_COL]
    num = pd.to_numeric(lab, errors="coerce")
    str_flag = ~lab.astype(str).str.strip().str.lower().isin({"benign", "normal", "0", ""})
    anomaly = np.where(num.notna().to_numpy(), num.fillna(0).to_numpy() != 0, str_flag.to_numpy())
    anomaly = anomaly.astype(np.int64)

    # Candidate features: numeric, not the label, not port/protocol identifiers.
    numeric = df.select_dtypes(include="number")
    candidates: list[str] = []
    for col in numeric.columns:
        low = col.lower()
        if col in (KAGGLE_LABEL_COL, "__ts") or "port" in low or "protocol" in low:
            continue
        vals = numeric[col].to_numpy(dtype=np.float64)
        if float(np.mean(~np.isfinite(vals))) > 0.5:
            continue  # more than half inf/NaN -> drop the column entirely
        candidates.append(col)
    if not candidates:
        raise ValueError(f"No usable numeric feature columns in {csv_path}")

    feats = numeric[candidates].astype(np.float64)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    feats = feats.fillna(0.0)  # safety net for pathological all-NaN survivors

    variances = feats.var().sort_values(ascending=False, kind="stable")
    top = list(variances.index[:max_raw_features])

    out = pd.DataFrame({"timestamp": df["__ts"].to_numpy()})
    for col in top:
        out[col] = feats[col].to_numpy(dtype=np.float64)
    out["anomaly"] = anomaly
    return out


def load_synthetic(path: str | Path) -> pd.DataFrame:
    """Read a generated ``k8s_metrics.csv`` back into a canonical frame (§3.1).

    Raises:
        FileNotFoundError: when the CSV does not exist.
        ValueError: when required columns are missing or the file is empty.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Synthetic metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Synthetic metrics CSV is empty: {csv_path}")
    if "timestamp" not in df.columns or "anomaly" not in df.columns:
        raise ValueError(f"{csv_path} is not a canonical metrics CSV (timestamp/anomaly missing)")
    feats = [c for c in df.columns if c not in ("timestamp", "anomaly")]
    if not feats:
        raise ValueError(f"{csv_path} contains no feature columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in feats:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).astype(np.int64)
    return df[["timestamp", *feats, "anomaly"]]


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Load the configured dataset as a canonical frame (paper §3.1).

    Dispatches on ``cfg["dataset"]["source"]``:

    * ``"synthetic"`` — read ``k8s_metrics.csv`` from ``paths.data_raw``,
      generating the telemetry first (:func:`generate_k8s_telemetry`) when the
      CSV does not exist yet.
    * ``"kaggle"`` — parse the cited Kaggle CSV via :func:`load_kaggle_traffic`.

    Raises:
        ValueError: for an unknown ``dataset.source``.
    """
    ds_cfg = cfg.get("dataset", {}) or {}
    source = str(ds_cfg.get("source", "synthetic")).strip().lower()
    if source == "synthetic":
        metrics_csv = resolve_path(cfg, cfg["paths"]["data_raw"]) / "k8s_metrics.csv"
        if not metrics_csv.exists():
            generate_k8s_telemetry(cfg)
        return load_synthetic(metrics_csv)
    if source == "kaggle":
        csv_rel = ds_cfg.get("kaggle_csv", "data/raw/balanced_shuffled_traffic.csv")
        return load_kaggle_traffic(
            resolve_path(cfg, csv_rel), int(ds_cfg.get("max_raw_features", 12))
        )
    raise ValueError(f"Unknown dataset.source {source!r}; expected 'synthetic' or 'kaggle'")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns of a canonical frame: everything but timestamp/anomaly."""
    return [c for c in df.columns if c not in ("timestamp", "anomaly")]
