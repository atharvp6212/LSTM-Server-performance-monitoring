"""Tests for selfheal.simulation.cluster_sim (closed-loop testbed, Steps 6-7).

SPEC test list: tick returns a canonical row; an injected memory_leak ramps
memory usage; restart_pod during an outage leads to recovery within a bounded
number of ticks.  Random background faults are disabled (fault_rate 0) so
every scenario is fully scripted and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from selfheal.data.dataset import CANONICAL_METRICS
from selfheal.simulation.cluster_sim import SimulatedCluster

SEED = 42


@pytest.fixture()
def quiet_cfg(cfg: dict) -> dict:
    """Config with spontaneous fault injection disabled."""
    cfg["dataset"]["synthetic"]["fault_rate"] = 0.0
    return cfg


def _tick_frame(cluster: SimulatedCluster, n: int) -> pd.DataFrame:
    return pd.concat([cluster.tick() for _ in range(n)], ignore_index=True)


def test_tick_returns_canonical_row(quiet_cfg: dict) -> None:
    """One tick -> one canonical telemetry row; sim time advances per cfg."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    t0 = float(cluster.current_time)
    row = cluster.tick()

    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    assert set(row.columns) == {"timestamp", "anomaly"} | set(CANONICAL_METRICS)
    assert pd.api.types.is_datetime64_any_dtype(row["timestamp"])
    values = row[CANONICAL_METRICS].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert int(row["anomaly"].iloc[0]) in (0, 1)

    tick_seconds = quiet_cfg["dataset"]["synthetic"]["tick_seconds"]
    assert float(cluster.current_time) - t0 == pytest.approx(tick_seconds)


def test_ticks_are_time_ordered_and_healthy_without_faults(quiet_cfg: dict) -> None:
    """With fault_rate 0 and no injection: increasing timestamps, no outage."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    frame = _tick_frame(cluster, 20)
    ts = frame["timestamp"].to_numpy()
    assert (ts[1:] > ts[:-1]).all(), "tick timestamps must strictly increase"
    assert int(frame["anomaly"].sum()) == 0
    assert cluster.has_active_fault() is False
    assert cluster.in_outage() is False


def test_injected_memory_leak_ramps_memory(quiet_cfg: dict) -> None:
    """A memory_leak fault builds up gradually (paper §3.2) after injection."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    baseline = _tick_frame(cluster, 10)

    pod = cluster.inject_fault("memory_leak")
    assert isinstance(pod, str) and pod
    assert cluster.has_active_fault() is True

    after = _tick_frame(cluster, 40)
    baseline_mem = float(baseline["memory_usage"].mean())
    ramped_mem = float(after["memory_usage"].tail(10).mean())
    assert ramped_mem > baseline_mem, (
        f"memory_usage must ramp above the healthy baseline "
        f"({ramped_mem:.3f} <= {baseline_mem:.3f})"
    )


def test_restart_pod_during_outage_recovers(quiet_cfg: dict) -> None:
    """Unhealed fault becomes an outage; restarting the faulty pod heals it."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    pod = cluster.inject_fault("memory_leak")

    # Unhealed, the ramp must eventually become an active outage (anomaly=1).
    for _ in range(500):
        cluster.tick()
        if cluster.in_outage():
            break
    assert cluster.in_outage(), "an unhealed fault must escalate to an outage"

    assert cluster.restart_pod(pod) is True

    # Healing the faulty pod must clear the outage within a bounded time.
    recovered = False
    for _ in range(200):
        row = cluster.tick()
        if not cluster.in_outage() and int(row["anomaly"].iloc[0]) == 0:
            recovered = True
            break
    assert recovered, "restart_pod on the faulty pod must end the outage"


def test_introspection_and_admin_hooks(quiet_cfg: dict) -> None:
    """unhealthiest_pod / top_pressure_metric / alert_admin honor the SPEC."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    cluster.tick()

    pod = cluster.unhealthiest_pod()
    assert isinstance(pod, str) and pod
    assert cluster.top_pressure_metric() in CANONICAL_METRICS
    assert cluster.alert_admin("unit-test alert") is True
    assert isinstance(cluster.scale_up(), bool)
    assert isinstance(cluster.drain_node(None), bool)


def test_maybe_inject_random_fault_probability_zero(quiet_cfg: dict) -> None:
    """probability 0.0 never injects and returns None."""
    cluster = SimulatedCluster(quiet_cfg, seed=SEED)
    for _ in range(10):
        assert cluster.maybe_inject_random_fault(0.0) is None
        cluster.tick()
    assert cluster.has_active_fault() is False
