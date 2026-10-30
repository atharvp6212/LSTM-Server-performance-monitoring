"""Healing actions: Action ∈ {RestartPod, RescheduleNode, ScaleUp, DrainNode, AlertAdmin}.

Implements paper Step 6 ("Self-Healing Integration") of *Self-Healing Cloud
Platforms Using LSTM-based Predictive Models for Early Fault Detection*
(IJCET 16(5) 2025): once the Step 5 decision rule fires, a healing action is
executed against the platform — through the real Kubernetes API
(:class:`KubernetesHealer`) or against the closed-loop simulated cluster
(:class:`SimulatedHealer`).

Import safety: this module never imports the ``kubernetes`` client at import
time — that import happens lazily inside :meth:`KubernetesHealer.__init__`,
and a :class:`KubernetesHealer` is only ever auto-instantiated by
:func:`make_healer` when ``cfg["healing"]["mode"] == "kubernetes"``.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # duck-typed at runtime; cluster_sim is owned by another module
    from selfheal.simulation.cluster_sim import SimulatedCluster

__all__ = [
    "HealingAction",
    "HealingResult",
    "BaseHealer",
    "KubernetesHealer",
    "SimulatedHealer",
    "make_healer",
]

logger = logging.getLogger("selfheal.healing")


class HealingAction(str, Enum):
    """The five healing actions of paper Step 6."""

    RESTART_POD = "restart_pod"
    RESCHEDULE_NODE = "reschedule_node"
    SCALE_UP = "scale_up"
    DRAIN_NODE = "drain_node"
    ALERT_ADMIN = "alert_admin"


@dataclass
class HealingResult:
    """Record of one executed healing action (paper Step 6 / Step 7 logging).

    Attributes:
        action: which :class:`HealingAction` ran.
        target: the target dict the action was executed against (pod/node names).
        success: whether the action completed without error.
        started_at: epoch seconds when execution started.
        completed_at: epoch seconds when execution finished.
        detail: human-readable description of what happened.
    """

    action: HealingAction
    target: dict
    success: bool
    started_at: float
    completed_at: float
    detail: str


class BaseHealer(ABC):
    """Template-method base for healers (paper Step 6).

    :meth:`execute` times the action, dispatches to the per-action handler of
    the concrete healer, and converts any raised exception into a failed
    :class:`HealingResult` — a healing attempt must never crash the control
    loop.
    """

    def execute(self, action: HealingAction, target: dict) -> HealingResult:
        """Execute one healing action against ``target`` and time it.

        ``target`` carries pod/node/deployment names (e.g. ``{"pod": "p-3"}``).
        Handlers return ``(success, detail)``; exceptions (including Kubernetes
        ``ApiException``) become ``success=False`` results.
        """
        target = dict(target or {})
        handlers: dict[HealingAction, Callable[[dict], tuple[bool, str]]] = {
            HealingAction.RESTART_POD: self._restart_pod,
            HealingAction.RESCHEDULE_NODE: self._reschedule_node,
            HealingAction.SCALE_UP: self._scale_up,
            HealingAction.DRAIN_NODE: self._drain_node,
            HealingAction.ALERT_ADMIN: self._alert_admin,
        }
        started_at = time.time()
        try:
            handler = handlers[HealingAction(action)]
            success, detail = handler(target)
        except Exception as exc:  # noqa: BLE001 — healing must never crash the loop
            success, detail = False, f"{type(exc).__name__}: {exc}"
        completed_at = time.time()
        result = HealingResult(
            action=HealingAction(action),
            target=target,
            success=bool(success),
            started_at=started_at,
            completed_at=completed_at,
            detail=str(detail),
        )
        logger.info(
            "healing action=%s success=%s target=%s detail=%s",
            result.action.value,
            result.success,
            result.target,
            result.detail,
        )
        return result

    # Per-action hooks — each returns (success, detail).
    @abstractmethod
    def _restart_pod(self, target: dict) -> tuple[bool, str]: ...

    @abstractmethod
    def _reschedule_node(self, target: dict) -> tuple[bool, str]: ...

    @abstractmethod
    def _scale_up(self, target: dict) -> tuple[bool, str]: ...

    @abstractmethod
    def _drain_node(self, target: dict) -> tuple[bool, str]: ...

    @abstractmethod
    def _alert_admin(self, target: dict) -> tuple[bool, str]: ...


class KubernetesHealer(BaseHealer):
    """Healing through the real Kubernetes API (paper Step 6, §3.4).

    The ``kubernetes`` client is imported lazily here — never at module import
    — so the rest of the platform works without a cluster or the package's
    native dependencies loaded. Never auto-instantiate this class unless
    ``cfg["healing"]["mode"] == "kubernetes"`` (enforced by :func:`make_healer`).

    Action mapping:
        RESTART_POD     -> delete the pod (its controller recreates it)
        RESCHEDULE_NODE -> evict the pod (scheduler places it elsewhere)
        SCALE_UP        -> patch the deployment's replicas +1 (capped)
        DRAIN_NODE      -> cordon the node + evict its non-DaemonSet pods
        ALERT_ADMIN     -> structured log line (+ optional webhook POST)
    """

    def __init__(self, cfg: dict) -> None:
        """Load kube configuration and build API clients.

        Uses ``cfg["healing"]["kubernetes"]["kubeconfig"]`` when set, otherwise
        the client's default loading chain (kubeconfig, then in-cluster).
        """
        # Lazy import per SPEC: the kubernetes client only loads when a real
        # cluster healer is actually constructed.
        from kubernetes import client as k8s_client  # noqa: PLC0415
        from kubernetes import config as k8s_config  # noqa: PLC0415
        from kubernetes.client.rest import ApiException  # noqa: PLC0415

        self._k8s = k8s_client
        self._ApiException = ApiException

        healing_cfg = cfg.get("healing", {}) or {}
        kube_cfg = healing_cfg.get("kubernetes", {}) or {}
        kubeconfig = kube_cfg.get("kubeconfig")
        if kubeconfig:
            k8s_config.load_kube_config(config_file=str(kubeconfig))
        else:
            try:  # default loading chain: kubeconfig first, then in-cluster
                k8s_config.load_kube_config()
            except Exception:  # noqa: BLE001
                k8s_config.load_incluster_config()

        self.core = k8s_client.CoreV1Api()
        self.apps = k8s_client.AppsV1Api()
        self.namespace: str = str(kube_cfg.get("namespace", "default"))
        self.max_replicas: int = int(kube_cfg.get("max_replicas", 10))
        self.alert_webhook: str | None = kube_cfg.get("alert_webhook")

    def _namespace_of(self, target: dict) -> str:
        return str(target.get("namespace", self.namespace))

    def _restart_pod(self, target: dict) -> tuple[bool, str]:
        """RESTART_POD: delete the pod; its controller recreates it."""
        pod = target.get("pod") or target.get("pod_name")
        if not pod:
            return False, "restart_pod: no 'pod' in target"
        namespace = self._namespace_of(target)
        try:
            self.core.delete_namespaced_pod(name=pod, namespace=namespace)
        except self._ApiException as exc:
            return False, f"restart_pod {namespace}/{pod} failed: {exc.reason}"
        return True, f"deleted pod {namespace}/{pod}; controller will recreate it"

    def _reschedule_node(self, target: dict) -> tuple[bool, str]:
        """RESCHEDULE_NODE: evict the pod so the scheduler re-places it."""
        pod = target.get("pod") or target.get("pod_name")
        if not pod:
            return False, "reschedule_node: no 'pod' in target"
        namespace = self._namespace_of(target)
        try:
            self._evict(pod, namespace)
        except self._ApiException as exc:
            return False, f"evict {namespace}/{pod} failed: {exc.reason}"
        return True, f"evicted pod {namespace}/{pod} for rescheduling"

    def _scale_up(self, target: dict) -> tuple[bool, str]:
        """SCALE_UP: patch deployment replicas +1, capped at max_replicas."""
        deployment = target.get("deployment") or target.get("deployment_name")
        if not deployment:
            return False, "scale_up: no 'deployment' in target"
        namespace = self._namespace_of(target)
        try:
            scale = self.apps.read_namespaced_deployment_scale(
                name=deployment, namespace=namespace
            )
            current = int(scale.spec.replicas or 0)
            desired = min(current + 1, self.max_replicas)
            if desired == current:
                return (
                    False,
                    f"deployment {namespace}/{deployment} already at "
                    f"max_replicas cap ({self.max_replicas})",
                )
            self.apps.patch_namespaced_deployment_scale(
                name=deployment,
                namespace=namespace,
                body={"spec": {"replicas": desired}},
            )
        except self._ApiException as exc:
            return False, f"scale_up {namespace}/{deployment} failed: {exc.reason}"
        return True, f"scaled {namespace}/{deployment}: {current} -> {desired} replicas"

    def _drain_node(self, target: dict) -> tuple[bool, str]:
        """DRAIN_NODE: cordon the node, then evict its non-DaemonSet pods."""
        node = target.get("node") or target.get("node_name")
        if not node:
            return False, "drain_node: no 'node' in target"
        try:
            self.core.patch_node(name=node, body={"spec": {"unschedulable": True}})
        except self._ApiException as exc:
            return False, f"cordon node {node} failed: {exc.reason}"
        evicted = failed = 0
        try:
            pods = self.core.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node}"
            )
        except self._ApiException as exc:
            return False, f"cordoned {node} but pod listing failed: {exc.reason}"
        for pod in pods.items:
            owners = pod.metadata.owner_references or []
            if any(ref.kind == "DaemonSet" for ref in owners):
                continue  # DaemonSet pods are node-bound; never evicted on drain
            try:
                self._evict(pod.metadata.name, pod.metadata.namespace)
                evicted += 1
            except self._ApiException:
                failed += 1
        return True, (
            f"cordoned node {node}; evicted {evicted} pods ({failed} evictions failed)"
        )

    def _alert_admin(self, target: dict) -> tuple[bool, str]:
        """ALERT_ADMIN: structured log line, plus optional webhook POST."""
        payload = {
            "alert": "self-healing",
            "detail": str(target.get("detail", "")),
            "target": {k: v for k, v in target.items() if k != "detail"},
            "ts": time.time(),
        }
        logger.warning("ALERT_ADMIN %s", json.dumps(payload, default=str))
        detail = f"admin alerted: {payload['detail'] or 'fault predicted'}"
        if self.alert_webhook:
            try:
                import urllib.request  # noqa: PLC0415

                request = urllib.request.Request(
                    self.alert_webhook,
                    data=json.dumps(payload, default=str).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5):
                    pass
                detail += f"; webhook {self.alert_webhook} notified"
            except Exception as exc:  # noqa: BLE001 — alert stays best-effort
                detail += f"; webhook failed: {type(exc).__name__}: {exc}"
        return True, detail

    def _evict(self, pod: str, namespace: str) -> None:
        """Create a pod eviction (used by reschedule and drain)."""
        eviction = self._k8s.V1Eviction(
            metadata=self._k8s.V1ObjectMeta(name=pod, namespace=namespace)
        )
        self.core.create_namespaced_pod_eviction(
            name=pod, namespace=namespace, body=eviction
        )


class SimulatedHealer(BaseHealer):
    """Healing against the closed-loop :class:`SimulatedCluster` (paper Step 6).

    The cluster object is duck-typed — only the SPEC hook methods are called:
    ``restart_pod``, ``reschedule_pod``, ``scale_up``, ``drain_node``,
    ``alert_admin``. The ``target`` dict carries pod/node names (``"pod"``,
    ``"node"``) and an optional ``"detail"`` string for alerts.
    """

    def __init__(self, cluster: "SimulatedCluster") -> None:
        self.cluster = cluster

    def _restart_pod(self, target: dict) -> tuple[bool, str]:
        pod = target.get("pod") or target.get("pod_name")
        ok = bool(self.cluster.restart_pod(pod))
        return ok, f"simulated restart_pod({pod or 'unhealthiest'}) -> {ok}"

    def _reschedule_node(self, target: dict) -> tuple[bool, str]:
        pod = target.get("pod") or target.get("pod_name")
        ok = bool(self.cluster.reschedule_pod(pod))
        return ok, f"simulated reschedule_pod({pod or 'unhealthiest'}) -> {ok}"

    def _scale_up(self, target: dict) -> tuple[bool, str]:
        ok = bool(self.cluster.scale_up())
        return ok, f"simulated scale_up() -> {ok}"

    def _drain_node(self, target: dict) -> tuple[bool, str]:
        node = target.get("node") or target.get("node_name")
        ok = bool(self.cluster.drain_node(node))
        return ok, f"simulated drain_node({node or 'auto'}) -> {ok}"

    def _alert_admin(self, target: dict) -> tuple[bool, str]:
        detail = str(target.get("detail", ""))
        ok = bool(self.cluster.alert_admin(detail))
        return ok, f"simulated alert_admin: {detail or 'fault predicted'}"


def make_healer(cfg: dict, cluster: "SimulatedCluster | None" = None) -> BaseHealer:
    """Build the healer selected by ``cfg["healing"]["mode"]`` (paper Step 6).

    ``"kubernetes"`` -> :class:`KubernetesHealer` (the only path that touches
    the real cluster API); ``"simulate"`` -> :class:`SimulatedHealer`, which
    requires ``cluster``.

    Raises:
        ValueError: unknown mode, or simulate mode without a cluster.
    """
    mode = str((cfg.get("healing", {}) or {}).get("mode", "simulate")).lower()
    if mode == "kubernetes":
        return KubernetesHealer(cfg)
    if mode == "simulate":
        if cluster is None:
            raise ValueError(
                "healing.mode == 'simulate' requires a SimulatedCluster instance"
            )
        return SimulatedHealer(cluster)
    raise ValueError(f"Unknown healing mode {mode!r}; expected 'simulate' or 'kubernetes'")
