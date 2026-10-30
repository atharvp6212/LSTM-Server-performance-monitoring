#!/usr/bin/env python
"""Run the paper's full closed self-healing loop on the simulated cluster.

Implements paper §3.4 / Steps 5-7 end to end (SPEC.md ``scripts/run_selfhealing.py``):
SimulatedCluster ticks -> rolling telemetry window -> LSTM ``FaultPredictor``
(Step 5 thresholding + decision rule) -> ``SelfHealingEngine`` executing
healing actions on the cluster (Step 6) -> ``MetricsTracker`` / ``FeedbackLoop``
recording MTTR, uptime, false-positive rate and every event (Step 7).

Usage (from the project root):
    python scripts/run_selfhealing.py [--config config.yaml] [--ticks 600]
                                      [--fault-prob 0.01] [--no-healing]

``--no-healing`` is the ablation: predictions are still made and logged, but no
healing action is executed — for comparing MTTR/uptime against the healed run.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TF logs (SPEC)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from selfheal.config import load_config, resolve_path, set_global_seeds
from selfheal.healing.actions import make_healer
from selfheal.healing.engine import SelfHealingEngine
from selfheal.monitoring.feedback import FeedbackLoop
from selfheal.monitoring.metrics import MetricsTracker
from selfheal.prediction.predictor import FaultPredictor
from selfheal.simulation.cluster_sim import SimulatedCluster


def build_parser() -> argparse.ArgumentParser:
    """CLI per SPEC.md: --config, --ticks, --fault-prob, --no-healing."""
    parser = argparse.ArgumentParser(
        description="Closed-loop LSTM self-healing simulation (paper Steps 5-7).")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: %(default)s)")
    parser.add_argument("--ticks", type=int, default=600,
                        help="Number of simulation ticks to run (default: %(default)s)")
    parser.add_argument("--fault-prob", type=float, default=0.01,
                        help="Per-tick probability of injecting a random fault "
                             "(default: %(default)s)")
    parser.add_argument("--no-healing", action="store_true",
                        help="Ablation: log predictions but take no healing action")
    return parser


def _fmt(value: object, spec: str = "{:.3f}") -> str:
    """Format a possibly-None numeric metric for the report table."""
    if value is None:
        return "n/a"
    try:
        return spec.format(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _fmt_actions(actions: object) -> str:
    """Render the actions-taken counter dict as `name=count, ...`."""
    if not isinstance(actions, dict) or not actions:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(actions.items()))


def render_report(report: dict, *, healing_on: bool, ticks: int, sim_seconds: float,
                  faults_injected: int, predictions: int, triggers: int,
                  saved_to: Path, outage_open: bool) -> str:
    """Build the final report table (paper Step 7 / §1: MTTR, uptime, FP rate)."""
    bar = "=" * 66
    sep = "-" * 66
    lines = [
        bar,
        "            SELF-HEALING CLOSED-LOOP RUN REPORT",
        bar,
        f"  Mode                 : {'HEALING ENABLED' if healing_on else 'NO-HEALING (ablation)'}",
        f"  Ticks simulated      : {ticks}  ({sim_seconds:.0f} s of sim time)",
        f"  Faults injected      : {faults_injected}",
        f"  Predictions made     : {predictions}  (triggered: {triggers})",
        sep,
        f"  MTTR (seconds)       : {_fmt(report.get('mttr_seconds'), '{:.1f}')}",
        f"  Uptime (%)           : {_fmt(report.get('uptime_percent'), '{:.2f}')}",
        f"  False positive rate  : {_fmt(report.get('false_positive_rate'))}",
        f"  Precision            : {_fmt(report.get('precision'))}",
        f"  Recall               : {_fmt(report.get('recall'))}",
        f"  Actions taken        : {_fmt_actions(report.get('actions_taken'))}",
        sep,
        f"  Report saved to      : {saved_to}",
    ]
    if outage_open:
        lines.append("  NOTE: run ended during an active outage (downtime still accruing).")
    if healing_on:
        lines.append("  NOTE: precision/recall are scored live against the fault population")
        lines.append("        healing itself shortens (faults resolve mid-ramp instead of")
        lines.append("        reaching sustained outage) -- not directly comparable to the")
        lines.append("        --no-healing run's recall. MTTR/uptime are the comparable metrics.")
    lines.append(bar)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Drive the closed loop per the SPEC.md run_selfhealing contract."""
    args = build_parser().parse_args(argv)
    if args.ticks <= 0:
        print("ERROR: --ticks must be positive.", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    set_global_seeds(cfg)

    cluster = SimulatedCluster(cfg)
    try:
        predictor = FaultPredictor.from_artifacts(cfg)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: could not load model artifacts ({exc}).\n"
              "Train first: python scripts/train_model.py --config "
              f"{args.config}", file=sys.stderr)
        return 2

    feedback = FeedbackLoop(resolve_path(cfg, cfg["paths"]["feedback_log"]))
    metrics = MetricsTracker(cfg)
    engine: SelfHealingEngine | None = None
    if not args.no_healing:
        healer = make_healer(cfg, cluster=cluster)
        engine = SelfHealingEngine(healer, cfg, feedback=feedback, metrics=metrics)

    window_size = int(predictor.window_size)
    buffer: deque[pd.DataFrame] = deque(maxlen=window_size)
    faults_injected: list[tuple[int, str]] = []
    predictions = 0
    triggers = 0
    outage_prev = False
    outage_started_at: float | None = None

    for tick_i in range(1, args.ticks + 1):
        injected = cluster.maybe_inject_random_fault(args.fault_prob)
        if injected is not None:
            faults_injected.append((tick_i, injected))

        buffer.append(cluster.tick())
        now = float(cluster.current_time)

        # Step 7 ground-truth bookkeeping: outage entry/exit -> MTTR samples.
        outage_now = cluster.in_outage()
        if outage_now and not outage_prev:
            metrics.record_fault_start(now)
            outage_started_at = now
        elif outage_prev and not outage_now:
            metrics.record_recovery(now)
            mttr = None if outage_started_at is None else now - outage_started_at
            feedback.log_event({
                "type": "outcome", "ts": now, "actual_fault": True,
                "mttr_seconds": mttr, "detail": "outage resolved",
            })
            outage_started_at = None
        outage_prev = outage_now

        if len(buffer) < window_size:
            continue

        # Step 5: predict fault probability over the rolling raw window,
        # threshold into a Decision.
        window_df = pd.concat(list(buffer), ignore_index=True)
        probability = float(predictor.predict_proba(window_df))
        decision = predictor.decide(probability)
        actual_fault = bool(cluster.has_active_fault())
        metrics.record_prediction(probability, decision.trigger,
                                  actual_fault=actual_fault, ts=now)
        predictions += 1
        if decision.trigger:
            triggers += 1

        if engine is not None:
            # Step 6: route the decision to a healing action on the cluster.
            target_pod = cluster.unhealthiest_pod()
            engine.handle(decision, {
                "target": {"pod": target_pod, "node": cluster.node_of(target_pod)},
                "top_metric": cluster.top_pressure_metric(),
                "timestamp": now,
            })
        else:
            # Ablation: predictions logged, no action taken (SPEC --no-healing).
            feedback.log_event({
                "type": "prediction", "ts": now, "probability": probability,
                "trigger": bool(decision.trigger), "severity": decision.severity,
                "actual_fault": actual_fault, "detail": "no-healing ablation",
            })

    report = metrics.report()
    reports_dir = resolve_path(cfg, cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "selfhealing_report.json"
    metrics.save(out_path)

    if faults_injected:
        injected_desc = ", ".join(f"tick {t}: {pod}" for t, pod in faults_injected)
        print(f"Injected faults -> {injected_desc}")
    print(render_report(
        report,
        healing_on=engine is not None,
        ticks=args.ticks,
        sim_seconds=args.ticks * float(cluster.tick_seconds),
        faults_injected=len(faults_injected),
        predictions=predictions,
        triggers=triggers,
        saved_to=out_path,
        outage_open=cluster.in_outage(),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
