"""Flask monitoring dashboard for the self-healing cloud platform.

Implements the visibility half of paper Step 7 ("Performance Monitoring and
Feedback Loop", IJCET 16(5) 2025, DOI 10.34218/IJCET_16_05_012, section 3.2):
it surfaces the offline evaluation metrics (``reports/metrics.json``), the
closed-loop runtime report produced by ``scripts/run_selfhealing.py``
(``reports/selfhealing_report.json`` — MTTR, uptime %, false-positive rate,
actions taken, cf. paper section 1), the evaluation figures (PNGs), and the
most recent feedback-loop events from the JSONL log written by
``selfheal.monitoring.feedback.FeedbackLoop``.

Routes (see SPEC.md, "dashboard/app.py"):
    GET /                 rendered dashboard (templates/index.html)
    GET /api/metrics      merged JSON of the two report files
    GET /api/events       last N feedback events (?limit=N, default 50)
    GET /reports/<file>   serves evaluation PNGs from the reports directory

Missing or unreadable report/feedback files always render friendly empty
states — never a 500. The page is fully self-contained (inline CSS, no CDN)
and auto-refreshes every 10 seconds via a meta-refresh tag.

Run from the project root:
    python dashboard/app.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

# Make `import selfheal` work when run as `python dashboard/app.py` (SPEC rule).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfheal.config import load_config, resolve_path  # noqa: E402

#: Figures written by ``selfheal.models.evaluate.evaluate_model`` (SPEC.md).
EVALUATION_PLOTS: list[tuple[str, str]] = [
    ("confusion_matrix.png", "Confusion matrix"),
    ("roc_curve.png", "ROC curve"),
    ("training_history.png", "Training history (loss + AUC)"),
    ("probability_timeline.png", "Fault-probability timeline"),
]

DEFAULT_EVENT_LIMIT = 50
MAX_EVENT_LIMIT = 1000


# --------------------------------------------------------------------------- #
# Small, defensive data-loading helpers (missing/corrupt files -> empty state) #
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from ``path``; return None if absent or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _load_events(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the last ``limit`` events from the feedback JSONL log.

    Uses the feedback event schema from SPEC.md (paper Step 7). Malformed
    lines are skipped; a missing file yields an empty list.
    """
    events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events[-max(limit, 0):] if limit > 0 else []


def _fmt_value(value: Any) -> str:
    """Render an arbitrary JSON value as a short human-readable string."""
    if value is None:
        return "—"  # em dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value == value else "NaN"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{k}: {_fmt_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value) if value else "none"
    return str(value)


def _metric_rows(data: dict[str, Any] | None) -> list[tuple[str, str]] | None:
    """Turn a report dict into (label, value) rows for the metrics tables."""
    if data is None:
        return None
    return [(str(key).replace("_", " "), _fmt_value(val)) for key, val in data.items()]


def _fmt_event_time(event: dict[str, Any]) -> str:
    """Format an event timestamp (epoch ``ts``, else ``logged_at``, else dash)."""
    ts = event.get("ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OverflowError, OSError, ValueError):
            pass
    logged = event.get("logged_at")
    return str(logged) if logged else "—"


def _event_rows(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Prepare feedback events (newest first) for the template table."""
    rows: list[dict[str, str]] = []
    for event in reversed(events):
        detail = event.get("detail")
        if not detail and isinstance(event.get("target"), dict):
            detail = _fmt_value(event["target"])
        rows.append(
            {
                "time": _fmt_event_time(event),
                "type": str(event.get("type", "—")),
                "probability": _fmt_value(event.get("probability")),
                "trigger": _fmt_value(event.get("trigger")),
                "severity": _fmt_value(event.get("severity")),
                "action": _fmt_value(event.get("action")),
                "success": _fmt_value(event.get("success")),
                "actual_fault": _fmt_value(event.get("actual_fault")),
                "detail": _fmt_value(detail),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# App factory                                                                  #
# --------------------------------------------------------------------------- #
def create_app(cfg: dict[str, Any]) -> Flask:
    """Build the dashboard Flask app bound to the configured report paths."""
    app = Flask(
        __name__, template_folder=str(Path(__file__).resolve().parent / "templates")
    )
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    reports_dir = resolve_path(cfg, paths.get("reports", "reports"))
    feedback_path = resolve_path(cfg, paths.get("feedback_log", "feedback/events.jsonl"))
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    dataset_source = str(dataset_cfg.get("source", "unknown"))

    def _available_plots() -> list[dict[str, Any]]:
        """List evaluation PNGs present in the reports dir (named ones first)."""
        plots: list[dict[str, Any]] = []
        named = {fname for fname, _ in EVALUATION_PLOTS}
        candidates = list(EVALUATION_PLOTS)
        try:
            extras = sorted(
                p.name for p in reports_dir.glob("*.png") if p.name not in named
            )
        except OSError:
            extras = []
        candidates += [(name, Path(name).stem.replace("_", " ")) for name in extras]
        for fname, title in candidates:
            fpath = reports_dir / fname
            try:
                if not fpath.is_file():
                    continue
                stamp = int(fpath.stat().st_mtime)
            except OSError:
                continue
            plots.append({"file": fname, "title": title, "stamp": stamp})
        return plots

    @app.get("/")
    def index() -> str:
        """Render the dashboard (paper Step 7: monitoring visibility)."""
        eval_metrics = _load_json(reports_dir / "metrics.json")
        heal_metrics = _load_json(reports_dir / "selfhealing_report.json")
        events = _load_events(feedback_path, DEFAULT_EVENT_LIMIT)
        return render_template(
            "index.html",
            generated_at=datetime.now(tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            dataset_source=dataset_source,
            eval_rows=_metric_rows(eval_metrics),
            heal_rows=_metric_rows(heal_metrics),
            plots=_available_plots(),
            events=_event_rows(events),
            event_limit=DEFAULT_EVENT_LIMIT,
        )

    @app.get("/api/metrics")
    def api_metrics() -> Any:
        """Merged JSON of metrics.json and selfhealing_report.json (SPEC)."""
        return jsonify(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "evaluation": _load_json(reports_dir / "metrics.json"),
                "selfhealing": _load_json(reports_dir / "selfhealing_report.json"),
            }
        )

    @app.get("/api/events")
    def api_events() -> Any:
        """Last N feedback events from the JSONL log (?limit=N, default 50)."""
        raw_limit = request.args.get("limit", DEFAULT_EVENT_LIMIT)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = DEFAULT_EVENT_LIMIT
        limit = max(1, min(limit, MAX_EVENT_LIMIT))
        events = _load_events(feedback_path, limit)
        return jsonify({"count": len(events), "limit": limit, "events": events})

    @app.get("/reports/<filename>")
    def report_file(filename: str) -> Any:
        """Serve an evaluation PNG from the reports dir (sanitized filename)."""
        safe = secure_filename(filename)
        if not safe or safe != filename or not safe.lower().endswith(".png"):
            abort(404)
        return send_from_directory(reports_dir, safe, max_age=0)

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception) -> Any:  # pragma: no cover - safety net
        """Return clean error responses; never leak tracebacks to the client."""
        if isinstance(error, HTTPException):
            return error
        app.logger.exception("Unhandled dashboard error")
        return (
            "<h1>Dashboard error</h1><p>An unexpected error occurred; "
            "check the server logs.</p>",
            500,
        )

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python dashboard/app.py --config config.yaml``."""
    parser = argparse.ArgumentParser(
        description="Self-healing platform monitoring dashboard (paper Step 7)."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml (project root)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument(
        "--port", type=int, default=5000, help="Bind port (SPEC default: 5000)."
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    app = create_app(cfg)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
