# SPEC — Self-Healing Cloud Platform Using LSTM-Based Predictive Models

Implementation of the paper: *Atharv S. Patil, Sanjay S. Patil (2025). "Self-Healing Cloud
Platforms Using LSTM-based Predictive Models for Early Fault Detection." IJCET 16(5), 177–187.
DOI: 10.34218/IJCET_16_05_012.*

This document is the **binding interface contract**. Every module must implement exactly these
public names and signatures so independently written modules integrate. Internal helpers are free.

## Environment
- Python 3.12, venv at `.venv/` (already provisioned).
- Installed: tensorflow-cpu 2.21 (keras 3), numpy 2.5, **pandas 3.0** (no deprecated APIs — no
  `df.append`, no chained-assignment writes; Copy-on-Write is on), scikit-learn 1.9, matplotlib,
  pyyaml, kubernetes, flask, pytest, joblib, psutil.
- Matplotlib must use the `Agg` backend (`matplotlib.use("Agg")` before pyplot import).
- All randomness seeded from `cfg["project"]["seed"]`.
- Every module: type hints + docstrings that cite the paper section/step it implements.
- All paths in config are relative to the project root. Resolve via `selfheal.config.resolve_path`.

## Project layout (files marked OWNER are written by exactly one build agent)

```
config.yaml                      (exists — read it; do not modify)
requirements.txt                 (exists)
SPEC.md                          (this file)
selfheal/
  config.py                      (exists — load_config()/resolve_path())
  data/
    dataset.py                   OWNER: data-agent
    preprocessing.py             OWNER: pipeline-agent
    windowing.py                 OWNER: pipeline-agent
    features.py                  OWNER: pipeline-agent
  models/
    lstm_model.py                OWNER: model-agent
    train.py                     OWNER: model-agent
    evaluate.py                  OWNER: model-agent
  prediction/
    predictor.py                 OWNER: healing-agent
  healing/
    actions.py                   OWNER: healing-agent
    engine.py                    OWNER: healing-agent
  monitoring/
    metrics.py                   OWNER: monitoring-agent
    feedback.py                  OWNER: monitoring-agent
  simulation/
    cluster_sim.py               OWNER: simulation-agent
scripts/
  generate_dataset.py            OWNER: data-agent
  train_model.py                 OWNER: model-agent
  evaluate_model.py              OWNER: model-agent
  run_selfhealing.py             OWNER: simulation-agent
  retrain_feedback.py            OWNER: monitoring-agent
dashboard/
  app.py                         OWNER: dashboard-agent
  templates/index.html           OWNER: dashboard-agent
tests/
  test_*.py                      OWNER: test-agent
data/raw/balanced_shuffled_traffic.csv   (exists — the cited Kaggle dataset [17])
```

Scripts are argparse CLIs run as `python scripts/<name>.py [--config config.yaml] ...` from the
project root. Each script must prepend the project root to `sys.path` so `import selfheal` works:
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`.

## Canonical data contract

A **canonical telemetry frame** is a `pd.DataFrame` with:
- column `timestamp` (datetime64, sorted ascending),
- N numeric feature columns (float64),
- column `anomaly` (int 0/1 ground-truth label).

`feature_columns(df)` = every column except `timestamp` and `anomaly`.

Array shapes downstream:
- windows `X`: `np.ndarray (M, w, d)` float32 — M windows, window size w, d raw features.
- engineered `F`: `np.ndarray (M, w, d')` float32.
- labels `y`: `np.ndarray (M,)` int — 1 if a fault occurs within `horizon` steps AFTER the window.

## Module contracts

### selfheal/config.py  (already written — for reference)
```python
def load_config(path: str | Path = "config.yaml") -> dict          # yaml.safe_load + PROJECT_ROOT injection
def resolve_path(cfg: dict, rel: str | Path) -> Path               # cfg["project"]["root"] / rel; mkdir parents of files NOT done here
def set_global_seeds(cfg: dict) -> None                            # seeds random, numpy, tensorflow
```

### selfheal/data/dataset.py — data-agent
Implements paper §3.1 (dataset) and the Table 1 schema.
```python
KAGGLE_LABEL_COL = "Label"
LOG_SCHEMA = ["timestamp", "namespace", "pod_name", "event_type", "message", "reason", "status", "anomaly_label"]
CANONICAL_METRICS = ["cpu_usage", "memory_usage", "network_io", "pod_restarts",
                     "error_event_rate", "warning_event_rate"]

def generate_k8s_telemetry(cfg: dict, out_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]
    # Returns (logs_df, metrics_df) and writes k8s_logs.csv + k8s_metrics.csv into out_dir
    # (default cfg["paths"]["data_raw"]).
    # logs_df: EXACTLY the LOG_SCHEMA columns (Table 1): ~cfg n_records rows of realistic K8s
    #   events (Normal / Warning / Error / anomaly-tagged), namespaces & pod names from cfg.
    # metrics_df: canonical telemetry frame at fixed tick interval per cfg, columns
    #   CANONICAL_METRICS + timestamp + anomaly, CLUSTER-AGGREGATED (one row per tick).
    # Faults must BUILD UP GRADUALLY (paper §3.2: "faults build up gradually but do not emerge
    #   suddenly"): e.g. memory-leak ramps, CPU runaway ramps, restart bursts, network saturation,
    #   each lasting many ticks before the labeled anomaly period; anomaly=1 during the fault window.
    # Log rows are generated consistently with the metric state (Error/Warning bursts near faults).

def load_kaggle_traffic(path: str | Path, max_raw_features: int = 12) -> pd.DataFrame
    # Parses the cited Kaggle CSV: sort by "Timestamp" -> canonical frame.
    # Select top `max_raw_features` numeric features by variance (excluding ports/protocol ids and
    #   any column with inf/nan share > 0.5; replace remaining inf with nan, then ffill/bfill).
    # anomaly = "Label". timestamp = parsed "Timestamp".

def load_synthetic(path: str | Path) -> pd.DataFrame
    # Reads k8s_metrics.csv back into a canonical frame.

def load_dataset(cfg: dict) -> pd.DataFrame
    # Dispatch on cfg["dataset"]["source"]: "synthetic" -> load_synthetic (generate first if the
    # CSV is absent), "kaggle" -> load_kaggle_traffic.

def feature_columns(df: pd.DataFrame) -> list[str]
```

### selfheal/data/preprocessing.py — pipeline-agent
Paper Step 1: cleaning + Min-Max normalization.
```python
class Scaler:
    # wraps sklearn MinMaxScaler over the feature columns only
    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Scaler"
    def transform(self, df: pd.DataFrame) -> pd.DataFrame        # returns copy, features scaled to [0,1]
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame
    def save(self, path: str | Path) -> None                     # joblib; persists feature_cols too
    @classmethod
    def load(cls, path: str | Path) -> "Scaler"
    feature_cols: list[str]

def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame
    # sort by timestamp, drop duplicate timestamps (keep first), replace ±inf with NaN,
    # impute per cfg["preprocessing"]["impute_strategy"] ("ffill" -> ffill().bfill(), "mean", "drop"),
    # clip features to finite float32 range. Never drops the anomaly column.

def preprocess(df: pd.DataFrame, cfg: dict, scaler: Scaler | None = None, fit: bool = True) -> tuple[pd.DataFrame, Scaler]
    # clean() then scale. fit=True fits a new Scaler (training); fit=False requires `scaler`.
```

### selfheal/data/windowing.py — pipeline-agent
Paper Step 2: sliding time window X_k = {x_k..x_{k+w-1}}.
```python
def make_windows(X: np.ndarray, y: np.ndarray, window_size: int, stride: int = 1, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]
    # X: (N, d) scaled features; y: (N,) anomaly per step.
    # Returns windows (M, w, d) float32 and labels (M,) where label_k = 1 iff
    # max(y[k+w : k+w+horizon]) == 1 (fault occurs within `horizon` steps after the window).
    # Windows whose full horizon extends past N are dropped (labels always well defined).

def window_dataframe(df: pd.DataFrame, cfg: dict, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]
    # convenience: pulls arrays from a canonical frame and calls make_windows with cfg["windowing"].
```

### selfheal/data/features.py — pipeline-agent
Paper Step 3: lag features, rolling stats, rate of change, utilization ratio, threshold flags.
The engineered output keeps the sequence form F_k ∈ R^(w×d') for the LSTM (paper Step 4).
```python
def engineered_dim(d: int, cfg: dict) -> int          # d' given raw dim d and cfg["features"]
def engineer_features(windows: np.ndarray, cfg: dict, limits: np.ndarray | None = None) -> np.ndarray
    # windows: (M, w, d) in [0,1]. Returns (M, w, d') float32 with channels appended per feature:
    #  a. lag channels: for l in 1..L, the series shifted by l within the window (padded with the
    #     window's first value) — L*d channels.
    #  b. rolling stats: per-window mean μ and std σ per feature, broadcast across w — 2*d channels.
    #  c. rate of change: per-step delta x_t − x_{t−1} (first step 0) — d channels.
    #  d. utilization ratio r = x / limit per feature; limits vector from cfg["features"]["limits"]
    #     in CANONICAL feature order when provided, else 1.0 for every feature (scaled data) — d channels.
    #  e. threshold flags f = (r > tau) as float — d channels.
    # Each block is included only if its cfg["features"] switch is on. Order: raw, lags, mean, std,
    # delta, ratio, flags. d' = engineered_dim(d, cfg) always.
def limits_vector(feature_cols: list[str], cfg: dict, scaler: "Scaler | None" = None) -> np.ndarray | None
    # Builds the per-feature limits array aligned with feature_cols from cfg["features"]["limits"].
    # If a scaler is given, limits are transformed into the scaled space (limit_scaled =
    # (limit - data_min) / (data_max - data_min), clipped to [1e-6, inf)). Features without a
    # configured limit get limit 1.0. Returns None if cfg["features"]["utilization_ratio"] is false.
```

### selfheal/models/lstm_model.py — model-agent
Paper Step 4 / Figure 1: Input → LSTM → Dropout → Dense(ReLU) → Dense(1, sigmoid); BCE loss.
```python
def build_lstm_model(input_shape: tuple[int, int], cfg: dict) -> "keras.Model"
    # keras.Sequential: Input(shape=input_shape) -> LSTM(cfg lstm_units) -> Dropout(cfg dropout)
    # -> Dense(cfg dense_units, relu) -> Dense(1, sigmoid).
    # compile(optimizer=Adam(cfg learning_rate), loss="binary_crossentropy",
    #         metrics=["accuracy", keras.metrics.Precision(name="precision"),
    #                  keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])
```

### selfheal/models/train.py — model-agent
```python
def chronological_split(F: np.ndarray, y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple
    # time-ordered split (no shuffle across the boundary): returns (F_tr, y_tr, F_va, y_va, F_te, y_te)

def train_model(cfg: dict) -> dict
    # Full pipeline: load_dataset -> preprocess(fit=True) -> window_dataframe -> limits_vector ->
    # engineer_features -> chronological_split -> build_lstm_model -> fit with EarlyStopping
    # (patience cfg, restore_best_weights, monitor val_auc mode max) + class_weight balanced when
    # cfg["model"]["class_weight"] == "balanced".
    # Saves into cfg["paths"]["artifacts"]:
    #   model.keras, scaler.joblib, meta.json {feature_cols, window_size, horizon, engineered_dim,
    #   threshold, trained_at, dataset_source, class_balance}, history.json (fit history lists).
    # Returns {"history": ..., "test": {...metrics...}, "artifacts_dir": str} — test metrics computed
    # on the held-out chronological test split: accuracy, precision, recall, f1, auc, confusion
    # matrix counts (tn, fp, fn, tp).
```

### selfheal/models/evaluate.py — model-agent
```python
def evaluate_model(cfg: dict) -> dict
    # Loads artifacts, rebuilds the SAME pipeline with the SAVED scaler (fit=False) and meta,
    # evaluates on the chronological test split. Writes to cfg["paths"]["reports"]:
    #   metrics.json (accuracy, precision, recall, f1, auc, tn/fp/fn/tp, threshold),
    #   confusion_matrix.png, roc_curve.png, training_history.png (loss+auc curves from history.json),
    #   probability_timeline.png (predicted fault probability vs true anomaly over the test span).
    # Returns the metrics dict.
```

### selfheal/prediction/predictor.py — healing-agent
Paper Step 5: thresholding and decision rule A_k.
```python
@dataclass
class Decision:
    probability: float
    trigger: bool                 # probability >= T
    severity: str                 # "none" | "alert" | "scale" | "restart" | "drain" (bands from cfg["decision"])

class FaultPredictor:
    @classmethod
    def from_artifacts(cls, cfg: dict) -> "FaultPredictor"   # loads model.keras, scaler.joblib, meta.json
    def predict_proba(self, raw_window: pd.DataFrame) -> float
        # raw_window: canonical frame slice of >= window_size rows (unscaled). Applies saved scaler,
        # takes the LAST window_size rows, engineers features with the saved limits, returns ŷ ∈ [0,1].
    def decide(self, probability: float) -> Decision
        # severity = highest band whose threshold <= probability (bands cfg["decision"]["severity_bands"]);
        # trigger = probability >= cfg["decision"]["fault_threshold"].
    window_size: int
    feature_cols: list[str]
```

### selfheal/healing/actions.py — healing-agent
Paper Step 6: Action ∈ {RestartPod, RescheduleNode, ScaleUp, DrainNode, AlertAdmin} via K8s API.
```python
class HealingAction(str, Enum): RESTART_POD, RESCHEDULE_NODE, SCALE_UP, DRAIN_NODE, ALERT_ADMIN

@dataclass
class HealingResult:
    action: HealingAction; target: dict; success: bool
    started_at: float; completed_at: float; detail: str

class BaseHealer(ABC):
    def execute(self, action: HealingAction, target: dict) -> HealingResult   # template: times + dispatch

class KubernetesHealer(BaseHealer):
    # Real cluster via `kubernetes` client. Loads kubeconfig from cfg["healing"]["kubernetes"]["kubeconfig"]
    # (or default loading chain). Implements: RESTART_POD -> delete pod (controller recreates);
    # RESCHEDULE_NODE -> evict pod; SCALE_UP -> patch deployment replicas +1 (capped);
    # DRAIN_NODE -> cordon node + evict its non-daemonset pods; ALERT_ADMIN -> structured log line
    # (and optional webhook URL from cfg if set). Every call wrapped: ApiException -> success=False.
    # NEVER auto-instantiated unless cfg["healing"]["mode"] == "kubernetes".

class SimulatedHealer(BaseHealer):
    def __init__(self, cluster: "SimulatedCluster")
    # Maps each action onto the SimulatedCluster hooks (restart_pod, reschedule_pod, scale_up,
    # drain_node, alert_admin) — target dict carries pod/node names.

def make_healer(cfg: dict, cluster: "SimulatedCluster | None" = None) -> BaseHealer
```

### selfheal/healing/engine.py — healing-agent
```python
class SelfHealingEngine:
    def __init__(self, healer: BaseHealer, cfg: dict, feedback: "FeedbackLoop | None" = None,
                 metrics: "MetricsTracker | None" = None)
    def handle(self, decision: Decision, context: dict) -> HealingResult | None
        # context: {"target": {...}, "top_metric": str | None, "timestamp": float}
        # No-op unless decision.trigger. Chooses action from decision.severity:
        #   alert -> ALERT_ADMIN, scale -> SCALE_UP, restart -> RESTART_POD, drain -> DRAIN_NODE;
        #   if severity == "restart" but top_metric == "network_io" -> RESCHEDULE_NODE (network
        #   pressure is placement-related). Enforces per-target cooldown cfg["healing"]["cooldown_seconds"]
        #   and cfg["healing"]["max_actions_per_hour"] (fall back to ALERT_ADMIN when throttled;
        #   ALERT_ADMIN itself is never throttled). Logs every prediction+action to feedback
        #   (see JSONL schema) and records into metrics tracker.
```

### selfheal/monitoring/metrics.py — monitoring-agent
Paper Step 7 + §1 ("MTTR, uptime percentage, false positive rates").
```python
class MetricsTracker:
    def __init__(self, cfg: dict | None = None)
    def record_prediction(self, probability: float, trigger: bool, actual_fault: bool | None = None, ts: float | None = None) -> None
    def record_fault_start(self, ts: float) -> None
    def record_recovery(self, ts: float) -> None          # closes the open fault -> one MTTR sample + downtime
    def record_action(self, result: "HealingResult") -> None
    # properties: mttr_seconds (mean, None if no samples), uptime_percent (observed span minus
    # downtime, 100.0 before any fault), false_positive_rate (FP/(FP+TN) over predictions with
    # known actual), precision, recall, actions_taken (dict per action), predictions_total
    def report(self) -> dict                               # all of the above, JSON-serializable
    def save(self, path: str | Path) -> None               # json dump of report()
```

### selfheal/monitoring/feedback.py — monitoring-agent
Paper Step 7: feedback loop, adaptive retraining.
```python
class FeedbackLoop:
    def __init__(self, path: str | Path)                   # JSONL file, append mode; creates parents
    def log_event(self, event: dict) -> None               # adds "logged_at" iso timestamp
    def load_events(self) -> list[dict]
    def should_retrain(self, cfg: dict) -> bool
        # True when events since last "retrain" event >= cfg["monitoring"]["retrain_every_events"]
        # OR observed false-positive rate among outcome-bearing events >
        # cfg["monitoring"]["fp_rate_retrain_threshold"] (with >= 20 outcome events).
    def mark_retrained(self, summary: dict) -> None        # logs {"type": "retrain", ...summary}

def retrain_from_feedback(cfg: dict) -> dict
    # Reruns train_model(cfg) (fresh data incl. any regenerated telemetry), marks the feedback loop,
    # returns the training summary.
```
**Feedback JSONL event schema** (every writer uses these keys):
`{"type": "prediction"|"healing"|"outcome"|"retrain", "ts": float_epoch, "probability": float?,
"trigger": bool?, "severity": str?, "action": str?, "target": dict?, "success": bool?,
"actual_fault": bool?, "mttr_seconds": float?, "detail": str?}`

### selfheal/simulation/cluster_sim.py — simulation-agent
Closed-loop testbed standing in for a live Kubernetes cluster (enables paper Step 6/7 end to end
without a real cluster).
```python
class SimulatedCluster:
    def __init__(self, cfg: dict, seed: int | None = None)
    # Simulates n_pods pods across nodes with per-pod cpu/mem/net/restart dynamics and injectable
    # fault modes: "memory_leak", "cpu_runaway", "crashloop", "network_saturation". Faults ramp up
    # over many ticks (gradual buildup) then, if unhealed, become an active outage (anomaly=1).
    def tick(self) -> pd.DataFrame
        # advances one tick (cfg dataset.synthetic.tick_seconds of sim time); returns ONE canonical
        # telemetry row (timestamp, CANONICAL_METRICS aggregated over pods, anomaly ground truth).
    def inject_fault(self, kind: str | None = None, pod: str | None = None) -> str   # returns pod name
    def maybe_inject_random_fault(self, probability: float) -> str | None
    # Healing hooks (used by SimulatedHealer) — each returns bool success and, when it addresses the
    # active fault's pod/node, clears or damps the fault:
    def restart_pod(self, pod: str | None = None) -> bool     # None -> unhealthiest pod
    def reschedule_pod(self, pod: str | None = None) -> bool
    def scale_up(self) -> bool
    def drain_node(self, node: str | None = None) -> bool
    def alert_admin(self, detail: str = "") -> bool           # always True; records the alert
    # Introspection:
    def unhealthiest_pod(self) -> str
    def top_pressure_metric(self) -> str                      # canonical metric name with highest utilization
    def has_active_fault(self) -> bool                        # ground truth (ramping or active)
    def in_outage(self) -> bool                               # ground truth outage (anomaly==1 now)
    current_time: float                                       # sim epoch seconds
```

### scripts — thin argparse CLIs (all accept `--config`, default `config.yaml`)
- `generate_dataset.py` (data-agent): `--records`, `--out`; calls generate_k8s_telemetry; prints summary.
- `train_model.py` (model-agent): `--source {synthetic,kaggle}` override; calls train_model; prints test metrics.
- `evaluate_model.py` (model-agent): calls evaluate_model; prints metrics + report paths.
- `retrain_feedback.py` (monitoring-agent): checks should_retrain (`--force` to skip check), runs retrain_from_feedback.
- `run_selfhealing.py` (simulation-agent): **the paper's full closed loop.**
  Flags: `--ticks N` (default 600), `--fault-prob p` (default 0.01 per tick), `--no-healing`
  (ablation: predictions logged but no actions — for comparing MTTR/uptime with healing off).
  Loop: SimulatedCluster.tick -> keep rolling buffer of last window_size rows -> once buffer full,
  FaultPredictor.predict_proba -> decide -> SelfHealingEngine.handle (context target = unhealthiest
  pod, top_metric = cluster.top_pressure_metric()) -> MetricsTracker records (actual_fault from
  cluster ground truth; fault_start on outage entry, recovery on outage exit).
  Ends by printing a report table (MTTR, uptime %, FP rate, precision, recall, actions taken),
  saving metrics to reports/selfhealing_report.json, and appending all events to the feedback log.

### dashboard/app.py — dashboard-agent
Flask app (port 5000) for paper Step 7 visibility. Routes:
- `GET /` — renders templates/index.html: current metrics table (reports/metrics.json +
  reports/selfhealing_report.json when present), recent feedback events table (last 50), and the
  evaluation PNGs. Missing files -> friendly empty states, never a 500.
- `GET /api/metrics` — merged JSON of the two report files.
- `GET /api/events?limit=N` — last N feedback events.
- `GET /reports/<file>` — serves PNGs from the reports dir (send_from_directory; filename sanitized).
No CDN dependencies; inline CSS. Run: `python dashboard/app.py --config config.yaml`.

### tests — test-agent (pytest, every test < ~20 s, no full training runs)
- test_config.py — load_config resolves root; seeds don't crash.
- test_dataset.py — synthetic generator: schema == LOG_SCHEMA, metrics canonical, anomalies present,
  gradual buildup (metric mean in the 10 ticks before an anomaly onset exceeds the global normal mean);
  kaggle loader on data/raw CSV: canonical, sorted, no NaN/inf, <= max_raw_features features.
- test_preprocessing.py — clean handles inf/nan/dups; Scaler roundtrip (save/load, transform range [0,1]).
- test_windowing.py — shapes, horizon labeling correctness on a hand-built y, stride handling.
- test_features.py — engineered_dim matches output; lag/delta/flag values verified on a tiny window.
- test_model.py — build_lstm_model output shape (None,1), compiles, one train step on random data.
- test_predictor.py — decide() band mapping incl. exact-threshold edges.
- test_healing.py — SimulatedHealer + engine: trigger routes to expected action; cooldown enforced;
  no action when trigger False.
- test_metrics.py — MTTR/uptime/FP-rate math on scripted event sequences.
- test_feedback.py — JSONL roundtrip; should_retrain thresholds.
- test_cluster_sim.py — tick returns canonical row; injected memory_leak ramps memory; restart_pod
  during outage leads to recovery within a bounded number of ticks.
- conftest.py — fixtures: tiny cfg dict (small window/records) built on tmp_path, seeded.

## Cross-cutting rules
- NO module writes files outside cfg paths. NO network calls at import time. NO real K8s calls
  unless cfg mode == "kubernetes".
- Keras 3 API (`import keras` or `from tensorflow import keras` — use `keras.saving`/`model.save("... .keras")`).
- Silence TF logs in scripts: set `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` before importing TF.
- `float32` for all model inputs.
- Import config as `from selfheal.config import load_config, resolve_path, set_global_seeds`.
- Cross-module imports allowed ONLY of the public names defined in this SPEC.
