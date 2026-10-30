# Self-Healing Cloud Platform Using LSTM-Based Predictive Models

Reference implementation of the research paper:

> **Atharv S. Patil, Sanjay S. Patil (2025).** *Self-Healing Cloud Platforms Using LSTM-based
> Predictive Models for Early Fault Detection.* International Journal of Computer Engineering and
> Technology (IJCET), 16(5), 177–187. DOI: [10.34218/IJCET_16_05_012](https://doi.org/10.34218/IJCET_16_05_012)

The system predicts impending faults in Kubernetes environments from time-series telemetry using an
LSTM network, and triggers proactive self-healing actions (pod restart, rescheduling, replica
scaling, node drain, admin alert) **before** faults escalate into outages. A feedback loop tracks
MTTR, uptime, false-positive behaviour and prediction reliability, and drives adaptive retraining.

## Paper → code mapping

| Paper section | What it describes | Implementation |
|---|---|---|
| §3.1, Table 1 | Kubernetes Anomaly/Log Dataset [17] | `data/raw/balanced_shuffled_traffic.csv` (the actual cited Kaggle dataset) + `selfheal/data/dataset.py` synthetic generator producing the exact Table 1 log schema (~85,000 entries) and cluster metrics |
| §3.2 / Step 1 | Cleaning + Min-Max normalization | `selfheal/data/preprocessing.py` |
| §3.2 / Step 2 | Sliding time window X_k | `selfheal/data/windowing.py` |
| §3.4 / Step 3 | Lag features, rolling μ/σ, rate of change Δ, utilization ratio r, threshold flags f (τ=0.9) | `selfheal/data/features.py` |
| §3.2 / Step 4, Fig. 1 | Input → LSTM → Dropout → Dense(ReLU) → Dense(sigmoid), binary cross-entropy | `selfheal/models/lstm_model.py`, `selfheal/models/train.py` |
| Step 5 | Fault thresholding & decision rule (ŷ ≥ T) | `selfheal/prediction/predictor.py` |
| §3.3 / Step 6 | Self-healing via Kubernetes APIs: RestartPod, RescheduleNode, ScaleUp, DrainNode, AlertAdmin | `selfheal/healing/actions.py`, `selfheal/healing/engine.py` |
| Step 7 | Performance monitoring (MTTR, uptime %, FP rate) + feedback loop & retraining | `selfheal/monitoring/metrics.py`, `selfheal/monitoring/feedback.py` |
| §1 (evaluation) | Accuracy, responsiveness, resource overhead | `selfheal/models/evaluate.py`, `reports/` |
| Closed loop (Fig. 1) | Full predict → heal → monitor → feedback cycle | `selfheal/simulation/cluster_sim.py`, `scripts/run_selfhealing.py` |

## Setup

```bash
python3 -m venv .venv                # python 3.12
.venv/bin/pip install -r requirements.txt
```

(If `python3-venv` is missing: `sudo apt install python3.12-venv`.)

## Quick start — the full paper pipeline

```bash
source .venv/bin/activate

# 1. Dataset (paper §3.1): generate the ~85k-entry Kubernetes telemetry + Table-1 log dataset
python scripts/generate_dataset.py

# 2. Train the LSTM fault predictor (paper Steps 1-4)
python scripts/train_model.py                 # synthetic K8s telemetry (default)
python scripts/train_model.py --source kaggle # or: the cited Kaggle dataset [17]

# 3. Evaluate (accuracy, precision, recall, F1, AUC + plots in reports/)
python scripts/evaluate_model.py

# 4. Run the closed self-healing loop (paper Steps 5-7) on the simulated cluster
python scripts/run_selfhealing.py --ticks 600
python scripts/run_selfhealing.py --ticks 600 --no-healing   # ablation: healing disabled

# 5. Adaptive retraining from the feedback log (paper Step 7)
python scripts/retrain_feedback.py            # add --force to skip the should_retrain check

# 6. Monitoring dashboard
python dashboard/app.py                       # http://localhost:5000
```

## Running against a real Kubernetes cluster

Set in `config.yaml`:

```yaml
healing:
  mode: kubernetes
  kubernetes:
    kubeconfig: null      # null = default kubeconfig chain (~/.kube/config, in-cluster, $KUBECONFIG)
    namespace: default
```

`KubernetesHealer` then executes the paper's action set through the official Kubernetes Python
client: pod deletion (controller recreation), pod eviction/rescheduling, deployment replica
scaling, node cordon + drain, and admin alerts (structured log + optional webhook). In the default
`simulate` mode no real API is ever touched.

## Configuration

Every parameter of the paper's algorithm lives in [config.yaml](config.yaml): window size *w*,
label horizon, lag count *L*, safety threshold τ, resource limits for utilization ratios, LSTM
units/dropout, decision threshold *T*, severity bands, healing cooldowns, and the feedback-loop
retraining triggers.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Project structure

```
selfheal/
  config.py                  configuration + seeding
  data/                      dataset loaders & synthetic generator, preprocessing,
                             sliding windows, feature engineering        (Steps 1-3)
  models/                    LSTM model, training, evaluation            (Step 4)
  prediction/                threshold decision rule                     (Step 5)
  healing/                   K8s + simulated healers, healing engine     (Step 6)
  monitoring/                MTTR/uptime/FP metrics, feedback loop       (Step 7)
  simulation/                simulated Kubernetes cluster testbed
scripts/                     CLI entry points for each stage
dashboard/                   Flask monitoring dashboard
tests/                       pytest suite
data/raw/                    datasets (cited Kaggle CSV + generated telemetry)
models/artifacts/            trained model, scaler, metadata
reports/                     evaluation metrics and plots
feedback/                    JSONL feedback event log
```
