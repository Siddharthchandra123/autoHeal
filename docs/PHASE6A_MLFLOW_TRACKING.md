# Phase 6A: Opt-in MLflow experiment tracking (AutoTune AI v0.8.7)

## Current verified boundary

- User independently ran the **v0.8.6** real CPU self-healing experiment in Windows VS Code on November 2, 2026: **207 passed, 1 skipped**, followed by two real PEFT LoRA training attempts using `sshleifer/tiny-gpt2` and a deliberately injected `overfit` stop on trial 1. Trial 2 completed normally.
- User's trial-1 and trial-2 validation losses: `10.365184783935547` and `10.365181922912598`, respectively. Difference ~`2.86e-6`, not meaningful model-quality evidence. `promotion_approved: false`.
- Phase 6A's additional MLflow code passed **216 local offline tests** (208 inherited + 8 new). These use a fake MLflow client and, for controller tests, a fake training boundary. **No live MLflow tracking-server session or database write has been verified** in this build environment.
- Hosted Meta advisor still unverified; mock default remains. Unsloth GPU remains optional and unverified. The earlier step-2-to-step-4 actual Trainer resume succeeded; exact `resume_proof.json` contents have not been supplied.

## What is added

`autotune/hf_mlflow.py` is an optional `MlflowClient` reporter integrated into `autotune/hf_selfheal.py` via `--mlflow-uri` and `--mlflow-experiment`. When opt-in is enabled, the controller creates one parent MLflow run and one child MLflow run per real training attempt. Each trial logs its validated config, scalar training and validation metrics, best validation loss, monitoring decisions, config hash, diagnosis, injection flag, and audit manifests. The controller logs the selection receipt, trial lineage, and non-promotion status.

**Intentional security and reproducibility boundaries:**

- If `--mlflow-uri` is omitted, no MLflow package import or network access occurs. An environment variable alone does not opt in.
- Neither weight tensors nor optimizer `.pt` files nor `.safetensors` are uploaded by this reporter. MLflow receives JSON/YAML metadata and hash manifests only; full weight files remain on the user's local disk. Model registration and remote artifact uploads are separate future, explicit actions.
- The existing local `controller_result.json`, JSONL event journal, and per-trial manifests remain the authoritative evidence. An enabled-but-failing tracker blocks the controller and requires a **new workspace**; it will not silently claim successful tracking or launch speculative retries.
- The current reporter logs metrics after an attempt finishes, not live during every training step. Real-time streaming is a later Phase 6B feature.
- The local web server should bind only to `127.0.0.1`. Avoid sending private dataset paths/configuration to a remote server without authorization and secure transport.

## Windows PowerShell acceptance test

**1. Apply the v0.8.7 patch without replacing existing `outputs` or `.venv`.**

**2. Install the optional MLflow dependency:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-mlflow.txt
.\.venv\Scripts\python.exe -m pytest -q
```

**3. In one VS Code PowerShell terminal start the local MLflow server:**

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Leave this terminal open. Open `http://127.0.0.1:5000` locally. MLflow's backend is local SQLite; it does not require Docker, Kubernetes or a GPU. Consult the MLflow tracking server guide: https://mlflow.org/docs/latest/tracking/server/.

**4. In a second terminal, run a NEW real CPU experiment with logging enabled:**

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml --execute --fault-inject-overfit --workspace outputs/hf-selfheal-mlflow-01 --mlflow-uri http://127.0.0.1:5000 --mlflow-experiment AutoTune-AI-CPU
```

Use a fresh workspace name if that folder already exists. The above is an **intentional fault-injection acceptance test**, not proof of natural overfitting. This will download or reuse cached tiny-GPT2 weights, train on CPU, and attempt to log to your local MLflow server. It will not overwrite previous successful experiments.

**5. Acceptance gates:**

- CLI `ok: true`, `attempt_count: 2`, `promotion_approved: false`.
- MLflow UI experiment `AutoTune-AI-CPU` contains **one parent controller run and two child trial runs** (`mlflow.parentRunId` tags).
- Each child has a config SHA256, scalar metrics and verified manifest artifacts; parent has final selected best validation loss, best-trial tag, and `controller_result.json`.
- No `.safetensors` or `.pt` weight upload in MLflow artifacts.
- Local `controller_events.jsonl` includes `mlflow_controller_created` and `mlflow_trial_created` with run IDs.

If the server is unavailable, inspect the local `controller_state.json` and `controller_events.jsonl` for `blocked`. **Do not rerun into the same directory**; start the server, investigate, then use a new workspace. Do not delete prior outputs.

## Next steps

Phase 6B: real MLflow server verification, live-step logging, read-only run comparison dashboards, failed-run reconciliation and offline exports. Phase 7: local CPU Kubernetes/Kubeflow Pipelines; Phase 8: Katib tuning. Hosted Meta reasoning and genuine naturally detected anomalies require their own verification.
