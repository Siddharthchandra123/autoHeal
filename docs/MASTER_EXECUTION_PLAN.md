# AutoTune AI v0.9.0 - Current roadmap (2026-11-02)

**Verified (live Windows):** Real CPU PyTorch, PEFT LoRA fine-tuning, adapter reload, checkpoint continuation, bounded fault-injected two-attempt recovery, and local MLflow accepted 3 runs. **Not live verified:** new read-only dashboard, real hosted Meta agent, actual Unsloth GPU, vulnerability scanning, Kubeflow, Katib.

| Priority | Stage | Acceptance | Status |
|---|---|---|---|
| 1 | Phase 6B FastAPI Viewer | Strict read-only local API, safe MLflow queries, tests | 238 Python tests passed; live Windows pending |
| 2 | Phase 6B Next.js Viewer | npm install, TypeScript build, three linked runs visible | Source ready; runtime build pending |
| 3 | S0 Security gates | pip-audit/npm audit, immutable model intake, dataset triage, deny/quarantine reports | Planned; not implemented |
| 4 | Phase 6C live monitoring | polling/streaming and richer experiment visualization | Planned |
| 5 | Phase 7 Kubernetes + Kubeflow | CPU training pipeline orchestrated in cluster | Planned |
| 6 | Phase 8 Katib | HPO trials coordinated, separate from agent diagnosis | Planned |
| 7 | Subsequent | Git review automation, authenticated job control, final held-out evaluation/promotion gates | Planned |

**Release hardening note:** MLflow optional dependency ranges now require `>=3.17.2,<4`, but we have not run pip-audit/npm audit, checked your installed version or shipped lockfiles.

**Immediate runbook:** `docs/PHASE6B_DASHBOARD_RUNBOOK.md`. **Security requirements:** loopback-only until authenticated; no arbitrary model uploads or artifact execution; do not infer meaningful improvement from tiny test losses.

---

# v0.8.9 — Phase 6A live verification and Phase 6B priority (2026-11-02)

**Documentation-only.** Source stays v0.8.7; user verified a successful local MLflow run using the Phase 6A CPU self-heal experiment. The console reported two successful HF training attempts and links for controller, trial-01 and trial-02; exact run IDs and limitations are in `AutoTune_AI_Phase6A_Live_Verification_v0.8.9.md`.

## Phase 6B implementation execution plan

| Workstream | Implementation | Acceptance gate |
|---|---|---|
| FastAPI read-only API | `GET /health`, `GET /experiments`, `GET /experiments/{id}/runs`, `GET /runs/{id}/metrics`, `GET /runs/{id}/audit` | Integration tests with fake MLflow + local journals; no credentials leaked |
| Next.js frontend | TypeScript App Router, experiment list/detail, loading/error states, responsive charts | TypeScript build, UI tests; operates on Windows CPU without Docker |
| Provenance and status | Clearly label injected fault vs natural diagnosis, config hashes, checkpoints, run status | Trial 1 shows "injected" annotation, no false model-promotion claim |
| MLflow hierarchy | Check `mlflow.parentRunId` and run tags from tracked records before showing a hierarchy | Parent-child relationship verified from provider metadata, not guessed by names |
| Read-only security | Environment-only server URI, same-origin/CORS, path allow-list, no raw arbitrary filesystem browse | Negative tests for traversal, missing runs, malformed IDs, server unavailable |
| Job controls (subsequent sprint) | Auth, validated POST job, background worker *process*, SSE events, cancel audit | Separate explicit acceptance step; not in read-only MVP |

## Follow-up proof and boundaries
1. Examine the user's MLflow UI tags, metrics, artifacts and parent-child runs (3 run links reported).
2. Inspect `outputs/gpt2-resume-probe-01/resume_proof.json` for optimizer/scheduler evidence.
3. Do not treat `~9.54e-7` tiny-GPT2 validation difference as an improvement claim; first diagnosis was fault-injected.
4. Preserve v0.8.8 dashboard/GPU architectural direction, but defer real GPU acceleration and Kubeflow/Katib until basic UI is tested.

---

# v0.8.8 frontend and GPU implementation addendum (2026-11-02)

**Documentation-only.** Existing working Python backend and historical outputs remain untouched.

### Near-term order

1. **6A acceptance:** run local MLflow SQLite server and inspect parent/child runs from new 2-attempt CPU experiment. Preserve prior outputs.
2. **6B UI foundation:** add `/frontend` Next.js App Router + TypeScript + Tailwind + shadcn/ui. Provide read-only `GET /health`, `GET /experiments` (FastAPI) initially; typed fetch client and error/loading states.
3. **6C job controls:** add `POST /training/jobs` with validated config and explicit execution consent; `GET /training/jobs/{id}` and SSE metrics; stop/cancel with audit trail. Run jobs in separate Python worker process, never in HTTP request or frontend.
4. **6D observability:** trend charts, retry/diagnosis timeline, checkpoints and config diff; integrate MLflow as optional visualization, local journal authoritative.
5. **GPU capability profiles:** `cpu`, `cuda16`, `cuda24`, `cuda32`, `rocm-mi300x`, `rocm-mi325x`, `rocm-mi350x`; profiles are logical labels, not automatically detected devices. Runtime gate queries actual hardware and available VRAM, version support, costs and backend readiness.
6. **7+ orchestrator:** Kubernetes/Kubeflow and Katib only after job API works locally. Remote AMD Instinct on supported Linux/ROCm. Preserve Ray Tune as alternative, not concurrent Katib optimizer.

### Acceptance checks

- UI compiles with TypeScript, typed job payloads and API error handling.
- Windows CPU launch with zero GPUs and zero Docker; can view prior experiments.
- Fake SSE stream e2e updates charts and displays controlled fault injection badge.
- GPU settings without GPU refuse execution gracefully; no unexpected model download or memory reservation.
- Remote GPU workers require actual training/adapter reload/checkpoint-resume verification before claims.
- API never returns access tokens, model/provider credentials or secrets.

### Hardware positioning

| Tier | GPU | Published memory | Use |
|---|---|---:|---|
| CPU | none | system RAM | development + tiny model LoRA |
| Entry NVIDIA | RTX 5060 Ti | 16 GB | LoRA/QLoRA trial |
| Practical NVIDIA | RTX 3090 / RTX 4090 | 24 GB | more training headroom |
| High-end NVIDIA | RTX 5090 | 32 GB | high-capacity single-GPU trials |
| AMD data center | MI300X | 192 GB | large model experimentation; Linux/ROCm |
| AMD data center | MI325X | 256 GB | enterprise training |
| AMD data center | MI350X | 288 GB | enterprise training |

Caution: VRAM numbers are hardware capacities, **not** guaranteed model fit. Evaluate model architecture, precision, quantization, context length, optimizer, sequence length, and microbatch per run.

---

# CURRENT CHECKPOINT - v0.8.7 | Phase 6A opt-in MLflow experiment tracking (2026-11-02)

**Executable source v0.8.7; 216/216 build-environment offline tests passed.** The user's v0.8.6 Phase 5E live Windows results are now VERIFIED: **207 passed / 1 skipped**, real HF PEFT LoRA trial 1 with **explicit fault-injected** overfit and trial 2 with a validated changed config, both real CPU training attempts complete. Trial 1 best validation loss `10.365184783935547`; trial 2 `10.365181922912598` (tiny `~2.86e-6` difference, no meaningful improvement claim). Final `ok:true`, `attempt_count:2`, `best_trial:2`, `promotion_approved:false`.

Phase 6A adds **optional**, explicitly requested `--mlflow-uri` (and experiment name) to `autotune.hf_selfheal` via new module `autotune/hf_mlflow.py`. Logs parent controller run, child runs per training attempt, config hashes, true metrics, fault-injection tags, validated checkpoint manifests and controller receipts. **Does not upload trained weights or optimizer files.** Local journaling stays authoritative. MLflow server was **NOT live tested**; 8 new tests exercise a fake client and fake trainer. MLflow failure blocks rather than silently claiming tracking succeeded. With flag absent, no MLflow import or remote call is made.

Windows next: see `docs/PHASE6A_MLFLOW_TRACKING.md`. Install `requirements-mlflow.txt`, run a localhost MLflow server on `127.0.0.1:5000`, then run a **fresh** CPU self-healing acceptance experiment with `--mlflow-uri http://127.0.0.1:5000`. Verify a controller parent and two child MLflow runs in the UI. Do not overwrite `outputs/hf-selfheal-injection-01`, `gpt2-cpu-01`, or `gpt2-resume-probe-01`. The original `resume_proof.json` field contents remain uninspected by the assistant. Hosted Meta AI and Unsloth GPU execution remain unverified.

Phase 6B next: verify real MLflow server, add live streaming and controlled reconciliation; later Kubeflow/Katib. Keep the project CPU-first.

---

# AutoTune AI Master Execution Plan - v0.8.6 (Phase 5E)

## Immediate development status (2026-11-02)

**Implemented:** CPU-first real Hugging Face PEFT LoRA recovery controller with bounded attempts, real Trainer telemetry bridge, policy-validated advisor patch, versioned YAML snapshots, checkpoint/adapter manifest verification, immutable past outputs and trial comparison. Added explicit, auditable test-only overfit fault injection.

**Tested locally:** 208/208 Python tests, of which 15 are new controller tests built around a mocked HF Trainer boundary and real integrity manifests. **Not live verified:** two-attempt Hugging Face training on Windows, live Meta API, MLflow, Kubeflow, Katib and optional Unsloth GPU. User's previous Windows suite: 192 passed, one skipped, plus real CPU LoRA, reload and pause/resume through step 4.

### Next immediate steps

1. Verify `outputs/gpt2-resume-probe-01/resume_proof.json` (begin step 2, end 4, optimizer state and scheduler step evidence).
2. Apply v0.8.6 patch without deleting `.venv` or any past `outputs` folders.
3. Run `pytest -q` and offline self-heal preflight on Windows.
4. Run a new controlled **real** Tiny-GPT2 two-attempt trial with `--fault-inject-overfit`, inspect both manifest files, controller journal and configs. Explicitly label trigger as synthetic; this confirms control flow, not natural anomaly detection or model quality.
5. Test a naturally diagnosed overfit/underfit on representative datasets and enforce an external holdout set to avoid tuning to the same validation split.
6. Add MLflow run/metric/artifact lineage and then a CPU Kubernetes/Kubeflow/Katib proof of concept.

### Guardrails

No unbounded retries (1..3). No arbitrary file/code edits from LLM. No promotion without independent holdout and approval. Starting a new trial after hyperparameter changes is NOT optimizer checkpoint resume. Existing workspaces are not reused. No unattended retries on an unknown/crashed training state.

---

# AutoTune AI - v0.8.5 progress checkpoint | 2 November 2026

**Scope:** Documentation-only status summary; no new training code, tests or deployment. Executable source: v0.8.3. Complete older roadmap is preserved below.

| Milestone | Current status | Evidence / verification boundary |
|---|---|---|
| Config/telemetry/monitoring | Verified in mock workflow | 15, 36 Windows tests; synthetic anomaly decisions |
| Durable recovery + Meta advisor | Verified with mock training | 48 and 77 tests; remote Meta API not tested |
| Real PyTorch training | Verified on Windows CPU | 95 tests; actual model weights and optimizer checkpoint |
| Hugging Face PEFT LoRA | Verified on Windows CPU | Tiny GPT-2, 128 trainable params, 10.3652 toy val loss |
| Adapter SHA audit + reload | Verified on Windows | 6/6 adapter files, 10/10 checkpoint files, 8 tokens generated |
| Controlled HF checkpoint continuation | Completed 2-to-4 step sequence | Inspect user `resume_proof.json` to confirm optimizer/scheduler fields |
| CPU BERT task adapter | Offline tested only | User live CPU classification training not yet reported |
| Optional Unsloth backend | GPU config preflight only | No actual supported-GPU training |
| Real HF autonomous diagnosis / policy retry | NOT complete | Phase 5E priority |
| MLflow/Kubeflow/Katib | NOT deployed | Phases 6-8 |
| Dashboard/Git PR/release evaluation | NOT complete | Phases 9-12 |

## Next step with acceptance order

1. Read `outputs/gpt2-resume-probe-01/resume_proof.json` (user Windows terminal): begin at 2, end at 4, optimizer state nonempty, scheduler advanced.
2. Build real HF Trainer adapter to typed monitoring and checkpoint receipt; preserve metric-step alignment.
3. Connect durable controller and bounded mock Meta advice; create NEW run config for changed hyperparameters (do not blindly resume incompatible optimizer).
4. Prove an automatic two-attempt CPU experiment and global best selection, including retry exhaustion, corruption, crash and tamper tests.
5. Add MLflow local server, followed by CPU Kubeflow Pipelines/Katib and then production-grade dashboard/evaluation.

**Latest test evidence:** user Windows `192 passed, 1 skipped in 22.26s` (build v0.8.3 separately reported 193 passed). CPU-first; no dedicated GPU needed.

---

# CURRENT STATUS / ACTION PLAN - v0.8.4, 2 November 2026

*This is an evidence checkpoint on v0.8.3 code. The complete archived master plan follows below; references to older pending gates there are superseded by this section.*

## Latest verified milestone

- Windows CPU pytest: **192 passed, 1 skipped** (22.26 s).
- Actual Hugging Face LoRA continuation: step 2 deliberately paused and saved to `outputs/gpt2-resume-probe-01/checkpoints/checkpoint-2`; next process resumed from checkpoint and completed **step 4/4**, reporting `ok: true` and `resume_proof.json`.
- Best reported validation loss **10.365182876586914** on a four-train/two-validation toy dataset; not a quality/performance claim.
- Real optimizer/scheduler state is guarded in source but the proof JSON values have not been supplied in chat. Inspect them before independently certifying their recorded values.
- Unsloth: GPU YAML static preflight accepted, not executed; CPU Unsloth training correctly flagged unavailable. No GPU training run.

## Critical path

| Priority | Activity | Acceptance evidence | Current state |
|---|---|---|---|
| 0 | Read `resume_proof.json` | begin step 2, end step 4; nonempty optimizer; scheduler step >=2 | User output pending |
| 1 | Real HF trainer adapter to monitored experiment | Event schema, real callback metrics, checkpoint/stop actions | Not implemented as end-to-end controller |
| 2 | Durable two-attempt CPU recovery | Trial 1 stop, verified stored state, bounded config patch, new trial, global best selection | Not yet live-tested |
| 3 | Safety + fault testing | Invalid advice, tampering, crash restart, exhausted retry budget fail closed | Partly unit-tested, live integration pending |
| 4 | MLflow experiment tracking | Params, metrics, lineage, artifacts inspectable in tracking UI | Pending |
| 5 | Kubernetes / Kubeflow on CPU | Actual local CPU jobs and pipeline integration | Pending; cluster resources required |
| 6 | Katib HPO | Controlled trial search and stop policies | Pending |
| 7 | Dashboard + evaluation gate | Auditability and untouched holdout quality review | Pending |
| Optional | GPU Unsloth backend | Real supported GPU end-to-end execution | Deferred |

## Safety and scope

Preserve the existing `outputs/gpt2-cpu-01` and `outputs/gpt2-resume-probe-01`. A changed learning rate, adapter layout or optimizer may invalidate full optimizer-state resume; new hyperparameter attempts normally start a fresh training run, unless exact checkpoint compatibility is established. Meta is advisory; policy engine decides. No arbitrary LLM code execution. No unbounded retries. Do not train on the final holdout set.

## Next actionable user inspection

```powershell
Get-Content outputs\gpt2-resume-probe-01\resume_proof.json -Raw
```

## Documentation release

v0.8.4 is **documentation only**, carried forward from v0.8.3 source, not a new verified training-code release. The accompanying handbook keeps the 45-page v0.8.3 contents and adds four evidence/implementation pages at the beginning.

---

# ARCHIVED EXECUTION PLAN FROM v0.8.3

## LATEST CHECKPOINT - v0.8.2 controlled CPU checkpoint resume (2026-11-01)

**User verification now complete:** Windows CPU `sshleifer/tiny-gpt2` LoRA training, 6/6 final adapter files verified, 10/10 full checkpoint files verified, and **fresh adapter inference generated eight tokens**. The user reported `ok: true` for read-only audit and `--reload`; best recorded val loss 10.365182876586914. **Actual optimizer resume is still NOT user-verified** (both user audits showed `actual_resume_executed: false`).

**New implementation:** a distinct CPU-only, two-epoch/four-step test (`config/lora_cpu_resume_probe.yaml`) adds `--pause-after-step 2` to the causal-LM CLI, a guarded paused-run resume contract and `resume_proof.json` with pre-continue optimizer/scheduler observation. The **completed original `outputs/gpt2-cpu-01` is never touched**. Changes are source-level and offline-tested only; live HF dependencies and downloads were not available in this build environment.

**Build evidence:** `185 passed` offline (166 inherited + 19 contract/CLI tests). **Next user action:** execute the two Windows commands in `docs/PHASE5D_OPTIMIZER_RESUME.md`; inspect paused state and real resume proof. No autonomous Meta-directed real-HF retry or production claim.

---

## v0.8.1 status update - real CPU LoRA run verified by user (2026-11-01)

**Actual outcome:** Public tiny-GPT2 CPU PEFT LoRA finished one-epoch training on Windows, returned best validation loss 10.365182876586914, and wrote a LoRA final adapter plus full Trainer checkpoint. The 4-train/2-validation toy dataset is a plumbing smoke only; no model-quality benchmark or meaningful overfitting determination.

**Added engineering:** A read-only `hf_audit_cli` verifies plan/provenance, status/receipt agreement, final adapter SHA-256 and full checkpoint SHA-256, with optional fresh PEFT reload. Its implementation and failure-paths pass offline tests; user-machine verification of the saved files is pending. Full checkpoint content verification is not optimizer resume.

**Next acceptance milestones in order:**

1. [ ] Run the read-only CLI on user's `outputs/gpt2-cpu-01` and inspect `ok`, file counts, SHA agreement.
2. [ ] Run `--reload` on user CPU and verify genuine base + adapter inference.
3. [ ] Stage a **separate interrupted training run**; prove actual optimizer, scheduler and RNG restoration and continued updates without touching the completed original.
4. [ ] Integrate the real HF callback with durable recovery state and actual two-attempt self-healing, policy guardrails and advisor; no automatic retrials until this step passes.
5. [ ] Implement real MLflow backend tracking and model registry; then progress to CPU Kubernetes/Kubeflow/Katib, dashboard and independent test-set evaluation.

**Testing:** 166 offline tests passed (153 inherited + 13 new); hosted model calls and genuine live optimizer resume remain unverified. See `docs/PHASE5C_ARTIFACT_AUDIT.md`.

---

## VERSION v0.8.0 - UNIVERSAL HUGGING FACE CPU TASK ROUTER (2026-11-01)

**Design decision:** Use task-specific adapters, not one universal training loop for all HF architectures. CPU-first implementation currently covers causal-LM PEFT LoRA and BERT-style sequence classification full fine-tuning. Additional model families require explicit adapter implementation and evaluation contracts.

| Milestone | Implementation | Evidence | Status |
|---|---|---|---|
| Universal HF request CLI | task routing, offline preflight, explicit training invocation | Offline model/dataset contracts | Implemented |
| Causal-LM LoRA adapter | existing Phase 5B PEFT workflow | Offline tests | Implemented; live CPU run pending |
| Classification adapter | new CPU Trainer, loss + accuracy, checkpoint manifests | 18 new offline tests | Implemented; live CPU run pending |
| End-to-end model training | download tiny public models, train, verify reload | Windows run logs needed | **Next immediate gate** |
| Real HF recovery | persisted attempt state -> Meta suggestion -> bounded retry | Not yet tested | Pending |
| Phase 6 | MLflow for multi-task metrics and artifact lineage | Not yet built | Pending |
| Phase 7-8 | CPU Kubernetes/Kubeflow Pipelines and Katib | Requires local cluster | Pending |
| Further adapters | vision, audio, seq2seq | Not built | Planned |

**Latest build evidence:** `python -m pytest -q`: **153 passed**, 18 added in v0.8.0. This is not proof of live Hugging Face downloads, trained classification heads or LoRA adapters. Preserve the earlier user's 95-test Windows real CPU PyTorch checkpoint evidence separately.

**Operational commands:** `python -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml` (offline); add `--execute --output outputs/bert-tiny-cpu-01` for real CPU classification. For causal language modeling, use `config/lora_public_cpu.yaml` with the same CLI. Full runbook: `docs/HF_OPEN_SOURCE_MODELS.md`.

---

## VERSION v0.7.2 - CPU-ONLY REVISED EXECUTION PLAN (2026-11-01)

**Constraint:** User has no dedicated GPU. Prior GPU fine-tune requirement is now OPTIONAL, not a prerequisite to phases 6-12. Phase 5B's mandatory new gate is small public-model PEFT training and reload on CPU; phase 6 can proceed with CPU training and logs.

| Order | Next executable goal | Acceptance gate | Status |
|---|---|---|---|
| 1 | Offline public-model CPU preflight | `hf.device=cpu`, data SHA, schedule and guard validation | Available; offline contract tested |
| 2 | CPU Hugging Face LoRA pilot | Nonempty actual validation losses, saved adapter manifest and reload | Awaiting real Windows run |
| 3 | Real CPU checkpoint resume | Verified full optimizer/scheduler/RNG checkpoint survives controlled restart | Awaiting real test |
| 4 | Connect trainer to durable recovery | True stop, versioned safe patch, bounded second attempt, best validation checkpoint | Not built |
| 5 | MLflow tracking locally | Runs, parameters, metrics, checkpoints, lineage inspectable via UI | Planned |
| 6 | CPU-compatible local Kubernetes/Kubeflow | Small training job and Katib trial run end-to-end | Planned; requires resources |
| 7 | Versioned Git changes, dashboard, evaluation, deployment | Guarded, traceable, reproducible local self-heal | Planned |
| Optional | Meta Llama on remote GPU, 4-bit QLoRA, GPU telemetry | Real hardware-specific tests | Deferred, not a blocker |

All evidence remains: Phase 5A Windows real CPU PyTorch 95 tests; Phase 5B v0.7 build-env offline 134 tests. v0.7.2 contains a small CPU-specific code patch plus docs; separate CPU HF run is not yet verified. See `docs/PHASE5B_CPU_FIRST_RUNBOOK.md`.

---

> **Current checkpoint: v0.7 / Phase 5B source pilot, 2026-11-01.** USER verified Phase 5A on Windows: 95 tests passed, real tiny GRU checkpoint SHA256 verified. Phase 5B added guarded Hugging Face LoRA/QLoRA code, 39 offline fault tests, **134/134 total tests passed in build container**. Meta model downloads, GPU kernels, Trainer optimizer restore and adapter inference are NOT live-tested. Phase 6 MLflow is next after the GPU verification gate.

> **Current checkpoint: v0.6 / Phase 5A, 2026-11-01.** 95 local tests passed (77 inherited + 18 new). Genuine offline CPU PyTorch next-byte training on a small GRU and loadable model+optimizer checkpoint are implemented. Meta Llama LoRA GPU fine-tuning is **not** yet verified. User validated v0.5 Phase 4 with 77/77 on Windows. Do not confuse tiny GRU CPU training with Llama fine-tuning.

# AutoTune AI - Master Execution Plan

**Version:** v0.7 | **Baseline date:** 2026-11-01 | **Delivery mode:** local Windows PowerShell / VS Code -> optional GPU -> Kubernetes/Kubeflow | **Status:** Phases 1-4 implemented and locally unit-tested in CPU mock; Phase 5 next.

## Mission and success criteria

Build an audited, bounded, self-healing model training and fine-tuning system. It observes **training and validation** signals; distinguishes possible overfitting, underfitting and divergence; proposes constrained remediations; selects the best valid checkpoint; versions training configurations; retries within budgets; and requires a promotion gate before production release. The controller must be useful with deterministic policies even if the Meta diagnostic API is offline.

**Minimum viable success:** A user starts a run from a validated YAML config and sees its metrics, diagnosis, limited retry actions, run provenance and best configuration. **Full success:** each attempt is independently scheduled through Kubeflow; Katib can propose trials; the Meta agent is advisory; MLflow logs and versioned artifacts provide auditability; task-specific held-out evaluation and model registry gate deployment.

## Architectural principles

1. Meta Muse model = advisory reasoning only; **never execute arbitrary model-generated Python or shell**.
2. Deterministic monitor and policy engine decide whether to stop, apply patches, and retry.
3. Primary orchestrator = Kubeflow Pipelines v2; optimizer = Kubeflow Katib. Ray Tune is optional alternative, **never control the same trial as Katib**.
4. Production retraining must have time, compute, maximum-trial, retry and cost ceilings.
5. Model/data provenance: config hash, dataset hashes, checkpoint IDs and model revision; only config fingerprint is implemented in Phase 1.
6. No reliance on test-set feedback to choose hyperparameters. Keep an untouched holdout for final evaluation.
7. Transparent event records, reproducible seeds where practical, approval before promotion, least-privilege secrets.
8. Training code changes require a Git branch and automated tests; default agent change surface is a bounded YAML patch.

## Thirteen implementation phases (0 through 12)

| Phase | Scope | Artifacts | Exit gate | Status |
|---|---|---|---|---|
| 0 | Requirements, design, documentation protocol | Roadmap, architecture Mermaid, PDF handbook, AGENTS.md | Aligned functional/safety contracts | Complete at v0.2 |
| 1 | Foundation & configuration contracts | Python package, YAML schema checks, diagnostics CLI, sample data, baseline tests | CPU demo completes, tests green, invalid configs rejected before run | **Complete - 15/15 tests** |
| 2 | Telemetry and signal quality | Typed metrics, decision severity/action/evidence, missing data and nonfinite handling, spikes and plateau, JSONL replay | Mock + synthetic cases, noisy/missing/nonfinite tests, audit evidence | **Complete - 36/36 tests total** |
| 3 | Recovery state machine | Local atomic state journal, simulated checkpoint/receipt integrity, idempotent reconciliation, bounded retries, operator recovery | 12 fault-injection/acceptance tests; 48 total passing | **Complete for CPU mock v0.4** |
| 4 | Meta agent advisor | Strict JSON outputs, rationale, bounded recommendations, numeric-only payload, offline fallback, replayable state | 29 added tests, 77 total passed locally, synthetic durable two-trial verification | **Complete for CPU mock v0.5; hosted API unverified** |
| 5 | Real training | 5A PyTorch CPU reference + 5B guarded Hugging Face LoRA pilot | 5A verified real `.pt`; 5B GPU adapter reload and exact Trainer resume gate | **5A user verified; 5B source/contracts tested (v0.7), GPU pending** |
| 6 | MLflow as experiment system | Tracking server, stores, run lineage, metrics and artifact linking | Run IDs, reproducible artifact metadata and comparisons | Planned |
| 7 | Kubernetes + Kubeflow | Container images, each attempt independent KFP job, credentials and resource requests | Fresh cluster executes pipeline with observed retries | Planned |
| 8 | Katib + optional Ray Tune | Search-space API, constraints, objective metrics, early-stop ownership | Automated valid trials, chosen optimizer recorded, no dual ownership | Planned |
| 9 | Git-based configuration changes | Config patch, branch, tests, commit / PR, rollback and approvals | Provenance + reproduction, malicious proposed change rejected | Planned |
| 10 | Unified control UI and ops | Dashboard, progress streams, audit timeline, GPU/cost signals, user overrides | End-to-end visibility, cancel and alert flows tested | Planned |
| 11 | Evaluation and policy gates | Holdout test, quality KPIs, privacy/security, throughput/cost budgets, model registry | No test leakage, deployment blocked unless gates pass | Planned |
| 12 | Deployment hardening & reference benchmark | CI/CD, docs, load tests, failure recovery, permissions, optional AMD ROCm path | Repeatable clean install and documented demo on real hardware | Planned |

## Dependencies and execution order

- Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 completed locally for mock CPU, establishing deterministic safety before hosted AI.
- Phase 4 after Phase 3; hosted model can be disabled or unavailable without stopping core function.
- Phase 5 can be smoke-tested alongside Phase 4 but its production enablement requires Phase 2/3 safety gates.
- Phase 6 before Phase 7, so KFP job boundaries retain tracking provenance.
- Phase 7 before Phase 8, because Katib's Kubernetes-native trials need a working cluster.
- Phase 9 before enabling autonomous repository changes in any shared environment.
- Phase 10 can be incrementally enhanced throughout; Phase 11 gates deployment; Phase 12 is release engineering.

## Component interfaces

| Producer | Consumer | Interface | Phase |
|---|---|---|---|
| User/config repo | Config validator | YAML -> validated mapping + content SHA256 | 1 |
| Trainer callbacks | Metrics stream | Evaluation event: step, paired train_loss/val_loss, optional gradient, LR and GPU peak | 2 |
| Metrics stream | Diagnostic engine | Windowed signal history -> state/severity/action/evidence | 2 |
| Diagnostic engine | Recovery controller | `none/overfit/underfit/diverged`, best candidate, stop hint | 3 |
| MLflow tracking | Meta diagnostic agent | Aggregated numeric metrics + allowlisted config; no raw data | 4,6 |
| Meta agent | Policy engine | Strict typed recommendation JSON, advisory only | 4 |
| Policy engine | Config writer | Allowlisted numeric patch -> versioned YAML | 3,9 |
| Recovery | Kubeflow Pipelines | New attempt config+data+model+artifact references | 7 |
| Katib | Attempt launcher | Candidate hyperparameters, search metadata | 8 |
| Evaluation | Model registry | Holdout score + signed policy decision | 11 |

## Current v0.3 baseline: what exists and what does not

**Implemented / tested locally:** Phase 1-2: typed metric streams, audit decisions, replay CLI, spike/plateau checks, configurable synthetic overfit and underfit tests; early stop; best simulated checkpoint metadata; bounded retries; policy allowlist and patch limits; per-run JSONL; environment doctor; baseline YAML contract; config fingerprint; optional MLflow and optional Meta agent hooks. Tests: 36 passed on Python 3.13 in the sandbox (15 baseline tests plus 21 Phase 2 tests). A PowerShell procedure is supplied for Python 3.11, but not run on the user's Windows system.

**Present but not validated on an actual deployment:** Hugging Face LoRA training adapter; Kubeflow v2 pipeline compile source; standalone Katib example; Meta API integration; MLflow remote tracking; Streamlit dashboard. **Not implemented:** distributed GPU job scheduling, production checkpoint resume, versioned Git PR automation, Ray Tune backend, evaluation/model registry, cost quota enforcement. Synthetic checkpoint JSON is not a model weight file.

## Phase 2-3 verification and remaining production backlog

- [x] Implement typed `MetricEvent` with positive steps, aligned train/validation evaluation pair, optional LR/grad/GPU fields.
- [x] Add diagnostic `Decision(state, explanation, severity, action, evidence)`; only `action=stop` requests termination.
- [x] Detect nonfinite losses/gradient norms, repeated simultaneous spikes, sustained overfitting, high-loss underfitting; warn on plateau and isolated spikes.
- [x] Treat missing data, duplicate steps and non-numeric metrics as insufficient evidence (do not falsely early-stop).
- [x] Audit each decision in `monitor_decisions.jsonl`; store standard JSONL with explicit nonfinite fault markers.
- [x] Provide `python -m autotune.monitor_cli --file ... --config ...` for offline replay.
- [x] Backward compatibility for Phase 1 history without explicit steps and unchanged synthetic recovery.
- [x] Locally verified 36/36 tests (Python 3.13) and mock overfitting recovery; reproduced trial 2 = 0.58 synthetic validation loss.
- [x] Phase 3 (local mock): durable RUNNING/STOPPED/COMPLETED/ABANDONED journal, idempotency key, safe resume and blocked state.
- [ ] Production: external QUEUED/STOPPING/FAILED/RETRYING transitions and Kubernetes cancellation acknowledgement.
- [x] Phase 3 (local mock): checksum-verified simulated JSON checkpoint markers, receipt crash injection and safe best selection.
- [ ] Production: loadable weight and optimizer checkpoint manifests, cancellation acknowledgement, atomic object-store artifact promotion.
- [x] Phase 3: explicit max-trial bounds, unsupported diagnosis blocking, explicit abandon operator path.
- [ ] Production: GPU OOM classification, external-job cancellation and GPU-hour/cost policy.
- [ ] Future instrumentation: correct HF train-loss/eval-step alignment, full task-metric schema, genuine GPU OOM, GPU/cost telemetry and clock validation.

**Honest scope:** Phase 2 checks a typed, per-evaluation loss pair and optional numeric signals; trainer timestamps come from audit logs. The HF adapter currently uses the latest logged training loss at evaluation time, not necessarily a same-step averaged loss. No OOM restart or live GPU telemetry integration was tested.

## Working style and update protocol

For every subsequent project prompt, increment or update the current phase version and revise: `docs/MASTER_EXECUTION_PLAN.md`, `docs/HANDBOOK_SOURCE.md` + built `AutoTune_AI_Handbook_vX.Y.pdf`, and `AGENTS.md`. Include diagram updates when topology changes and test evidence whenever a claim of completion is made. Use a dated checkpoint in AGENTS.md and say explicitly which integrations were not executed. The project cannot automatically work between prompts; documentation is updated when we work on it together.

## Setup prerequisites

- Windows 11; VS Code; Python >=3.11; PowerShell; PyYAML; pytest; Git.
- Later: GPU/PyTorch model licensing; ROCm-compatible stack or CUDA; Docker/Kubernetes cluster; Kubeflow/MLflow installations; external Meta API credentials stored as environment secrets.
- No paid Meta API usage, GPU requirement or Kubernetes cluster is needed to execute Phase 1-3 CPU mock tests.

## Milestones, verification, and stop conditions

- Reproduce Phase 1 with `python -m autotune.doctor --mode mock`, `python -m pytest -q`, `python -m autotune.orchestrator --mode mock --no-writeback`.
- Require reviewers to verify claims about real training separately from synthetic tests.
- Stop recovery upon failed config validation, exceeded budgets, unavailable or corrupted artifacts, invalid metrics, repeated failures, or an unsafe agent patch.

## Official reference documents (checked 2026-11-01)

- Kubeflow Pipelines compilation: https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/compile-a-pipeline/
- Kubeflow Katib overview: https://www.kubeflow.org/docs/components/katib/overview/
- Katib early stopping: https://www.kubeflow.org/docs/components/katib/user-guides/early-stopping/
- MLflow experiment tracking: https://mlflow.org/docs/latest/ml/tracking

## Phase 3 implementation checkpoint - 2026-11-01 (v0.4)

- New `autotune/recovery.py` local durable controller: atomic JSON journal, write-before-run attempt state, SHA256 mock checkpoint receipt, best validation checkpoint verification, stable idempotency keys, exclusive local file lock and bounded attempts.
- New `autotune/recovery_cli.py`, `run_phase3.ps1`, `docs/PHASE3_RUNBOOK.md`, and `docs/diagrams/durable_recovery.mmd` (+ Graphviz/SVG).
- Reconciliation permits only validated results; uncertain `RUNNING` without receipt blocks until explicit operator abandon. No automatic retry of diverged/unknown states, no automatic repository overwrite.
- **Verified locally:** 48/48 tests passed on Python 3.13 (36 existing + 12 new), synthetic overfit -> deterministic patch -> next completed run, best val loss 0.58 (mock only).
- User previously confirmed v0.3 **36 passed in 0.53s** on Windows PowerShell, Python 3.13; v0.4 confirmed on Windows: 48 tests passed in 1.85s, experiment runs\20261101T141716Z; 2 attempts, second completed, best mock validation loss 0.58.
- External model weights/checkpoint resume, distributed locking/Kubernetes job reconciliation, real GPU retries and Meta hosted advice remain future work.

### Phase 4 detailed acceptance criteria

1. Meta advisor uses a strict numeric-only schema with JSON parsing and unknown-key rejection; no arbitrary code or shell instructions accepted.
2. No secrets, raw samples, or model-generated untrusted paths are transferred to advisor.
3. Offline fallback remains fully operational, preserving Phase 3 journal and receipts.
4. Advisory call has bounded timeout, retry policy, audit trail, and safeguards against prompt injection.
5. New tests verify malformed, oversized, adversarial, partial and contradictory responses; replay should remain deterministic for saved proposals.
6. Real hosted Meta connection only marked verified when an actual authenticated request succeeds on user-approved credentials.

## Phase 4 implementation checkpoint - 2026-11-01 (v0.5)

- User independently confirmed Phase 3 on Windows: **48 passed in 1.85s**, durable experiment `runs\20261101T141716Z`, statuses stopped/completed, SHA-256 checkpoint digest, best synthetic validation loss 0.58.
- Added `autotune/advisor.py` strict parser and bounded Model API connector; `autotune/advisor_cli.py` read-only advisor preview; `config/agent_mock.yaml` and `config/agent_meta.example.yaml`; integrated source/patch/snapshot into durable recovery.
- Advisor input sends only last eight numeric metric signals and approved numeric training parameters; output rejects extra keys, duplicate keys, nonfinite values, conflicting diagnosis, policy violations. No raw text or credentials in user message.
- Remote provider uses `MODEL_API_KEY` from environment and Meta Chat Completions (Muse Spark 1.3), with 20s timeout, no client retries. Missing credentials or invalid reply -> bounded deterministic fallback, not a false success claim. No live API call was executed.
- Advice is stored with input config SHA-256, checkpoint SHA-256, output config SHA-256 and record hash. Resume replays an existing validated record instead of asking the model again. Tamper detection blocks.
- **Verified locally:** 77/77 tests passed (48 inherited + 29 Phase 4), and CPU end-to-end mock advisor -> bounded trial 2 completed. Real API/GPU/Kubeflow/MLflow remote stack still unverified.
- New assets: `run_phase4.ps1`, `docs/PHASE4_RUNBOOK.md`, `docs/diagrams/meta_advisor.mmd` and SVG/PDF handbook v0.5.

### Phase 5 acceptance criteria - real training integration

1. Confirm hardware backend, GPU driver, VRAM and model access/licensing; never assume CUDA can run on an AMD integrated GPU.
2. Tiny representative dataset with immutable train/validation partitions, reproducible seed and dataset hash.
3. Actual HF Trainer training/evaluation callbacks have aligned step-window losses; monitor and advisor operate on true metrics, not a scripted sequence.
4. A checkpoint contains loadable model/adapter weights and required metadata; resume compatibility verified, including optimizer/scheduler when applicable.
5. Stop, audit and retry under a finite budget; compare new validation results with previous best; preserve independent held-out test for model promotion.
6. Re-run all tests on user's Windows environment and document model/backend/runtime details and observed measurements.


## Phase 5A technical execution record (v0.6)

- `autotune/real_training.py`: validate local JSONL train/validation, fingerprint split bytes, tokenize next-byte targets, genuinely train with AdamW, evaluate per epoch, monitor, atomically save best weights and optimizer state.
- `autotune/real_recovery.py`: independent `real-torch-v0.6` state schema, real checkpoint receipts, SHA256 and `weights_only` validation, policy-gated retry for diagnosed overfit/underfit, explicit blocked states for uncertainty and failure; best selection remains a promotion *candidate*, not an approved release.
- `autotune/real_cli.py`, `config/real_cpu.yaml`, `requirements-real-cpu.txt`, `run_phase5.ps1`, `docs/PHASE5_RUNBOOK.md`: offline Windows reproducibility path.
- Tests: `tests/test_real_phase5.py` adds 18 cases; **95/95** local tests passed with torch CPU. A genuine 4-epoch smoke test lowered observed validation loss from 5.6072 to 5.5675. This is a smoke check, not a quality benchmark.
- **Critical remaining gate (Stage 5B):** run pretrained Meta Llama LoRA on licensed model + compatible GPU, verify actual PEFT adapter state and inference reload, checkpoint/resume with optimizer state, update data/weights provenance and compare independent validation metrics. No Meta/GPU API has been used in Phase 5A.
- Continue to Phase 6 only after Stage 5B has either passed or been explicitly deferred in the documented scope.


## Phase 5B implementation checkpoint and work breakdown (v0.7)

- [x] Fix local PyTorch missing NumPy warning in install requirements.
- [x] Build offline `lora_cli` plan with train/validation SHA256, train/validation duplicate prevention, bounded maximum optimizer steps, and evaluation-count preflight.
- [x] Isolate HF/PEFT optional imports, requiring explicit `--execute` to download or train.
- [x] Provide Meta Llama LoRA and public tiny-GPT2 CPU configs. Require approved Meta license to retrieve gated model.
- [x] Add QLoRA 4-bit code path with a separate hardware/library compatibility gate.
- [x] Wire fixed-interval train logging to validation metrics, Phase 2 monitor and graceful Trainer early stop.
- [x] Implement SHA256 manifests for whole Trainer checkpoint (adapter, optimizer, scheduler, RNG) and final adapter. Provide explicit trusted resume and reload entrypoints.
- [x] Add 39 offline contract/safety tests, **134 total passing** in build environment. These do not run Hugging Face Trainer.
- [ ] Run genuine PEFT LoRA CPU public-model smoke, verify adapter reload and actual checkpoint format.
- [ ] Run licensed Meta Llama single-GPU fine-tune, reload a final adapter and validate interrupted Trainer checkpoint/optimizer resume.
- [ ] Benchmark memory, training throughput, and collect GPU telemetry; do not infer gains from fabricated data.
- [ ] Connect trusted HF checkpoint protocol to durable recovery and Meta advisor (separate safety milestone).
- [ ] Deploy persistent MLflow server + artifact store (Phase 6), then Kubeflow/Katib production infra (Phases 7-8).

**Next execution order:** 5B dry-run on Windows -> public tiny-model optional CPU smoke -> GPU/ROCm or CUDA preflight -> Meta model access -> full LoRA training -> checkpoint/resume integrity and adapter reload -> Phase 6 MLflow. Change the default `hf.revision: main` to a specific commit SHA for reproducible production runs. Never run Katib and Ray Tune simultaneously for one trial.

**Source links (checked 2026-11-01):**
- Meta model access: https://dev.meta.ai/llama/docs/getting-the-models/hugging-face
- PEFT adapter training & saving: https://huggingface.co/docs/transformers/peft
- Trainer resume/checkpoints: https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer_recipes.md
- bitsandbytes ROCm support: https://huggingface.co/docs/bitsandbytes/installation


## v0.8.3 - optional Unsloth GPU backend (2026-11-02)

- **Decision:** use Hugging Face + PEFT as the default CPU training backend. Add Unsloth Core / TRL as a separate opt-in, fail-closed accelerator backend when a supported NVIDIA / ROCm GPU is present.
- **Implemented:** experimental `unsloth_cli.py` and `unsloth_backend.py`, safe repository-ID validation, offline provenance preflight, real GPU hardware gate, per-evaluation monitoring and artifact integrity hooks, one-trial budget; sample GPU config and runbook.
- **Verification:** local suite 193 passed (185 inherited + 8 new). No supported GPU in build environment; zero live Unsloth training. Avoid speedup claims.
- **Next:** complete user Windows real HF optimizer-state resume verification (v0.8.2) before implementing agent-driven HF restarts. Defer Unsloth end-to-end GPU validation to compatible remote hardware; then verify TRL API, checkpoints, adapter reload and retry semantics.
- **Acceptance gate for Unsloth:** model/capability license check, actual GPU training, full checkpoint manifest and fresh adapter reload, paired metrics / stop callback, explicit training resume proof, and bounded policy-driven retry. None of the latter GPU-dependent gates are currently completed.

## Security Lifecycle Sprint S0 - checkpoint v0.8.10 (2026-11-02)

**Status:** proposed only. No production vulnerability scanner is deployed and no new source tests run at this documentation checkpoint. Source remains v0.8.7.

### P0: before any shared dashboard or remote training
1. Dependency and SBOM intake: pin versions, run pip-audit/OSV in CI and local dev, record triage and blocks for reviewed critical/high vulnerabilities.
2. Trusted model intake: pin Hugging Face commit SHA; deny implicit remote code and untrusted pickles; prefer safetensors; verify license, origin and provenance.
3. Dataset security: validate license and origin; detect PII/secrets; label anomaly/poisoning suspicion flags; publish redacted audit record; fail closed on unresolved serious findings.
4. Logging and secrets: keep local-only MLflow until access control is designed; redact configuration/log artifacts and never upload raw text or tokens to advisor.

### P1: before model promotion
5. Separate training-health anomalies from security alerts in a structured event schema; quarantine suspicious datasets/checkpoints, rather than automatic retry.
6. Evaluate robust behavior with held-out adversarial, privacy memorization and trigger/backdoor tests; register outcomes as model security test receipts.
7. Test agent prompt-injection resistance and numeric-only policy enforcement, plus API path/URL controls.

### P2: before multiuser infrastructure
8. Authenticate FastAPI and MLflow, secure TLS/reverse proxy, role permissions, audit retention and incident-response flow; Kubernetes RBAC/network policies before distributed jobs.

### Evidence/acceptance criteria
- Produce one artifact each for dependency scan, model provenance, dataset scan, security-event journal and security evaluation; failures must be reproducible in tests.
- No security gate can be bypassed by a Meta generated recommendation. No model promotion from a toy-data-only result.
- Maintain CPU-first Windows PowerShell setup and historical artifacts.

### Source cross-check
Reviewed v0.8.7 modules: `lora_contract.py`, `metrics.py`, `advisor.py`, `hf_selfheal.py`, `hf_mlflow.py`, `hf_audit_cli.py`, `hf_trainer.py`. No additional runtime security checks were executed.


## 2026-11-02 v0.8.11 - Interview and vulnerability explanation checkpoint

This documentation-only update clarifies what to say in an interview: AutoTune AI currently detects training-health anomalies with its monitor, uses MLflow for experiment records and SHA256 for artifact mismatch detection, and enforces bounded hyperparameter patches. These are not equivalent to comprehensive cybersecurity vulnerability scanning.

### Planned security acceptance sequence
1. Add pip-audit and dependency inventory with known advisory findings, remediation status and fail-closed policy for critical issues.
2. Pin approved Hugging Face model revisions and check permitted artifact formats and provenance.
3. Add dataset PII and poisoning/label-anomaly triage with quarantine paths and operator decisions.
4. Add security events to audit trail, separate from training anomalies; do not automatically retrain on suspected hostile input.
5. Complete read-only FastAPI/Next.js dashboard with least-privilege and local-only defaults.

**Acceptance evidence required:** unit tests, malicious fixtures, a real scanner execution report, audit provenance, and user Windows verification. Merely describing a scanner or installing it is not the same as integrating it.

No source code changes or new test results in this checkpoint.


---

# v0.12.0 - Phase 7 Knowledge Engine delivery checkpoint (2026-11-04)

## Completed source milestones

| Workstream | Current state | Exit caveat |
|---|---|---|
| 7A ingestion | Implemented; TXT/MD/CSV/JSON/JSONL/PDF/DOCX, hashes, SQLite, chunking | OCR for image PDFs is not present |
| 7B retrieval | FTS5 lexical, optional local sentence-transformer embeddings, hybrid RRF | Model download/index operation not live-tested |
| 7C RAG | Local Ollama answers with exact quote + chunk ID verification | Lexical overlap is not entailment; Ollama live call unverified |
| 7D SFT data | Grounded Q/A generator, regex hygiene, source-document-disjoint split, messages JSONL, response-only loss | Basic PII detection; generated content still needs review |
| 7E orchestration | Three-model sequential registry, pinned revisions, two LR trials, validation-based token-F1 filter plus a single final test run for the selected candidate, no auto-promotion | No real candidate training/benchmark executed in this build |
| 7F dashboard | Next.js page and API job polling in source | npm install timed out; typecheck/build/browser acceptance pending |

## Build validation

`PYTHONPATH=. pytest -q`: **338 passed in 9.31 seconds** in the build environment. That result covers automated Python contracts; it does not prove the user Windows runtime, external Ollama/Hugging Face downloads, model training quality, GGUF conversion or browser UI.

## Architecture / operating principles

1. Keep RAG and fine-tuning distinct: RAG retrieves changing facts; SFT teaches desired task/response behaviour.
2. Keep source documents disjoint between train, validation and held-out test.
3. Validate exact evidence quotes and reject weakly supported answer pairs, but retain human review for generated datasets.
4. Use response-only loss masking and model-specific chat templates.
5. Run candidate trials sequentially for the CPU-first baseline; pin immutable revisions.
6. Evaluate candidates on the same held-out domain examples; token F1 is a preliminary lexical screen, not semantic correctness.
7. If no candidate passes, record `NO_QUALIFIED_MODEL`; do not lower gates or force a winner.
8. No production auto-promotion. Require human/domain, privacy/security and safety approval.
9. Run API on `127.0.0.1` only; prototype has no authentication.

## Next acceptance actions on the user's Windows system

1. Extract v0.12.0 to a fresh folder; preserve existing project outputs, `.venv*`, MLflow DB and adapter artifacts.
2. Execute `run_phase7.ps1`; verify `/api/knowledge/health`, upload/list a small text file, and lexical search.
3. Start Ollama locally; verify Q/A generation and dataset manifest with at least four distinct source documents.
4. Optionally install semantic dependencies and explicitly build/index embeddings.
5. Run orchestrator preflight; inspect all data-quality gate issues and candidate config plans.
6. Execute only one candidate first on CPU with enough RAM/disk; verify pinned revision, training receipts and held-out comparison.
7. Run npm install/typecheck/build and browser acceptance on Windows; capture failures or results.
8. Human-review scorecards and source data before model export; production deployment remains blocked until a broader independent evaluation, security hardening, authentication and monitoring exist.
