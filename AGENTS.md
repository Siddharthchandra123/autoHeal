# CURRENT CHECKPOINT - v0.9.0 (2026-11-02) - Phase 6B, read-only dashboard

**IMPORTANT: Start here to resume work.** The latest executable project release is **v0.9.0**. Earlier AGENTS history continues below. The project is CPU-first on Windows VS Code and uses local MLflow. Preserve all previously successful user experiment folders under `outputs/`; none are in the source archive.

## User-verified checkpoints carried forward

- User ran 207 tests passing / 1 skipped for v0.8.6, and a controlled real-HF Tiny GPT-2 two-attempt CPU recovery run: trial-01 overfit was **fault-injected**, trial-02 finished; both validation losses approx 10.36518; selected trial 2, promotion_approved false. No significant model improvement is established.
- User ran a successful two-stage CPU pause at step 2 / resume to step 4; full optimizer state proof needs deeper independent validation of `resume_proof.json` (not supplied in conversation).
- User audited 6 LoRA adapter files and 10 checkpoint files and reloaded adapter into a fresh Tiny GPT-2 model, generating 8 tokens.
- User ran a local MLflow server on 127.0.0.1:5000 and recorded three real runs (one parent controller, two training trials). Live run links are in the user's earlier output. Do not claim the new dashboard has been verified live.

## New Phase 6B implementation

- Added `autotune/viewer_api.py`: FastAPI **read-only** facade to the existing MLflow Tracking SDK, exposing health, experiment list, runs, individual run, bounded metric history. MLflow client initialized lazily. This API never imports or launches training.
- Added `web/`: Next.js App Router + TypeScript server-rendered dashboard with overview, experiment listing, run comparisons, metric plots and allowlisted metadata. No training controls, authentication, or artifact download endpoints.
- Added Windows runbook `docs/PHASE6B_DASHBOARD_RUNBOOK.md`, diagram sources `docs/diagrams/viewer_phase6b_v090.*`, optional `requirements-viewer.txt`. MLflow optional requirements now specify `mlflow>=3.17.2,<4` as a minimum patched baseline; no automated CVE audit or environment verification is claimed.
- Local API security: only explicit `http://127.0.0.1:<port>` MLflow tracking URL accepted (no URI userinfo, query, fragments or paths), fixed target at startup, numeric experiment IDs / 32-char hex run IDs, strict metric-name allowlist, response metadata allowlist, bounded metric results, localhost CORS, TrustedHostMiddleware, sanitized errors. Local-only posture, **NOT** production-grade authentication.
- **Build verification:** `238 passed` Python tests on this source snapshot (216 inherited, 22 new viewer tests), using fake MLflow client for API contract tests. Tests do not prove actual MLflow network integration or Next.js runtime. The `npm install` attempt in the build environment timed out; frontend build/typecheck remains unverified on Windows.

## Mandatory next actions

1. Extract full v0.9.0 ZIP as a clean project or overlay patch on the existing Windows project after backup, **preserving `outputs/`, `.venv`, and `mlflow.db`**.
2. Start local MLflow port 5000, install `requirements-viewer.txt`, launch FastAPI on port 8000, run `GET /api/experiments` and verify the three previously recorded runs appear.
3. On the Windows VS Code machine, in `web/`, run `npm install`, `npm run typecheck`, `npm run build`, `npm run dev`; verify `http://127.0.0.1:3000` and experiment/run detail pages. Capture output before marking live complete.
4. Implement **S0 security scanning gates** next: reproducible dependency lockfile and pip-audit/npm audit outputs, trusted immutable HF model revision checks and safe serialization, dataset security/PII triage, proper block/quarantine and audit events.
5. Then Phase 6C live telemetry and dashboard interactions guarded by authentication; Phase 7 Kubeflow and Phase 8 Katib. Do not claim these work yet.

## Safety and truthfulness

- No UI or API secrets or Python arbitrary code execution, no wildcard CORS and no route for user-supplied fetch URLs. Local MLflow, API and Next servers **must** bind to 127.0.0.1 because there is NO user authentication.
- Viewer is not a vulnerability scanner; training anomalies != security threats. P0 scanning is on the roadmap, NOT implemented.
- Previously held PDFs are cumulative history; this release adds a new handbook appendix. Each substantive project change should update handbook PDF, this AGENTS.md, and master plan in the current response.

---

# CURRENT CHECKPOINT — v0.8.9 | Phase 6A MLflow LIVE VERIFIED (2026-11-02)

**Status:** User's Windows CPU run with MLflow localhost succeeded for both real Hugging Face PEFT LoRA attempts. Three MLflow run links were emitted (controller + trial-01 + trial-02). `outputs/hf-selfheal-mlflow-02` completed with `ok:true`, `attempt_count:2`, `best_trial:2`, `promotion_approved:false`.

- Trial 1 val loss 10.365182876586914 (`fault_injected:true`, stop diagnosis `overfit`), trial 2 val loss 10.365181922912598 (`fault_injected:false`, stop diagnosis null). The difference ~9.54e-7 is not meaningful quality evidence.
- Run IDs: trial-01 `dca1c1ac15d8498991fc40b2ddc98d6f`, trial-02 `7f914d30a4214e4f9b15841453e341f7`, controller `f6c7c00ccfed4a5b80449f22cff972df`. UI inspection of parent-child linking, metrics and artifact views remains open; don't claim verified tags or visual hierarchy from console links alone.
- The earlier nonempty workspace error was correctly blocked. A NEW workspace was used, preserving historical artifacts.
- **Verification distinction:** The user previously ran `207 passed, 1 skipped` on Windows. Executable source remains v0.8.7 with 216 build-environment offline tests. This v0.8.9 release is a **documentation-only** checkpoint, not a new code/test release. v0.8.8 TypeScript/Next.js frontend plan remains adopted.
- **Next:** inspect live MLflow UI run relationships, then implement read-only FastAPI endpoints + Next.js experiment dashboard (Phase 6B); maintain CPU-first / no GPU or Docker dependency; no promotion without independent evaluation.
- **Still pending:** detailed optimizer/scheduler `resume_proof.json` fields; naturally detected real training problems (not fault injection), hosted Meta API, Unsloth on GPU, Kubeflow/Katib orchestration and deployment.
- **Project rules:** always update cumulative handbook PDF, roadmap and AGENTS.md on AutoTune project iterations. Never claim background document updates or overwrite `outputs/*`.

---

# CURRENT CHECKPOINT — v0.8.8 | GPU positioning + TypeScript UI (2026-11-02)

**Documentation-only architectural decision; no source code updated; no new code tests.** Previous executable source remains **v0.8.7**, whose optional MLflow adapter passed **216/216 build-environment offline tests**. User last verified **v0.8.6** on Windows with **207 passed, 1 skipped** and a real 2-attempt CPU Hugging Face LoRA workflow using explicit injected overfit on attempt 1. Phase 6A live MLflow server remains UNVERIFIED.

## Decisions
- **Main UI: Next.js / React / TypeScript** with Tailwind, shadcn/ui, chart library (Recharts or ECharts). Streamlit optional for developer exploration only. Keep entire training engine, Meta agent interface, policy validation, checkpoint recovery and MLflow integration in Python FastAPI.
- **CPU-first is mandatory for the user's current Windows 11 setup**, without a dedicated GPU. The frontend never needs a GPU. Do not require Docker or Kubernetes to operate the base UI or MLflow.
- **Model/backend selection:** Hugging Face Transformers + PEFT by default; optional Unsloth GPU fast-path after capability gating; Katib primary HPO for Kubeflow, Ray Tune optional alternative only.
- **GPU positioning:** entry NVIDIA RTX 5060 Ti **16GB**, practical RTX 3090/4090 **24GB**, premium RTX 5090 **32GB**; data-center AMD MI300X **192GB**, MI325X **256GB**, MI350X **288GB**. These are examples/capacity tiers, not performance benchmarks or claims of fit for every model. Check actual memory needs, model license, CUDA/ROCm and model version compatibility before use. AMD Instinct is Linux/ROCm; AMD's 2026 Unsloth documentation supports Radeon Windows/WSL for certain architectures, but no Unsloth GPU test was run.
- **Next sprint:** do Phase 6A MLflow localhost acceptance test first; then bootstrap `/frontend` Next.js app and FastAPI read-only experiment/job APIs and SSE progress. Only after that integrate GPU worker profiles and remote cluster jobs.
- **Security:** UI never receives provider secrets. Job specs must be validated and access-controlled; add GPU cost/attempt budgets, no browser-side training execution. Log fault injection explicitly.
- **Documentation deliverables:** handbook `AutoTune_AI_Handbook_v0.8.8.pdf` extends cumulative v0.8.7 with hardware table, architecture figure and frontend implementation checklist; roadmap and AGENTS.md updated.

Sources reviewed 2026-11-02: https://nextjs.org/docs/app/api-reference/config/typescript ; https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements ; https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html ; https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html ; https://www.nvidia.com/en-in/geforce/10-series/10-series-specs/

---

# CURRENT CHECKPOINT - v0.8.7 | Phase 6A opt-in MLflow experiment tracking (2026-11-02)

**Executable source v0.8.7; 216/216 build-environment offline tests passed.** The user's v0.8.6 Phase 5E live Windows results are now VERIFIED: **207 passed / 1 skipped**, real HF PEFT LoRA trial 1 with **explicit fault-injected** overfit and trial 2 with a validated changed config, both real CPU training attempts complete. Trial 1 best validation loss `10.365184783935547`; trial 2 `10.365181922912598` (tiny `~2.86e-6` difference, no meaningful improvement claim). Final `ok:true`, `attempt_count:2`, `best_trial:2`, `promotion_approved:false`.

Phase 6A adds **optional**, explicitly requested `--mlflow-uri` (and experiment name) to `autotune.hf_selfheal` via new module `autotune/hf_mlflow.py`. Logs parent controller run, child runs per training attempt, config hashes, true metrics, fault-injection tags, validated checkpoint manifests and controller receipts. **Does not upload trained weights or optimizer files.** Local journaling stays authoritative. MLflow server was **NOT live tested**; 8 new tests exercise a fake client and fake trainer. MLflow failure blocks rather than silently claiming tracking succeeded. With flag absent, no MLflow import or remote call is made.

Windows next: see `docs/PHASE6A_MLFLOW_TRACKING.md`. Install `requirements-mlflow.txt`, run a localhost MLflow server on `127.0.0.1:5000`, then run a **fresh** CPU self-healing acceptance experiment with `--mlflow-uri http://127.0.0.1:5000`. Verify a controller parent and two child MLflow runs in the UI. Do not overwrite `outputs/hf-selfheal-injection-01`, `gpt2-cpu-01`, or `gpt2-resume-probe-01`. The original `resume_proof.json` field contents remain uninspected by the assistant. Hosted Meta AI and Unsloth GPU execution remain unverified.

Phase 6B next: verify real MLflow server, add live streaming and controlled reconciliation; later Kubeflow/Katib. Keep the project CPU-first.

---

# CURRENT CHECKPOINT - v0.8.6 | Phase 5E real Hugging Face self-healing controller (2026-11-02)

**Executable source v0.8.6.** Phase 5E implements an integrated, CPU-first serial recovery controller. Previous v0.8.5 was docs-only. Read `docs/PHASE5E_REAL_HF_SELFHEAL.md` before working on the new controller.

- **New components:** `autotune/hf_selfheal.py` with offline preflight and explicit `--execute` real-HF mode; `config/lora_selfheal_cpu.yaml` (two-attempt budget); `tests/test_hf_selfheal_v086.py`; explicit fault-injection hook in existing `autotune/lora_gpu.py`.
- **Local test evidence:** `208 passed` in the build environment (193 prior + 15 new). The 15 new tests use a fake Trainer boundary with *real* SHA-256 manifests; this is **not** evidence that a live two-attempt HF run completed.
- **Live Windows evidence remains:** `192 passed, 1 skipped` for previous source, original real HF Tiny-GPT2 LoRA training, fresh adapter reload, explicit step-2-to-step-4 resume. The `resume_proof.json` *contents have not been shared*; optimizer/scheduler restoration details remain subject to this check.
- **Integrated flow:** actual `run_lora` receives metrics, monitor may issue overfit/underfit stop, trainer saves adapters/checkpoints, controller verifies manifest, uses Meta/mock advisor with validated numeric-only patch, runs fresh `trial-02`, and compares validation loss without promotion. Critical divergence does NOT trigger autoremediation.
- **Test-only path:** `--fault-inject-overfit` at the first real evaluation provides deterministic stop for integration testing; logged explicitly as synthetic. Never claim natural overfitting from that test. `--execute` required, CPU only, no GPU.
- **Windows next commands:** `python -m pytest -q` then `python -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml`; for actual two-attempt exercise `python -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml --execute --fault-inject-overfit --workspace outputs/hf-selfheal-injection-01` (use a FRESH output directory). Preserve `gpt2-cpu-01` and `gpt2-resume-probe-01`.
- **Safety:** no in-place checkpoint resume with changed LoRA settings; per-attempt isolated config/output; max 3 attempts (sample uses 2); strict diagnosis/parameter limits; config and dataset provenance; best checkpoint and adapter reverified; errors block rather than silently retry. Mock advisor by default, hosted Meta API unverified.
- **Not done:** actual Windows live 2-attempt HF execution, independent holdout evaluation, MLflow tracking server, Kubeflow/Katib, PR workflow, reliable remote job recovery, production deployment, actual GPU Unsloth. Next priority validate live Phase 5E and `resume_proof.json` contents.
- **Docs protocol:** keep cumulative handbook PDF, roadmap, and AGENTS.md updated during AutoTune AI project turns; no background updates.

---

# CURRENT CHECKPOINT - v0.8.5 | Project progress status (2026-11-02)

**Documentation-only update.** The latest executable source remains v0.8.3; v0.8.4 and v0.8.5 are evidence and documentation snapshots. The inherited v0.8.4 log follows intact below.

## Verified status and evidence

- **Windows / VS Code / CPU only:** user reported `192 passed, 1 skipped in 22.26s` from latest project; do not imply 193 tests passed on that machine. v0.8.3 build suite had 193/193 passing locally.
- **Phase 1-2:** config validation; monitoring overfitting/underfitting, nonfinite loss, plateau, gradient metrics; synthetic recovery: best simulated val loss 0.58; not a model-quality benchmark.
- **Phase 3:** durable CPU-mock recovery with journal, hash and bounded retries; user verified 48 tests.
- **Phase 4:** mock Meta advisor/policy and deterministic recommendation replay, 77 tests user verified; hosted Meta API NOT live-verified.
- **Phase 5A:** user verified real PyTorch GRU CPU (95 tests), best val loss 5.5674948692; actual `.pt` weight/optimizer checkpoint.
- **Phase 5B-5C:** user verified Hugging Face tiny-GPT2 CPU LoRA: 4 train/2 val rows; 128 trainable parameters; best validation loss 10.365182876586914; final adapter, checkpoint-2. Read-only audit verified 6 adapter and 10 checkpoint files; fresh adapter reload generated 8 tokens.
- **Phase 5D:** user paused run at step 2; second process explicitly continued from `checkpoint-2` through step 4/4 and reported `resume_proof` path. **Exact optimizer and scheduler proof fields have not been shared.** Need to inspect before certifying those values. This is not bitwise optimizer equivalence proof.
- **Hugging Face universal task router:** CPU text classification adapter implementation offline tested, no user live BERT fine-tune verified.
- **Unsloth:** offline GPU config accepted only based on config; actual GPU runtime unverified; CPU Unsloth training blocked; HF PEFT remains default.

## What remains incomplete

1. **Immediate:** read `outputs/gpt2-resume-probe-01/resume_proof.json` and confirm begin/end steps and nonempty optimizer/scheduler state.
2. **Phase 5E:** real Hugging Face monitoring -> safe diagnosis -> validated Meta mock/fallback advice -> versioned NEW trial -> checkpoint comparison in an automatically controlled CPU two-attempt experiment. Not yet end-to-end tested.
3. **Phase 6:** persistent MLflow server + run/metric/artifact/lineage verification. Optional MLflow hooks do not count as completed integration.
4. **Phases 7-8:** CPU Kubernetes/Kubeflow Pipelines and Katib deployment and real job reconciliation. Ray Tune optional, never simultaneous with Katib for the same trial.
5. **Phases 9-12:** PR-based config automation, dashboard, independent holdout evaluation, model registry, deployment, fault injection and operational validation.
6. Optional remote GPU: Meta Llama/Unsloth/QLoRA and hardware-specific benchmarking. **No GPU is required for the priority development path.**

## Immediate command (Windows PowerShell)

```powershell
Get-Content outputs\gpt2-resume-probe-01\resume_proof.json -Raw
```

## Project continuation rules

- Keep `.venv`, `outputs/gpt2-cpu-01`, and `outputs/gpt2-resume-probe-01` intact; do not retrain completed runs in place.
- Never equate a synthetic loss improvement, toy 4/2 evaluation, config preflight, or file integrity check with useful model quality or complete production readiness.
- Apply only allowlisted configuration patches, verify provenance, cap retries, and preserve independent holdout data.
- Update AGENTS.md, full cumulative PDF handbook and master execution plan when completing future project prompts. No automatic background changes are implied.

---

## LATEST CHECKPOINT - v0.8.4 | Windows real checkpoint continuation (2026-11-02)

**Authoritative most-recent status.** No source-code modification in v0.8.4: this is an evidence + documentation checkpoint on v0.8.3 source. Do not mark a subsequent phase complete from this user log alone.

- **Windows environment:** VS Code / PowerShell; `.venv`; CPU-only training; no GPU. User ran `python -m pytest -q`: **192 passed, 1 skipped in 22.26 seconds**. Do not silently report this as 193 passed or assume reason for skip.
- **Real Hugging Face PEFT resume exercise:** `config/lora_cpu_resume_probe.yaml`; `sshleifer/tiny-gpt2`; output `outputs/gpt2-resume-probe-01`. First process ran with `--pause-after-step 2` and returned `{ok: true, status: paused, global_step: 2, target_step: 4}`, saving `checkpoints/checkpoint-2`. A subsequent process used explicit `--resume-checkpoint outputs/gpt2-resume-probe-01/checkpoints/checkpoint-2` and returned `ok: true`, `resume_proof` path, adapter directory, and completed progress **4/4**. Best validation loss was **10.365182876586914**; only 4 training and 2 validation examples; not a meaningful generalization benchmark.
- **Resume evidence caution:** `outputs/gpt2-resume-probe-01/resume_proof.json` exists according to CLI output, but its contents have NOT been supplied. Inspect `checkpoint_step=2`, `trainer_begin_step=2`, `trainer_end_step=4`, `optimizer_state_nonempty_at_resume=true`, and `scheduler_last_epoch_at_resume>=2` before asserting those specific recorded values. Source enforces nonempty optimizer state at train-begin and schedule restoration. This is not bitwise-equivalence proof.
- **Unsloth preflight:** `config/lora_unsloth_gpu.example.yaml` returned `ok=true`, `executable_on_configured_device=true`, `requires_gpu=true`, `live_gpu_training_verified=false`. This is CONFIG-BASED feasibility, not a detected/available GPU. CPU config `config/lora_public_cpu.yaml` returned `executable_on_configured_device=false`, preserving Hugging Face PEFT on CPU. No actual Unsloth GPU training tested.
- **Next exact user command:** `Get-Content outputs\gpt2-resume-probe-01\resume_proof.json -Raw`. Do not retrain or overwrite completed output folders.
- **Next engineering objective (Phase 5E):** Connect the *real* HF LoRA callback and checkpoints to existing MonitorStream, Meta advisor (mock/deterministic fallback), strict patch policy and durable recovery controller. Run a bounded two-attempt CPU training test. Distinguish controlled/synthetic anomaly from genuine metric deterioration and separately evaluate final model quality. Then MLflow, CPU Kubernetes/Kubeflow/Katib, dashboard and optional GPU Unsloth.
- **Files produced in this documentation checkpoint:** `AutoTune_AI_Handbook_v0.8.4.pdf` (4-page current evidence insert + 45-page inherited handbook); `AutoTune_AI_AGENTS_v0.8.4.md`; `AutoTune_AI_Master_Execution_Plan_v0.8.4.md`; diagram and checkpoint notes. **No new code tests were executed in this documentation update.**

---

## LATEST CHECKPOINT - v0.8.3 Optional Unsloth GPU backend (2026-11-02)

- **User request:** Can AutoTune AI integrate Unsloth? YES as an OPTIONAL backend. Current development hardware is Windows + CPU only. Preserve verified Hugging Face PEFT LoRA CPU workflow; Unsloth is NOT a drop-in acceleration for CPU fine-tuning. Official Unsloth CPU chat/data workflows do not establish GPU-style CPU training support.
- **New files:** `autotune/unsloth_backend.py` (offline plan, safe HF model-id validation, CUDA/ROCm GPU guard, experimental Unsloth FastLanguageModel + TRL SFTTrainer path, shared MonitorStream and artifact manifest hooks); `autotune/unsloth_cli.py`; `config/lora_unsloth_gpu.example.yaml`; `tests/test_unsloth_backend_v083.py`; `docs/UNSLOTH_OPTIONAL_BACKEND.md`; diagram sources. Default `autotune.hf_models_cli` unchanged.
- **Build verification:** 193/193 local tests passed (185 inherited + 8 new) in 6.68s on CPU. GPU config preflight passed; CPU-only Unsloth execute failed closed BEFORE download/output creation. **No real Unsloth GPU training, checkpoint resume, performance claim, API call, or agent-driven retry has been tested.** GPU runtime is experimental and might need dependency/version compatibility changes.
- **Safe current commands:** `python -m autotune.unsloth_cli --config config/lora_unsloth_gpu.example.yaml` (offline preflight), and `python -m autotune.unsloth_cli --config config/lora_public_cpu.yaml` (explicit unsupported device explanation). Do NOT run Unsloth `--execute` on CPU. Keep `.venv` and `outputs/gpt2-cpu-01` untouched.
- **Priority remains:** Run v0.8.2 paused-run optimizer resume verification on Windows, then integrate real HF training with the durable controller. Unsloth GPU execution can be validated later on supported remote compute (e.g., appropriately configured AMD MI300X). Then evaluate MLflow/Kubeflow/Katib.
- **Living-doc contract:** Updated master plan and 42-page inherited handbook plus v0.8.3 appendix/diagram. No background automation implied.

---

## LATEST CHECKPOINT - v0.8.2 controlled CPU checkpoint resume (2026-11-01)

**User verification now complete:** Windows CPU `sshleifer/tiny-gpt2` LoRA training, 6/6 final adapter files verified, 10/10 full checkpoint files verified, and **fresh adapter inference generated eight tokens**. The user reported `ok: true` for read-only audit and `--reload`; best recorded val loss 10.365182876586914. **Actual optimizer resume is still NOT user-verified** (both user audits showed `actual_resume_executed: false`).

**New implementation:** a distinct CPU-only, two-epoch/four-step test (`config/lora_cpu_resume_probe.yaml`) adds `--pause-after-step 2` to the causal-LM CLI, a guarded paused-run resume contract and `resume_proof.json` with pre-continue optimizer/scheduler observation. The **completed original `outputs/gpt2-cpu-01` is never touched**. Changes are source-level and offline-tested only; live HF dependencies and downloads were not available in this build environment.

**Build evidence:** `185 passed` offline (166 inherited + 19 contract/CLI tests). **Next user action:** execute the two Windows commands in `docs/PHASE5D_OPTIMIZER_RESUME.md`; inspect paused state and real resume proof. No autonomous Meta-directed real-HF retry or production claim.

---

## LATEST CHECKPOINT: v0.8.1 - FIRST REAL HF LoRA SUCCESS + AUDIT GATE (2026-11-01)

- **User-reported independent Windows verification:** Completed real CPU fine-tune of public `sshleifer/tiny-gpt2` through `python -m autotune.hf_models_cli --config config/lora_public_cpu.yaml --execute --output outputs/gpt2-cpu-01`. Console result `ok:true`, 4 train / 2 validation rows, one epoch, 2 training steps; train loss about 10.4, validation loss **10.365182876586914**, trainable PEFT parameters **128**, no early-stop (`stop_diagnosis: null`). Saved `outputs/gpt2-cpu-01/final_adapter` and `outputs/gpt2-cpu-01/checkpoints/checkpoint-2`. Adapter weight SHA-256: `d7aabd74886d97b0d83136441891a2442dc8f99266177e7ff05a52c55570e0d1`.
- **New v0.8.1 engineering:** Added read-only `autotune/hf_audit_cli.py` for finished LoRA runs: verifies immutable plan against current config/data, successful completion receipt, final adapter manifest, full checkpoint manifest/optimizer/scheduler/RNG hashes, expected paths and receipt-to-state agreement. Optional `--reload` runs fresh PEFT adapter inference *only after* verification. Added `tests/test_hf_audit_v081.py`, audit runbook, architecture flow diagram, and updated documentation.
- **Local test evidence:** `python -m pytest -q` -> **166 passed** (153 inherited + 13 new) in the build container. Offline tests did not invoke live HF weights or backend training.
- **User action on current output:** `.\.venv\Scripts\python.exe -m autotune.hf_audit_cli --config config/lora_public_cpu.yaml --run outputs/gpt2-cpu-01`; then same command with `--reload` for optional real CPU adapter inference.
- **Remaining acceptance gates:** user has NOT yet returned audit/reload results; optimizer/scheduler state resume was NOT executed, real HF agent-driven retry was NOT executed, Meta hosted API and MLflow/Kubeflow/Katib remain unverified. **Never resume the already-completed output in place.** Prepare an isolated interrupted checkpoint test. Training quality is not established from 4/2 toy samples.
- **Status conventions:** Distinguish `implemented`, `offline tested`, `user-verified real training`, and `not yet tested on user's machine`. Never represent a file-integrity audit as actual resume execution or a loss value as a model-quality benchmark.
- **Documentation protocol:** keep the full handbook PDF, master roadmap and this AGENTS.md updated per project prompt. Do not promise unscheduled background work.

---

## LATEST CHECKPOINT: v0.8.0 - HUGGING FACE TASK ADAPTERS (2026-11-01)

- **User request:** Extend AutoTune AI to train multiple open-source Hugging Face models without a dedicated GPU; retain existing project, handbook, and authoritative AGENTS history.
- **Delivered:** Unified `autotune/hf_models_cli.py` with offline preflight and explicit `--execute`/`--verify`. Two adapters: (1) existing `causal-lm` PEFT LoRA, e.g., `sshleifer/tiny-gpt2` on CPU; (2) NEW `text-classification` full CPU fine-tune, e.g., `google/bert_uncased_L-2_H-128_A-2`, in `autotune/hf_classification.py`. New YAML, JSONL train/val example, runbook, Hugging Face task routing architecture diagram and handbook revision.
- **Tests in build environment:** `python -m pytest -q` -> **153 passed** (135 inherited + 18 new). Offline classification plan/label/split/parameter validation; model ID safeguards; SHA-256 model-manifest verification; dependency-free preflight. `python -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml` succeeds without model download.
- **Verification boundary:** Transformers, PEFT, datasets absent from build runtime; **no real HF BERT classification or GPT-2 PEFT training executed or benchmarked**. Genuine earlier small GRU CPU training was user-verified. Hosted Meta model, MLflow tracking server, Kubeflow/Katib, live HF auto-retry remain unverified. Public model access can require network and license checks.
- **Next exact user steps (Windows PowerShell):** install `requirements-lora.txt` in `.venv-hf`, run `python -m pytest -q`, preflight, then `python -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml --execute --output outputs/bert-tiny-cpu-01`. Verify using `--verify outputs/bert-tiny-cpu-01/best_model`; for LoRA run corresponding config `config/lora_public_cpu.yaml`. See `docs/HF_OPEN_SOURCE_MODELS.md`.
- **Next engineering milestone:** Validate real CPU BERT and tiny GPT-2 PEFT train/reload/resume; then integrate task adapters with Phase 3 durable recovery loop, Meta diagnosis and Phase 6 MLflow. Vision/audio plugin interfaces planned, not shipped.
- **Operational contract:** At every active AutoTune AI project iteration, update handbook PDF, `docs/MASTER_EXECUTION_PLAN.md`, and `AGENTS.md` including tests, limits and next step. No background updates.

---

## LATEST CHECKPOINT: v0.7.2 - CPU-FIRST PIVOT (2026-11-01)

- **User constraint:** no dedicated GPU; Windows + VS Code + PowerShell is the primary execution environment. **Do not block on GPU.**
- **Phase 5B default:** `config/lora_public_cpu.yaml` uses public `sshleifer/tiny-gpt2` on CPU with PEFT LoRA, no quantization. `--execute --allow-cpu` must be explicit; actual online HF run and adapter reload have NOT yet been reported or verified. Offline preflight downloads nothing.
- **GPU-only items optional/deferred:** Meta Llama large-model fine-tune, 4-bit QLoRA, GPU telemetry and GPU benchmarks. Meta Muse hosted diagnostic agent can be accessed from CPU through API when separately configured. Hosted call remains unverified.
- **Execution goal:** real CPU LoRA -> validate/reload adapter -> full Trainer checkpoint-resume -> integrate metrics and policy with durable recovery -> MLflow -> CPU Kubernetes/Kubeflow/Katib -> dashboard and final evaluation. No time estimate or guaranteed speed on CPU.
- **Code changes v0.7.2:** HF Trainer `use_cpu=True` for `hf.device: cpu`, corrected LoRA CLI CPU instructions, one CPU guidance regression test; no online model training in this build. Source code from v0.7 otherwise unchanged.
- **Historical verified evidence:** Phase 5A 95 Windows tests and real tiny GRU CPU run. Phase 5B v0.7 134 local offline tests. **Never call an unexecuted HF pilot 'verified'.**
- **Command/checkpoint:** read `docs/PHASE5B_CPU_FIRST_RUNBOOK.md`; use `python -m autotune.lora_cli --config config/lora_public_cpu.yaml` then optional `--execute --allow-cpu` in fresh HF venv.
- **Docs protocol:** on future AutoTune AI prompts revise handbook, `AGENTS.md` and execution plan in that active turn, version them, and preserve the tested/planned distinction.

---

<!-- AUTO TUNE PROJECT LATEST CHECKPOINT - READ FIRST -->
**LATEST CHECKPOINT - 2026-11-01, v0.7 / Phase 5B source pilot:** USER independently verified Phase 5A on Windows: `95 passed in 34.36s`, best real PyTorch tiny-GRU validation loss `5.567494869232178`, experiment `runs_real\20261101T151327Z`, best checkpoint `trial-01/best_checkpoint.pt`, SHA256 `093223fd97f7fcc085e7a274036e24c82f46d0406d98c28433bd6328b85b3f3e`. PyTorch warning `No module named numpy` did not stop training; `requirements-real-cpu.txt` now includes NumPy. **New v0.7**: `autotune/lora_contract.py`, `lora_cli.py`, `lora_gpu.py`, two HF configs, 39 new offline tests (134/134 build-container total), and checkpoint integrity manifests. **Do NOT claim actual Meta Llama/PEFT run or GPU validation:** Transformers/PEFT/datasets not installed in test container, no gated model downloaded, no GPU used. HF LoRA path is a separate pilot, NOT wired to durable self-healing recovery / Katib / Kubernetes. Next verification: Windows preflight -> optional public LoRA CPU -> supported GPU licensed Meta Llama LoRA/reload and optimizer checkpoint resume -> Phase 6 MLflow. Every project iteration update this AGENTS.md, master plan and generated PDF handbook; no background processing.

## v0.7 Phase 5B implementation and trust boundary

- **New files:** `autotune/lora_contract.py` (offline plan, dataset SHA256 and no-overlap, step estimate, manifest write/verify), `autotune/lora_gpu.py` (optional HF Trainer + PEFT LoRA/QLoRA, evaluation-aligned monitor, checkpoint manifests, explicit resume/reload), `autotune/lora_cli.py` (default dry-run; `--execute` required), `config/lora_meta_gpu.yaml`, `config/lora_public_cpu.yaml`, `requirements-lora.txt`, `docs/PHASE5B_RUNBOOK.md`, `run_phase5b.ps1`, `tests/test_lora_phase5b.py`, and `docs/diagrams/lora_gpu_bridge.{mmd,dot,svg}`.
- **Evidence:** inherited 95/95 + 39 new offline contract/fault tests -> **134 passed**. Tests check malformed configs, missing/out-of-range settings, data leakage, missing validation schedule, checkpoint/optimizer/RNG hash integrity, tampering, provenance, exclusive local lock, missing GPU, EOS/pad label mask, NaN JSONL and dry-run no-download behavior. These tests use FAKE adapter bytes for manifest contract; **no full adapter checkpoint was generated by the Hugging Face Trainer**.
- **The Phase 5A real-weight recovery loop is NOT interchangeable** with the Phase 5B Hugging Face checkpoint. Phase 5B saves a Trainer checkpoint + adapter manifest; the final adapter alone does not restore optimizer or scheduler. Resume requires an explicit trusted in-run checkpoint, exact config/data hashes, and an operator verifying no competing process. Hashes are corruption checks, not signatures or malware protection.
- **GPU integration readiness:** default `meta-llama/Llama-3.2-1B-Instruct` requires Meta license/HF account access; use a supported GPU host, not assume Windows iGPU works. `sshleifer/tiny-gpt2` CPU alternative requires network download. QLoRA (4bit) requires compatible bitsandbytes/CUDA-or-ROCm; optional path unverified. `trust_remote_code=False` in model/tokenizer loading.
- **No live GPU auto-heal yet:** Meta Muse agent not used by HF trainer and no automatic tuning/retry; it only monitors real losses, stops and saves safely. Do not claim production readiness or model quality from tiny datasets.
- **To run locally:** `python -m pytest -q`; `python -m autotune.lora_cli --config config/lora_meta_gpu.yaml` (preflight no model download). For full steps see `docs/PHASE5B_RUNBOOK.md`.
- **Next exact phase:** Verify public smoke or Meta GPU, checkpoint reload and resumed optimizer reproducibility; then build Phase 6 MLflow tracking and connect real HF events with recovery controller. Keep old mock and real-CPU paths intact.

<!-- Phase 5A latest checkpoint - read first -->
**ARCHIVED CHECKPOINT (2026-11-01, v0.6):** User verified Phase 4 on Windows with 77/77. Phase 5A has now been implemented and verified in the build environment: 95/95 tests, real offline PyTorch tiny-byte-GRU train/validation and loadable `.pt` best model/optimizer tensors. The 4-epoch CPU smoke validation loss 5.6072 -> 5.5675 is not a quality benchmark. Meta Llama LoRA GPU fine-tune, hosted Meta API, remote MLflow and Kubeflow are NOT validated. Next Stage 5B: real GPU LoRA adapter, metric alignment, weights/optimizer resume and test. Each future iteration must update handbook PDF, roadmap and this file.

# AGENTS.md - AutoTune AI project memory

> Authoritative checkpoint for continuing AutoTune AI. Read this file before making changes; do not infer production capability from a synthetic demo.

## Identity, goal and operating agreement

- **Project:** AutoTune AI - bounded agentic self-healing MLOps for Meta-model fine-tuning.
- **Current baseline:** v0.7, 2026-11-01. **Phase:** 1-4 CPU mock complete, Phase 5A independently verified on Windows (95 tests); Phase 5B offline source/contracts verified (134 tests), live Meta Llama GPU validation outstanding.
- **Primary development host:** Windows 11 + VS Code + PowerShell, CPU-first; optional supported GPU and later Kubeflow Kubernetes cluster.
- **Document contract on EVERY future project iteration/prompt:** update this file, `docs/MASTER_EXECUTION_PLAN.md`, handbook source `docs/HANDBOOK_SOURCE.md`, and regenerate an up-to-date versioned PDF handbook with relevant diagrams, concepts, implementation notes and verification. Work occurs only during active requests, not asynchronously. If a PDF cannot be generated, disclose that clearly.
- **User goal:** Observe metrics during training and fine-tuning, detect over/underfitting or instability, stop and retain best valid checkpoint, ask Meta model for remedial changes, constrain changes via deterministic policy, version code/config, retry safely, and evaluate before promotion.

## Architecture decisions - do not reverse casually

1. Kubeflow Pipelines v2 owns orchestration, Katib is PRIMARY hyperparameter optimizer. Ray Tune is OPTIONAL alternative, never operate both on the same trial.
2. Meta Muse/Spark is an advisory diagnostic agent only, not a shell/script executor, direct YAML writer or trusted scheduler.
3. Policy engine validates all proposed numeric hyperparameter patches with hard allowlists, ranges and diagnosis-aware directions.
4. Prefer configuration changes to arbitrary Python source edits. Any future source change needs an isolated Git branch, tests, and review/approval.
5. Evaluation-based early stop; checkpoint/weights saved before restart; no automatic production deployment. Never tune on final holdout test data.
6. Finite retries, GPU-hour and cost budgets, robust failure/idempotency and human review gates are required before production.
7. Agent API receives only aggregate statistics and parameter values, never training samples, tokens, credentials, or PII.
8. Synthetic runs are demonstrators, not measured ML results, and `best_checkpoint.json` is metadata only.

## Repo topology

- `autotune/config.py` - v0.2 configuration contract and SHA256 fingerprint.
- `autotune/doctor.py` - v0.2 local prerequisite/checks CLI, JSON output.
- `autotune/metrics.py` - Phase 2 typed MetricEvent, evidence-backed Decision, MonitorStream; anomaly/warning/stop rules.
- `autotune/monitor_cli.py` - replay prior metrics.jsonl offline and report all decisions.
- `autotune/mock_trainer.py` - CPU-only synthetic simulation; Phase 2 audit integration.
- `autotune/hf_trainer.py` - optional, UNVERIFIED Meta/HF LoRA GPU adapter; NOT integrated with real CPU recovery.
- `autotune/real_training.py` - Phase 5A real PyTorch byte-level GRU model, dataset identity, offline training, loadable best weight + optimizer checkpoint.
- `autotune/real_recovery.py` / `autotune/real_cli.py` - Phase 5A separate SHA-verified durable real-weight controller and CLI; `run_phase5.ps1`, `config/real_cpu.yaml`, `docs/PHASE5_RUNBOOK.md`.
- `autotune/orchestrator.py` - legacy non-durable local bounded loop; still supported for Phase 1-2 compatibility.
- `autotune/recovery.py` + `autotune/recovery_cli.py` - **Phase 3** durable CPU-mock controller, atomic journal/receipt, checkpoint digest, explicit resume and operator-only abandon.
- `run_phase3.ps1`, `docs/PHASE3_RUNBOOK.md` - Windows Phase 3 verification and incident/runbook commands.
- `autotune/policy.py` - deterministic patch validation and fallback policy.
- `autotune/meta_agent.py` - legacy optional Meta client for non-durable controller, live unverified.
- `autotune/advisor.py` + `autotune/advisor_cli.py` - Phase 4 strict Meta advice contract, numeric-only outbound fields, mock/hosted providers and deterministic fallback; the durable recovery controller uses this module.
- `config/agent_mock.yaml` / `agent_meta.example.yaml` - explicit provider selection. `run_phase4.ps1` and `docs/PHASE4_RUNBOOK.md` - smoke and Windows runbook.
- `autotune/tracking.py` - JSON-compliant local JSONL with explicit nonfinite fault markers + optional unverified MLflow.
- `config/baseline.yaml` - immutable baseline config for the demo.
- `pipelines/compile.py` / `pipelines/katib-mock-example.yaml` - examples not deployed to Kubeflow.
- `docs/MASTER_EXECUTION_PLAN.md`, `docs/HANDBOOK_SOURCE.md`, `docs/diagrams/*.mmd`, `scripts/build_handbook.py` - living docs pipeline.
- `tests/` - deterministic tests.

## Verified checkpoints

### 2026-11-01: v0.5 - Phase 4 model advisor integrated
- User confirmed Phase 3 locally on Windows 11/Python 3.13: **48 passed in 1.85s**. `run_phase3.ps1` terminated successfully; experiment `runs\20261101T141716Z`, attempts `stopped`, `completed`, best synthetic validation loss 0.58, verified simulated checkpoint hash. This is user-provided output.
- Phase 4 added strict `autotune/advisor.py` and read-only `autotune/advisor_cli.py`. Fixed public numeric-only payload, rejection of duplicate/unknown JSON fields and contradictory diagnoses, independent policy validation, no agent shell/file edit powers.
- Meta API transport uses official OpenAI-compatible `https://api.meta.ai/v1`, model `muse-spark-1.3`, `MODEL_API_KEY` from environment; 20s timeout, no client retries. **No real Meta API call has been verified.** Offline mock does not use network or credits.
- Durable mock recovery now stores `trial-XX/advisor_decision.json` with source, status, bounded patch, input/output config digests, verified checkpoint digest and an additional record digest. Replays validated snapshots on resume; tampering blocks. Default Phase 3 baseline remains unchanged (advisor disabled).
- **Local test evidence: 77/77 passed** on Python 3.13 (48 inherited + 29 new), including fake transport, hostile model output, missing-key fallback, replay and tamper checks; two-trial mock recovery preserved best loss 0.58. At the time of that historical checkpoint, user Windows verification of Phase 4 was pending; subsequent user confirmation is recorded above.
- No actual trained weights, production-ready integrity guarantee against malicious disk access, live GPU benchmarks, hosted Meta inference, or Kubeflow API calls have been verified.
- **Next:** Phase 5 real GPU HF LoRA training; prepare actual data/weights and checkpoint validation, with a hardware-specific environment plan. Maintain PDF + AGENTS + roadmap update each project prompt.


### 2026-11-01: v0.4 - Phase 3 durable recovery
- User confirmed Phase 2 locally with Python 3.13, **36 passed in 0.53s**, doctor `ok: true`, experiment `runs\20261101T140459Z`, best mock validation loss 0.58 at trial 2.
- Added atomic `recovery_state.json` journal, stable attempt idempotency key, local exclusive lock, simulated marker SHA256 and atomic `result_receipt.json`.
- Resume with valid receipt reconciles once, without retraining; uncertain RUNNING without receipt blocks and can be explicitly abandoned only after external job status verification, consuming trial budget. Corrupt receipt/checkpoint and changed baseline block safely.
- New entrypoint: `python -m autotune.recovery_cli --config config/baseline.yaml --workspace runs` and `--resume runs/EXPERIMENT`. Run `run_phase3.ps1` on Windows.
- **Local test result 48/48 passing on Python 3.13**; 12 Phase 3 fault-injection tests added. Synthetic trial 1 overfits, trial 2 finishes with 0.58 mock loss. Not verified on user's Windows host yet.
- No real weight checkpoint or distributed exactly-once guarantee: Phase 3 validates MOCK JSON markers; use fresh trial for next run. HF Trainer/Meta API not wired to this journal yet.
- By default durable controller does **not** write optimized.yaml; `--writeback` is explicit. No production promotion or test evaluation.
- At the v0.4 checkpoint, Phase 4 was the next task. It is now complete for CPU mock in v0.5; Phase 5 is next.



### 2026-11-01: v0.3 - Phase 2 telemetry and diagnostics
- User independently confirmed Phase 1 run: **15 passed in 0.36s**; trial 1 overfit stopped at step 6; trial 2 finished, selected best simulated validation loss 0.58, local experiment `runs\20261101T133811Z`. This is user-provided output, not a GPU benchmark.
- Expanded `autotune/metrics.py`: typed `MetricEvent` (positive step, paired losses, optional timestamp/LR/gradient norm/GPU peak); `Decision` with `state`, `severity`, `action`, `evidence`; backward-compatible legacy input.
- Conservative decision rules: nonfinite losses/gradient stop; sustained overfit/high-loss underfit stop; repeated joint loss spikes stop; isolated spikes, low-loss plateaus and large finite gradient norm warn; malformed/missing/duplicate-step events warn without stop.
- Wired `MonitorStream` into mock and unverified HF callback; per-trial `monitor_decisions.jsonl` records every decision; JSONL logger normalizes nonfinite numeric values to null plus explicit fault markers.
- New CLI: `python -m autotune.monitor_cli --file runs/.../trial-01/metrics.jsonl --config config/baseline.yaml`; replay requires no retraining.
- Added 21 tests (`tests/test_monitor_phase2.py`) including offline replay, nonfinite JSONL, warning/stop behavior, noisy curves and integration audit.
- **Verified locally: 36 passed in 0.18s** with Python 3.13 (`python -m pytest -q`); mock recovery reproduced overfit at step 6 and best trial 2 (0.58 simulated validation loss). Not validated on the user's Windows machine yet.
- Added `docs/diagrams/telemetry_flow.mmd` + DOT/SVG and expanded versioned handbook PDF.
- **Limitations:** The HF callback takes latest train loss from `on_log`, which may not reflect a precisely matched evaluation-step window; no live Meta API/Kubeflow/GPU tests. Nonfinite loss is safely retained as a signal; do not treat warnings as automatic stop.

### 2026-11-01: v0.2 - Phase 1
- Unpacked and reviewed original `AutoTune_AI_Starter_v0.1.zip`; preserved original as reference.
- Added `autotune/config.py` with validated YAML contracts: model/data/train/monitor/safety/agent/simulator, nonfinite rejection, discrete LoRA-r constraints, and stable SHA256 fingerprint.
- Added `autotune/doctor.py`: Python/dependency/config checks and JSON summary; for `hf` also checks key packages and dataset paths.
- Wired `orchestrator.run()` to validate config before creating a run directory, and wrote `config_sha256` into each new manifest.
- Added 7 tests for config, fingerprints, invalid values and side-effect-free failures.
- **Test result:** `15 passed in 0.12s` (local Python 3.13); `python -m autotune.doctor --mode mock` returned `ok: true`.
- **Simulation result:** trial-01 early stop (`overfit`, best validation loss 1.57); trial-02 finished (best validation loss 0.58). Values are deterministic mock loss curves, not a training benchmark.
- Run command used `--no-writeback` to avoid changing the reference baseline during verification.

### Before v0.2: v0.1 starting point
- Mock controller, overfit/underfit heuristics, local JSONL tracking, optional Meta agent and MLflow hooks, optional HF Trainer/LoRA code, sample Kubeflow compilation source, standalone Katib example, Streamlit dashboard.
- Prior evidence: 8 tests, synthetic recovery only. No actual KFP cluster/GPU / Meta API validation.

- **Archived prior documentation:** 13-page `AutoTune_AI_Handbook_v0.2.pdf`; current updated documentation is 19-page v0.4, now with a durable recovery flow diagram. Keep historical doc versions only outside current release ZIP to avoid ambiguity.

## What is not done

- Real on-GPU fine-tuning and actual model checkpoint resume.
- Hosted Meta Muse API call verification.
- Full Kubeflow installation/deployment, Katib integration to controller, Ray Tune implementation.
- Persistent tracking infra or object-store migration, Git PR auto-change, held-out evaluation/registry, dashboard end-to-end proof, granular GPU/cost budgets.
- Current phase 1 config contract is Python validation, **not** a published JSON Schema or API model.

## Next exactly

**Phase 5 - Real training & model weight checkpoints:** HF/PEFT GPU smoke, reliable train/eval alignment, actual model checkpoint metadata and controlled stop/resume. Optional hosted Meta key verification as a separate explicit test.

## Local reproduction (Windows PowerShell from project root)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m autotune.doctor --mode mock
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.orchestrator --mode mock --no-writeback
```

## Safety / failure rules

- Never let Meta advisory output execute shell, Python or arbitrary file edits. Reject unexpected keys and invalid values.
- Preserve baseline YAML, logs and hashes, and label all synthetic losses.
- Do not write benchmark claims, recovery improvements, or production readiness based on mock data.
- Do not silently change orchestration ownership between Katib and Ray Tune.
- Protect credentials with environment variables/secret stores and keep them out of logs and Git.
- Never claim completion if tests or live services weren't actually run.

## Phase 4 run commands (Windows VS Code PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.advisor_cli --provider mock --diagnosis overfit
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/agent_mock.yaml --workspace runs
```

Opt-in live test after separately installing `openai`: `python -m autotune.advisor_cli --provider meta --diagnosis overfit`, using a secret `MODEL_API_KEY` in environment. Do not ask user to share the key or claim success if output is fallback.


## Phase 5A audit (v0.6, 2026-11-01)

- **New**: genuine CPU PyTorch byte language-model training (embedding, GRU, head), JSONL validation, train+validation byte hash provenance, per-epoch real train and validation loss, Phase 2 stream monitor, actual model+optimizer checkpoint tensors. No pretrained Llama weights are present in this path.
- **Separate recovery**: `autotune/real_recovery.py` uses `real-torch-v0.6` schema and `best_checkpoint.pt`; checksum and config/dataset/epoch/loss verified before selection using `torch.load(weights_only=True)`. Never pass this receipt to Phase 3 mock-only validator.
- **Recovery behavior**: completed resume idempotent and revalidates real checkpoint; unknown RUNNING without a receipt blocks; tampering or changed input data blocks. A forced-threshold test exercises bounded advisor/fallback retries on genuine tensors, but its underfit diagnosis is artificial.
- **Verification**: 95/95 build-env tests passed (18 new). 4-epoch local training produced observed train loss 5.5887 -> 5.5384; validation loss 5.6072 -> 5.5675; selected actual checkpoint. At this historical checkpoint, user Windows verification was pending; subsequent user output confirmed Phase 5A (see v0.7 header). PyTorch 2.10.0+cpu available in build environment.
- **CLI**: `python -m autotune.real_cli --config config/real_cpu.yaml --workspace runs_real`; `python -m pytest -q`; PowerShell `run_phase5.ps1`. Install `requirements-real-cpu.txt` first. Work from repo root so relative paths resolve.
- **Known limits**: no real Meta Llama fine-tuning, no LoRA in CPU GRU backend (`lora_dropout` only maps to normal GRU output dropout), no per-batch crash-resume, no GPU OOM/reconciliation, no final holdout model quality gate, no production promotion, no remote MLflow validation. Existing Hugging Face adapter is optional and unverified.
- **Next**: Stage 5B on a supported GPU, enable Meta Llama LoRA with PEFT, validate adapter reload/inference and proper checkpoint/optimizer resume; then Phase 6 MLflow server.

## 2026-11-02: v0.8.10 - Security lifecycle checkpoint (documentation only)

- **Source baseline:** v0.8.7 executable source; v0.8.9 latest prior documentation. User verified Phase 6A MLflow on localhost via 3 run links (parent + 2 real HF trials). First overfit diagnosis was fault-injected, Meta advisor mock. No new executable code or security tests in v0.8.10.
- **Existing inspected controls:** `lora_contract.py` validates JSONL, nonempty samples, duplicate/cross-split texts, SHA256 fingerprints and preflight configuration. `metrics.py` monitors loss/gradient irregularities. `advisor.py` strict structured numeric-only recommendations and `policy.py` bounds. `hf_selfheal.py` per-attempt YAML, fresh workspace and exclusive lock, 1-3 attempt bounds. `hf_audit_cli.py` and checkpoint manifests check file hashes. `hf_mlflow.py` logs metrics/metadata and small artifacts rather than model weights by default.
- **Critical distinction:** loss overfit/underfit/divergence is training-health detection and **cannot prove data poisoning, backdoors, model exfiltration, or vulnerability exploitation**. SHA256 detects mismatches against an expected manifest; it cannot certify that the baseline source was benign. No automatic security-CVE/PII/poisoning/robustness scanners currently operate in training.
- **Open P0 findings (not remediated):** mutable HF revision (`main` allowed); package version ranges without formal known-vulnerability scan; lack of explicit untrusted-checkpoint import policy beyond hashes; missing dataset PII/security triage; future API/MLflow production auth and network guards.
- **Proposed gates:** ingest -> provenance + duplicates + PII + suspicious-label review; model -> immutable revision and trusted safe serialization; dependencies -> pip-audit/SBOM; live training -> separate training metrics and security events; agent -> structured bounded outputs; artifacts -> trusted provenance plus SHA256; evaluation -> poisoning/privacy/robustness tests; release -> human/security approval and rollback.
- **Incident policy:** training anomaly may stop and bounded-retry; suspected compromise must block/quarantine and escalate rather than automatically retrain on possibly hostile inputs. Do not upload private datasets/credentials to MLflow or diagnostic APIs.
- **Next engineering phase:** security-sprint P0 (dependency scan, model-intake policy, dataset/PII gate, tests) then Phase 6B read-only FastAPI/Next.js dashboard. Preserve CPU-first development and existing outputs.
- **Artifacts:** `AutoTune_AI_Handbook_v0.8.10.pdf` cumulative, `AutoTune_AI_AGENTS_v0.8.10.md`, `AutoTune_AI_Master_Execution_Plan_v0.8.10.md`, documentation ZIP.
- **Security sources:** NIST AI 100-2 e2025; OWASP Secure AI Model Ops Cheat Sheet and OWASP ML Security Top Ten draft; Hugging Face pickle scanning; MLflow network security; pip-audit.


## 2026-11-02: v0.8.11 - Viva / interviewer explanation (documentation only)

- **User request:** Explain as the student who implemented AutoTune AI how the system works, what detects vulnerabilities, and how findings are corrected.
- **Verified implementation narrative:** CPU Hugging Face Transformers + PEFT LoRA training; custom loss/gradient monitor; bounded recovery controller with strict agent numeric-schema and policy-validated changes; SHA-256 for expected artifacts; local JSON journal and MLflow run tracking.
- **User evidence:** Phase 5E fault-injected two-attempt real Tiny GPT-2 CPU experiment completed; local MLflow created a controller and two trial records; the loss difference is negligible and model promotion remains unapproved. Reported 207 tests passed / 1 skipped. Earlier adapter reload was successful and checkpoint continuation reached step 4 from step 2.
- **Security boundary:** MLflow is NOT a vulnerability scanner. Training anomalies are NOT proof of adversarial compromise. Current SHA-256 hashes identify mismatches, not original source trust. Real Meta advisor API has not been verified; mock advisor was used.
- **Planned vulnerability monitoring:** pip-audit for dependency CVEs, pinned trusted HF model revisions and safe checkpoint intake, PII/poisoning/robustness checks, authenticated dashboard/MLflow access for multi-user deployment. These are NOT presently implemented and no CVE has been claimed found.
- **Response distinction:** Training problem -> safe stop, verified checkpoint, bounded parameter patch and retry. Security suspicion -> block/quarantine, preserve evidence, human review, no automatic retry on suspicious inputs.
- **Artifacts:** AutoTune_AI_Handbook_v0.8.11.pdf (cumulative + interview appendix), AutoTune_AI_AGENTS_v0.8.11.md, AutoTune_AI_Master_Execution_Plan_v0.8.11.md, AutoTune_AI_Viva_QA_v0.8.11.md and documentation ZIP. No source files changed and no new tests executed.
- **Next priority:** P0 security scanning / data and model intake policy, then read-only FastAPI + Next.js dashboard. Preserve existing outputs and CPU-first workflow.


## 2026-11-02: v0.9.1 - Local Swagger UI troubleshooting + opt-in patch

- **User symptom:** Uvicorn started successfully at `127.0.0.1:8000`, but `GET /`, `GET /docs` returned 404.
- **Verified root cause in v0.9.0 source:** `autotune/viewer_api.py` deliberately created FastAPI with `docs_url=None`, `redoc_url=None`, `openapi_url=None`; the root route was never defined. A 404 here was expected, not an MLflow connectivity error.
- **Existing working API routes:** `GET /api/health`, `/api/experiments`, `/api/experiments/{id}/runs`, `/api/runs/{id}`, `/api/runs/{id}/metrics/{metric}`. `/api/health` returns `mlflow: not probed`; it cannot alone prove the backend is connected.
- **v0.9.1 patch:** Optional Swagger/OpenAPI docs gated behind environment variable `AUTOTUNE_ENABLE_DOCS=1`. Documentation remains **off by default** and `redoc` remains off. Never expose localhost services publicly or assume Swagger supplies auth.
- **Windows action:** After applying patch: ` $env:AUTOTUNE_ENABLE_DOCS='1'` then restart `python -m uvicorn autotune.viewer_api:app --host 127.0.0.1 --port 8000`; open `http://127.0.0.1:8000/docs`. Disable by removing that variable and restarting.
- **Verification performed in artifact build:** Existing + new viewer tests 27/27 passed; full project Python tests 243/243 passed. These were executed in the build environment, **not** on user's Windows computer. Live browser acceptance is still pending.
- **Limits:** No frontend npm build, no new live MLflow connection test, no job submission endpoints, no security scanner integration. Existing runs and model artifacts are unchanged.
- **New files:** `tests/test_viewer_docs_v091.py`, `docs/PHASE6B_SWAGGER_TROUBLESHOOTING_v0.9.1.md`; changed `autotune/viewer_api.py`.
- **Next:** User verifies `/docs`, `/api/health`, and `/api/experiments` on Windows; then proceed to Next.js viewer browser integration and P0 security gates.

## 2026-11-02: v0.9.2 - Hugging Face SmolLM2 LoRA -> Ollama CPU bridge

- **User request:** Fine-tune a small Ollama-compatible open-source language model from Hugging Face; no dedicated GPU, Windows VS Code / PowerShell.
- **Chosen model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache 2.0, 135M, Llama-family). Train original HF weights via existing `hf_models_cli` PEFT LoRA CPU trainer, NOT by modifying an Ollama quantized model. Dataset in `sample_data/smollm2_ollama_{train,validation}.jsonl` (16/4 small synthetic educational examples formatted with ChatML user/assistant tokens). Config `config/lora_smollm2_ollama_cpu.yaml` (CPU, no 4-bit training quantization, q_proj/v_proj, 8 expected update steps).
- **New source:** `autotune/ollama_bridge.py` CLI `plan`, `verify`, `merge`, `verify-merge`, `modelfile`, optional `gguf`. Verifies existing v0.7 adapter manifest and original base model, merges to fresh Safetensors directory and writes SHA256 file manifest; creates Ollama Modelfile; supports optional llama.cpp converter with safe subprocess argv and explicit local converter path. Does NOT auto-download model weights on preflight or replace previous outputs.
- **Tests:** 15 new offline bridge tests, **258 Python tests passed in build environment**; no new real SmolLM2 fine-tune, actual merge, llama.cpp GGUF conversion, Ollama create, or Ollama runtime test performed. User Windows acceptance pending.
- **Current architecture:** Existing model monitoring, durable retry controller, MLflow and read-only viewer are retained. This is an additive deployment path and does NOT yet wire SmolLM2 export into the real self-heal multi-trial controller or MLflow artifact registration.
- **Security:** refuse unexpected model ID and wrong CPU backend, reject nonempty output folders and symlinks, reject modified merged artifacts, `trust_remote_code=False`, only merge verified PEFT adapter. Model is currently revision `main` (P0 to pin a full immutable HF commit before sensitive use). SHA256 is an integrity check, not provenance/benign-code proof. Never load third-party checkpoint pickle directly.
- **Immediate user steps:** apply v0.9.2 patch preserving `.venv`, `outputs`, `mlflow.db`; `python -m autotune.ollama_bridge plan --config config/lora_smollm2_ollama_cpu.yaml`; real `hf_models_cli --execute` into `outputs/smollm2-ollama-01`; `ollama_bridge verify`; `merge`; `verify-merge`; `modelfile --merged`; `ollama create` and `ollama run`. GGUF optional fallback; full instructions in `docs/PHASE6C_HF_TO_OLLAMA_CPU_v0.9.2.md`.
- **Next:** user live-verifies CPU fine-tune/adapter merge/Ollama import; then add registered artifact evaluation + security gates + automated self-healing for this backend; pending Next.js live dashboard verification and Kubeflow/Katib orchestration remain.
