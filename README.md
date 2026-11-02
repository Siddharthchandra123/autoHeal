# AutoTune AI v0.9.0 - Phase 6B Local Experiment Dashboard

The new *read-only* FastAPI + Next.js/TypeScript viewer displays existing MLflow experiment runs and bounded metrics. Localhost-only URL validation, input allowlists, CORS/host checks and metadata redaction are implemented. Existing Hugging Face training and artifacts are unchanged. Python tests are verified; **Next.js install/build and a live Windows MLflow-to-viewer run have not yet been verified**.

Read [Phase 6B Runbook](docs/PHASE6B_DASHBOARD_RUNBOOK.md). In three local PowerShell terminals, launch the MLflow server on port 5000, the viewer API on 8000, and Next.js on 3000. Never expose these unauthenticated development services publicly.

---

## v0.8.7: Opt-in MLflow tracking (Phase 6A)

The user's prior v0.8.6 real two-attempt CPU self-healing run was independently verified on Windows (207 passed, 1 skipped; first stop intentionally injected). This release adds optional MLflow lineage with controller/child trial runs, metrics and metadata, not model-weight upload. **216 offline tests passed in the build environment; live MLflow server integration is awaiting Windows verification.**

See [Phase 6A MLflow runbook](docs/PHASE6A_MLFLOW_TRACKING.md). A local MLflow server requires no GPU or Docker. This integration is off by default, and uses explicit `--mlflow-uri`.

## NEW: Phase 5E v0.8.6 - Real HF CPU autonomous recovery controller

**Offline integration tests: 208 passed; live two-attempt CPU training still needs user verification.** This release adds `autotune.hf_selfheal`: a bounded pipeline that connects real Hugging Face LoRA training -> paired validation monitoring -> strict agent recommendation -> versioned new training attempt -> hash-verified artifact comparison. Controlled synthetic fault injection is separately labeled and must not be confused with natural model overfitting.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml
# Controlled REAL HF two-attempt CPU acceptance, fresh workspace required:
.\.venv\Scripts\python.exe -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml --execute --fault-inject-overfit --workspace outputs/hf-selfheal-injection-01
```

Read `docs/PHASE5E_REAL_HF_SELFHEAL.md` for safety gates, diagrams, artifacts, and limitations. Preserve all prior outputs. No GPU needed. No MLflow/Kubeflow/Unsloth live training is claimed.

---

## v0.8.2: CPU-only controlled Trainer-resume exercise

Your existing `gpt2-cpu-01` run must remain untouched. The user has independently verified its artifact hashes and fresh LoRA adapter reload on Windows. For the *next* milestone, use `config/lora_cpu_resume_probe.yaml` and the two commands in `docs/PHASE5D_OPTIMIZER_RESUME.md` to intentionally pause a **new** four-step run after step two, then resume it from the full Trainer checkpoint. Build-only status: 185 offline tests passed; live optimizer resume is not yet verified.

## IMPORTANT - CPU-first mode (v0.7.2)

No dedicated GPU is required for the current development plan. Start with `config/lora_public_cpu.yaml` and the instructions in `docs/PHASE5B_CPU_FIRST_RUNBOOK.md`. Phase 5A real CPU PyTorch is user-verified; Phase 5B Hugging Face LoRA CPU actual training remains unverified until executed. GPU-specific Meta Llama/QLoRA is optional.

---

# AutoTune AI v0.7 - Phase 5B LoRA pilot (CPU tests verified; GPU UNVERIFIED)

**New verification checkpoint, November 1, 2026:** The user verified Phase 5A on Windows/Python 3.13: **95 tests passed in 34.36s**, actual PyTorch tiny-GRU training finished, best validation loss 5.567494869232178 and loadable tensor checkpoint SHA256 `093223fd97f7fcc085e7a274036e24c82f46d0406d98c28433bd6328b85b3f3e`. PyTorch emitted a nonfatal **missing NumPy warning**; `requirements-real-cpu.txt` now includes NumPy.

**Phase 5B (new):** LoRA/QLoRA Hugging Face pipeline, validated offline plan, train/validation leakage detection, bounded evaluation cadence, graceful early stop, complete Trainer checkpoint/adapter hash manifests, explicit resume/reload, and CLI safeguards. **134/134 tests passed in build container.** Hugging Face, PEFT and model weights were NOT installed/downloaded here, and Meta license/GPU/QLoRA/Trainer checkpoint format have **not** been live-tested. No autonomous GPU retry is claimed.

## Quick start - Windows VS Code (no GPU or online model needed)

```powershell
.\.venv\Scripts\python.exe -m pip install numpy
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.lora_cli --config config/lora_meta_gpu.yaml
.\.venv\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml
```

These are OFFLINE plan checks, not actual Meta fine-tuning. `run_phase5b.ps1` runs the tests and offline plan; invoke with an appropriate PowerShell execution policy or call commands directly. For an optional public tiny-GPT2 CPU LoRA smoke or real GPU run, follow [Phase 5B Runbook](docs/PHASE5B_RUNBOOK.md) and use explicit `--execute` only after verifying hardware/model permissions.

**Living documentation:** `AGENTS.md`, `docs/MASTER_EXECUTION_PLAN.md`, `docs/HANDBOOK_SOURCE.md`, and `AutoTune_AI_Handbook_v0.7.pdf` are updated in every active project iteration.

---

# AutoTune AI v0.6 - Phase 5A real PyTorch CPU reference

**Current:** 95 local tests passed (77 inherited + 18 new). Previous Phase 4 mock was confirmed by the user on Windows with 77 tests passing. Phase 5A creates REAL, loadable PyTorch model and optimizer tensors using a tiny, randomly initialized byte-level language model. It does **not** fine-tune Meta Llama.

## Windows + VS Code / PowerShell

Extract the ZIP; open `AutoTune_AI_Phase5` as the VS Code workspace root.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-real-cpu.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.real_cli --config config/real_cpu.yaml --workspace runs_real
```

Or use `.\run_phase5.ps1` once PowerShell execution policies permit trusted local scripts. If unsigned scripts are blocked, use the Python commands above rather than changing organization policies. The commands use only local sample data and PyTorch CPU; no model download or Meta API key.

**Files to inspect** in the newest `runs_real/<run-id>` folder:

- `trial-01/metrics.jsonl` and `monitor_decisions.jsonl`: actual per-epoch metrics and decisions.
- `trial-01/best_checkpoint.pt`: REAL tensors (model+optimizer), atomically written after validation improvement.
- `trial-01/result_receipt.json`: expected SHA256 and config/data binding.
- `recovery_state.json`, `recovery_events.jsonl`, `active.yaml`: durable run audit and selected config.

To verify an existing complete run without retraining, pass `--resume runs_real/<run-id>` with the same baseline config/data. If a prior run was left RUNNING without receipt, recovery blocks for operator inspection. Do not force-retry unknown active work.

The earlier **mock demonstration** is still available with `python -m autotune.recovery_cli --config config/agent_mock.yaml --workspace runs`; it uses simulated checkpoint metadata and is intentionally separate.

## Project status

| Phase | Verified level |
|---|---|
| 1-4 | CPU mock 77 tests, user validated Phase 4 on Windows |
| 5A | Genuine offline CPU PyTorch model and optimizer state, 18 additional tests |
| 5B | Meta Llama LoRA on supported GPU, not yet tested |
| 6+ | MLflow server, Kubeflow, Katib, Git automation, UI, release gates - planned |

`autotune/hf_trainer.py` contains an **optional, unverified** Hugging Face/PEFT LoRA adapter. Do not treat it as load-tested or connected to real-weight recovery. Read `docs/PHASE5_RUNBOOK.md`, `AGENTS.md`, and `docs/MASTER_EXECUTION_PLAN.md` before progressing.

### Docs update protocol

For every future project iteration, update `AGENTS.md`, master plan, handbook Markdown, Mermaid/Graphviz diagrams as appropriate, regenerate the versioned PDF handbook, and record actual tests and unverified features. Work only advances in active project requests, not automatically in the background.

## New in v0.8.0 — train multiple Hugging Face model types (CPU-first)

The new entrypoint `python -m autotune.hf_models_cli` supports `causal-lm` LoRA and `text-classification` full fine-tuning. Each task has its own validated input contract, and both support CPU operation for appropriately small models. Default CLI behavior is offline preflight. **No automatic Hugging Face weight downloads occur without `--execute`.**

```powershell
.\.venv-hf\Scripts\python.exe -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml
.\.venv-hf\Scripts\python.exe -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml --execute --output outputs/bert-tiny-cpu-01
```

See `docs/HF_OPEN_SOURCE_MODELS.md` for tasks, dataset format, model changes, checkpoint verification and limitations. Other model types (ViT, Whisper, seq2seq) are roadmap items, not yet executable adapters. The hosted Meta advisor and HF automatic retraining loop are not yet connected.


## v0.8.1 - Verify the first real public CPU LoRA run

A user's Windows terminal successfully executed small-model PEFT LoRA training with `sshleifer/tiny-gpt2` (four training and two validation rows, final eval loss 10.3652). This confirms real CPU training and checkpoint writing, **not** inference reload or optimizer resume. Avoid interpreting this tiny smoke as trained-model quality.

Use the new read-only audit command on the **existing** output without rerunning training:

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_audit_cli --config config/lora_public_cpu.yaml --run outputs/gpt2-cpu-01
```

After a successful audit, optional real adapter reload:

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_audit_cli --config config/lora_public_cpu.yaml --run outputs/gpt2-cpu-01 --reload
```

`--reload` may read/download cached Hugging Face model weights. This does not test optimizer resume. See `docs/PHASE5C_ARTIFACT_AUDIT.md`.


## Optional Unsloth GPU backend (v0.8.3)

Unsloth was added as an **experimental and opt-in** GPU training adapter. Windows CPU users should continue using the verified Hugging Face PEFT backend. To inspect without downloading or training:

```powershell
.\.venv\Scripts\python.exe -m autotune.unsloth_cli --config config/lora_unsloth_gpu.example.yaml
```

GPU execution is intentionally blocked when no supported CUDA/ROCm training GPU is visible. Read `docs/UNSLOTH_OPTIONAL_BACKEND.md` before installing dependencies or using `--execute`. Unsloth runtime and real agent recovery have not yet been GPU-tested.
