# AutoTune AI v0.14.0 — Phase 8 Katib HPO

Phase 8 adds offline-first Kubeflow Katib Experiment generation, bounded hyperparameter contracts, a single-trial worker and fail-closed trial summaries. Start with the synthetic CPU mock backend; it is a control-plane smoke test, not a model benchmark.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.katib_cli render --spec config/katib_cpu.yaml --output pipelines/autotune_katib_experiment.yaml
```

Full runbook: [`docs/PHASE8_KATIB_v0.14.0.md`](docs/PHASE8_KATIB_v0.14.0.md). Offline suite: **367 passed**. Real Kubernetes/Katib and HF worker acceptance remain pending.

---

# AutoTune AI v0.13.1 - Knowledge Engine Dataset Generation Fix

**Release checkpoint: 9 November 2026.** v0.13.1 fixes the slow/stuck dataset-generation workflow from v0.13.0. The dataset-build API now queues a background job and returns immediately; the dashboard polls persistent job status and displays progress. Existing supervised JSONL data such as Databricks Dolly is recognized and imported directly into structured SFT messages, without asking a local LLM to regenerate examples. Train/validation/test splits for a one-file supervised dataset are disjoint by source record and retain the original source document and JSONL line provenance.

The v0.13.0 Knowledge Engine scale features remain: 512 MiB upload limit, 100M extracted-character limit by default, disk-staged upload streaming, incremental extraction, batched SQLite/FTS writes, retry/progress, source offsets, and bounded semantic indexing/search.

## Quick start - Windows / PowerShell

Stop any old API/frontend terminals first. Overlay the v0.13.1 files while preserving your existing `knowledge_store` directory, then run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_phase7.ps1
```

In a second terminal:

```powershell
Set-Location .\web
npm run dev
```

The first start creates a lightweight `.venv-phase7` with the API/document dependencies only; this environment does not install PyTorch. On later starts, use `-SkipInstall`. Open `http://127.0.0.1:3000/knowledge`; API docs are at `http://127.0.0.1:8010/docs`.

## Dataset-generation behavior

- **Dolly / other supervised JSONL:** if indexed JSONL records contain `instruction`/`response` (Dolly), `prompt`/`completion`, `question`/`answer`, `input`/`output`, or structured `messages`, the engine normalizes and imports those examples directly. Ollama is not called. Context is appended to the user message where present. Duplicate message pairs and basic PII patterns are rejected.
- **Unlabelled documents:** for PDF, DOCX, TXT, Markdown and other knowledge-only sources, the engine continues to use local Ollama for grounded question/answer generation. Every accepted generated answer must still pass the exact-source-quote and lexical-support checks. This workflow requires at least four distinct source documents for document-disjoint splits.
- Generation is a background job. `POST /api/knowledge/datasets/generate` returns HTTP 202 and a `job_id`; poll `GET /api/knowledge/datasets/generation-jobs/{job_id}`. The dashboard does this automatically and reports stage, processed chunks/records, accepted pairs and rejected pairs.
- The default Ollama chunk limit is now 100 instead of 500, and generation stops once its target is reached and all three splits contain sufficient examples.

## Dolly upload workflow

1. Upload and index `databricks-dolly-15k.jsonl` through the Knowledge Engine.
2. Click **Build fine-tuning dataset**.
3. Watch the background job progress. For recognized Dolly JSONL, the UI reports `direct_import`; Ollama does not need to be running.
4. Preview the train, validation and test splits, then run **Preflight** before any model download or training.

For a single JSONL source, splitting is by individual source record, not by document. The manifest records both the record IDs used as the leakage boundary and the original source document ID. This is appropriate for individually labelled examples such as Dolly; it does not claim document-disjointness for one-file data.

## API endpoints added

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/knowledge/datasets/generate` | Start a dataset generation/import job (HTTP 202) |
| GET | `/api/knowledge/datasets/generation-jobs/{job_id}` | Poll job stage, progress, result or error |

## Verification boundary

`PYTHONPATH=. python -m pytest -q` -> **351 passed** in the build environment. The Knowledge Engine page TSX transpiled with zero syntax diagnostics using the installed TypeScript compiler. A full npm/Next.js typecheck/build and browser acceptance were not run. No live Ollama generation, actual Windows upload, Hugging Face model download, or fine-tuning was executed in this release validation. The service remains an unauthenticated local research prototype and must stay bound to `127.0.0.1`.

The existing `AutoTune_AI_Handbook_v0.13.0.pdf` documents the v0.13.0 architecture; this release's dataset-job and Dolly-specific behavior is documented in [`docs/KNOWLEDGE_ENGINE_DATASET_GENERATION_V0131.md`](docs/KNOWLEDGE_ENGINE_DATASET_GENERATION_V0131.md).
