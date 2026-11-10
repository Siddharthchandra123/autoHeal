# AutoTune AI v0.12.0 - Phase 7 Knowledge Engine

**Checkpoint: 4 November 2026.** Phase 7 now connects local document ingestion, hybrid retrieval, grounded RAG, structured SFT dataset creation, response-only LoRA training, sequential candidate trials, held-out scoring, model artifact export and a Next.js dashboard.

## 1. Phase 7 scope and current delivery

| Sub-phase | Capability | Implementation state |
|---|---|---|
| 7A | Document upload, extraction, hashing, SQLite metadata and chunking | Implemented; offline tests pass |
| 7B | FTS5 keyword retrieval, optional CPU sentence-transformer embeddings and hybrid rank fusion | Implemented; semantic weights download only after explicit indexing |
| 7C | Local Ollama RAG with exact-quote citation verification | Implemented; lexical support is a screen, not entailment proof |
| 7D | Local Ollama Q/A generation, exact-source-quote validation, PII-pattern rejection, source-disjoint splits, structured `messages` JSONL and response-only SFT loss | Implemented; heuristic filters need human review |
| 7E | Sequential candidate model registry, immutable Hub revision pinning, bounded learning-rate trials, base-vs-adapter held-out scoring, shortlist/NO_QUALIFIED_MODEL, verified Ollama bridge | Implemented in code; real downloads/training/evaluation not run in the build environment |
| 7F | Next.js Knowledge Engine page for upload, search, ask, generation and job control | Implemented in source; npm install timed out in the build environment, so browser build/runtime is not verified |

## 2. Architecture

```text
Upload files
  -> format extraction + SHA-256 dedupe + SQLite metadata
  -> overlapping source chunks + FTS5 index
  -> optional sentence-transformer vectors (CPU; explicit indexing)
  -> hybrid retrieval (reciprocal-rank fusion)
  -> grounded answer (local Ollama; exact source quote checked)
  -> grounded Q/A generation (local Ollama; exact quote + lexical support checks)
  -> source-document-disjoint train / validation / test
  -> structured `messages` JSONL
  -> response-only LoRA SFT (prompt labels = -100)
  -> sequential model + learning-rate trials
  -> base/adapter held-out token-F1 comparison
  -> qualified shortlist or NO_QUALIFIED_MODEL
  -> adapter integrity check / merge / Modelfile / optional GGUF
  -> human production approval
```

### Knowledge store layout

```text
knowledge_store/
  uploads/                 original files keyed by content-derived document id
  processed/               source passage exports
  datasets/<dataset-id>/   train.jsonl, validation.jsonl, test.jsonl, dataset_manifest.json
  jobs/                    persistent job status JSON
  training_runs/<job-id>/  preflight, configs, trials, receipts and evaluation reports
  knowledge.sqlite3        documents, chunks, FTS5 and optional embedding vectors
```

The API uses `AUTOTUNE_KNOWLEDGE_HOME` to change the storage root. Back up this entire folder before upgrades or cleanup.

## 3. Windows setup (PowerShell only)

From the extracted project root in VS Code PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_phase7.ps1
```

This creates `.venv-phase7`, installs the ingestion/API requirements and starts the API on `127.0.0.1:8010`.

Optional semantic indexing dependencies:

```powershell
.\run_phase7.ps1 -InstallSemanticSearch
```

Optional Hugging Face training dependencies:

```powershell
.\run_phase7.ps1 -InstallTrainingDependencies
```

Training dependency installation can consume substantial disk. For CPU-only Windows training, make sure the installed PyTorch build is CPU-compatible. The candidate registry currently runs sequentially on CPU and may be slow. Do not install GPU/ROCm/CUDA variants unless the target machine actually has a supported accelerator.

Optional Next.js dashboard (Node.js/npm required):

```powershell
.\run_phase7.ps1 -StartDashboard
```

Open `http://127.0.0.1:3000/knowledge` and the API docs at `http://127.0.0.1:8010/docs`. Or run the UI manually from `web` using `npm install` and `npm run dev`.

### Local Ollama setup

Install Ollama separately and pull a small instruction model. For example:

```powershell
ollama pull qwen2.5:0.5b
ollama list
```

Ollama must be running locally at `http://127.0.0.1:11434`. The Knowledge Engine deliberately rejects non-loopback Ollama endpoints. The tag entered in the UI must match an installed local model. The Ollama model creates candidate Q/A examples; it is not one of the HF fine-tuning candidates.

## 4. Upload and retrieval workflow

Supported formats: TXT, Markdown, CSV, JSON, JSONL, PDF and DOCX. Upload size is capped at 20 MB per file; extracted content is capped at 5,000,000 characters; PDF extraction is capped at 500 pages. Scanned/image-only PDFs are not OCR'd; extractable text is required. Native PDF/DOCX parsers are installed by `requirements-knowledge.txt`.

Upload via the dashboard or:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8010/api/knowledge/health
```

API routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/knowledge/health` | API health and capability list |
| POST | `/api/knowledge/documents` | Multipart document upload |
| GET | `/api/knowledge/documents` | Document list, status and chunk counts |
| POST | `/api/knowledge/export` | Export source passages; explicitly not SFT-ready |
| POST | `/api/knowledge/search` | FTS5, semantic or hybrid retrieval |
| POST | `/api/knowledge/index` | Explicitly create CPU embedding index |
| POST | `/api/knowledge/ask` | Grounded RAG answer from local Ollama |
| POST | `/api/knowledge/datasets/generate` | Create structured grounded Q/A dataset |
| GET | `/api/knowledge/datasets` | List generated datasets |
| GET | `/api/knowledge/datasets/{dataset_id}/preview?split=train&limit=8` | Preview up to 50 structured examples from one split |
| POST | `/api/knowledge/training/jobs` | Start training preflight or execution job |
| GET | `/api/knowledge/training/jobs/{job_id}` | Poll durable job status/result |

### Retrieval modes

- `lexical`: SQLite FTS5.
- `semantic`: sentence-transformer cosine similarity. Requires explicit `/index` call and optional dependency/model download.
- `hybrid`: reciprocal-rank fusion across keyword and semantic rankings.
- `auto`: attempts hybrid and falls back to lexical if embeddings are unavailable.

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, running on CPU. Call `/api/knowledge/index` explicitly to download/cache the model and index current chunks. New documents can still be searched lexically; re-run indexing to embed them. The implementation stores embedding bytes, dimension, model name and hash in SQLite.

### RAG behaviour

The RAG system sends retrieved passages to a local Ollama model and asks for structured JSON with source chunk IDs and exact quotes. The server verifies that each quoted string exists in the cited chunk. It also calculates a simple lexical-overlap support ratio. This can reject some unsupported answers but is not a semantic entailment model and cannot guarantee factual correctness. RAG citations identify the uploaded file and quote.

## 5. Grounded SFT dataset factory

The dataset generator asks local Ollama for up to five Q/A pairs per chunk. It rejects examples when:

- required fields or length bounds fail;
- the evidence quote is not an exact contiguous substring of the source chunk;
- answer terms have less than 45% lexical support in the quote;
- the question duplicates an earlier question;
- a basic PII/secret regex is matched (email, phone-like sequences, SSN-like values, AWS access-key and OpenAI-key-shaped strings).

These PII checks are heuristics only. They do not detect every secret, identifier, personal detail or confidential business datum. Review your sources and generated data before training. Do not upload sensitive or regulated content to this local prototype unless your local policy permits it.

The factory groups examples by `source_id`, assigns complete source documents to exactly one of train/validation/test, and verifies source-disjoint sets. It requires at least four source documents and at least two accepted examples in every split; the additional source is necessary to provide at least two train documents under the 80/10/10 document split. The manifest records split counts, source-document IDs, the generating Ollama tag, seed, rejection counts, data format and SHA-256 hashes for each JSONL split.

The orchestrator additionally verifies JSONL structure, manifest counts, source provenance and duplicate/leakage checks before training. The generated dataset is structured as:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source_ids":["..."],"source_file":"...","chunk_id":"...","evidence_quote":"...","evidence_support_ratio":0.91}
```

Do not treat `fine_tuning_ready: true` as a claim of expert review. It means the generator's automated structural/evidence filters passed.

## 6. Response-only LoRA training

For structured `messages` data, each candidate tokenizer renders the conversation with its own chat template. The trainer separately renders the prompt with `add_generation_prompt=True` and the full conversation with `add_generation_prompt=False`; it verifies prompt token IDs are an exact prefix of the full sequence. Prompt/system/user tokens receive label `-100`; assistant completion tokens and response EOS are supervised. Padding labels are also `-100`. Long examples that exceed configured `max_length` are rejected instead of silently truncating target answers.

Legacy text-format datasets continue using the old all-token-loss path. `hf.loss_mode=response_only` requires structured messages. This response-only path is unit-tested with a fake tokenizer/collator and validated through plan contracts; actual multi-model training is not run in this build environment.

## 7. Autonomous multi-model orchestration

Candidate registry: `config/knowledge_candidates.json`.

| Candidate | Hub model | Device | Notes |
|---|---|---|---|
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | CPU | Thinking mode disabled for consistent instruction output |
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | CPU | Small instruction baseline |
| SmolLM2 360M Instruct | `HuggingFaceTB/SmolLM2-360M-Instruct` | CPU | Small candidate for resource-constrained testing |

Default search trials use learning rates 2e-4 and 1e-4; max train steps 100; 3 epochs; LoRA rank 8 / alpha 16 / dropout 0.05; sequential loading so one model is resident at a time. The settings are an initial configurable pilot policy, not universal optimal hyperparameters. Candidate model revisions are resolved to immutable Hub commit SHAs at execution time. Offline preflight uses `main` as a placeholder and does not download weights.

The default execution row-count gates are 100 train / 20 validation / 20 test examples. These are minimum pilot gates, not guarantees of statistical power or quality. `--allow-small-data` / API `allow_small_data` exists only for an explicitly opted-in smoke test; it never grants production eligibility. Use substantially more curated examples for serious domain adaptation.

Preflight (no model downloads / training):

```powershell
.\.venv-phase7\Scripts\python.exe -m autotune.knowledge_orchestrator --dataset knowledge_store\datasets\<dataset-id> --output outputs\knowledge-preflight
```

Execute (downloads model snapshots and trains sequentially on CPU):

```powershell
.\.venv-phase7\Scripts\python.exe -m autotune.knowledge_orchestrator --dataset knowledge_store\datasets\<dataset-id> --output outputs\knowledge-run-01 --execute --allow-cpu
```

Execute one candidate first to validate memory/time footprint:

```powershell
.\.venv-phase7\Scripts\python.exe -m autotune.knowledge_orchestrator --dataset knowledge_store\datasets\<dataset-id> --output outputs\knowledge-smoke --execute --allow-cpu --max-candidates 1 --allow-small-data
```

The smoke override must not be used to assert model quality. A candidate trial trains an adapter and then reloads the base and adapter separately to avoid holding both model copies in memory. The objective filter compares their predictions on the same held-out test examples using normalized token F1 and exact match. It chooses the best learning-rate trial within each model and qualifies candidates using validation F1 and improvement thresholds. The validation set selects one model architecture; only that selected model is evaluated once against the final test split. A failed final test produces `NO_QUALIFIED_MODEL`. Raw validation loss is not compared across different model families/tokenizers.

Output includes `events.jsonl`, per-candidate records, per-trial configs/runs/evaluations, `experiment_result.json` and selection result. If no candidate passes, result is `NO_QUALIFIED_MODEL`. Selection is a shortlist only: token F1 is lexical overlap, not robust semantic evaluation or a safety certificate. The orchestrator always sets `promotion_approved=false` and `production_eligible=false`.

## 8. Verify and export to Ollama

Use the selected `config_path` and `run_dir` from `experiment_result.json`:

```powershell
.\.venv-phase7\Scripts\python.exe -m autotune.ollama_bridge verify --config <config_path> --run <run_dir>
.\.venv-phase7\Scripts\python.exe -m autotune.ollama_bridge merge --config <config_path> --run <run_dir> --output outputs\knowledge-run-01\merged
.\.venv-phase7\Scripts\python.exe -m autotune.ollama_bridge verify-merge --merged outputs\knowledge-run-01\merged
.\.venv-phase7\Scripts\python.exe -m autotune.ollama_bridge modelfile --merged outputs\knowledge-run-01\merged --output outputs\knowledge-run-01\Modelfile
ollama create autotune-knowledge:pilot -f outputs\knowledge-run-01\Modelfile
```

The bridge checks adapter/base-model identity and artifact hashes and supports generic registered base models. A GGUF conversion is optional; it requires a compatible local `llama.cpp` converter script and should be staged to a new output directory. Ollama import success is not proof of model quality.

## 9. Security, data and release boundaries

- Bind the API to `127.0.0.1`; the prototype has no user authentication, authorization, tenant isolation or production secret-management layer.
- It uses `TrustedHostMiddleware`, localhost-only CORS origins, file allowlists, a 20 MB limit, content hashes, sanitized basenames and local Ollama URL allowlisting. These are development hardening measures, not a security audit.
- FTS5 and SQLite are local; there is no managed vector database or multi-user access.
- Uploaded documents and generated prompts are untrusted data. RAG and dataset generation prompts say source passages are data, not instructions, but prompt injection resistance is not guaranteed.
- Model downloads require internet and Hugging Face access. License/usage terms must be checked by the operator for each project/use case.
- The current evaluation threshold is an initial filter. Production requires broader independently curated tests, safety/security reviews, human evaluation, domain-owner signoff, monitoring and rollback.
- Do not expose this unauthenticated service on `0.0.0.0` or a public host.

## 10. Verification record and unverified work

**Build verification:** `PYTHONPATH=. pytest -q` -> **338 passed in 9.31 seconds** in the build environment. This includes inherited AutoTune tests and offline tests for retrieval, grounded-data checks, source-disjoint splits, response-only collator, orchestrator selection/test isolation, API route contracts, tampered-split hash rejection, job-state persistence and Ollama bridge generalization.

**Not verified in this build:** live Windows PowerShell startup; real PDF/DOCX extraction on Windows; live Ollama calls; sentence-transformer model download/indexing; Hugging Face Hub revision resolution; downloading weights for the three candidate models; actual multi-model LoRA runs; base-vs-adapter held-out inference; Ollama merge/import/GGUF export; Next.js `npm install`, typecheck, build and browser acceptance. The npm install attempt in the build environment timed out. Do not represent these as successful runs.

## 11. Troubleshooting

- **Port 8010 in use:** identify the process before stopping anything; do not kill unknown Python processes. Change the port in your own startup command and update the dashboard API endpoint only if needed.
- **Ollama connection refused:** open Ollama, run `ollama list`, and verify `http://127.0.0.1:11434` responds locally.
- **No semantic embeddings:** run `run_phase7.ps1 -InstallSemanticSearch`, then call the Build semantic index control. Search auto mode safely falls back to FTS5 until vectors exist.
- **Not enough accepted pairs:** add distinct, text-rich source documents; inspect rejection counters; improve the local Q/A generation model or reduce noise in source material. Do not lower evidence filters merely to get a larger dataset.
- **Dataset source split fails:** provide at least four distinct documents and enough independent examples so every split has at least two records. More documents are strongly preferred.
- **Training exceeds RAM/time:** run one candidate via `--max-candidates 1`, reduce max length/steps in `config/knowledge_candidates.json`, or use a supported remote accelerator environment after validating its runtime. The current CPU path is intended for small candidates and can be slow.
- **`NO_QUALIFIED_MODEL`:** keep it. Inspect per-candidate score and dataset quality; do not force promotion or tune the test threshold just to produce a winner.

---

**Release statement:** AutoTune AI v0.12.0 delivers the Phase 7 source architecture and offline-tested integration. It is a local research/dev prototype, not a production-ready multi-user fine-tuning service. Live model training and end-to-end browser acceptance remain operator-side verification steps.
