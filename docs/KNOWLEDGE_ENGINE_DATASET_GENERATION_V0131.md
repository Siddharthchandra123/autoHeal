# AutoTune AI v0.13.1 - Dataset Generation and Dolly Import Runbook

## Why v0.13.0 looked stuck

The v0.13.0 UI issued one synchronous HTTP request and kept its button in `Generating…` until the response returned. The backend called local Ollama once per selected chunk, sequentially, with a 180-second timeout per call and a maximum of 500 chunks. There was no dataset-generation progress endpoint, so a slow Ollama call looked like a frozen page. A supervised dataset such as Dolly also arrived as one JSONL file, while the original document-level split algorithm required four distinct source documents; after the long generation pass, that input shape could not satisfy the old split rule.

## v0.13.1 behavior

1. `POST /api/knowledge/datasets/generate` returns HTTP 202 with a `job_id` immediately.
2. `GET /api/knowledge/datasets/generation-jobs/{job_id}` returns `queued`, `running`, `completed`, `failed` or `interrupted`, along with progress and the final result/error.
3. The dashboard polls every 1.5 seconds and displays the current stage, chunk/record progress, accepted pairs and rejected pairs.
4. A single generation job is allowed per local API process. This prevents duplicate clicks from starting multiple expensive Ollama runs.
5. The default Ollama chunk budget is 100 instead of 500. Grounded generation stops after it reaches the requested pair count and every split has enough examples.

## Dolly / existing instruction dataset workflow

The engine inspects retained, indexed `.jsonl` uploads before it calls Ollama. It recognizes these common source schemas:

- Dolly: `instruction`, optional `context`, `response`, optional `category`.
- `prompt` / `completion`.
- `question` / `answer`.
- `input` / `output`.
- Structured `messages` with valid `system`, `user`, and `assistant` roles.

For Dolly, the normalized user message contains the instruction and, when non-empty, a `Context:` section. The assistant message contains the original response. Category metadata is retained. Exact duplicate conversations and configured PII-like patterns are rejected. The original JSONL file and line number remain in the provenance fields.

### Split policy

A single uploaded Dolly JSONL file is one document, so a document-disjoint split would put all examples into the same partition. v0.13.1 therefore assigns each unique JSONL record a stable source-record ID derived from the uploaded document ID, line number and message digest. A deterministic hash of that ID and the split seed assigns the example to train, validation or test. The manifest records:

- `source_unit_kind: "record"`
- `source_units`: per-split record IDs, which are checked for overlap
- `source_documents`: original upload document IDs for provenance; a single ID may correctly appear in all three splits
- `source_disjoint: true`: means source-unit disjointness; for Dolly the unit is a record, not an entire uploaded document

For Ollama-generated Q/A from ordinary documents, the source unit remains a document and split assignment is document-disjoint.

## Windows operation

1. Stop the old API and frontend terminals with `Ctrl+C`. The old synchronous request does not migrate into the new job system; it may be abandoned when the old server stops.
2. Back up `knowledge_store` before overlaying the patch. Do not delete `knowledge_store/uploads` or `knowledge_store/knowledge.sqlite3`.
3. Extract the v0.13.1 patch into the existing project root, preserving relative paths and allowing file overwrite, or use the full v0.13.1 project ZIP.
4. Start the API in PowerShell:

```powershell
.\run_phase7.ps1
```

The first run creates `.venv-phase7` and installs only the Knowledge Engine/API dependencies (`requirements-knowledge.txt`), not PyTorch. This keeps document ingestion and Dolly JSONL import independent of the blocked `torch\lib\shm.dll` seen in the separate training environment. On subsequent starts, use `-SkipInstall`.

5. Start the dashboard in a second PowerShell terminal:

```powershell
Set-Location .\web
npm run dev
```

6. Open `http://127.0.0.1:3000/knowledge`, refresh the document list, and click **Build fine-tuning dataset** again.

## How to read job progress

| Field | Meaning |
|---|---|
| `stage` | `starting`, `importing_structured_records`, `preparing_grounded_generation`, `generating_grounded_pairs`, `finalizing_dataset`, or terminal state |
| `processed_records` | Number of structured JSONL records processed for direct import |
| `processed_chunks` / `total_chunks` | Chunks processed / selected for Ollama Q/A generation |
| `accepted_pairs` | Records or generated Q/A examples retained after validation |
| `rejected_pairs` | Generated examples rejected for duplicate, unsupported evidence, length, or PII reasons |
| `result` | Output directory, train/validation/test counts, split type, and generation mode |
| `error` | Structured failure type and concise message |

If the API process restarts while a generation job is still running, the job is reported as `interrupted`; start a new job. Completed dataset directories are retained.

## Quick checks

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/knowledge/health
Invoke-RestMethod http://127.0.0.1:8010/api/knowledge/documents
Invoke-RestMethod http://127.0.0.1:8010/api/knowledge/datasets
```

For direct Dolly import, neither Ollama nor PyTorch is required for the dataset-building step. The `model` request field remains in the common API schema because the same endpoint also supports grounded generation from unlabelled documents. Fine-tuning later still requires the separate training dependencies and Hugging Face model downloads.

## Limits and caveats

- Direct import currently reads retained JSONL uploads only. CSV/JSON supervised data import is not included in this fix.
- The direct importer supports common Q/A schemas, not arbitrary column mapping or automatic label repair.
- Record-level split boundaries stop exact record reuse across splits. They do not guarantee no semantic similarity between examples from the same upstream dataset; deduplication is exact-message based in this release.
- Ollama generation from unlabelled documents still requires four distinct source documents for document-disjoint train/validation/test.
- Background job state is stored on disk but execution is in-process; keep a single local API worker. No distributed worker queue or job cancellation is included.
- The local API has no authentication. Keep it bound to `127.0.0.1`.
