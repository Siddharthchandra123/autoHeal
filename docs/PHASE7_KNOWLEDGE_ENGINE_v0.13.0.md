# AutoTune AI v0.13.0 - Knowledge Engine Scale & Reliability

**Release checkpoint: 9 November 2026.** This release hardens Phase 7 for large document collections. It adds disk-staged streaming uploads, incremental extraction, bounded chunking, batched SQLite/FTS writes, ingestion progress and retry, source character offsets, additive database migration, and memory-bounded semantic indexing/search.

## 1. What's new in v0.13.0

| Area | v0.12.0 behavior | v0.13.0 behavior |
|---|---|---|
| Upload limit | 20 MiB | 512 MiB default; environment-configurable |
| Extracted-text limit | 5 million chars | 100 million chars default; environment-configurable |
| HTTP upload | One full `UploadFile.read()` into RAM | 1 MiB blocks staged to disk with incremental SHA-256 |
| Extraction | Complete extracted string | Incremental TXT/MD, CSV, JSONL, large JSON, PDF-page and DOCX-paragraph extraction |
| Chunking | Complete `list[str]` | Bounded rolling buffer and iterator |
| Persistence | One transaction for all chunks | 500-chunk batches by default; progress persisted per batch |
| Recovery | Failed document could require re-upload | Retained original upload and retry endpoint |
| Traceability | Document ID and chunk number | Also normalized source character start/end offsets |
| Semantic indexing | All missing chunks fetched at once | SQLite cursor + bounded encoder batches |
| Semantic search | All vectors loaded and scored in RAM | Cursor batches + exact top-k heap; vector SHA-256 checked |
| Dashboard | Static 20 MiB label | API-reported limits, document states, progress and retry action |

## 2. Resource limits and configuration

Defaults:

```text
AUTOTUNE_MAX_FILE_BYTES=536870912       # 512 MiB
AUTOTUNE_MAX_CHARS=100000000             # 100 million extracted characters
AUTOTUNE_CHUNK_SIZE=1200
AUTOTUNE_CHUNK_OVERLAP=160
AUTOTUNE_MAX_PDF_PAGES=10000
AUTOTUNE_INGEST_BATCH_CHUNKS=500
```

The Python process reads these variables at startup. `.env.example` documents the settings; it is not automatically loaded by Python. To override values in PowerShell before starting the API:

```powershell
$env:AUTOTUNE_MAX_FILE_BYTES = '1073741824' # 1 GiB
$env:AUTOTUNE_MAX_CHARS = '150000000'
$env:AUTOTUNE_CHUNK_SIZE = '1200'
$env:AUTOTUNE_CHUNK_OVERLAP = '160'
$env:AUTOTUNE_INGEST_BATCH_CHUNKS = '500'

.\run_phase7.ps1 -SkipInstall
```

The upload-size and extracted-character limits are independent. A compressed PDF or DOCX can expand substantially after extraction; the extracted-character cap is checked while fragments are consumed. Do not raise these caps without checking free disk space and SQLite/WAL storage.

### File-format handling

- TXT/Markdown: UTF-8 (BOM-aware), with Windows-1252 fallback after an incremental validation pass; reads in 1 MiB blocks.
- CSV: row-wise parsing; individual fields can be up to 100 million characters.
- JSONL: record-wise parsing; a single line is limited to 10 million characters to avoid a pathological one-record allocation.
- JSON: files up to 16 MiB use the standard JSON parser. Larger JSON files use `ijson` and should have a root array or root object. Large JSON arrays are emitted one element at a time; root object members are emitted as one-key objects.
- PDF: page-wise extraction, default cap 10,000 pages. Scanned/image-only PDFs still require OCR; OCR is not built in.
- DOCX: paragraph-wise extraction from `word/document.xml`; headers, footers, embedded objects and images are not extracted in this pass.

`ijson` is included in `requirements-knowledge.txt` for large JSON. PDF and DOCX support uses `pypdf` and `python-docx` dependencies already present in that file; the DOCX v0.13 path uses the ZIP/XML stream reader.

## 3. Ingestion lifecycle

The normal HTTP route stages the upload to `knowledge_store/uploads/<uuid>.uploading` using 1 MiB blocks while calculating SHA-256. It checks the byte cap before accepting more data. After hash-based duplicate detection, the staged file is atomically renamed to a generated UUID filename and a document row is created with `queued` status.

FastAPI background processing changes the status to `indexing`, streams extracted fragments into a rolling chunker, and inserts each bounded chunk batch into SQLite and FTS5. The original uploaded file is retained. At successful completion, status becomes `indexed`. If extraction or indexing fails, partial chunk/FTS records are removed and status becomes `failed`; the error message is retained for diagnostics and retry.

Possible document states:

- `queued`: upload accepted, waiting for the background worker.
- `indexing`: extraction/chunking and batched database writes are active.
- `indexed`: complete and available to retrieval.
- `failed`: processing failed; inspect `error` and retry after correcting the cause.

A process crash can leave a document in `indexing`. The same retry endpoint can reprocess `queued`, `indexing` or `failed` documents. Retry deletes partial chunk rows and rebuilds them from the retained original file.

## 4. API additions and changes

Base URL: `http://127.0.0.1:8010`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/knowledge/health` | Version, capability list and configured resource caps |
| GET | `/api/knowledge/stats` | Document, extracted-character, chunk and embedding totals; counts by status |
| POST | `/api/knowledge/documents` | Streaming upload; returns accepted document ID and initial status |
| GET | `/api/knowledge/documents` | Documents including progress, chunk count, status and error |
| POST | `/api/knowledge/documents/{document_id}/retry` | Queue a failed/incomplete document for re-indexing |
| POST | `/api/knowledge/index` | Explicit CPU semantic indexing in bounded batches |
| POST | `/api/knowledge/search` | Lexical, semantic, hybrid or automatic retrieval |
| POST | `/api/knowledge/ask` | Grounded local Ollama Q&A with exact quote checks |

Example health response includes the effective limits, so the dashboard can display runtime settings rather than relying only on hardcoded labels.

### `POST /api/knowledge/documents`

Multipart field: `file`. The endpoint returns an initial result similar to:

```json
{
  "id": "<32-character-document-id>",
  "filename": "architecture.md",
  "bytes": 10500321,
  "characters": 0,
  "chunks": 0,
  "status": "queued",
  "duplicate": false
}
```

Processing occurs in FastAPI background work after the response. Poll `GET /api/knowledge/documents` or `/api/knowledge/stats` until the document becomes `indexed` or `failed`.

### Retry

```powershell
$body = @{ } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8010/api/knowledge/documents/<document-id>/retry'
```

A retry is accepted only for `failed`, `queued` or `indexing` states. Indexed documents cannot be retried through this action.

## 5. Retrieval scalability and provenance

Semantic indexing no longer loads every pending chunk into one Python list. A SQLite cursor returns a bounded batch, the CPU encoder processes it, and vectors are committed to SQLite. Semantic query evaluation streams the vector table and maintains only the best `top_k` results in a heap. Each stored vector digest is checked before scoring; rows with incorrect dimension, invalid values or failed digest checks are skipped.

Semantic search remains exact brute-force cosine scoring. Its memory footprint is bounded, but query latency still grows linearly with the number of indexed chunks. For very large corpora, a later release should add an approximate-nearest-neighbor backend (for example FAISS/HNSW) while preserving the same chunk IDs and source provenance.

Each chunk stores `char_start` and `char_end` offsets in the normalized extracted text. These offsets are useful for source navigation and audit; they are not original PDF page coordinates or byte offsets. Search results and verified RAG citations return the chunk index and character offsets when available.

## 6. Windows setup

From VS Code PowerShell in the extracted `AutoTune_AI` folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_phase7.ps1
```

For the separate frontend process, use a second terminal:

```powershell
Set-Location .\web
npm install
npm run dev
```

Keep both terminals open. API: `http://127.0.0.1:8010/docs`. Dashboard: `http://127.0.0.1:3000/knowledge`.

## 7. Tests and verification boundary

The v0.13.0 build environment ran:

```powershell
PYTHONPATH=. pytest -q
```

Result: **346 passed**. Added tests cover default resource limits, incremental chunk emission, source offsets, oversized extraction fail-closed behavior, retry, upload API/statistics, retry rejection for indexed documents, additive migration from the v0.12 schema and health capability reporting.

The dashboard TSX was transpiled with the installed TypeScript compiler and reported zero syntax diagnostics. A full Next.js typecheck/build and browser acceptance are not claimed because this package build did not install the frontend dependencies. A real 100-million-character upload, live Ollama generation, live sentence-transformer downloads/indexing and model fine-tuning were not run in this environment.

## 8. Operational and security notes

- Back up the entire `knowledge_store` directory before upgrades. SQLite uses WAL; stop the API cleanly before manually copying the database and its `-wal`/`-shm` companions.
- Keep this unauthenticated development API bound to `127.0.0.1`. Do not expose it to a LAN or public network.
- A 100M-character corpus can create tens of thousands of chunks and substantial SQLite/FTS/vector storage. Keep sufficient free disk; monitor the `knowledge_store` directory.
- The API's synchronous Python compatibility helper `ingest(filename, data)` still accepts an in-memory `bytes` value for tests/scripts. The HTTP upload route is the disk-staged streaming path.
- A 512 MiB multipart upload may consume temporary disk space in the web server's upload spool and a second copy in the knowledge store. Keep free disk headroom above the raw file size.
- PDF text extraction is page-wise but each extracted page is materialized; a single extremely large page can still consume memory.
- CSV rows and JSONL lines are parsed as individual records; JSONL lines have a 10M-character cap.
- SQLite FTS5 and semantic exact-scan are local single-node components, not a multi-tenant or distributed vector service.
- This release does not add OCR, comprehensive PII/secret detection, formal semantic entailment, authentication or production multi-user controls.

**Release statement:** v0.13.0 makes the ingestion/indexing code path memory-bounded for common large text corpora and adds recovery/provenance support. The configured 100M-character limit is not a claim that every 100M-character format or compressed document will process within a particular RAM/time budget. Validate with representative documents on the target Windows machine before relying on the ceiling.
