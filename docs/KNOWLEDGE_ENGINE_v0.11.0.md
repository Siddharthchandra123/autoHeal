# AutoTune AI Knowledge Engine v0.11.0 — Phase 7A

## Goal
Create a local, auditable knowledge ingestion layer that prepares source-grounded material for later RAG and supervised fine-tuning workflows.

## Implemented in this checkpoint
- Local FastAPI service: `autotune.knowledge_engine`.
- Upload formats: TXT, Markdown, CSV, JSON, JSONL; PDF and DOCX when optional parsers are installed.
- File-size limit 20 MB; extracted-text limit 5 million characters; PDF page limit 500.
- Filename path normalization, SHA-256 duplicate detection, random storage IDs, extension allowlist, and local SQLite metadata.
- Chunking with overlap; source document ID, original filename, chunk index and content digest retained.
- JSONL export of source passages for review and downstream dataset creation.
- No automatic answer invention, no training launch, no model promotion.

## Run on Windows PowerShell
From the `AutoTune_AI` project root:

```powershell
py -3.13 -m venv .venv-knowledge
.\.venv-knowledge\Scripts\python.exe -m pip install -r requirements-knowledge.txt
.\.venv-knowledge\Scripts\python.exe -m uvicorn autotune.knowledge_engine:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010/docs`. Upload through `POST /api/knowledge/documents`, list with `GET /api/knowledge/documents`, and export source passages with `POST /api/knowledge/export`.

By default data is stored under `knowledge_store/` in the current project directory. Set `AUTOTUNE_KNOWLEDGE_HOME` to a dedicated local folder before starting the service if desired. Do not expose this unauthenticated development service to a public network.

## API
- `GET /api/knowledge/health`
- `POST /api/knowledge/documents` — multipart field `file`
- `GET /api/knowledge/documents`
- `POST /api/knowledge/export` — writes `knowledge_store/processed/knowledge_passages.jsonl`

## Important data semantics
The export is a **source-passage corpus**, not an instruction-tuning dataset. It is explicitly marked `fine_tuning_ready: false`. To become SFT-ready, a subsequent phase must create task-specific user/assistant examples grounded in passages, retain source references, validate answers, detect duplicates and leakage, and reserve a held-out evaluation set. Human review may be required for generated examples, especially in high-stakes domains.

## Not implemented / not verified
- Semantic/vector search and RAG answering.
- OCR for scanned PDFs or image extraction from documents.
- Automatic question-answer generation.
- Response-only SFT data loader integration.
- Multi-model autonomous fine-tuning orchestration.
- Dashboard UI and live Windows upload verification.
- Authentication, multi-user isolation, cloud/object storage, malware scanning and enterprise deployment.
- PDF/DOCX parser execution in this build environment unless optional dependencies are installed.

## Offline verification
`python -m pytest -q tests/test_knowledge_engine_v011.py` — 6 passed in the build environment. These tests cover TXT/JSONL ingestion, chunk creation, duplicate detection, input rejection, traceability and export semantics. They do not constitute a live browser upload or training benchmark.
