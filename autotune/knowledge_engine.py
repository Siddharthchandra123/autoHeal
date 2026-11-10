"""AutoTune AI v0.13.0 scalable local Knowledge Engine.

Provides upload/ingestion, SQLite FTS5 search, optional semantic indexing, grounded
local-Ollama RAG and SFT dataset generation, plus guarded sequential training jobs.
This unauthenticated development API must remain bound to loopback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get('AUTOTUNE_KNOWLEDGE_HOME', 'knowledge_store')).resolve()
UPLOADS = ROOT / 'uploads'
PROCESSED = ROOT / 'processed'
DB = ROOT / 'knowledge.sqlite3'
MAX_FILE_BYTES = int(os.environ.get('AUTOTUNE_MAX_FILE_BYTES', str(512 * 1024 * 1024)))
DATASETS = ROOT / 'datasets'
MAX_CHARS = int(os.environ.get('AUTOTUNE_MAX_CHARS', '100000000'))
ALLOWED = {'.txt', '.md', '.csv', '.json', '.jsonl', '.pdf', '.docx'}
CHUNK_SIZE = int(os.environ.get('AUTOTUNE_CHUNK_SIZE', '1200'))
CHUNK_OVERLAP = int(os.environ.get('AUTOTUNE_CHUNK_OVERLAP', '160'))
MAX_PDF_PAGES = int(os.environ.get('AUTOTUNE_MAX_PDF_PAGES', '10000'))


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB, timeout=30)
    con.execute('PRAGMA busy_timeout=30000')
    con.execute('PRAGMA foreign_keys=ON')
    return con


def _init() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('''CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_type TEXT NOT NULL,
            sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, chars INTEGER NOT NULL,
            status TEXT NOT NULL, uploaded_at TEXT NOT NULL, storage_name TEXT NOT NULL,
            error TEXT, processed_chars INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)''')
        con.execute('CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(sha256)')
        con.execute('''CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            text TEXT NOT NULL, sha256 TEXT NOT NULL,
            char_start INTEGER NOT NULL DEFAULT 0, char_end INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(document_id) REFERENCES documents(id))''')
        doc_columns = {row[1] for row in con.execute('PRAGMA table_info(documents)')}
        for name, declaration in (
            ('processed_chars', 'INTEGER NOT NULL DEFAULT 0'),
            ('chunk_count', 'INTEGER NOT NULL DEFAULT 0'),
            ('updated_at', 'TEXT'),
        ):
            if name not in doc_columns:
                con.execute(f'ALTER TABLE documents ADD COLUMN {name} {declaration}')
        chunk_columns = {row[1] for row in con.execute('PRAGMA table_info(chunks)')}
        for name, declaration in (
            ('char_start', 'INTEGER NOT NULL DEFAULT 0'),
            ('char_end', 'INTEGER NOT NULL DEFAULT 0'),
            ('metadata_json', "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if name not in chunk_columns:
                con.execute(f'ALTER TABLE chunks ADD COLUMN {name} {declaration}')
        con.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_ordinal ON chunks(document_id,ordinal)')
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, source_file, text, tokenize='unicode61')")


def make_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    '''Compatibility helper; production ingestion uses the streaming iterator.'''
    from .knowledge_ingest import iter_text_chunks
    return [row[0] for row in iter_text_chunks((text,), size=size, overlap=overlap)]


def ingest(filename: str, data: bytes) -> dict[str, Any]:
    '''Synchronous compatibility wrapper used by tests and local scripts.'''
    from .knowledge_ingest import ingest_bytes
    return ingest_bytes(filename, data)


def retry_document(document_id: str) -> dict[str, Any]:
    _init()
    if not re.fullmatch(r'[0-9a-f]{32}', document_id):
        raise ValueError('Invalid document ID')
    with _connect() as con:
        row = con.execute('SELECT status FROM documents WHERE id=?', (document_id,)).fetchone()
    if not row:
        raise FileNotFoundError('Document not found')
    if row[0] not in {'failed', 'queued', 'indexing'}:
        raise ValueError('Only failed or incomplete documents can be retried')
    from .knowledge_ingest import process_document
    return process_document(document_id)


def list_documents() -> list[dict]:
    _init()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute(
            'SELECT d.id,d.filename,d.file_type,d.sha256,d.bytes,d.chars,d.status,d.uploaded_at,d.error, '
            'd.processed_chars,d.updated_at,d.chunk_count '
            'FROM documents d ORDER BY d.uploaded_at DESC')]


def export_corpus(path: str | Path) -> dict[str, Any]:
    """Export source-grounded passages as JSONL; not yet instruction/answer pairs."""
    _init()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con, target.open('w', encoding='utf-8') as out:
        rows = con.execute('''SELECT d.id,d.filename,c.ordinal,c.text,c.sha256,c.char_start,c.char_end
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.status='indexed' ORDER BY d.uploaded_at,c.ordinal''')
        count = 0
        for doc_id, filename, ordinal, chunk, digest, char_start, char_end in rows:
            out.write(json.dumps({'source_id': doc_id, 'source_file': filename,
                                  'chunk_index': ordinal, 'text': chunk, 'sha256': digest, 'char_start': char_start, 'char_end': char_end}, ensure_ascii=False) + '\n')
            count += 1
    return {'path': str(target), 'records': count, 'format': 'source_passages_jsonl',
            'fine_tuning_ready': False,
            'note': 'Passages require reviewed instruction/response generation and held-out splitting before SFT.'}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default='auto', pattern='^(auto|lexical|semantic|hybrid)$')
    model_name: str | None = Field(default=None, max_length=200)


class IndexRequest(BaseModel):
    model_name: str | None = Field(default=None, max_length=200)
    batch_size: int = Field(default=32, ge=1, le=128)


class DatasetRequest(BaseModel):
    model: str = Field(min_length=1, max_length=160)
    ollama_url: str = Field(default='http://127.0.0.1:11434', max_length=300)
    max_chunks: int = Field(default=100, ge=1, le=5000)
    min_pairs: int = Field(default=200, ge=4, le=100000)
    seed: int = Field(default=42, ge=0, le=2147483647)




class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str = Field(min_length=1, max_length=160)
    ollama_url: str = Field(default='http://127.0.0.1:11434', max_length=300)
    top_k: int = Field(default=5, ge=1, le=15)
    mode: str = Field(default='auto', pattern='^(auto|lexical|semantic|hybrid)$')

class TrainingJobRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=80)
    mode: str = Field(default='preflight', pattern='^(preflight|execute)$')
    allow_cpu: bool = False
    allow_small_data: bool = False
    max_candidates: int | None = Field(default=None, ge=1, le=3)


def create_app() -> FastAPI:
    _init()
    app = FastAPI(title='AutoTune AI Knowledge Engine', version='0.13.1', docs_url='/docs')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:3000','http://localhost:3000'],
                       allow_methods=['GET','POST'], allow_headers=['Content-Type'], allow_credentials=False)

    @app.get('/api/knowledge/health')
    def health():
        return {'status': 'ready', 'version': '0.13.1', 'mode': 'local-knowledge-rag-dataset-factory',
                'limits': {'max_file_bytes': MAX_FILE_BYTES, 'max_extracted_characters': MAX_CHARS, 'chunk_size': CHUNK_SIZE, 'chunk_overlap': CHUNK_OVERLAP},
                'capabilities': ['streaming-ingestion','batched-sqlite-indexing','lexical-search','memory-bounded-semantic-search','optional-semantic-search','grounded-sft-generation','background-dataset-jobs','direct-jsonl-sft-import','source-record-disjoint-splits']}

    @app.post('/api/knowledge/documents')
    async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        from .knowledge_ingest import register_staged_upload, process_document
        name = str(file.filename or 'upload.txt').replace('\\', '/').split('/')[-1]
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(415, 'Unsupported file type.')
        UPLOADS.mkdir(parents=True, exist_ok=True)
        staged = UPLOADS / f"{uuid.uuid4().hex}.uploading"
        digest = hashlib.sha256()
        size = 0
        try:
            with staged.open('wb') as out:
                while True:
                    block = await file.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_FILE_BYTES:
                        raise ValueError(f'File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB upload limit.')
                    digest.update(block)
                    out.write(block)
            result = register_staged_upload(name, staged, size, digest.hexdigest())
            if not result.get('duplicate'):
                background_tasks.add_task(process_document, result['id'])
            return result
        except ValueError as exc:
            staged.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get('/api/knowledge/stats')
    def stats():
        _init()
        with _connect() as con:
            docs = con.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0),COALESCE(SUM(chars),0) FROM documents').fetchone()
            chunks = con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            by_status = {row[0]: row[1] for row in con.execute('SELECT status,COUNT(*) FROM documents GROUP BY status')}
            try:
                embeddings = con.execute('SELECT COUNT(*) FROM chunk_embeddings').fetchone()[0]
            except sqlite3.OperationalError:
                embeddings = 0
        return {'documents': docs[0], 'stored_bytes': docs[1], 'extracted_characters': docs[2],
                'chunks': chunks, 'embeddings': embeddings, 'documents_by_status': by_status,
                'limits': {'max_file_bytes': MAX_FILE_BYTES, 'max_extracted_characters': MAX_CHARS}}

    @app.post('/api/knowledge/documents/{document_id}/retry')
    def retry(document_id: str, background_tasks: BackgroundTasks):
        if not re.fullmatch(r'[0-9a-f]{32}', document_id):
            raise HTTPException(400, 'Invalid document ID')
        with _connect() as con:
            row = con.execute('SELECT status FROM documents WHERE id=?', (document_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Document not found')
        if row[0] not in {'failed', 'queued', 'indexing'}:
            raise HTTPException(409, 'Only failed or incomplete documents can be retried')
        from .knowledge_ingest import process_document
        background_tasks.add_task(process_document, document_id)
        return {'id': document_id, 'status': 'queued'}

    @app.get('/api/knowledge/documents')
    def documents():
        return {'documents': list_documents()}

    @app.post('/api/knowledge/export')
    def export():
        target = PROCESSED / 'knowledge_passages.jsonl'
        return export_corpus(target)

    @app.post('/api/knowledge/search')
    def search(body: SearchRequest):
        try:
            from .retrieval import search as retrieve
            return retrieve(body.query, body.top_k, body.mode, body.model_name or __import__('autotune.retrieval', fromlist=['DEFAULT_EMBEDDING_MODEL']).DEFAULT_EMBEDDING_MODEL)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc


    @app.post('/api/knowledge/ask')
    def ask(body: AskRequest):
        try:
            from .rag import answer_question
            return answer_question(body.question, model=body.model, top_k=body.top_k,
                                   mode=body.mode, ollama_url=body.ollama_url)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/knowledge/index')
    def index(body: IndexRequest):
        try:
            from .retrieval import index_embeddings, DEFAULT_EMBEDDING_MODEL
            return index_embeddings(body.model_name or DEFAULT_EMBEDDING_MODEL, body.batch_size)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/knowledge/datasets/generate', status_code=202)
    def generate_dataset(body: DatasetRequest):
        """Queue a dataset build and return immediately; use the job route for progress."""
        try:
            from .dataset_jobs import create_job
            return create_job(DATASETS, model=body.model, ollama_url=body.ollama_url,
                              max_chunks=body.max_chunks, min_pairs=body.min_pairs, seed=body.seed)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/knowledge/datasets/generation-jobs/{job_id}')
    def dataset_generation_job(job_id: str):
        try:
            from .dataset_jobs import get_job
            return get_job(DATASETS, job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Dataset generation job not found') from exc

    @app.get('/api/knowledge/datasets/{dataset_id}/preview')
    def preview_dataset(dataset_id: str, split: str = 'train', limit: int = 10):
        if not re.fullmatch(r'dataset-[0-9]{8}-[0-9]{6}', dataset_id):
            raise HTTPException(400, 'Invalid dataset ID')
        if split not in {'train', 'validation', 'test'} or not 1 <= limit <= 50:
            raise HTTPException(400, 'split must be train/validation/test and limit must be 1..50')
        base = DATASETS.resolve()
        candidate_folder = DATASETS / dataset_id
        if candidate_folder.is_symlink():
            raise HTTPException(404, 'Dataset not found')
        folder = candidate_folder.resolve()
        if folder.parent != base or not folder.is_dir():
            raise HTTPException(404, 'Dataset not found')
        manifest_path = folder / 'dataset_manifest.json'
        data_path = folder / f'{split}.jsonl'
        if not manifest_path.is_file() or manifest_path.is_symlink() or not data_path.is_file() or data_path.is_symlink():
            raise HTTPException(404, 'Dataset split not found')
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            examples = []
            with data_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        examples.append(json.loads(line))
                    if len(examples) >= limit:
                        break
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(400, 'Dataset preview could not be read') from exc
        return {'dataset_id':dataset_id, 'split':split, 'limit':limit,
                'declared_count':manifest.get('counts',{}).get(split),
                'source_disjoint':manifest.get('source_disjoint'),
                'source_unit_kind':manifest.get('source_unit_kind', 'document'), 'examples':examples}

    @app.get('/api/knowledge/datasets')
    def datasets():
        DATASETS.mkdir(parents=True, exist_ok=True)
        items = []
        for folder in sorted(DATASETS.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True):
            manifest = folder / 'dataset_manifest.json'
            if folder.is_dir() and not folder.is_symlink() and manifest.is_file():
                try:
                    obj = json.loads(manifest.read_text(encoding='utf-8'))
                    items.append({'path':str(folder),'counts':obj.get('counts'),
                                  'source_disjoint':obj.get('source_disjoint'),
                                  'source_unit_kind':obj.get('source_unit_kind', 'document'),
                                  'created_at_utc':obj.get('created_at_utc')})
                except (OSError, json.JSONDecodeError):
                    continue
        return {'datasets':items}

    @app.post('/api/knowledge/training/jobs')
    def start_training_job(body: TrainingJobRequest):
        if not re.fullmatch(r'dataset-[0-9]{8}-[0-9]{6}', body.dataset_id):
            raise HTTPException(400, 'Invalid dataset ID')
        base = DATASETS.resolve()
        folder = (DATASETS / body.dataset_id).resolve()
        if folder.parent != base or folder.is_symlink() or not folder.is_dir():
            raise HTTPException(404, 'Dataset not found')
        try:
            from .knowledge_jobs import create_job
            return create_job(ROOT, dataset_dir=folder, mode=body.mode,
                              allow_cpu=body.allow_cpu, max_candidates=body.max_candidates,
                              allow_small_data=body.allow_small_data)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/knowledge/training/jobs/{job_id}')
    def training_job(job_id: str):
        try:
            from .knowledge_jobs import get_job
            return get_job(ROOT, job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Job not found') from exc

    return app


app = create_app()
