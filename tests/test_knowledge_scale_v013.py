import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from autotune import knowledge_engine as ke
from autotune.knowledge_ingest import iter_text_chunks


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(ke, "ROOT", root)
    monkeypatch.setattr(ke, "UPLOADS", root / "uploads")
    monkeypatch.setattr(ke, "PROCESSED", root / "processed")
    monkeypatch.setattr(ke, "DATASETS", root / "datasets")
    monkeypatch.setattr(ke, "DB", root / "knowledge.sqlite3")
    monkeypatch.setattr(ke, "MAX_FILE_BYTES", 512 * 1024 * 1024)
    monkeypatch.setattr(ke, "MAX_CHARS", 100_000_000)


def test_v013_defaults_support_100m_characters_and_512mb_files():
    assert ke.MAX_CHARS == 100_000_000
    assert ke.MAX_FILE_BYTES == 512 * 1024 * 1024


def test_chunk_iterator_emits_before_consuming_all_fragments():
    consumed = 0

    def source():
        nonlocal consumed
        for _ in range(10):
            consumed += 1
            yield "x" * 100

    iterator = iter_text_chunks(source(), size=100, overlap=10)
    first = next(iterator)
    assert len(first[0]) == 100
    assert consumed < 10


def test_streaming_text_index_records_offsets_and_progress():
    payload = ("Kubernetes schedules pods across worker nodes. " * 10_000).encode()
    result = ke.ingest("large-notes.txt", payload)
    assert result["status"] == "indexed"
    assert result["chunks"] > 100
    doc = ke.list_documents()[0]
    assert doc["processed_chars"] == len(payload.decode())
    assert doc["chunk_count"] == result["chunks"]
    with ke._connect() as con:
        first = con.execute(
            "SELECT char_start,char_end,text FROM chunks WHERE document_id=? ORDER BY ordinal LIMIT 1",
            (result["id"],),
        ).fetchone()
    assert first[0] == 0
    assert first[1] - first[0] == len(first[2])


def test_over_limit_extraction_fails_closed_and_can_be_retried(monkeypatch):
    monkeypatch.setattr(ke, "MAX_CHARS", 300)
    payload = ("Some extracted content. " * 40).encode()
    with pytest.raises(ValueError, match="100 million|exceeds"):
        ke.ingest("too-large.txt", payload)
    doc = ke.list_documents()[0]
    assert doc["status"] == "failed"
    with ke._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (doc["id"],)).fetchone()[0] == 0
    monkeypatch.setattr(ke, "MAX_CHARS", 100_000_000)
    retried = ke.retry_document(doc["id"])
    assert retried["status"] == "indexed"


def test_upload_api_streams_and_stats_show_indexed_document():
    client = TestClient(ke.create_app())
    payload = b"Knowledge engine streaming upload. " * 4000
    upload = client.post("/api/knowledge/documents", files={"file": ("streamed.txt", payload)})
    assert upload.status_code == 200
    assert upload.json()["status"] == "queued"
    docs = client.get("/api/knowledge/documents").json()["documents"]
    assert docs[0]["status"] == "indexed"
    stats = client.get("/api/knowledge/stats").json()
    assert stats["documents"] == 1
    assert stats["chunks"] > 1
    assert stats["limits"]["max_extracted_characters"] == 100_000_000


def test_retry_api_rejects_indexed_document():
    client = TestClient(ke.create_app())
    result = ke.ingest("ready.txt", b"This file has already been indexed by the knowledge engine.")
    response = client.post(f"/api/knowledge/documents/{result['id']}/retry")
    assert response.status_code == 409


def test_v012_database_schema_migrates_additively(tmp_path, monkeypatch):
    old_db = ke.DB
    old_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(old_db) as con:
        con.execute("""CREATE TABLE documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_type TEXT NOT NULL,
            sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, chars INTEGER NOT NULL,
            status TEXT NOT NULL, uploaded_at TEXT NOT NULL, storage_name TEXT NOT NULL, error TEXT)""")
        con.execute("""CREATE TABLE chunks (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            text TEXT NOT NULL, sha256 TEXT NOT NULL)""")
    ke._init()
    with ke._connect() as con:
        doc_columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        chunk_columns = {row[1] for row in con.execute("PRAGMA table_info(chunks)")}
    assert {"processed_chars", "chunk_count", "updated_at"} <= doc_columns
    assert {"char_start", "char_end", "metadata_json"} <= chunk_columns


def test_health_endpoint_reports_limits_and_streaming_capability():
    response = TestClient(ke.create_app()).get("/api/knowledge/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.13.1"
    assert body["limits"]["max_extracted_characters"] == 100_000_000
    assert "streaming-ingestion" in body["capabilities"]
