"""Hybrid retrieval over Knowledge Engine chunks.

Lexical search uses SQLite FTS5 and is available offline. Semantic search is an
opt-in enhancement using sentence-transformers; no embedding model is downloaded
implicitly. Embeddings are stored in SQLite with model/version provenance.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from functools import lru_cache
import os
import sqlite3
from pathlib import Path
from typing import Any

from . import knowledge_engine as ke

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "AUTOTUNE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def _init() -> None:
    ke._init()
    with ke._connect() as con:
        try:
            con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED, document_id UNINDEXED, source_file, text,
                tokenize='unicode61')""")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite FTS5 is required for local retrieval") from exc
        con.execute("""CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT NOT NULL, model_name TEXT NOT NULL, dimension INTEGER NOT NULL,
            vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL,
            PRIMARY KEY(chunk_id, model_name))""")
        count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if not count:
            con.execute("""INSERT INTO chunks_fts(chunk_id,document_id,source_file,text)
                SELECT c.id,c.document_id,d.filename,c.text
                FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE d.status='indexed'""")


def _tokens(text: str) -> list[str]:
    return [x for x in __import__('re').findall(r"[\w'-]{2,}", text.lower())]


def lexical_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    _init()
    terms = _tokens(query)
    if not terms:
        return []
    # Quote all user terms so FTS syntax cannot be injected.
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:24])
    with ke._connect() as con:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("""SELECT f.chunk_id,f.document_id,f.source_file,
                f.text,bm25(chunks_fts) AS rank,c.ordinal,c.char_start,c.char_end
                FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id
                JOIN documents d ON d.id=c.document_id
                WHERE chunks_fts MATCH ? AND d.status='indexed' ORDER BY rank LIMIT ?""", (match, max(1,min(50,top_k)))).fetchall()
        except sqlite3.OperationalError:
            return []
    results = []
    for row in rows:
        results.append({"chunk_id": row["chunk_id"], "document_id": row["document_id"],
                        "source_file": row["source_file"], "chunk_index": row["ordinal"],
                        "text": row["text"], "char_start": row["char_start"], "char_end": row["char_end"],
                        "score": float(-row["rank"]), "method": "fts5"})
    return results


@lru_cache(maxsize=1)
def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic retrieval requires: pip install -r requirements-knowledge-ml.txt") from exc
    return SentenceTransformer(model_name, device="cpu")


def index_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32) -> dict:
    """Index missing chunks in bounded batches; memory use is O(batch_size)."""
    if not isinstance(model_name, str) or not model_name or len(model_name) > 200:
        raise ValueError("Invalid embedding model name")
    _init()
    import numpy as np
    batch_size = max(1, min(128, int(batch_size)))
    model = _load_embedder(model_name)
    with ke._connect() as read_con:
        total = read_con.execute("""SELECT COUNT(*) FROM chunks c
            LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id AND e.model_name=?
            JOIN documents d ON d.id=c.document_id WHERE e.chunk_id IS NULL AND d.status='indexed'""",
            (model_name,)).fetchone()[0]
        cursor = read_con.execute("""SELECT c.id,c.text FROM chunks c
            JOIN documents d ON d.id=c.document_id
            LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id AND e.model_name=?
            WHERE e.chunk_id IS NULL AND d.status='indexed' ORDER BY c.rowid""", (model_name,))
        inserted = 0
        while batch := cursor.fetchmany(batch_size):
            vectors = model.encode([r[1] for r in batch], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            rows = []
            for (chunk_id, _), vector in zip(batch, vectors):
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim != 1 or not np.all(np.isfinite(array)):
                    continue
                packed = array.tobytes()
                rows.append((chunk_id, model_name, int(len(array)), packed, hashlib.sha256(packed).hexdigest()))
            if rows:
                with ke._connect() as write_con:
                    write_con.executemany("INSERT OR REPLACE INTO chunk_embeddings VALUES (?,?,?,?,?)", rows)
                inserted += len(rows)
    remaining = max(0, total - inserted)
    return {"model": model_name, "indexed": inserted, "remaining": remaining,
            "total_missing_at_start": total, "device": "cpu",
            "note": "Embedding indexing streams SQLite rows in bounded batches; vectors remain local."}


def semantic_search(query: str, top_k: int = 5, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[dict]:
    """Exact cosine search with bounded memory using a streaming top-k heap."""
    _init()
    import numpy as np
    top_k = max(1, min(50, int(top_k)))
    with ke._connect() as con:
        sample = con.execute("SELECT dimension FROM chunk_embeddings WHERE model_name=? LIMIT 1",
                             (model_name,)).fetchone()
        if not sample:
            raise RuntimeError("No embeddings indexed for this model. Call /api/knowledge/index first.")
        cursor = con.execute("""SELECT c.id,c.document_id,d.filename,c.ordinal,c.text,c.char_start,c.char_end,
            e.dimension,e.vector,e.vector_sha256 FROM chunk_embeddings e
            JOIN chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.document_id
            WHERE e.model_name=? AND d.status='indexed' ORDER BY c.rowid""", (model_name,))
        model = _load_embedder(model_name)
        q = np.asarray(model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0], dtype=np.float32)
        heap: list[tuple[float, int, dict]] = []
        sequence = 0
        while rows := cursor.fetchmany(512):
            for row in rows:
                vector_blob = row[8]
                if row[7] != len(q) or hashlib.sha256(vector_blob).hexdigest() != row[9]:
                    continue
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                score = float(np.dot(q, vector))
                if not math.isfinite(score):
                    continue
                item = {"chunk_id": row[0], "document_id": row[1], "source_file": row[2],
                        "chunk_index": row[3], "text": row[4], "char_start": row[5], "char_end": row[6], "score": score,
                        "method": "sentence-transformers"}
                sequence += 1
                entry = (score, sequence, item)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        return [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]


def search(query: str, top_k: int = 5, mode: str = "auto", model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("Query must contain 1..4000 characters")
    if mode not in {"auto", "lexical", "semantic", "hybrid"}:
        raise ValueError("mode must be auto, lexical, semantic, or hybrid")
    lexical = lexical_search(query, top_k)
    if mode == "lexical":
        return {"query": query, "mode": "lexical", "results": lexical}
    semantic = []
    semantic_error = None
    if mode in {"auto", "semantic", "hybrid"}:
        try:
            semantic = semantic_search(query, top_k, model_name)
        except Exception as exc:
            semantic_error = f'{type(exc).__name__}: {str(exc)[:240]}'
            if mode == "semantic":
                raise RuntimeError(semantic_error) from exc
    if mode == "semantic":
        return {"query": query, "mode": "semantic", "results": semantic}
    if not semantic:
        return {"query": query, "mode": "lexical-fallback",
                "results": lexical, "semantic_available": False, "semantic_note": semantic_error}
    # Reciprocal rank fusion is robust to incomparable raw FTS/cosine scores.
    fused: dict[str, dict] = {}
    for rank, row in enumerate(lexical, 1):
        fused[row["chunk_id"]] = {**row, "fusion": 1 / (60 + rank)}
    for rank, row in enumerate(semantic, 1):
        current = fused.setdefault(row["chunk_id"], {**row, "fusion": 0.0})
        current["fusion"] += 1 / (60 + rank)
        current["semantic_score"] = row["score"]
    final = sorted(fused.values(), key=lambda x: x["fusion"], reverse=True)[:max(1,min(50,top_k))]
    return {"query": query, "mode": "hybrid", "results": final, "semantic_available": True}
