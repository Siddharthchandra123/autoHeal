"""Memory-bounded document ingestion for AutoTune AI Knowledge Engine.

Uploads are staged to disk, extracted incrementally, chunked with bounded buffers,
and committed to SQLite/FTS5 in batches. The raw upload is retained for retry.
"""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import uuid
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_P = f"{{{W_NS}}}p"
W_T = f"{{{W_NS}}}t"
BATCH_CHUNKS = int(os.environ.get("AUTOTUNE_INGEST_BATCH_CHUNKS", "500"))
READ_BYTES = 1024 * 1024
JSON_SMALL_FILE_BYTES = 16 * 1024 * 1024
JSONL_MAX_LINE_CHARS = 10_000_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encoding(path: Path) -> str:
    """Validate UTF-8 incrementally; fall back to Windows-1252 without loading the file."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    try:
        with path.open("rb") as handle:
            while block := handle.read(READ_BYTES):
                decoder.decode(block, final=False)
            decoder.decode(b"", final=True)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def _text_chunks(path: Path) -> Iterator[str]:
    encoding = _encoding(path)
    with path.open("r", encoding=encoding, errors="strict", newline="") as handle:
        while block := handle.read(READ_BYTES):
            yield block


def _iter_docx(path: Path) -> Iterator[str]:
    """Stream paragraph text from the main DOCX document XML part."""
    try:
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                raise ValueError("DOCX is missing word/document.xml")
            with archive.open("word/document.xml") as stream:
                for event, element in ET.iterparse(stream, events=("end",)):
                    if element.tag == W_P:
                        paragraph = "".join(node.text or "" for node in element.iter(W_T))
                        if paragraph.strip():
                            yield paragraph + "\n"
                        element.clear()
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError("Invalid or corrupted DOCX file") from exc


def _iter_pdf(path: Path) -> Iterator[str]:
    from . import knowledge_engine as ke
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support is optional. Install requirements-knowledge.txt") from exc
    try:
        reader = PdfReader(str(path), strict=True)
        if len(reader.pages) > ke.MAX_PDF_PAGES:
            raise ValueError(f"PDF exceeds the {ke.MAX_PDF_PAGES:,}-page limit")
        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text:
                yield f"\n[Page {page_number}]\n{text}\n"
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {type(exc).__name__}") from exc


def _iter_json(path: Path) -> Iterator[str]:
    size = path.stat().st_size
    encoding = _encoding(path)
    if size <= JSON_SMALL_FILE_BYTES:
        with path.open("r", encoding=encoding) as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            for item in obj:
                yield json.dumps(item, ensure_ascii=False) + "\n"
        elif isinstance(obj, dict):
            yield json.dumps(obj, ensure_ascii=False, indent=2)
        else:
            yield json.dumps(obj, ensure_ascii=False)
        return

    try:
        import ijson
    except ImportError as exc:
        raise ValueError(
            "Large JSON extraction requires ijson. Install requirements-knowledge.txt, "
            "or convert this file to JSONL."
        ) from exc
    with path.open("rb") as handle:
        first = b""
        while not first:
            block = handle.read(1)
            if not block:
                break
            if not block.isspace():
                first = block
        handle.seek(0)
        if first == b"[":
            iterator = ijson.items(handle, "item")
            for item in iterator:
                yield json.dumps(item, ensure_ascii=False) + "\n"
        elif first == b"{":
            for key, value in ijson.kvitems(handle, ""):
                yield json.dumps({key: value}, ensure_ascii=False) + "\n"
        else:
            raise ValueError("Large JSON must have an object or array at its root")


def iter_extracted_fragments(filename: str, path: Path) -> Iterator[str]:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        yield from _text_chunks(path)
    elif ext == ".csv":
        csv.field_size_limit(100_000_000)
        with path.open("r", encoding=_encoding(path), newline="") as handle:
            for row in csv.reader(handle):
                yield " | ".join(cell.strip() for cell in row) + "\n"
    elif ext == ".jsonl":
        with path.open("r", encoding=_encoding(path)) as handle:
            for line_number, line in enumerate(handle, 1):
                if len(line) > JSONL_MAX_LINE_CHARS:
                    raise ValueError(f"JSONL line {line_number} exceeds {JSONL_MAX_LINE_CHARS:,} characters")
                if line.strip():
                    try:
                        yield json.dumps(json.loads(line), ensure_ascii=False) + "\n"
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
    elif ext == ".json":
        yield from _iter_json(path)
    elif ext == ".pdf":
        yield from _iter_pdf(path)
    elif ext == ".docx":
        yield from _iter_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def iter_text_chunks(fragments: Iterable[str], size: int = 1200, overlap: int = 160):
    """Yield (chunk, normalized_start, normalized_end) using a bounded buffer."""
    if not 100 <= size <= 10_000 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap")
    buffer = ""
    base_offset = 0
    saw_content = False

    def emit(force: bool):
        nonlocal buffer, base_offset
        while len(buffer) > size or (force and buffer):
            limit = min(size, len(buffer))
            end = limit
            if limit == size and len(buffer) > size:
                boundary = max(buffer.rfind("\n", 0, limit), buffer.rfind(". ", 0, limit),
                               buffer.rfind(" ", 0, limit))
                if boundary > size // 2:
                    end = boundary + (1 if buffer[boundary:boundary + 1] == "\n" else 1)
            raw = buffer[:end]
            leading = len(raw) - len(raw.lstrip())
            content = raw.strip()
            if content:
                start_at = base_offset + leading
                yield content, start_at, start_at + len(content)
            if end >= len(buffer):
                base_offset += len(buffer)
                buffer = ""
                break
            advance = max(1, end - overlap)
            base_offset += advance
            buffer = buffer[advance:]

    for fragment in fragments:
        if not isinstance(fragment, str) or not fragment:
            continue
        fragment = fragment.replace("\x00", "")
        fragment = re.sub(r"[ \t]+", " ", fragment)
        fragment = re.sub(r"\n{3,}", "\n\n", fragment)
        if fragment.strip():
            saw_content = True
        buffer += fragment
        yield from emit(False)
    yield from emit(True)
    if not saw_content:
        return


def register_staged_upload(filename: str, staged_path: Path, size: int, digest: str) -> dict:
    from . import knowledge_engine as ke

    ke._init()
    safe_name = str(filename or "upload.txt").replace("\\", "/").split("/")[-1]
    ext = Path(safe_name).suffix.lower()
    if ext not in ke.ALLOWED:
        staged_path.unlink(missing_ok=True)
        raise ValueError(f"Allowed formats: {', '.join(sorted(ke.ALLOWED))}")
    if size <= 0 or size > ke.MAX_FILE_BYTES:
        staged_path.unlink(missing_ok=True)
        raise ValueError(f"File must be between 1 byte and {ke.MAX_FILE_BYTES // (1024 * 1024)} MB")

    with ke._connect() as con:
        duplicate = con.execute("SELECT id,filename,status FROM documents WHERE sha256=?", (digest,)).fetchone()
    if duplicate:
        staged_path.unlink(missing_ok=True)
        return {"id": duplicate[0], "filename": duplicate[1], "status": duplicate[2], "duplicate": True}

    document_id = uuid.uuid4().hex
    stored_name = f"{document_id}{ext}"
    destination = ke.UPLOADS / stored_name
    os.replace(staged_path, destination)
    now = _now()
    try:
        with ke._connect() as con:
            con.execute(
                """INSERT INTO documents
                (id,filename,file_type,sha256,bytes,chars,status,uploaded_at,storage_name,error,
                 processed_chars,chunk_count,updated_at)
                VALUES (?,?,?,?,?,0,'queued',?,?,NULL,0,0,?)""",
                (document_id, safe_name, ext, digest, size, now, stored_name, now),
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"id": document_id, "filename": safe_name, "file_type": ext, "sha256": digest,
            "bytes": size, "characters": 0, "chunks": 0, "status": "queued", "duplicate": False}


def _insert_batch(document_id: str, filename: str, rows: list[tuple], processed_chars: int):
    from . import knowledge_engine as ke

    fts_rows = [(row[0], document_id, filename, row[3]) for row in rows]
    with ke._connect() as con:
        con.executemany(
            "INSERT INTO chunks(id,document_id,ordinal,text,sha256,char_start,char_end,metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?)", rows
        )
        con.executemany(
            "INSERT INTO chunks_fts(chunk_id,document_id,source_file,text) VALUES (?,?,?,?)", fts_rows
        )
        con.execute("UPDATE documents SET processed_chars=?,chunk_count=?,updated_at=? WHERE id=?",
                    (processed_chars, con.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()[0], _now(), document_id))


def process_document(document_id: str) -> dict:
    """Incrementally extract and index one already-registered upload."""
    from . import knowledge_engine as ke

    ke._init()
    with ke._connect() as con:
        row = con.execute("SELECT filename,file_type,storage_name,status FROM documents WHERE id=?", (document_id,)).fetchone()
    if not row:
        raise FileNotFoundError("Document not found")
    filename, ext, stored_name, status = row
    if status == "indexed":
        return {"id": document_id, "filename": filename, "status": status, "duplicate": True}
    source = (ke.UPLOADS / stored_name).resolve()
    if source.parent != ke.UPLOADS.resolve() or source.is_symlink() or not source.is_file():
        raise ValueError("Stored source file is missing or unsafe")

    with ke._connect() as con:
        con.execute("DELETE FROM chunks_fts WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        con.execute("UPDATE documents SET status='indexing',error=NULL,chars=0,processed_chars=0,chunk_count=0,updated_at=? WHERE id=?",
                    (_now(), document_id))

    total_chars = 0
    chunk_count = 0
    batch: list[tuple] = []
    try:
        def counted_fragments():
            nonlocal total_chars
            for fragment in iter_extracted_fragments(filename, source):
                total_chars += len(fragment)
                if total_chars > ke.MAX_CHARS:
                    raise ValueError(f"Extracted text exceeds {ke.MAX_CHARS:,} characters")
                yield fragment

        for text, char_start, char_end in iter_text_chunks(counted_fragments(), ke.CHUNK_SIZE, ke.CHUNK_OVERLAP):
            chunk_id = uuid.uuid4().hex
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            batch.append((chunk_id, document_id, chunk_count, text, digest, char_start, char_end, "{}"))
            chunk_count += 1
            if len(batch) >= max(10, min(2000, BATCH_CHUNKS)):
                _insert_batch(document_id, filename, batch, total_chars)
                batch = []
        if batch:
            _insert_batch(document_id, filename, batch, total_chars)
        if chunk_count == 0:
            raise ValueError("No extractable text found. Scanned PDFs need OCR before upload.")
        with ke._connect() as con:
            con.execute("UPDATE documents SET status='indexed',chars=?,processed_chars=?,chunk_count=?,error=NULL,updated_at=? WHERE id=?",
                        (total_chars, total_chars, chunk_count, _now(), document_id))
        return {"id": document_id, "filename": filename, "file_type": ext,
                "characters": total_chars, "chunks": chunk_count, "status": "indexed", "duplicate": False}
    except Exception as exc:
        with ke._connect() as con:
            con.execute("DELETE FROM chunks_fts WHERE document_id=?", (document_id,))
            con.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            con.execute("UPDATE documents SET status='failed',chars=?,processed_chars=?,chunk_count=0,error=?,updated_at=? WHERE id=?",
                        (total_chars, total_chars, f"{type(exc).__name__}: {str(exc)[:900]}", _now(), document_id))
        raise


def ingest_bytes(filename: str, data: bytes) -> dict:
    from . import knowledge_engine as ke

    ke.UPLOADS.mkdir(parents=True, exist_ok=True)
    if not data:
        raise ValueError(f"File must be between 1 byte and {ke.MAX_FILE_BYTES // (1024 * 1024)} MB")
    if len(data) > ke.MAX_FILE_BYTES:
        raise ValueError(f"File must be between 1 byte and {ke.MAX_FILE_BYTES // (1024 * 1024)} MB")
    staged = ke.UPLOADS / f"{uuid.uuid4().hex}.uploading"
    digest = hashlib.sha256(data).hexdigest()
    try:
        staged.write_bytes(data)
        result = register_staged_upload(filename, staged, len(data), digest)
        if result.get("duplicate"):
            return result
        indexed = process_document(result["id"])
        return {**result, **indexed, "duplicate": False}
    finally:
        staged.unlink(missing_ok=True)
