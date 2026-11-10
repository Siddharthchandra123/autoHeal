"""Evidence-grounded instruction dataset factory backed by a local Ollama model.

Generated answers are accepted only when the proposed evidence quote is an exact
substring of a retrieved source chunk and the answer has lexical support in that
quote. Splits are source-document-disjoint to reduce memorization leakage.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import knowledge_engine as ke
from .retrieval import lexical_search

STOP = {"the","and","for","with","that","this","from","what","when","where","which","who","why","how","are","is","was","were","can","could","should","would","will","into","their","there","have","has","had","does","did","not","you","your","about","than","then","also","such","use","used","using","more","most","some","any","all","each","other","one","two","three","a","an","of","to","in","on","at","by","as","or","be","it","its"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _content_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9._/-]{1,}", s.lower()) if t not in STOP}


def _local_ollama_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Only a local Ollama HTTP endpoint on localhost is allowed")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL cannot contain credentials, query or fragment")
    return url.rstrip("/") + "/api/chat"


def _ollama_json(model: str, context: str, ollama_url: str, timeout: int = 180) -> list[dict]:
    prompt = f"""Create at most 5 high-quality question/answer examples from ONLY the supplied source text.
Return JSON only with this schema: {{\"pairs\":[{{\"question\":\"...\",\"answer\":\"...\",\"evidence_quote\":\"an exact contiguous quote from the source\"}}]}}.
Rules: answer must be directly supported by evidence_quote; do not add outside facts; ask specific useful questions; answers should be concise and complete; no duplicates; do not mention that you are an AI. If the source is insufficient, return an empty pairs array.

SOURCE TEXT:
{context[:10000]}"""
    payload = {"model": model, "stream": False, "format": "json",
               "messages": [{"role": "user", "content": prompt}],
               "options": {"temperature": 0.1, "num_ctx": 8192}}
    req = urllib.request.Request(_local_ollama_url(ollama_url),
        data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = json.loads(response.read(8 * 1024 * 1024))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Local Ollama generation failed: {type(exc).__name__}") from exc
    text = raw.get("message", {}).get("content", "")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON despite JSON format mode") from exc
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, list):
        raise RuntimeError("Ollama response missing pairs array")
    return pairs


def _validate_pair(pair: Any, chunk: str) -> tuple[dict | None, str | None]:
    if not isinstance(pair, dict):
        return None, "not_object"
    q, a, quote = pair.get("question"), pair.get("answer"), pair.get("evidence_quote")
    if not all(isinstance(x, str) for x in (q, a, quote)):
        return None, "missing_fields"
    q, a, quote = q.strip(), a.strip(), quote.strip()
    if not (12 <= len(q) <= 600 and 12 <= len(a) <= 2500 and 20 <= len(quote) <= 2000):
        return None, "length_gate"
    pii_patterns = (r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
                    r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)",
                    r"\b\d{3}-\d{2}-\d{4}\b", r"\bAKIA[0-9A-Z]{16}\b",
                    r"\bsk-[A-Za-z0-9]{20,}\b")
    if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in pii_patterns for value in (q,a,quote)):
        return None, "sensitive_data_pattern"
    if quote not in chunk:
        return None, "evidence_not_exact_source_quote"
    evidence_terms = _content_tokens(quote)
    answer_terms = _content_tokens(a)
    if len(answer_terms) < 2 or not answer_terms:
        return None, "answer_too_short"
    support = len(answer_terms & evidence_terms) / len(answer_terms)
    if support < 0.45:
        return None, "answer_weakly_grounded"
    if "as an ai" in a.lower() or "i cannot" in a.lower() and "source" not in a.lower():
        return None, "generic_or_refusal_answer"
    return {"question": q, "answer": a, "evidence_quote": quote,
            "evidence_support_ratio": round(support, 4)}, None


def _split_docs(doc_ids: list[str], seed: int) -> dict[str, str]:
    if len(doc_ids) < 4:
        raise ValueError("At least 4 distinct source documents are required for useful source-disjoint train/validation/test splits")
    ordered = sorted(set(doc_ids), key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    n = len(ordered)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    assignment = {doc: "train" for doc in ordered}
    for doc in ordered[:n_test]: assignment[doc] = "test"
    for doc in ordered[n_test:n_test+n_val]: assignment[doc] = "validation"
    return assignment


def _structured_messages(row: Any) -> tuple[list[dict] | None, dict]:
    """Normalize common supervised JSONL schemas (including Dolly) to messages."""
    if not isinstance(row, dict):
        return None, {}
    metadata: dict[str, Any] = {}
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        if (len(messages) >= 2 and isinstance(messages[-1], dict) and
                messages[-1].get("role") == "assistant" and
                any(isinstance(m, dict) and m.get("role") == "user" for m in messages)):
            normalized = []
            for item in messages:
                if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
                    return None, {}
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    return None, {}
                normalized.append({"role": item["role"], "content": content.strip()})
            return normalized, metadata
    pairs = (
        ("instruction", "response"),
        ("prompt", "completion"),
        ("question", "answer"),
        ("input", "output"),
    )
    user_value = assistant_value = None
    for user_key, assistant_key in pairs:
        if isinstance(row.get(user_key), str) and isinstance(row.get(assistant_key), str):
            user_value, assistant_value = row[user_key].strip(), row[assistant_key].strip()
            break
    if user_value is None:
        return None, {}
    context = row.get("context")
    if isinstance(context, str) and context.strip():
        user_value = f"{user_value}\n\nContext:\n{context.strip()}"
    if not user_value or not assistant_value:
        return None, {}
    category = row.get("category")
    if isinstance(category, str) and category.strip():
        metadata["category"] = category.strip()[:160]
    return ([{"role": "user", "content": user_value},
             {"role": "assistant", "content": assistant_value}], metadata)


def _message_digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def import_structured_jsonl(*, output_dir: str | Path, seed: int = 42,
                             min_pairs: int = 200, progress_callback=None) -> dict | None:
    """Import existing supervised JSONL (Dolly-style) without regenerating it with an LLM.

    Each JSONL record is treated as an independent source unit. This is necessary
    for a single-file dataset such as Dolly: document-level splits would put every
    row into one split, while record-level splits keep individual examples disjoint.
    """
    ke._init()
    import sqlite3
    with sqlite3.connect(ke.DB) as con:
        sources = con.execute("""SELECT id,filename,storage_name FROM documents
            WHERE status='indexed' AND file_type='.jsonl' ORDER BY uploaded_at,filename""").fetchall()
    if not sources:
        return None

    supplied_out = Path(output_dir)
    if supplied_out.is_symlink():
        raise ValueError("Dataset output cannot be a symlink")
    out = supplied_out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Dataset output is nonempty: {out}. Select a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    split_paths = {name: out / f"{name}.jsonl" for name in ("train", "validation", "test")}
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in split_paths.items()}
    counts = {name: 0 for name in handles}
    source_units = {name: set() for name in handles}
    source_documents = {name: set() for name in handles}
    seen_messages: set[str] = set()
    rejected: dict[str, int] = {}
    valid_total = 0
    structured_sources = 0
    pii_patterns = (r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
                    r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)",
                    r"\bAKIA[0-9A-Z]{16}\b", r"\bsk-[A-Za-z0-9]{20,}\b")
    try:
        for document_id, filename, storage_name in sources:
            path = (ke.UPLOADS / storage_name).resolve()
            if path.parent != ke.UPLOADS.resolve() or path.is_symlink() or not path.is_file():
                rejected["missing_or_unsafe_source"] = rejected.get("missing_or_unsafe_source", 0) + 1
                continue
            source_had_rows = False
            with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        rejected["invalid_jsonl_row"] = rejected.get("invalid_jsonl_row", 0) + 1
                        continue
                    messages, metadata = _structured_messages(raw)
                    if messages is None:
                        rejected["unsupported_row_schema"] = rejected.get("unsupported_row_schema", 0) + 1
                        continue
                    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
                    assistant_text = messages[-1]["content"]
                    if len(user_text) < 8 or len(assistant_text) < 2 or len(user_text) > 12000 or len(assistant_text) > 12000:
                        rejected["length_gate"] = rejected.get("length_gate", 0) + 1
                        continue
                    if any(re.search(pattern, user_text + "\n" + assistant_text, flags=re.IGNORECASE)
                           for pattern in pii_patterns):
                        rejected["sensitive_data_pattern"] = rejected.get("sensitive_data_pattern", 0) + 1
                        continue
                    digest = _message_digest(messages)
                    if digest in seen_messages:
                        rejected["duplicate_example"] = rejected.get("duplicate_example", 0) + 1
                        continue
                    seen_messages.add(digest)
                    source_had_rows = True
                    record_id = f"{document_id}:{line_number}:{digest[:16]}"
                    bucket = int(hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
                    split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
                    row = {"messages": messages, "source_ids": [record_id],
                           "source_unit_kind": "record", "source_record_id": record_id,
                           "source_document_id": document_id, "source_file": filename,
                           **metadata}
                    handles[split].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[split] += 1
                    source_units[split].add(record_id)
                    source_documents[split].add(document_id)
                    valid_total += 1
                    if progress_callback and valid_total % 250 == 0:
                        progress_callback({"stage": "importing_structured_records", "processed_records": valid_total,
                                           "total_records": None, "accepted_pairs": valid_total,
                                           "current_source": filename})
            if source_had_rows:
                structured_sources += 1
        for handle in handles.values():
            handle.flush()
            handle.close()

        if valid_total == 0:
            for path in [*split_paths.values()]:
                path.unlink(missing_ok=True)
            out.rmdir()
            return None
        if valid_total < min_pairs:
            raise ValueError(
                f"Found {valid_total} valid instruction/response records in uploaded JSONL, "
                f"but the minimum is {min_pairs}. Upload a larger supervised dataset or lower the minimum for a pilot."
            )
        if min(counts.values()) < 2:
            raise ValueError("Structured data split produced fewer than two examples in a split; add more records or change the seed.")
        if progress_callback:
            progress_callback({"stage": "finalizing_dataset", "processed_records": valid_total,
                               "accepted_pairs": valid_total, "total_records": valid_total})
        file_hashes = {split: _sha256_file(path) for split, path in split_paths.items()}
        unit_lists = {split: sorted(values) for split, values in source_units.items()}
        document_lists = {split: sorted(values) for split, values in source_documents.items()}
        manifest = {
            "schema": "autotune-knowledge-dataset-v1", "model": "direct-structured-import",
            "generation_method": "direct_sft_import", "seed": seed,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "counts": counts, "source_units": unit_lists, "source_unit_kind": "record",
            "source_documents": document_lists, "file_hashes": file_hashes,
            "source_disjoint": True, "data_format": "messages", "rejected": rejected,
            "candidate_pairs": valid_total, "fine_tuning_ready": True,
            "note": "Existing instruction/response records were normalized directly. Splits are record-disjoint within the original source document; no Q/A content was regenerated by an LLM. Review licensing and quality before training."
        }
        (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if progress_callback:
            progress_callback({"stage": "completed", "processed_records": valid_total,
                               "accepted_pairs": valid_total, "total_records": valid_total})
        return {"output_dir": str(out), "counts": counts, "source_disjoint": True,
                "source_unit_kind": "record", "source_documents": structured_sources,
                "rejected": rejected, "manifest": str(out / "dataset_manifest.json"),
                "fine_tuning_ready": True, "generation_mode": "direct_import",
                "quality_note": manifest["note"]}
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        # Preserve a failed output directory for debugging only when it contains a manifest.
        for path in [*split_paths.values(), out / "dataset_manifest.json"]:
            path.unlink(missing_ok=True)
        try:
            out.rmdir()
        except OSError:
            pass
        raise


def generate_dataset(*, model: str, ollama_url: str = "http://127.0.0.1:11434",
                     output_dir: str | Path = "knowledge_store/datasets/latest", seed: int = 42,
                     max_chunks: int = 100, min_pairs: int = 200, progress_callback=None) -> dict:
    """Generate candidate Q/A data and export source-disjoint messages JSONL."""
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", model):
        raise ValueError("Invalid Ollama model tag")
    if not 1 <= max_chunks <= 5000 or not 0 <= seed <= 2**31-1 or not 4 <= min_pairs <= 100000:
        raise ValueError("Invalid generation limits")
    ke._init()
    import sqlite3
    with sqlite3.connect(ke.DB) as con:
        chunks = con.execute("""SELECT c.id,c.document_id,d.filename,c.text,c.sha256
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.status='indexed' ORDER BY d.uploaded_at,c.ordinal LIMIT ?""", (max_chunks,)).fetchall()
    if not chunks:
        raise ValueError("Upload and index source documents before generating a dataset")
    candidates, rejected = [], {}
    seen_questions = set()
    seen_chunk_hashes = set()
    total_chunks = len(chunks)
    for chunk_number, (chunk_id, document_id, filename, text, chunk_hash) in enumerate(chunks, 1):
        if chunk_hash in seen_chunk_hashes:
            continue
        seen_chunk_hashes.add(chunk_hash)
        for raw in _ollama_json(model, text, ollama_url):
            pair, reason = _validate_pair(raw, text)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            key = _norm(pair["question"])
            if key in seen_questions:
                rejected["duplicate_question"] = rejected.get("duplicate_question", 0) + 1
                continue
            seen_questions.add(key)
            candidates.append({**pair, "chunk_id": chunk_id, "source_id": document_id,
                               "source_file": filename})
        if progress_callback:
            progress_callback({"stage": "generating_grounded_pairs", "processed_chunks": chunk_number,
                               "total_chunks": total_chunks, "accepted_pairs": len(candidates),
                               "rejected_pairs": sum(rejected.values()), "current_source": filename})
        # Stop once the target is met and every split has at least two examples.
        if len(candidates) >= min_pairs and len({x["source_id"] for x in candidates}) >= 4:
            current_assignment = _split_docs([x["source_id"] for x in candidates], seed)
            split_counts = {name: 0 for name in ("train", "validation", "test")}
            for candidate in candidates:
                split_counts[current_assignment[candidate["source_id"]]] += 1
            if min(split_counts.values()) >= 2:
                break
    if len(candidates) < min_pairs:
        raise ValueError(f"Only {len(candidates)} grounded pairs passed validation; minimum is {min_pairs}. Improve source quality or add documents.")
    assignment = _split_docs([x["source_id"] for x in candidates], seed)
    splits = {"train": [], "validation": [], "test": []}
    for item in candidates:
        split = assignment[item["source_id"]]
        splits[split].append({
            "messages": [{"role":"user", "content":item["question"]},
                         {"role":"assistant", "content":item["answer"]}],
            "source_ids": [item["source_id"]], "source_file": item["source_file"],
            "chunk_id": item["chunk_id"], "evidence_quote": item["evidence_quote"],
            "evidence_support_ratio": item["evidence_support_ratio"]
        })
    if min(len(splits[x]) for x in splits) < 2:
        raise ValueError("Each source-disjoint split must contain at least 2 examples; add more distinct documents or richer source material")
    supplied_out = Path(output_dir)
    if supplied_out.is_symlink():
        raise ValueError("Dataset output cannot be a symlink")
    out = supplied_out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Dataset output is nonempty: {out}. Select a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_sets = {name: sorted({sid for row in rows for sid in row["source_ids"]}) for name, rows in splits.items()}
    if any(set(source_sets[a]) & set(source_sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise RuntimeError("Internal source leakage check failed")
    file_hashes = {split: _sha256_file(out / f"{split}.jsonl")
                   for split in ("train", "validation", "test")}
    manifest = {"schema":"autotune-knowledge-dataset-v1", "model":model,
        "seed":seed, "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts":{k:len(v) for k,v in splits.items()}, "source_documents":source_sets,
        "source_units":source_sets, "source_unit_kind":"document",
        "file_hashes":file_hashes, "source_disjoint":True, "rejected":rejected, "candidate_pairs":len(candidates),
        "fine_tuning_ready":True, "data_format":"messages",
        "note":"Generated by a local model; every pair passed exact-quote and lexical support checks. Human review is still recommended before production."}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir":str(out), "counts":manifest["counts"], "source_disjoint":True,
            "source_unit_kind":"document", "rejected":rejected, "manifest":str(out / "dataset_manifest.json"),
            "fine_tuning_ready":True, "quality_note":manifest["note"]}



def create_dataset(*, model: str, ollama_url: str = "http://127.0.0.1:11434",
                   output_dir: str | Path = "knowledge_store/datasets/latest", seed: int = 42,
                   max_chunks: int = 100, min_pairs: int = 200, progress_callback=None) -> dict:
    """Prefer direct SFT import for JSONL instruction datasets; otherwise use Ollama RAG generation."""
    direct = import_structured_jsonl(output_dir=output_dir, seed=seed,
                                     min_pairs=min_pairs, progress_callback=progress_callback)
    if direct is not None:
        return direct
    if progress_callback:
        progress_callback({"stage": "preparing_grounded_generation", "processed_chunks": 0,
                           "total_chunks": 0, "accepted_pairs": 0})
    generated = generate_dataset(model=model, ollama_url=ollama_url, output_dir=output_dir,
                                 seed=seed, max_chunks=max_chunks, min_pairs=min_pairs,
                                 progress_callback=progress_callback)
    generated["generation_mode"] = "ollama_grounded_generation"
    return generated
