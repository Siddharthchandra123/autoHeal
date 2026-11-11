from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .profiler import profile_messages
from .classifier import classify_overflow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_jsonl_dataset(
    dataset_dir,
    output_dir,
    tokenizer,
    *,
    max_length: int,
    chat_template_kwargs=None,
    splits=("train", "validation", "test"),
):
    """Create immutable-source, model-tokenizer-specific prepared splits.

    Fitting examples are copied without changing their content. Oversized or
    malformed examples are quarantined with their original row and provenance.
    No truncation or synthetic answer generation is performed.
    """
    source = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / "dataset_manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("Source dataset_manifest.json is required")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != "autotune-knowledge-dataset-v1":
        raise ValueError("Unsupported source dataset manifest")

    adaptive = {
        "schema": "autotune-adaptive-sequence-manifest-v1",
        "source_dir": str(source),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "max_length": max_length,
        "chat_template_kwargs": chat_template_kwargs or {},
        "policy": "profile_and_quarantine_no_truncation_no_automatic_splitting",
        "splits": {},
    }
    derived_manifest = {
        "schema": "autotune-knowledge-dataset-v1",
        "source_disjoint": True,
        "data_format": "messages",
        "source_unit_kind": source_manifest.get("source_unit_kind", "document"),
        "counts": {},
        "file_hashes": {},
        "source_units": {},
        "parent_dataset_manifest_sha256": adaptive["source_manifest_sha256"],
        "adaptive_sequence_manifest": "adaptive_manifest.json",
    }

    for split in splits:
        source_file = source / f"{split}.jsonl"
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        prepared_file = output / f"{split}.jsonl"
        quarantine_file = output / f"{split}.quarantine.jsonl"
        counts = Counter()
        lengths = []
        retained_sources = set()

        with source_file.open("r", encoding="utf-8") as src, \
             prepared_file.open("w", encoding="utf-8") as good, \
             quarantine_file.open("w", encoding="utf-8") as bad:
            for line_no, raw in enumerate(src, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                        raise ValueError("invalid_jsonl_schema")
                    source_ids = row.get("source_ids")
                    if not isinstance(source_ids, list) or not source_ids or any(
                        not isinstance(item, str) or not item for item in source_ids
                    ):
                        raise ValueError("missing_or_invalid_source_ids")
                    profile = profile_messages(
                        tokenizer, row["messages"], max_length, chat_template_kwargs
                    )
                except Exception as exc:
                    bad.write(json.dumps({
                        "source_split": split,
                        "source_line": line_no,
                        "reason": "invalid_or_unprofileable",
                        "error": str(exc)[:500],
                        "raw": raw.rstrip("\n"),
                    }, ensure_ascii=False) + "\n")
                    counts["quarantined"] += 1
                    counts["invalid_or_unprofileable"] += 1
                    continue

                lengths.append(profile["total_tokens"])
                if profile["fits"]:
                    good.write(json.dumps(row, ensure_ascii=False) + "\n")
                    retained_sources.update(source_ids)
                    counts["accepted"] += 1
                else:
                    reason = classify_overflow(profile)
                    bad.write(json.dumps({
                        "source_split": split,
                        "source_line": line_no,
                        "reason": reason,
                        "profile": profile,
                        "row": row,
                    }, ensure_ascii=False) + "\n")
                    counts["quarantined"] += 1
                    counts[reason] += 1

        digest = _sha256(prepared_file)
        adaptive["splits"][split] = {
            "source_file": str(source_file),
            "source_sha256": _sha256(source_file),
            "prepared_file": str(prepared_file),
            "prepared_sha256": digest,
            "quarantine_file": str(quarantine_file),
            "quarantine_sha256": _sha256(quarantine_file),
            "source_rows": counts["accepted"] + counts["quarantined"],
            "accepted_rows": counts["accepted"],
            "quarantined_rows": counts["quarantined"],
            "counts_by_reason": dict(counts),
            "max_observed_tokens": max(lengths, default=0),
            "mean_observed_tokens": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        }
        derived_manifest["counts"][split] = counts["accepted"]
        derived_manifest["file_hashes"][split] = digest
        derived_manifest["source_units"][split] = sorted(retained_sources)

    adaptive["training_ready"] = all(
        info["accepted_rows"] >= 2 for info in adaptive["splits"].values()
    )
    (output / "adaptive_manifest.json").write_text(
        json.dumps(adaptive, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(derived_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return adaptive
