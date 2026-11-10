from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
from .profiler import profile_messages
from .classifier import classify_overflow

def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def prepare_jsonl_dataset(dataset_dir, output_dir, tokenizer, *, max_length,
                          chat_template_kwargs=None,
                          splits=("train", "validation", "test")):
    source, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir(): raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema":"autotune-adaptive-sequence-manifest-v1", "source_dir":str(source),
        "max_length":max_length, "chat_template_kwargs":chat_template_kwargs or {},
        "policy":"profile_and_quarantine_no_truncation_no_automatic_splitting", "splits":{}}
    for split in splits:
        src_path = source / f"{split}.jsonl"
        if not src_path.is_file(): raise FileNotFoundError(src_path)
        good_path, bad_path = output/f"{split}.jsonl", output/f"{split}.quarantine.jsonl"
        counts, lengths = Counter(), []
        with src_path.open(encoding="utf-8") as src, good_path.open("w", encoding="utf-8") as good, bad_path.open("w", encoding="utf-8") as bad:
            for line_no, raw in enumerate(src, 1):
                if not raw.strip(): continue
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                        raise ValueError("invalid_jsonl_schema")
                    profile = profile_messages(tokenizer, row["messages"], max_length, chat_template_kwargs)
                except Exception as exc:
                    bad.write(json.dumps({"source_split":split,"source_line":line_no,"reason":"invalid_or_unprofileable","error":str(exc)[:500],"raw":raw.rstrip()}, ensure_ascii=False)+"\\n")
                    counts["quarantined"] += 1; counts["invalid_or_unprofileable"] += 1
                    continue
                lengths.append(profile["total_tokens"])
                if profile["fits"]:
                    good.write(json.dumps(row, ensure_ascii=False)+"\\n"); counts["accepted"] += 1
                else:
                    reason = classify_overflow(profile)
                    bad.write(json.dumps({"source_split":split,"source_line":line_no,"reason":reason,"profile":profile,"row":row}, ensure_ascii=False)+"\\n")
                    counts["quarantined"] += 1; counts[reason] += 1
        manifest["splits"][split] = {"source_file":str(src_path),"source_sha256":_sha256(src_path),
            "prepared_file":str(good_path),"quarantine_file":str(bad_path),
            "source_rows":sum(counts.values()),"accepted_rows":counts["accepted"],
            "quarantined_rows":counts["quarantined"],"counts_by_reason":dict(counts),
            "max_observed_tokens":max(lengths, default=0),
            "mean_observed_tokens":round(sum(lengths)/len(lengths),2) if lengths else 0}
    manifest["training_ready"] = all(v["quarantined_rows"] == 0 for v in manifest["splits"].values())
    (output/"adaptive_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\\n",encoding="utf-8")
    return manifest
