import json
from pathlib import Path

from autotune.adaptive_data import prepare_jsonl_dataset, profile_messages, classify_overflow


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return "\\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": text.split()}


def make_source(root):
    root.mkdir()
    rows_by_split = {}
    for split in ("train", "validation", "test"):
        rows = [
            {"messages":[{"role":"user","content":f"{split} short question"},
                         {"role":"assistant","content":f"{split} short answer"}],
             "source_ids":[f"{split}-source-1"]},
            {"messages":[{"role":"user","content":f"{split} another short question"},
                         {"role":"assistant","content":f"{split} another short answer"}],
             "source_ids":[f"{split}-source-2"]},
            {"messages":[{"role":"user","content":" ".join(["longcontext"]*40)},
                         {"role":"assistant","content":"answer"}],
             "source_ids":[f"{split}-source-3"]},
        ]
        rows_by_split[split] = rows
        (root/f"{split}.jsonl").write_text(
            "".join(json.dumps(row)+chr(10) for row in rows), encoding="utf-8")
    import hashlib
    manifest = {
        "schema":"autotune-knowledge-dataset-v1", "source_disjoint":True,
        "data_format":"messages", "source_unit_kind":"record",
        "counts":{k:len(v) for k,v in rows_by_split.items()},
        "file_hashes":{k:hashlib.sha256((root/f"{k}.jsonl").read_bytes()).hexdigest()
                       for k in rows_by_split},
        "source_units":{k:sorted({sid for row in rows for sid in row["source_ids"]})
                        for k,rows in rows_by_split.items()},
    }
    (root/"dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_profile_uses_rendered_template_and_classifies():
    tokenizer = FakeTokenizer()
    profile = profile_messages(tokenizer, [
        {"role":"user","content":"hello"}, {"role":"assistant","content":"world"}], 20)
    assert profile["fits"]
    assert classify_overflow(profile) == "fits"


def test_preparation_quarantines_overflow_and_writes_valid_derived_manifest(tmp_path):
    source, output = tmp_path/"source", tmp_path/"prepared"
    make_source(source)
    result = prepare_jsonl_dataset(source, output, FakeTokenizer(), max_length=12)
    assert result["splits"]["train"]["accepted_rows"] == 2
    assert result["splits"]["train"]["quarantined_rows"] == 1
    assert (output/"train.quarantine.jsonl").read_text(encoding="utf-8").count(chr(10)) == 1
    from autotune.knowledge_orchestrator import _dataset_paths
    data = _dataset_paths(output)
    assert data["counts"] == {"train":2, "validation":2, "test":2}
    assert "longcontext" in (output/"train.quarantine.jsonl").read_text(encoding="utf-8")


def test_preparation_never_changes_source(tmp_path):
    source, output = tmp_path/"source", tmp_path/"prepared"
    make_source(source)
    before = {p.name:p.read_bytes() for p in source.iterdir()}
    prepare_jsonl_dataset(source, output, FakeTokenizer(), max_length=12)
    after = {p.name:p.read_bytes() for p in source.iterdir()}
    assert before == after
