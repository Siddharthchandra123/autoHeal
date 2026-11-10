import json
import time
from pathlib import Path

import pytest

from autotune import knowledge_engine as ke
from autotune.dataset_factory import import_structured_jsonl
from autotune.knowledge_orchestrator import _dataset_paths


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(ke, "ROOT", root)
    monkeypatch.setattr(ke, "UPLOADS", root / "uploads")
    monkeypatch.setattr(ke, "PROCESSED", root / "processed")
    monkeypatch.setattr(ke, "DATASETS", root / "datasets")
    monkeypatch.setattr(ke, "DB", root / "knowledge.sqlite3")


def _dolly_jsonl(rows=240):
    return "\n".join(json.dumps({
        "instruction": f"Explain the documented engineering concept number {i} in practical terms.",
        "context": f"Concept {i} refers to a distinct system component used for reliable data processing.",
        "response": f"Concept {i} is a system component used for reliable data processing and operational consistency.",
        "category": "closed_qa",
    }) for i in range(rows)).encode("utf-8")


def test_dolly_jsonl_is_imported_directly_with_record_disjoint_splits(tmp_path):
    uploaded = ke.ingest("databricks-dolly-15k.jsonl", _dolly_jsonl(240))
    output = tmp_path / "dataset-20261109-120000"
    result = import_structured_jsonl(output_dir=output, seed=42, min_pairs=200)

    assert result is not None
    assert result["generation_mode"] == "direct_import"
    assert result["source_unit_kind"] == "record"
    assert sum(result["counts"].values()) == 240
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_unit_kind"] == "record"
    assert manifest["source_disjoint"] is True
    assert manifest["source_documents"]["train"] == [uploaded["id"]]

    data = _dataset_paths(output)
    assert data["counts"]["train"] > 0
    assert data["source_unit_kind"] == "record"
    split_ids = {}
    for split in ("train", "validation", "test"):
        split_ids[split] = set()
        with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                split_ids[split].update(row["source_ids"])
                assert row["source_document_id"] == uploaded["id"]
                assert row["messages"][-1]["role"] == "assistant"
    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["validation"] & split_ids["test"])


def test_non_sft_jsonl_is_not_misclassified(tmp_path):
    ke.ingest("events.jsonl", b'{"timestamp":"2026-01-01","event":"started"}\n')
    result = import_structured_jsonl(output_dir=tmp_path / "candidate", min_pairs=4)
    assert result is None
    assert not (tmp_path / "candidate").exists()


def test_direct_import_rejects_under_minimum(tmp_path):
    ke.ingest("small-dolly.jsonl", _dolly_jsonl(20))
    with pytest.raises(ValueError, match="minimum is 200"):
        import_structured_jsonl(output_dir=tmp_path / "small", min_pairs=200)


def test_dataset_generation_api_returns_background_job_immediately(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from autotune import dataset_jobs

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_create_dataset(**kwargs):
        started.set()
        release.wait(3)
        return {"counts": {"train": 8, "validation": 2, "test": 2},
                "generation_mode": "test", "source_unit_kind": "record"}

    monkeypatch.setattr("autotune.dataset_factory.create_dataset", fake_create_dataset)
    monkeypatch.setattr(ke, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(ke, "ROOT", tmp_path / "store")
    monkeypatch.setattr(ke, "UPLOADS", tmp_path / "store" / "uploads")
    monkeypatch.setattr(ke, "PROCESSED", tmp_path / "store" / "processed")
    monkeypatch.setattr(ke, "DB", tmp_path / "store" / "knowledge.sqlite3")

    response = TestClient(ke.create_app()).post("/api/knowledge/datasets/generate", json={
        "model": "qwen2.5:0.5b", "max_chunks": 20, "min_pairs": 4
    })
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert started.wait(2)
    status = TestClient(ke.app).get(f"/api/knowledge/datasets/generation-jobs/{job['job_id']}")
    assert status.status_code == 200
    assert status.json()["status"] in {"running", "queued"}
    release.set()
    for _ in range(50):
        status = TestClient(ke.app).get(f"/api/knowledge/datasets/generation-jobs/{job['job_id']}")
        if status.json()["status"] == "completed":
            break
        time.sleep(0.05)
    assert status.json()["status"] == "completed"


def test_full_api_flow_builds_dolly_dataset_without_ollama(tmp_path):
    from fastapi.testclient import TestClient

    ke.ingest("databricks-dolly-15k.jsonl", _dolly_jsonl(240))
    client = TestClient(ke.create_app())
    response = client.post("/api/knowledge/datasets/generate", json={
        "model": "qwen2.5:0.5b", "max_chunks": 100, "min_pairs": 200, "seed": 42
    })
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = response.json()["status"]
    for _ in range(100):
        if status not in {"queued", "running"}:
            break
        time.sleep(0.05)
        poll = client.get(f"/api/knowledge/datasets/generation-jobs/{job_id}")
        assert poll.status_code == 200
        status_body = poll.json()
        status = status_body["status"]
    assert status == "completed", status_body
    assert status_body["result"]["generation_mode"] == "direct_import"
    listing = client.get("/api/knowledge/datasets").json()["datasets"]
    assert len(listing) == 1
    assert listing[0]["source_unit_kind"] == "record"
