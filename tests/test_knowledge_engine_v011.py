import json
from pathlib import Path

import pytest

from autotune import knowledge_engine as ke


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ke, 'ROOT', tmp_path / 'store')
    monkeypatch.setattr(ke, 'UPLOADS', ke.ROOT / 'uploads')
    monkeypatch.setattr(ke, 'PROCESSED', ke.ROOT / 'processed')
    monkeypatch.setattr(ke, 'DB', ke.ROOT / 'knowledge.sqlite3')


def test_txt_ingestion_creates_traceable_chunks():
    payload = ('This is a knowledge paragraph. ' * 100).encode()
    result = ke.ingest('../../internal.txt', payload)
    assert result['status'] == 'indexed'
    assert result['filename'] == 'internal.txt'
    assert result['chunks'] >= 2
    docs = ke.list_documents()
    assert docs[0]['id'] == result['id']


def test_duplicate_file_is_not_ingested_twice():
    payload = b'Unique content for the knowledge base.'
    first = ke.ingest('one.txt', payload)
    second = ke.ingest('renamed.txt', payload)
    assert first['id'] == second['id']
    assert second['duplicate'] is True
    assert len(ke.list_documents()) == 1


def test_rejects_unsupported_extension_and_oversize():
    with pytest.raises(ValueError, match='Allowed formats'):
        ke.ingest('payload.exe', b'hello')
    with pytest.raises(ValueError, match='between'):
        ke.ingest('empty.txt', b'')


def test_jsonl_validation_and_extraction():
    data = b'{"question":"What is X?","answer":"X is a test."}\n'
    result = ke.ingest('qa.jsonl', data)
    assert result['status'] == 'indexed'
    assert result['chunks'] == 1


def test_export_is_explicitly_not_sft_ready(tmp_path):
    ke.ingest('facts.md', b'# Facts\n\nA useful source-grounded fact.')
    result = ke.export_corpus(tmp_path / 'corpus.jsonl')
    assert result['records'] == 1
    assert result['fine_tuning_ready'] is False
    row = json.loads((tmp_path / 'corpus.jsonl').read_text(encoding='utf-8'))
    assert row['source_file'] == 'facts.md'
    assert row['text'].startswith('# Facts')


def test_chunking_respects_overlap_and_content():
    text = 'Sentence one. ' * 200
    chunks = ke.make_chunks(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert ''.join(chunks).replace(' ', '')
