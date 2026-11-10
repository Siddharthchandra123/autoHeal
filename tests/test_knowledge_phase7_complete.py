import json
from pathlib import Path

import pytest

from autotune import knowledge_engine as ke
from autotune import retrieval
from autotune.dataset_factory import _validate_pair, _split_docs
from autotune.lora_contract import make_plan, LoraContractError


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    root = tmp_path / 'store'
    monkeypatch.setattr(ke, 'ROOT', root)
    monkeypatch.setattr(ke, 'UPLOADS', root / 'uploads')
    monkeypatch.setattr(ke, 'PROCESSED', root / 'processed')
    monkeypatch.setattr(ke, 'DATASETS', root / 'datasets')
    monkeypatch.setattr(ke, 'DB', root / 'knowledge.sqlite3')


def test_fts_search_is_available_and_traceable():
    ke.ingest('network.txt', b'Kubernetes coordinates containers using pods and deployments.')
    result = retrieval.search('Kubernetes pods', mode='lexical')
    assert result['results']
    assert result['results'][0]['source_file'] == 'network.txt'
    assert 'pods' in result['results'][0]['text']


def test_new_upload_is_added_to_existing_fts_index():
    ke.ingest('first.txt', b'Vector databases store embeddings for semantic search.')
    retrieval.lexical_search('embeddings')
    ke.ingest('second.txt', b'Kubernetes schedules pods across cluster worker nodes.')
    results = retrieval.lexical_search('Kubernetes worker')
    assert results and results[0]['source_file'] == 'second.txt'


def test_evidence_gate_requires_exact_quote_and_answer_support():
    source = 'Apache Iceberg is an open table format for huge analytic datasets.'
    good, reason = _validate_pair({
        'question':'What is Apache Iceberg used for?',
        'answer':'Apache Iceberg is an open table format for huge analytic datasets.',
        'evidence_quote':'Apache Iceberg is an open table format for huge analytic datasets.'}, source)
    assert reason is None and good['evidence_support_ratio'] == 1.0
    bad, reason = _validate_pair({
        'question':'Who created the project and when?',
        'answer':'It was created by a team in 2015.',
        'evidence_quote':'Apache Iceberg is an open table format for huge analytic datasets.'}, source)
    assert bad is None and reason == 'answer_weakly_grounded'


def test_split_is_source_disjoint_and_requires_four_docs():
    split = _split_docs(['a','b','c','d','e','f'], 42)
    assert set(split.values()) == {'train','validation','test'}
    assert len(split) == 6
    with pytest.raises(ValueError, match='At least 4'):
        _split_docs(['one','two'], 42)


def test_response_only_plan_accepts_structured_messages(tmp_path):
    train = tmp_path / 'train.jsonl'; valid = tmp_path / 'valid.jsonl'
    train.write_text('\n'.join(json.dumps({'messages':[{'role':'user','content':f'Question {i}?'},{'role':'assistant','content':f'Answer {i}.'}]}) for i in range(2)), encoding='utf-8')
    valid.write_text('\n'.join(json.dumps({'messages':[{'role':'user','content':f'Validation question {i}?'},{'role':'assistant','content':f'Validation answer {i}.'}]}) for i in range(2)), encoding='utf-8')
    import yaml
    cfg = yaml.safe_load(Path('config/lora_public_cpu.yaml').read_text(encoding='utf-8'))
    cfg['model']['name'] = 'Qwen/Qwen3-0.6B'
    cfg['data']['train_file'] = str(train); cfg['data']['validation_file'] = str(valid)
    cfg['train']['per_device_train_batch_size'] = 1
    cfg['train']['gradient_accumulation_steps'] = 1
    cfg['train']['num_train_epochs'] = 3
    cfg['hf'].update({'target_modules':['q_proj','v_proj'],'loss_mode':'response_only','chat_template_kwargs':{'enable_thinking':False}})
    cfg_path = tmp_path / 'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    plan = make_plan(cfg_path)
    assert plan['data_format'] == 'messages'
    assert plan['loss_mode'] == 'response_only'
    assert plan['chat_template_kwargs'] == {'enable_thinking': False}


def test_response_only_rejects_legacy_text_data(tmp_path):
    import yaml
    cfg = yaml.safe_load(Path('config/lora_public_cpu.yaml').read_text(encoding='utf-8'))
    cfg['hf']['loss_mode'] = 'response_only'
    cfg_path = tmp_path / 'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(LoraContractError, match='structured messages'):
        make_plan(cfg_path)


def test_response_only_collator_masks_prompt_and_padding():
    import torch
    from autotune.lora_gpu import _build_collator

    class Tokenizer:
        padding_side = 'right'
        def pad(self, features, padding=True, return_tensors='pt'):
            width=max(len(x['input_ids']) for x in features)
            ids=[]; masks=[]
            for item in features:
                pad=width-len(item['input_ids'])
                ids.append(item['input_ids']+[0]*pad)
                masks.append(item['attention_mask']+[0]*pad)
            return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks)}

    batch = _build_collator(Tokenizer(), response_only=True)([
        {'input_ids':[1,2,3],'attention_mask':[1,1,1],'labels':[-100,-100,3]},
        {'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1],'labels':[-100,-100,3,4]},
    ])
    assert batch['labels'].tolist() == [[-100,-100,3,-100],[-100,-100,3,4]]


def test_pii_patterns_are_rejected_from_generated_sft():
    source='For support contact alex@example.com about the Kubernetes deployment.'
    row, reason = _validate_pair({'question':'What is the contact email for Kubernetes support?',
        'answer':'Contact alex@example.com for Kubernetes support.',
        'evidence_quote':source}, source)
    assert row is None and reason == 'sensitive_data_pattern'


def test_dataset_factory_writes_structured_source_disjoint_splits(tmp_path, monkeypatch):
    from autotune import dataset_factory

    source_texts = {
        'a.txt':'Kubernetes schedules application pods across worker nodes in a cluster.',
        'b.txt':'Apache Iceberg is an open table format for large analytical datasets.',
        'c.txt':'SQLite stores structured records in local relational database tables.',
        'd.txt':'LoRA fine tuning updates a small set of trainable adapter parameters.'
    }
    for filename, text in source_texts.items():
        ke.ingest(filename, text.encode('utf-8'))

    def fake_generation(model, context, ollama_url, timeout=180):
        return [
            {'question':f'What does this source say about {context.split()[0]}?',
             'answer':context,
             'evidence_quote':context},
            {'question':f'Which detail is recorded for {context.split()[1]}?',
             'answer':context,
             'evidence_quote':context}
        ]
    monkeypatch.setattr(dataset_factory, '_ollama_json', fake_generation)
    result = dataset_factory.generate_dataset(model='test:local', output_dir=tmp_path/'dataset', min_pairs=4)
    manifest = json.loads((tmp_path/'dataset'/'dataset_manifest.json').read_text(encoding='utf-8'))
    assert result['fine_tuning_ready'] is True
    assert manifest['source_disjoint'] is True
    assert len(manifest['source_documents']['train']) == 2
    assert len(manifest['source_documents']['validation']) == 1
    assert len(manifest['source_documents']['test']) == 1
    for split in ('train','validation','test'):
        rows = [json.loads(line) for line in (tmp_path/'dataset'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) >= 2
        assert rows[0]['messages'][-1]['role'] == 'assistant'
        assert rows[0]['source_ids']


def test_semantic_index_and_search_with_injected_encoder(monkeypatch):
    import numpy as np
    from autotune import retrieval

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
            vectors = []
            for text in texts:
                vectors.append(np.asarray([1.0, 0.0] if 'Kubernetes' in text else [0.0, 1.0], dtype=np.float32))
            return np.asarray(vectors)

    ke.ingest('cluster.txt', b'Kubernetes schedules pods across worker nodes.')
    monkeypatch.setattr(retrieval, '_load_embedder', lambda model_name: FakeEncoder())
    indexed = retrieval.index_embeddings('unit-test-encoder')
    found = retrieval.search('Kubernetes pods', mode='semantic', model_name='unit-test-encoder')
    assert indexed['indexed'] >= 1
    assert found['results'][0]['source_file'] == 'cluster.txt'
    assert found['results'][0]['method'] == 'sentence-transformers'


def test_rag_rejects_unverifiable_quotes_and_accepts_exact_quote(monkeypatch):
    import io
    from autotune import rag

    chunk = {'chunk_id':'chunk-1','source_file':'cluster.txt','text':'Kubernetes schedules pods across worker nodes.'}
    monkeypatch.setattr(rag, 'search', lambda *args, **kwargs: {'mode':'lexical','results':[chunk]})

    class FakeResponse:
        def __init__(self, body): self.body=body
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def read(self, size=-1): return json.dumps({'message':{'content':json.dumps(self.body)}}).encode()

    good = {'answer':'Kubernetes schedules pods across worker nodes.',
            'citations':[{'chunk_id':'chunk-1','quote':'Kubernetes schedules pods across worker nodes.'}]}
    monkeypatch.setattr(rag.urllib.request, 'urlopen', lambda *args, **kwargs: FakeResponse(good))
    accepted = rag.answer_question('What schedules pods?', model='qwen2.5:0.5b')
    assert accepted['insufficient_evidence'] is False
    assert accepted['citations'][0]['source_file'] == 'cluster.txt'

    bad = {'answer':'Kubernetes schedules pods across worker nodes.',
           'citations':[{'chunk_id':'chunk-1','quote':'The system uses advanced machine learning to detect fraud.'}]}
    monkeypatch.setattr(rag.urllib.request, 'urlopen', lambda *args, **kwargs: FakeResponse(bad))
    rejected = rag.answer_question('What schedules pods?', model='qwen2.5:0.5b')
    assert rejected['insufficient_evidence'] is True
    assert rejected['citations'] == []
