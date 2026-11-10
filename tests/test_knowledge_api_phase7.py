import io
import json

from fastapi.testclient import TestClient

from autotune import knowledge_engine as ke


def test_upload_and_search_api(tmp_path, monkeypatch):
    root=tmp_path/'store'
    monkeypatch.setattr(ke,'ROOT',root)
    monkeypatch.setattr(ke,'UPLOADS',root/'uploads')
    monkeypatch.setattr(ke,'PROCESSED',root/'processed')
    monkeypatch.setattr(ke,'DATASETS',root/'datasets')
    monkeypatch.setattr(ke,'DB',root/'knowledge.sqlite3')
    client=TestClient(ke.create_app())
    up=client.post('/api/knowledge/documents',files={'file':('..\\network.txt',b'Kubernetes schedules pods on worker nodes.')})
    assert up.status_code==200
    assert up.json()['filename']=='network.txt'
    result=client.post('/api/knowledge/search',json={'query':'Kubernetes worker nodes','mode':'lexical'})
    assert result.status_code==200
    assert result.json()['results'][0]['source_file']=='network.txt'


def test_job_api_rejects_path_traversal_dataset_id(tmp_path, monkeypatch):
    root=tmp_path/'store'
    monkeypatch.setattr(ke,'ROOT',root)
    monkeypatch.setattr(ke,'UPLOADS',root/'uploads')
    monkeypatch.setattr(ke,'PROCESSED',root/'processed')
    monkeypatch.setattr(ke,'DATASETS',root/'datasets')
    monkeypatch.setattr(ke,'DB',root/'knowledge.sqlite3')
    client=TestClient(ke.create_app())
    response=client.post('/api/knowledge/training/jobs',json={'dataset_id':'..\\outside','mode':'preflight'})
    assert response.status_code==400


def test_dataset_preview_rejects_unsafe_ids_and_returns_examples(tmp_path, monkeypatch):
    root=tmp_path/'store'
    monkeypatch.setattr(ke,'ROOT',root)
    monkeypatch.setattr(ke,'UPLOADS',root/'uploads')
    monkeypatch.setattr(ke,'PROCESSED',root/'processed')
    monkeypatch.setattr(ke,'DATASETS',root/'datasets')
    monkeypatch.setattr(ke,'DB',root/'knowledge.sqlite3')
    dataset=root/'datasets'/'dataset-20261104-120000'
    dataset.mkdir(parents=True)
    row={'messages':[{'role':'user','content':'What is the source fact?'},{'role':'assistant','content':'This is the source answer.'}],
         'source_ids':['doc-a'],'evidence_quote':'This is the source answer.'}
    (dataset/'train.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    (dataset/'dataset_manifest.json').write_text(json.dumps({'counts':{'train':1},'source_disjoint':True}),encoding='utf-8')
    client=TestClient(ke.create_app())
    bad=client.get('/api/knowledge/datasets/..%2Foutside/preview')
    assert bad.status_code in {400,404}
    good=client.get('/api/knowledge/datasets/dataset-20261104-120000/preview?split=train&limit=5')
    assert good.status_code==200
    assert good.json()['examples'][0]['messages'][-1]['content']=='This is the source answer.'
