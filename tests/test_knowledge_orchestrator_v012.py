import json
from pathlib import Path

import pytest

from autotune.knowledge_orchestrator import _dataset_paths, preflight, execute


def make_dataset(root: Path, *, leak=False):
    root.mkdir()
    rows = {}
    ids = {'train':'doc-train','validation':'doc-val','test':'doc-test'}
    for split in ('train','validation','test'):
        rows[split] = []
        for i in range(2):
            source = ids[split]
            if leak and split == 'test': source = ids['train']
            rows[split].append({
                'messages':[{'role':'user','content':f'{split} prompt {i} ?'},
                            {'role':'assistant','content':f'{split} answer {i}.'}],
                'source_ids':[source], 'evidence_quote':f'{split} answer {i}.'})
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows[split]), encoding='utf-8')
    import hashlib
    manifest = {'schema':'autotune-knowledge-dataset-v1','source_disjoint':True,'data_format':'messages',
        'counts':{k:len(v) for k,v in rows.items()},
        'file_hashes':{k:hashlib.sha256((root/f'{k}.jsonl').read_bytes()).hexdigest() for k in rows},
        'source_documents':{k:sorted({sid for row in v for sid in row['source_ids']}) for k,v in rows.items()}}
    (root/'dataset_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')


def registry_file(path: Path):
    registry = {
        'schema':'autotune-knowledge-candidates-v1',
        'candidates':[{'id':'qwen-test','model':'Qwen/Qwen3-0.6B','target_modules':['q_proj','v_proj'],
            'chat_template_kwargs':{'enable_thinking':False},'device':'cpu','quantization':'none','enabled':True}],
        'defaults':{'max_length':512,'max_steps':100,'epochs':3,'eval_steps':2,
            'batch_size':1,'gradient_accumulation_steps':1,'lora_r':8,'lora_alpha':16,
            'lora_dropout':0.05,'learning_rates':[0.0002],'max_test_examples':10,
            'minimum_token_f1':0.3,'minimum_improvement':0.02,
            'minimum_train_rows':100,'minimum_validation_rows':20,'minimum_test_rows':20}}
    path.write_text(json.dumps(registry),encoding='utf-8')
    return path


def test_dataset_contract_validates_real_source_disjointness(tmp_path):
    root=tmp_path/'dataset';make_dataset(root)
    data=_dataset_paths(root)
    assert data['counts']=={'train':2,'validation':2,'test':2}


def test_dataset_contract_rejects_manifest_claim_that_hides_source_leakage(tmp_path):
    root=tmp_path/'dataset';make_dataset(root,leak=True)
    with pytest.raises(ValueError,match='Source-unit leakage'):
        _dataset_paths(root)


def test_dataset_contract_rejects_tampered_split_file_hash(tmp_path):
    root=tmp_path/'dataset';make_dataset(root)
    with (root/'train.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'messages':[{'role':'user','content':'tamper?'},{'role':'assistant','content':'tampered'}], 'source_ids':['doc-train']})+'\n')
    with pytest.raises(ValueError, match='manifest row count mismatch|split file hash mismatch'):
        _dataset_paths(root)


def test_preflight_builds_all_candidate_trial_configs_without_download(tmp_path):
    root=tmp_path/'dataset';make_dataset(root)
    reg=registry_file(tmp_path/'registry.json')
    result=preflight(root,output_dir=tmp_path/'preflight',registry_path=reg)
    assert result['mode']=='PREFLIGHT_ONLY'
    assert result['trial_count']==1
    assert result['candidates'][0]['plan']['data_format']=='messages'
    assert result['candidates'][0]['plan']['loss_mode']=='response_only'
    assert result['data_quality_gate']['ready'] is False


def test_execute_blocks_small_datasets_without_explicit_smoke_override(tmp_path):
    root=tmp_path/'dataset';make_dataset(root)
    reg=registry_file(tmp_path/'registry.json')
    with pytest.raises(ValueError,match='data-quality gate'):
        execute(root,output_dir=tmp_path/'execute',registry_path=reg,allow_cpu=True)


def execute_with_mocked_models(tmp_path, monkeypatch, *, test_passes=True):
    from autotune import knowledge_orchestrator as ko
    from autotune import lora_gpu

    root = tmp_path / 'dataset'
    make_dataset(root)
    registry = {
        'schema':'autotune-knowledge-candidates-v1',
        'candidates':[{'id':'qwen-test','model':'Qwen/Qwen3-0.6B','target_modules':['q_proj','v_proj'],
            'chat_template_kwargs':{'enable_thinking':False},'device':'cpu','quantization':'none','enabled':True}],
        'defaults':{'max_length':512,'max_steps':100,'epochs':3,'eval_steps':2,
            'batch_size':1,'gradient_accumulation_steps':1,'lora_r':8,'lora_alpha':16,
            'lora_dropout':0.05,'learning_rates':[0.0002,0.0001],'max_test_examples':10,
            'minimum_token_f1':0.3,'minimum_improvement':0.02,
            'minimum_train_rows':2,'minimum_validation_rows':2,'minimum_test_rows':2}}
    reg = tmp_path / 'registry.json'
    reg.write_text(json.dumps(registry), encoding='utf-8')
    monkeypatch.setattr(ko, '_pin_revision', lambda model: 'a' * 40)
    def fake_train(config, output, allow_cpu=False):
        Path(output).mkdir(parents=True, exist_ok=True)
        return {'ok':True,'adapter_dir':str(Path(output)/'adapter'),'best_val_loss':1.0}
    monkeypatch.setattr(lora_gpu, 'run_lora', fake_train)
    eval_calls = []

    def fake_evaluate(*, model_id, revision, adapter_dir, test_file, max_examples, chat_template_kwargs):
        split = Path(test_file).name
        eval_calls.append((split, adapter_dir, revision))
        if split == 'validation.jsonl':
            score = 0.5 if 'trial-01' in adapter_dir else 0.6
            base = 0.45
        else:
            score = 0.55 if test_passes else 0.25
            base = 0.50
        return {'model':model_id,'revision':revision,
            'base':{'mean_token_f1':base,'exact_match_rate':0.1},
            'candidate':{'mean_token_f1':score,'exact_match_rate':0.2},
            'token_f1_delta':score-base,'examples':[],'metric':'token_f1'}

    monkeypatch.setattr(ko, 'evaluate_candidate', fake_evaluate)
    result = ko.execute(root, output_dir=tmp_path / 'execution', registry_path=reg,
                        allow_cpu=True, allow_small_data=True)
    return result, eval_calls


def test_execute_selects_on_validation_and_uses_test_once(tmp_path, monkeypatch):
    result, eval_calls = execute_with_mocked_models(tmp_path, monkeypatch, test_passes=True)
    assert [x[0] for x in eval_calls] == ['validation.jsonl','validation.jsonl','test.jsonl']
    assert 'trial-02' in eval_calls[-1][1]
    assert all(x[2] == 'a' * 40 for x in eval_calls)
    assert result['selection']['status'] == 'qualified_candidate_found'
    assert result['selection']['test_token_f1'] == 0.55
    assert result['promotion_approved'] is False


def test_execute_returns_no_qualified_model_when_final_test_fails(tmp_path, monkeypatch):
    result, eval_calls = execute_with_mocked_models(tmp_path, monkeypatch, test_passes=False)
    assert [x[0] for x in eval_calls].count('test.jsonl') == 1
    assert result['selection']['status'] == 'NO_QUALIFIED_MODEL'
    assert result['promotion_approved'] is False
