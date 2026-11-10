"""Offline security and acceptance tests: no real Ollama or model training."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autotune import eval_gate as eg
from autotune.ollama_compare import EvaluationError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'sample_data/eval_gate_pilot.jsonl'
POLICY = ROOT / 'config/evaluation_gate.yaml'
KNOWN = [ROOT / n for n in eg.DEFAULT_KNOWN]


def mock_ollama(path, payload=None):
    if path == '/api/tags':
        return {'models': [{'name': 'smollm2:135m-instruct-fp16', 'digest': 'abcdef1234567890',
                            'details': {'format': 'gguf'}},
                           {'name': 'autotune-smollm2:gguf-v095', 'digest': '9876543210fedcba',
                            'details': {'format': 'gguf'}}]}
    assert path == '/api/generate'
    assert payload['raw'] is True and payload['options']['num_predict'] == 384
    return {'response': 'A sampled MLOps answer, for an offline test.', 'done': True, 'eval_count': 35,
            'total_duration': 3500000, 'done_reason': 'stop'}


def generated(tmp_path, monkeypatch, fake=mock_ollama):
    monkeypatch.setattr(eg, 'local_api', fake)
    dest = tmp_path / 'generation'
    result = eg.execute(DATA, POLICY, KNOWN, dest)
    assert result['prompts'] == 60
    return dest


def fill_review(path, a=4, b=4, critical='no', reviewer='reviewer-01'):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in eg.SCORE_FIELDS:
            row[field] = str(a if field.startswith('A_') else b)
        row['A_critical'] = critical
        row['B_critical'] = 'no'
        row['reviewer_id'] = reviewer
        row['review_notes'] = 'Reviewed against expected MLOps criteria.'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=eg.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_offline_preflight_exactly_60_and_digest():
    with patch.object(eg, 'local_api', side_effect=AssertionError('no network in preflight')):
        res = eg.preflight(DATA, POLICY, KNOWN)
    assert res['prompts'] == 60
    assert set(res['categories']) == set(eg.CATEGORIES)
    assert set(res['categories'].values()) == {10}
    assert res['model_promotion_approved'] is False
    assert len(res['dataset_sha256']) == 64


def test_policy_freeze(tmp_path):
    content = POLICY.read_text().replace('num_predict: 384', 'num_predict: 128')
    file = tmp_path / 'wrong.yaml'
    file.write_text(content)
    with pytest.raises(EvaluationError, match='frozen'):
        eg.load_policy(file)


def test_yaml_no_python_exec(tmp_path):
    file = tmp_path / 'bad.yaml'
    file.write_text('!!python/object/apply:os.system ["echo unsafe"]')
    with pytest.raises(EvaluationError):
        eg.load_policy(file)


def test_dataset_invalid_count(tmp_path):
    file = tmp_path / 'short.jsonl'
    file.write_text(DATA.read_text().splitlines()[0] + '\n')
    with pytest.raises(EvaluationError, match='60'):
        eg.check_dataset(file, eg.load_policy(POLICY))


def test_dataset_duplicate_id(tmp_path):
    entries = [json.loads(s) for s in DATA.read_text().splitlines()]
    entries[-1]['id'] = entries[0]['id']
    file = tmp_path / 'duplicate.jsonl'
    file.write_text(''.join(json.dumps(i) + '\n' for i in entries))
    with pytest.raises(EvaluationError, match='duplicated'):
        eg.check_dataset(file, eg.load_policy(POLICY))


def test_dataset_near_duplicate(tmp_path):
    entries = [json.loads(s) for s in DATA.read_text().splitlines()]
    entries[1]['prompt'] = entries[0]['prompt'] + ' Indeed?'
    file = tmp_path / 'duplicate.jsonl'
    file.write_text(''.join(json.dumps(i) + '\n' for i in entries))
    with pytest.raises(EvaluationError, match='Near-duplicate'):
        eg.check_dataset(file, eg.load_policy(POLICY))


def test_data_leakage_rejected(tmp_path):
    first = json.loads(DATA.read_text().splitlines()[0])
    leak = tmp_path / 'leaked.jsonl'
    leak.write_text(json.dumps({'prompt': first['prompt']}) + '\n')
    with pytest.raises(EvaluationError, match='overlaps'):
        eg.check_leakage(eg.check_dataset(DATA, eg.load_policy(POLICY)), [leak])


def test_known_source_required():
    with pytest.raises(EvaluationError, match='Must provide'):
        eg.check_leakage([], [])


def test_missing_source_is_fail_closed(tmp_path):
    with pytest.raises(EvaluationError, match='missing'):
        eg.preflight(DATA, POLICY, [tmp_path/'does-not-exist.jsonl'])


def test_execute_mock_generates_blind_review_and_no_promotion(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    file = json.loads((folder/'evaluation_results.json').read_text())
    assert file['status'].endswith('pending_human_review')
    assert file['models']['candidate']['digest'] == '9876543210fedcba'
    assert file['samples'][0]['baseline']['maybe_truncated'] is False
    assert file['inference']['num_predict'] == 384
    assert file['model_promotion_approved'] is False
    with (folder/'human_review.csv').open(encoding='utf-8-sig', newline='') as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 60
    assert rows[0]['answer_A'].startswith('A sampled MLOps answer')
    assert rows[0]['A_correctness'] == ''


def test_reject_reuse_output(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    with pytest.raises(EvaluationError, match='must not exist'):
        eg.execute(DATA, POLICY, KNOWN, folder)


def test_model_missing_digest_fails_before_outputs(tmp_path, monkeypatch):
    def fake(path, payload=None):
        return {'models': [{'name': 'smollm2:135m-instruct-fp16', 'digest': None},
                           {'name': 'autotune-smollm2:gguf-v095', 'digest': None}]}
    monkeypatch.setattr(eg, 'local_api', fake)
    folder = tmp_path/'dest'
    with pytest.raises(EvaluationError, match='digest'):
        eg.execute(DATA, POLICY, KNOWN, folder)
    assert not folder.exists()


def test_truncation_flag(tmp_path, monkeypatch):
    def fake(path, payload=None):
        if path == '/api/tags': return mock_ollama(path, payload)
        answer = mock_ollama(path, payload)
        answer['eval_count'] = 384
        answer['done_reason'] = 'length'
        return answer
    folder = generated(tmp_path, monkeypatch, fake)
    file = json.loads((folder/'evaluation_results.json').read_text())
    assert file['samples'][0]['candidate']['maybe_truncated'] is True


def test_invalid_model_answer_rejected(tmp_path, monkeypatch):
    def bad(path, payload=None):
        result = mock_ollama(path, payload)
        if path == '/api/generate': result['done'] = False
        return result
    monkeypatch.setattr(eg, 'local_api', bad)
    with pytest.raises(EvaluationError, match='Incomplete'):
        eg.execute(DATA, POLICY, KNOWN, tmp_path/'bad')


def test_pending_review_blocks_promotion(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    res = eg.grade(folder/'evaluation_results.json',folder/'human_review.csv',tmp_path/'report')
    assert res['pending'] == 60 and res['status'] == 'review_incomplete_blocked'
    assert res['promotion_approved'] is False
    assert (tmp_path/'report/category_summary.csv').exists()


def test_critical_failure_blocks_even_high_scores(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    rows = fill_review(folder/'human_review.csv')
    rows[0]['A_critical'] = 'yes'
    with (folder/'human_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=eg.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    res = eg.grade(folder/'evaluation_results.json',folder/'human_review.csv',tmp_path/'report')
    assert res['promotion_approved'] is False
    assert res['status'] == 'pilot_not_qualified_blocked'
    assert any('critical' in msg for msg in res['reasons']) or any('paired' in msg for msg in res['reasons'])


def test_neutral_tie_blocks_and_never_promotes(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    fill_review(folder/'human_review.csv', a=4, b=4)
    res = eg.grade(folder/'evaluation_results.json',folder/'human_review.csv',tmp_path/'report')
    assert res['pending'] == 0 and res['paired_correctness_gain'] == 0
    assert res['promotion_approved'] is False
    assert res['status'] == 'pilot_not_qualified_blocked'


def test_promising_result_still_requires_human_approval(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    file = folder/'human_review.csv'
    rows = fill_review(file, a=4, b=4)
    manifest = json.loads((folder/'evaluation_results.json').read_text())
    for item in rows:
        first = eg.identify_sides(item['id'], manifest['dataset_sha256'])
        baseline_side = 'A' if first == 'baseline' else 'B'
        candidate_side = 'B' if first == 'baseline' else 'A'
        item[f'{baseline_side}_correctness'] = '3'
        item[f'{candidate_side}_correctness'] = '4'
    with file.open('w', encoding='utf-8-sig', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=eg.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    res = eg.grade(folder/'evaluation_results.json',file,tmp_path/'report')
    assert res['status'] == 'pilot_thresholds_met_manual_approval_still_required'
    assert res['promotion_approved'] is False
    assert res['automated_training_advisor_authorized'] is False
    assert res['paired_bootstrap_95pct_interval'][0] >= .9


def test_review_tamper_detected(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    rows = fill_review(folder/'human_review.csv')
    rows[0]['answer_B'] = 'changed after generation'
    with (folder/'human_review.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer = csv.DictWriter(h, fieldnames=eg.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(EvaluationError, match='Tampered'):
        eg.grade(folder/'evaluation_results.json',folder/'human_review.csv',tmp_path/'report')


def test_invalid_grade_refused(tmp_path, monkeypatch):
    folder = generated(tmp_path, monkeypatch)
    rows = fill_review(folder/'human_review.csv')
    rows[1]['A_safety'] = '5'
    with (folder/'human_review.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer = csv.DictWriter(h, fieldnames=eg.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(EvaluationError, match='0-4'):
        eg.grade(folder/'evaluation_results.json',folder/'human_review.csv',tmp_path/'report')


def test_blind_assignment_deterministic():
    s = 'a'*64
    assert eg.identify_sides('train_01', s) == eg.identify_sides('train_01', s)
    assert eg.identify_sides('train_01', s) in ('baseline', 'candidate')


def test_no_side_effects_preflight_output(tmp_path):
    with patch.object(eg, 'local_api', side_effect=RuntimeError('unexpected')):
        preflight = eg.preflight(DATA, POLICY, KNOWN)
    assert preflight['status'] == 'preflight_ok'
    assert list(tmp_path.iterdir()) == []


def test_spreadsheet_formula_injection_is_escaped(tmp_path, monkeypatch):
    def unsafe(path, payload=None):
        output = mock_ollama(path, payload)
        if path == '/api/generate':
            output['response'] = '=HYPERLINK("http://invalid.example", "danger")'
        return output
    folder = generated(tmp_path, monkeypatch, unsafe)
    rows = fill_review(folder/'human_review.csv')
    assert rows[0]['answer_A'].startswith("'=HYPERLINK")
    result = eg.grade(folder/'evaluation_results.json', folder/'human_review.csv', tmp_path/'scores')
    assert result['promotion_approved'] is False
