"""Held-out task evaluation for an HF causal LM and a PEFT LoRA adapter."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return 0.0
    from collections import Counter
    p, r = Counter(pred), Counter(ref)
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('messages'), list):
                raise ValueError(f'{path}:{line_no}: expected structured messages')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: test split is empty')
    return rows


def _generate(model: Any, tokenizer: Any, messages: list[dict], *, max_new_tokens: int,
              template_kwargs: dict) -> str:
    import torch
    prompt_messages = messages[:-1] if messages and messages[-1].get('role') == 'assistant' else messages
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False,
        add_generation_prompt=True, **template_kwargs)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=False, add_special_tokens=False)
    if inputs['input_ids'].shape[-1] > 4096:
        raise ValueError('Held-out prompt exceeds 4096 tokens; prepare shorter evaluation prompts')
    inputs = {k: v.to('cpu') for k,v in inputs.items()}
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    completion = output[0, inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def evaluate_candidate(*, model_id: str, revision: str, adapter_dir: str,
                       test_file: str, max_examples: int = 300,
                       max_new_tokens: int = 160, chat_template_kwargs: dict | None = None) -> dict:
    """Compare base and adapter on the same source-disjoint held-out test set."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _read_jsonl(test_file)[:max_examples]
    template_kwargs = chat_template_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer requires pad or EOS token')
    def load_base():
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
            torch_dtype=torch.float32, trust_remote_code=False)
        model.to('cpu')
        return model

    base_rows = []
    base_model = load_base()
    for index, row in enumerate(rows):
        reference = str(row['messages'][-1]['content'])
        base_prediction = _generate(base_model, tokenizer, row['messages'],
            max_new_tokens=max_new_tokens, template_kwargs=template_kwargs)
        base_rows.append({'index':index, 'prediction':base_prediction,
                          'token_f1':token_f1(base_prediction,reference),
                          'exact_match':_tokens(base_prediction)==_tokens(reference)})
    del base_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tuned_model = load_base()
    adapter = PeftModel.from_pretrained(tuned_model, adapter_dir, is_trainable=False)
    adapter.to('cpu')
    tuned_rows = []
    for index, row in enumerate(rows):
        reference = str(row['messages'][-1]['content'])
        tuned_prediction = _generate(adapter, tokenizer, row['messages'],
            max_new_tokens=max_new_tokens, template_kwargs=template_kwargs)
        tuned_rows.append({'index':index, 'prediction':tuned_prediction,
                           'token_f1':token_f1(tuned_prediction,reference),
                           'exact_match':_tokens(tuned_prediction)==_tokens(reference)})
    del adapter, tuned_model
    gc.collect()

    def summary(values):
        scores = [float(x['token_f1']) for x in values]
        exact = [bool(x['exact_match']) for x in values]
        return {'examples':len(values), 'mean_token_f1':sum(scores)/len(scores),
                'exact_match_rate':sum(exact)/len(exact)}
    base_summary, tuned_summary = summary(base_rows), summary(tuned_rows)
    delta = tuned_summary['mean_token_f1'] - base_summary['mean_token_f1']
    return {'model':model_id, 'revision':revision, 'base':base_summary,
            'candidate':tuned_summary, 'token_f1_delta':delta,
            'examples':tuned_rows, 'metric':'token_f1',
            'note':'Objective lexical overlap on held-out domain Q/A; not a human quality or safety certification.'}
