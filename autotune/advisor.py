"""Phase 4 constrained Meta Muse advisor, deterministic offline mock, and policy gate.

Agent outputs are untrusted data. No shell, file paths, code edits, or execution
permissions are exposed. Only aggregate, allowlisted numeric signals leave host.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from typing import Any, Callable

from .policy import fallback_patch, validate_patch

ALLOWED_METRIC_FIELDS = ('step', 'train_loss', 'val_loss', 'learning_rate', 'grad_norm')
ALLOWED_DIAGNOSES = ('overfit', 'underfit')
MAX_RESPONSE_CHARS = 8192
MAX_HISTORY = 8

class AdvisorError(ValueError):
    """Model reply, input, or provider does not satisfy the trust boundary."""


@dataclass(frozen=True)
class Advice:
    provider: str
    source: str
    diagnosis: str
    patch: dict
    reason: str
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


def _unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise AdvisorError('Duplicate JSON key')
        obj[key] = value
    return obj


def public_metrics(history: list[dict]) -> list[dict]:
    """Allowlist ONLY finite numeric metrics; silently exclude arbitrary text/data."""
    if not isinstance(history, list):
        raise AdvisorError('History must be a list')
    safe = []
    for entry in history[-MAX_HISTORY:]:
        if not isinstance(entry, dict):
            continue
        row = {}
        for key in ALLOWED_METRIC_FIELDS:
            val = entry.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                row[key] = val
        if row:
            safe.append(row)
    return safe


def public_parameters(train: dict) -> dict:
    """Parameter contract, excluding arbitrary config fields and secret-like data."""
    from .policy import LIMITS
    if not isinstance(train, dict):
        raise AdvisorError('Training parameters must be a mapping')
    safe = {}
    for key in LIMITS:
        value = train.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            safe[key] = value
    return safe


def parse_reply(raw: str, diagnosis: str, train: dict) -> tuple[dict, str]:
    """Require exactly {patch, rationale?, diagnosis?}. Reject extra keys/instructions.

    Validation is independent of the LLM's own diagnosis or rationale.
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_RESPONSE_CHARS:
        raise AdvisorError('Missing or oversized model reply')
    try:
        data = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=lambda x: (_ for _ in ()).throw(AdvisorError('Nonfinite JSON number')))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AdvisorError('Model did not return strict JSON') from exc
    if not isinstance(data, dict) or set(data) - {'patch', 'rationale', 'diagnosis'} or 'patch' not in data:
        raise AdvisorError('Unexpected advisor reply schema')
    if 'diagnosis' in data and data['diagnosis'] != diagnosis:
        raise AdvisorError('Advisor diagnosis contradicts monitoring engine')
    rationale = data.get('rationale', '')
    if not isinstance(rationale, str) or len(rationale) > 320 or any(ord(c) < 32 for c in rationale):
        raise AdvisorError('Rationale is not short plain text')
    try:
        patch = validate_patch(data['patch'], train, diagnosis)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise AdvisorError(f'Policy rejected advisor patch: {exc}') from exc
    return patch, rationale


def _mock_reply(diagnosis: str, train: dict) -> str:
    patch = fallback_patch(diagnosis, train)
    return json.dumps({'diagnosis': diagnosis, 'patch': patch, 'rationale': 'Synthetic offline advice for contract verification.'})


def _meta_reply(diagnosis: str, history: list[dict], train: dict, model: str) -> str:
    key = os.environ.get('MODEL_API_KEY', '')
    if not key:
        raise AdvisorError('MODEL_API_KEY missing; no network request was sent')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AdvisorError('Optional dependency openai is not installed') from exc
    safe_data = {'diagnosis': diagnosis, 'recent_metrics': public_metrics(history),
                 'train_hyperparameters': public_parameters(train),
                 'patch_keys': ['learning_rate', 'weight_decay', 'lora_dropout',
                                'num_train_epochs', 'lora_r', 'gradient_accumulation_steps']}
    client = OpenAI(base_url='https://api.meta.ai/v1', api_key=key, timeout=20.0, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'developer', 'content': (
                'You are an advisory model in a safety-bounded ML training controller. '
                'Output ONE JSON object, with exactly patch and optional diagnosis/rationale. '
                'patch must map allowed hyperparameter names to numeric values. '
                'Never output commands, paths, scripts, secrets or raw dataset requests. '
                'All changes are independently validated and might be rejected.')},
            {'role': 'user', 'content': json.dumps(safe_data, allow_nan=False)},
        ],
    )
    choices = getattr(response, 'choices', [])
    if not choices:
        raise AdvisorError('Meta response had no choices')
    content = choices[0].message.content
    if not isinstance(content, str):
        raise AdvisorError('Meta response content was not plain text')
    return content


def advise(diagnosis: str, history: list[dict], train: dict,
           *, provider: str = 'mock', model: str = 'muse-spark-1.3',
           response_fn: Callable[[str, list[dict], dict, str], str] | None = None) -> Advice:
    """Return approved advice or *separately validated* deterministic fallback.

    Never allow a failed remote call to authorize arbitrary parameter mutations.
    The deterministic fallback is used only for supported stable diagnoses.
    """
    if diagnosis not in ALLOWED_DIAGNOSES:
        raise AdvisorError(f'Unsupported autonomous diagnosis: {diagnosis}')
    if provider not in ('mock', 'meta'):
        raise AdvisorError('Provider must be mock or meta')
    # Validate local inputs before making any external request.
    public_metrics(history)
    public_parameters(train)
    try:
        if response_fn is not None:
            raw = response_fn(diagnosis, public_metrics(history), public_parameters(train), model)
        elif provider == 'mock':
            raw = _mock_reply(diagnosis, train)
        else:
            raw = _meta_reply(diagnosis, history, train, model)
        patch, rationale = parse_reply(raw, diagnosis, train)
        return Advice(provider, f'{provider}-agent', diagnosis, patch, rationale, 'approved')
    except Exception as exc:
        # Deliberately omit error messages: some remote errors can echo input or secrets.
        fallback = fallback_patch(diagnosis, train)
        try:
            valid = validate_patch(fallback, train, diagnosis)
        except (ValueError, TypeError, ZeroDivisionError) as failure:
            raise AdvisorError(f'Agent failed and no safe fallback exists: {type(failure).__name__}') from failure
        return Advice(provider, 'deterministic-fallback', diagnosis, valid,
                      f'{type(exc).__name__}; fallback policy applied', 'fallback')
