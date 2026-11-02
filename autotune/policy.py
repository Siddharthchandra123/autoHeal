"""Deterministic remediation and strict allowlist for AI-suggested patches."""
from copy import deepcopy
import math

LIMITS = {
    'learning_rate': (1e-6, 5e-4),
    'weight_decay': (0.0, 0.3),
    'lora_dropout': (0.0, 0.35),
    'num_train_epochs': (1, 12),
    'lora_r': (4, 64),
    'gradient_accumulation_steps': (1, 64),
}
INTEGERS = {'num_train_epochs', 'lora_r', 'gradient_accumulation_steps'}


def fallback_patch(state: str, train: dict) -> dict:
    if state == 'overfit':
        return {
            'lora_dropout': round(min(0.35, float(train['lora_dropout']) + 0.1), 3),
            'weight_decay': round(min(0.3, float(train['weight_decay']) + 0.02), 4),
            'learning_rate': float(train['learning_rate']) * 0.8,
        }
    if state == 'underfit':
        return {
            'learning_rate': float(train['learning_rate']) * 1.5,
            'lora_dropout': round(max(0.0, float(train['lora_dropout']) - 0.03), 3),
        }
    return {}


def validate_patch(patch: dict, current: dict, diagnosis: str | None=None) -> dict:
    if not isinstance(patch, dict) or not patch:
        raise ValueError('Patch must be a nonempty object')
    valid = {}
    for key, value in patch.items():
        if key not in LIMITS or key not in current:
            raise ValueError(f'Unauthorized training parameter: {key}')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Parameter {key} must be a finite number')
        lo, hi = LIMITS[key]
        if not lo <= value <= hi:
            raise ValueError(f'{key} outside approved range [{lo}, {hi}]')
        if key in INTEGERS:
            if int(value) != value or (key == 'lora_r' and int(value) not in {4,8,16,32,64}):
                raise ValueError(f'Invalid discrete value for {key}')
            value = int(value)
        if key == 'learning_rate' and not 0.5 <= float(value)/float(current[key]) <= 2.0:
            raise ValueError('Learning-rate change exceeds 2x per iteration')
        if key == 'lora_dropout' and abs(float(value)-float(current[key])) > 0.15:
            raise ValueError('Dropout change exceeds 0.15 per iteration')
        if diagnosis == 'overfit':
            if ((key == 'learning_rate' and value > current[key])
                or (key in {'weight_decay','lora_dropout'} and value < current[key])
                or (key == 'num_train_epochs' and value > current[key])):
                raise ValueError(f'{key} change contradicts overfit protection policy')
        if diagnosis == 'underfit':
            if ((key == 'learning_rate' and value < current[key])
                or (key == 'lora_dropout' and value > current[key])):
                raise ValueError(f'{key} change contradicts underfit protection policy')
        valid[key] = value
    if all(current[k] == v for k,v in valid.items()):
        raise ValueError('Patch makes no changes')
    return valid


def apply_patch(config: dict, patch: dict) -> dict:
    validated = validate_patch(patch, config['train'])
    result = deepcopy(config)
    result['train'].update(validated)
    return result
