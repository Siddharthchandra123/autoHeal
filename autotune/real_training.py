"""Phase 5: genuine CPU PyTorch next-byte language-model training.

Educational reference backend: REAL gradient updates and loadable tensor checkpoints;
NOT a Meta Llama/LoRA fine-tune or performance benchmark. No network calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .config import config_sha256
from .metrics import MonitorStream
from .mock_trainer import RunResult
from .tracking import Tracker


class RealTrainingError(RuntimeError):
    pass


def dataset_identity(config: dict) -> str:
    """Bind train/validation bytes plus their roles; no raw examples to the agent."""
    digest = hashlib.sha256(b'autotune-byte-lm-v1\0')
    fingerprints = []
    for role, key in [('train', 'train_file'), ('validation', 'validation_file')]:
        path = Path(config['data'][key])
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 16_000_000:
            raise RealTrainingError(f'{role} data missing, symlinked or too large for CPU pilot: {path}')
        content = path.read_bytes()
        if not content:
            raise RealTrainingError(f'{role} data empty')
        fingerprints.append(hashlib.sha256(content).hexdigest())
        digest.update(role.encode() + b'\0' + bytes.fromhex(fingerprints[-1]))
    if fingerprints[0] == fingerprints[1]:
        raise RealTrainingError('Train and validation content must differ')
    return digest.hexdigest()


def read_texts(path: str | Path) -> list[str]:
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RealTrainingError(f'Invalid JSONL line {line_no}: {exc}') from exc
        if not isinstance(obj, dict) or not isinstance(obj.get('text'), str) or len(obj['text'].strip()) < 3:
            raise RealTrainingError(f'JSONL line {line_no} requires nonempty text field')
        records.append(obj['text'])
    if len(records) < 2:
        raise RealTrainingError('At least 2 examples required in each split')
    return records


def make_model(hidden: int = 32, dropout: float = 0.05):
    import torch
    from torch import nn

    class ByteLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 256)

        def forward(self, tokens):
            h, _ = self.gru(self.emb(tokens))
            return self.head(self.drop(h))

    return ByteLM()


def batchify(examples: list[str], max_length: int):
    """Make fixed-length byte inputs, using -100 on padding targets."""
    import torch
    seq_len = min(int(max_length), 96)
    x = torch.zeros((len(examples), seq_len), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, text in enumerate(examples):
        raw = list(text.encode('utf-8')[:seq_len + 1])
        tokens = min(len(raw) - 1, seq_len)
        if tokens < 2:
            raise RealTrainingError('Tokenized record too short')
        x[i, :tokens] = torch.tensor(raw[:tokens])
        y[i, :tokens] = torch.tensor(raw[1:tokens + 1])
    return x, y


def load_verified_checkpoint(checkpoint: str | Path, *, expected_sha256: str,
                             expected_config_sha256: str, expected_data_sha256: str) -> dict[str, Any]:
    """Safe torch deserialization with content, config and data binding.

    Callers must control the path: torch weights_only=True mitigates unsafe pickle,
    but resource limits and provenance are still required for untrusted files.
    """
    import torch
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64_000_000:
        raise RealTrainingError('Missing, linked, or oversized model checkpoint')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RealTrainingError('Model checkpoint SHA256 mismatch')
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RealTrainingError(f'Unable to read safe weights checkpoint: {exc}') from exc
    if not isinstance(obj, dict) or obj.get('schema') != 'autotune-tiny-byte-lm-v1':
        raise RealTrainingError('Unexpected checkpoint schema')
    if obj.get('config_sha256') != expected_config_sha256 or obj.get('dataset_sha256') != expected_data_sha256:
        raise RealTrainingError('Config/dataset checkpoint identity mismatch')
    if not isinstance(obj.get('epoch'), int) or obj['epoch'] < 1:
        raise RealTrainingError('Invalid saved epoch')
    value = obj.get('val_loss')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RealTrainingError('Invalid checkpoint validation loss')
    if not isinstance(obj.get('model_state'), dict) or not isinstance(obj.get('optimizer_state'), dict):
        raise RealTrainingError('Missing model/optimizer state')
    model = make_model(hidden=obj.get('hidden_dim'), dropout=obj.get('dropout', 0.0))
    try:
        model.load_state_dict(obj['model_state'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(obj['optimizer_state'])
    except Exception as exc:
        raise RealTrainingError(f'Checkpoint weights/optimizer not loadable: {exc}') from exc
    return obj


def train(config: dict, trial_dir: Path, trial: int) -> RunResult:
    import torch
    from torch.nn import functional as F

    if config['model']['name'] != 'autotune/tiny-byte-gru':
        raise RealTrainingError('CPU real trainer only supports autotune/tiny-byte-gru')
    torch.manual_seed(2026 + trial)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    data_hash = dataset_identity(config)
    train_text = read_texts(config['data']['train_file'])
    validation_text = read_texts(config['data']['validation_file'])
    train_x, train_y = batchify(train_text, config['model']['max_length'])
    valid_x, valid_y = batchify(validation_text, config['model']['max_length'])
    tr = config['train']
    # This tiny GRU is not LoRA; its dropout is a lightweight stand-in to exercise policy changes.
    model = make_model(dropout=float(tr['lora_dropout']))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr['learning_rate']), weight_decay=float(tr['weight_decay']))
    tracker = Tracker(trial_dir, trial, config)
    monitor = MonitorStream(trial_dir / 'monitor_decisions.jsonl', config['monitor'])
    checkpoint = trial_dir / 'best_checkpoint.pt'
    best_loss, best_step = math.inf, 0
    history, state, reason = [], 'completed', 'Training finished without monitored early stop'
    batch_size = int(tr['per_device_train_batch_size'])
    for epoch in range(1, int(tr['num_train_epochs']) + 1):
        model.train()
        ordered = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(2026 + trial + epoch))
        running, tokens = 0.0, 0
        for indexes in ordered.split(batch_size):
            logits = model(train_x[indexes])
            labels = train_y[indexes]
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RealTrainingError('Nonfinite CPU training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int((labels != -100).sum().item())
            running += float(loss.item()) * n
            tokens += n
        train_loss = running / tokens
        model.eval()
        with torch.inference_mode():
            validation_logits = model(valid_x)
            val_loss = float(F.cross_entropy(validation_logits.flatten(0, 1), valid_y.flatten(), ignore_index=-100).item())
        row = {'step': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        tracker.log(epoch, train_loss, val_loss, extra={'learning_rate': float(tr['learning_rate'])})
        if math.isfinite(val_loss) and val_loss < best_loss:
            best_loss, best_step = val_loss, epoch
            payload = {'schema': 'autotune-tiny-byte-lm-v1', 'config_sha256': config_sha256(config),
                       'dataset_sha256': data_hash, 'epoch': epoch, 'val_loss': best_loss,
                       'hidden_dim': 32, 'dropout': float(tr['lora_dropout']),
                       'model_state': model.state_dict(), 'optimizer_state': opt.state_dict()}
            temp = trial_dir / '.checkpoint-writing.tmp'
            torch.save(payload, temp)
            os.replace(temp, checkpoint)
        decision = monitor.observe(row)
        if decision.should_stop:
            state, reason = decision.state, decision.explanation
            break
    tracker.finish(state, reason)
    if best_step == 0:
        raise RealTrainingError('No valid checkpoint could be produced')
    return RunResult(state, reason, float(best_loss), int(best_step), str(checkpoint), history)
