"""Optional, GPU-gated Unsloth adapter for causal language-model fine-tuning.

Unsloth is NEVER imported during preflight or on unsupported hardware.
This source path is a GPU-only integration candidate; live GPU training and
checkpoint resume are not verified.  The stable Hugging Face CPU path remains
unchanged and is the default backend.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path

from .config import load_config
from .lora_contract import LoraContractError, make_plan, verify_manifest, write_manifest, _write_json_atomic
from .metrics import MonitorStream


class UnslothIntegrationError(RuntimeError):
    """Unsupported environment, incompatible configuration or runtime failure."""


def make_unsloth_plan(config_path: str | Path) -> dict:
    """Offline: validate the existing LoRA/HF contract without importing Unsloth."""
    base = make_plan(config_path)
    cfg = load_config(config_path, mode='hf')
    name = base['model']
    if (len(name) > 200 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?', name)
            or '..' in name):
        raise UnslothIntegrationError('Unsloth model name must be a safe Hugging Face repository ID')
    if base['device'] != 'gpu':
        return {
            'ok': True, 'mode': 'UNSLOTH OFFLINE PREFLIGHT', 'backend': 'unsloth',
            'executable_on_configured_device': False, 'reason': 'Unsloth Core fine-tuning requires a supported accelerator. CPU training stays on Hugging Face PEFT.',
            'model': base['model'], 'model_revision': base['revision'],
            'config_sha256': base['config_sha256'], 'train_rows': base['train_rows'],
            'validation_rows': base['validation_rows'], 'requires_gpu': True,
            'live_gpu_training_verified': False,
        }
    return {
        'ok': True, 'mode': 'UNSLOTH OFFLINE PREFLIGHT', 'backend': 'unsloth',
        'executable_on_configured_device': True,
        'reason': 'Configuration targets a GPU. Actual PyTorch GPU availability and Unsloth/TRL compatibility must pass at execution.',
        'model': base['model'], 'model_revision': base['revision'],
        'config_sha256': base['config_sha256'], 'train_sha256': base['train_sha256'],
        'validation_sha256': base['validation_sha256'],
        'train_rows': base['train_rows'], 'validation_rows': base['validation_rows'],
        'estimated_steps': base['estimated_steps'], 'quantization': base['quantization'],
        'target_modules': base['target_modules'], 'requires_gpu': True,
        'live_gpu_training_verified': False,
    }


def assert_gpu_available(plan: dict) -> dict:
    """Mandatory fail-closed runtime gate; PyTorch ROCm also exposes torch.cuda."""
    if plan['device'] != 'gpu':
        raise UnslothIntegrationError('Unsloth fine-tuning is GPU-only in this integration. Use the existing Hugging Face PEFT CPU backend.')
    try:
        import torch
    except ImportError as exc:
        raise UnslothIntegrationError('PyTorch is missing; install a supported GPU build in a separate environment.') from exc
    if not torch.cuda.is_available():
        raise UnslothIntegrationError('No supported NVIDIA/ROCm training GPU detected by PyTorch. No training started.')
    return {'gpu_name': torch.cuda.get_device_name(0), 'pytorch_cuda_runtime': str(torch.version.cuda),
            'pytorch_hip_runtime': str(torch.version.hip)}


def run_unsloth(config_path: str | Path, output: str | Path) -> dict:
    """Experimental GPU-only training path. Never modifies original CPU outputs.

    This path is NOT wired to automatic agent-driven retries; the controller's
    existing safety policy must be separately integrated and GPU-tested.
    """
    make_unsloth_plan(config_path)  # Includes the stricter repository-ID safety gate.
    base = make_plan(config_path)
    assert_gpu_available(base)  # Always before package imports/model downloads.
    cfg = load_config(config_path, mode='hf')
    out = Path(output).resolve()
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise UnslothIntegrationError('Output must be a fresh empty directory; existing experiments are never overwritten.')
    if int(cfg['safety']['max_trials']) > 1:
        # There is no actual autonomous retry integration here yet.
        raise UnslothIntegrationError('Unsloth experimental runner must use safety.max_trials=1 until recovery is GPU-verified.')
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback
        from datasets import load_dataset
        import torch
    except (ImportError, RuntimeError) as exc:
        raise UnslothIntegrationError('Unsloth/TRL GPU runtime unavailable; use an isolated environment and official install instructions.') from exc

    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    # No automatic resume of interrupted Unsloth jobs. A human must investigate.
    try:
        _write_json_atomic(out / 'plan.json', base)
        _write_json_atomic(out / 'run_state.json', {'status': 'running', 'backend': 'unsloth', 'schema': 'experimental-v0.8.3'})
        max_len = int(cfg['model']['max_length'])
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base['model'], max_seq_length=max_len,
            load_in_4bit=(base['quantization'] == '4bit'), trust_remote_code=False)
        tr = cfg['train']
        model = FastLanguageModel.get_peft_model(
            model, r=int(tr['lora_r']), target_modules=base['target_modules'],
            lora_alpha=int(tr['lora_alpha']), lora_dropout=float(tr['lora_dropout']),
            bias='none', use_gradient_checkpointing='unsloth', random_state=base['seed'])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable < 1:
            raise UnslothIntegrationError('Unsloth adapter did not expose any trainable parameters.')
        ds = load_dataset('json', data_files=base['data_files'])
        if 'text' not in ds['train'].column_names or 'text' not in ds['validation'].column_names:
            raise UnslothIntegrationError('Expected train/validation JSONL rows containing text')

        class RecoveryAwareCallback(TrainerCallback):
            def __init__(self):
                self.monitor = MonitorStream(out / 'monitor_decisions.jsonl', cfg['monitor'])
                self.log_window = []
                self.prev_eval_step = 0
                self.stop_diagnosis = None

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    self.log_window.append((int(state.global_step), float(logs['loss'])))
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics or 'eval_loss' not in metrics:
                    return control
                step = int(state.global_step)
                losses = [v for s, v in self.log_window if self.prev_eval_step < s <= step]
                self.prev_eval_step = step
                self.log_window = [(s, v) for s, v in self.log_window if s > step]
                if not losses:
                    return control  # Never fabricate paired training loss.
                paired = {'step': step, 'train_loss': sum(losses)/len(losses),
                          'val_loss': float(metrics['eval_loss'])}
                with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({**paired, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}, allow_nan=False) + '\n')
                decision = self.monitor.observe(paired)
                if decision.should_stop:
                    self.stop_diagnosis = decision.state
                    control.should_training_stop = True
                    control.should_save = True
                return control

        callback = RecoveryAwareCallback()
        args = SFTConfig(
            output_dir=str(out / 'checkpoints'), dataset_text_field='text', max_length=max_len,
            max_steps=int(base['estimated_steps']), learning_rate=float(tr['learning_rate']),
            per_device_train_batch_size=int(tr['per_device_train_batch_size']),
            per_device_eval_batch_size=int(tr['per_device_train_batch_size']),
            gradient_accumulation_steps=int(tr['gradient_accumulation_steps']),
            weight_decay=float(tr['weight_decay']), eval_strategy='steps', save_strategy='steps',
            eval_steps=int(tr['eval_steps']), save_steps=int(tr['eval_steps']),
            logging_steps=1, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, save_total_limit=3, report_to='none', seed=int(base['seed']),
            bf16=bool(torch.cuda.is_bf16_supported()), fp16=not torch.cuda.is_bf16_supported(),
            dataset_num_proc=1)
        trainer = SFTTrainer(model=model, train_dataset=ds['train'], eval_dataset=ds['validation'],
                             args=args, processing_class=tokenizer, callbacks=[callback])
        trainer.train()
        best = trainer.state.best_model_checkpoint
        if not best:
            raise UnslothIntegrationError('No evaluated best model checkpoint was selected')
        # Verify the exact full training state before accepting a checkpoint.
        write_manifest(best, base, kind='trainer_checkpoint')
        verify_manifest(best, base, kind='trainer_checkpoint')
        final = out / 'final_adapter'
        if final.exists():
            raise UnslothIntegrationError('Final adapter path already exists')
        final.mkdir()
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))
        manifest = write_manifest(final, base, kind='final_adapter')
        verify_manifest(final, base, kind='final_adapter')
        best_score = float(trainer.state.best_metric)
        if not math.isfinite(best_score):
            raise UnslothIntegrationError('Missing or non-finite validation metric')
        result = {'ok': True, 'mode': 'EXPERIMENTAL UNSLOTH GPU', 'model': base['model'],
                  'best_val_loss': best_score, 'best_checkpoint': str(best), 'adapter_dir': str(final),
                  'adapter_sha256': manifest['files'].get('adapter_model.safetensors'),
                  'trainable_parameters': trainable, 'stop_diagnosis': callback.stop_diagnosis,
                  'note': 'No automatic agent retries. Actual checkpoint resumption is not verified.'}
        _write_json_atomic(out / 'result_receipt.json', result)
        _write_json_atomic(out / 'run_state.json', {'status': 'completed', 'backend': 'unsloth'})
        return result
    except BaseException as exc:
        _write_json_atomic(out / 'run_state.json', {'status': 'blocked', 'backend': 'unsloth', 'error_type': type(exc).__name__})
        raise
