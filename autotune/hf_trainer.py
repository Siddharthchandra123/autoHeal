"""Single-GPU HF Trainer + PEFT LoRA pilot. Requires authorized model and GPU."""
from pathlib import Path
from .metrics import MonitorStream
from .mock_trainer import RunResult
from .tracking import Tracker
import re


def _target_modules(config: dict) -> list[str]:
    """Use model-specific LoRA targets from the pinned training config."""
    values = config.get("hf", {}).get("target_modules", ["q_proj", "v_proj"])
    if (not isinstance(values, list) or not values or
            any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", x) for x in values) or
            len(set(values)) != len(values)):
        raise ValueError("hf.target_modules must be a non-empty unique list of safe module names")
    return values


def train(config: dict, trial_dir: Path, trial: int) -> RunResult:
    # Imports intentionally deferred so mock testing needs no torch/CUDA installation.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling)
    trcfg = config['train']
    tracker = Tracker(trial_dir, trial, config)
    history = []
    monitor = MonitorStream(trial_dir / "monitor_decisions.jsonl", config["monitor"])

    class MonitorCallback(TrainerCallback):
        def __init__(self):
            self.train_loss = None
            self.extra = {}
            self.state = 'completed'
            self.reason = 'Training completed without monitored stop condition'
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                self.train_loss = float(logs['loss'])
            if logs:
                for key in ('learning_rate','grad_norm','epoch'):
                    if key in logs and logs[key] is not None:
                        self.extra[key] = float(logs[key])
            return control
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and 'eval_loss' in metrics and self.train_loss is not None:
                item = {'step':int(state.global_step), 'train_loss':self.train_loss,
                        'val_loss':float(metrics['eval_loss'])}
                if 'grad_norm' in self.extra:
                    item['grad_norm'] = self.extra['grad_norm']
                history.append(item)
                gpu_memory = (torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0)
                tracker.log(**item, extra={**self.extra,'gpu_peak_allocated_gib': gpu_memory})
                decision = monitor.observe(item)
                if decision.should_stop:
                    self.state, self.reason = decision.state, decision.explanation
                    control.should_training_stop = True
                    control.should_save = True
            return control

    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset('json', data_files={
        'train':config['data']['train_file'],
        'validation':config['data']['validation_file']})
    if 'text' not in data['train'].column_names or 'text' not in data['validation'].column_names:
        raise ValueError('Each JSONL record must have a text field')
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=int(config['model']['max_length']))
    tokenized = data.map(tokenize, batched=True, remove_columns=data['train'].column_names)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(trcfg['lora_r']),
        lora_alpha=int(trcfg['lora_alpha']), lora_dropout=float(trcfg['lora_dropout']),
        target_modules=_target_modules(config)))
    args = TrainingArguments(
        output_dir=str(trial_dir/'checkpoints'),
        learning_rate=float(trcfg['learning_rate']),
        num_train_epochs=int(trcfg['num_train_epochs']),
        per_device_train_batch_size=int(trcfg['per_device_train_batch_size']),
        per_device_eval_batch_size=int(trcfg['per_device_train_batch_size']),
        gradient_accumulation_steps=int(trcfg['gradient_accumulation_steps']),
        weight_decay=float(trcfg['weight_decay']),
        eval_strategy='steps', save_strategy='steps',
        eval_steps=int(trcfg['eval_steps']), save_steps=int(trcfg['eval_steps']),
        logging_strategy='steps', logging_steps=max(1,int(trcfg['eval_steps'])//2),
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=2, report_to='none', fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=42)
    callback = MonitorCallback()
    trainer = Trainer(model=model, args=args,
        train_dataset=tokenized['train'], eval_dataset=tokenized['validation'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False),
        processing_class=tokenizer, callbacks=[callback])
    try:
        trainer.train()
        if not history:
            raise RuntimeError('No evaluation occurred; reduce eval_steps or increase dataset/training steps')
        best = min(history, key=lambda x:x['val_loss'])
        output = trial_dir/'best-adapter'
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        tracker.finish(callback.state,callback.reason)
        return RunResult(callback.state,callback.reason,float(best['val_loss']),int(best['step']),str(output),history)
    except Exception:
        tracker.finish('failed','Training exception')
        raise
