"""Deterministic CPU-only test harness. Metrics are SYNTHETIC, not benchmark results."""
from dataclasses import dataclass
from pathlib import Path
from .metrics import MonitorStream
from .tracking import Tracker, write_json

@dataclass
class RunResult:
    state: str
    reason: str
    best_val_loss: float
    best_step: int
    checkpoint: str
    history: list[dict]


def train(config: dict, trial_dir: Path, trial: int) -> RunResult:
    tracker = Tracker(trial_dir, trial, config)
    monitor = MonitorStream(trial_dir / "monitor_decisions.jsonl", config["monitor"])
    history = []
    best_val = float('inf')
    best_step = 0
    scenario = config['simulator']['scenario']
    lr = float(config['train']['learning_rate'])
    dropout = float(config['train']['lora_dropout'])
    state, reason = 'completed', 'All simulated evaluation steps completed'
    for i in range(1, 11):
        if scenario == 'overfit' and dropout < .12:
            tr = max(.35, 2.0 - .17*i)
            vl = 1.85 - .07*min(i, 4) + .11*max(i-4, 0)
        elif scenario == 'underfit' and lr < .00015:
            tr, vl = 1.7-.005*i, 1.78-.004*i
        else:
            tr = max(.48, 1.95 - .18*i)
            vl = max(.58, 1.95 - .145*i + .004*max(i-7,0)**2)
        row = {'step': i, 'train_loss': tr, 'val_loss': vl}
        history.append(row)
        tracker.log(i,tr,vl,extra={'learning_rate': lr, 'lora_dropout':dropout})
        if vl < best_val:
            best_val, best_step = vl, i
        decision = monitor.observe(row)
        if decision.should_stop:
            state, reason = decision.state, decision.explanation
            print(f'EARLY STOP: {state} — {reason}', flush=True)
            break
    checkpoint = trial_dir/'best_checkpoint.json'
    write_json(checkpoint, {'note':'SIMULATED checkpoint metadata, NOT model weights', 'best_step':best_step, 'best_val_loss':best_val, 'train':config['train']})
    tracker.finish(state,reason)
    return RunResult(state,reason,best_val,best_step,str(checkpoint),history)
