"""Local JSONL logs with optional MLflow experiment tracking."""
from pathlib import Path
import json
import os
import math
from datetime import datetime, timezone


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


def event(file: Path, payload: dict):
    file.parent.mkdir(parents=True, exist_ok=True)
    item = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **payload}
    with file.open('a',encoding='utf8') as f:
        f.write(json.dumps(item) + '\n')


class Tracker:
    def __init__(self, trial_dir: Path, trial: int, config: dict):
        self.trial_dir = trial_dir
        self.file = trial_dir / 'metrics.jsonl'
        self.mlflow = None
        self.run = None
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir/'input_config.json', config)
        try:
            if os.environ.get('MLFLOW_TRACKING_URI'):
                import mlflow
                mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
                mlflow.set_experiment(config['project'])
                self.run = mlflow.start_run(run_name=f'trial-{trial}')
                mlflow.log_params({k: str(v) for k,v in config['train'].items()})
                self.mlflow = mlflow
        except Exception as exc:
            print('[MLflow disabled]', repr(exc))
            self.mlflow = None

    def log(self, step: int, train_loss: float, val_loss: float, extra: dict | None=None):
        """Write strictly JSON-compliant metric rows, including fault markers.

        Non-finite values are represented by null and a `nonfinite_metrics` list.
        The monitor replay command recognizes this marker as critical evidence.
        """
        nonfinite = []
        def safe_float(name, value):
            number = float(value)
            if not math.isfinite(number):
                nonfinite.append(name)
                return None
            return round(number, 8)

        row = {'step': step, 'train_loss': safe_float('train_loss', train_loss),
               'val_loss': safe_float('val_loss', val_loss)}
        if row['train_loss'] is not None and row['val_loss'] is not None:
            row['generalization_gap'] = round(row['val_loss'] - row['train_loss'], 8)
            if row['val_loss'] < 70:
                row['validation_perplexity'] = round(math.exp(row['val_loss']), 8)
        for key, value in (extra or {}).items():
            if isinstance(value, (int,float)) and not isinstance(value,bool):
                row[key] = safe_float(key, value)
        if nonfinite:
            row['nonfinite_metrics'] = nonfinite
        event(self.file, row)
        print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
        if self.mlflow:
            self.mlflow.log_metrics({k:v for k,v in row.items()
                                     if k not in ('step','nonfinite_metrics') and isinstance(v,(int,float))},
                                    step=step)

    def finish(self, status: str, reason: str):
        if self.mlflow:
            self.mlflow.set_tags({'autotune.status': status, 'autotune.reason': reason[:200]})
            self.mlflow.end_run(status='FAILED' if status == 'failed' else 'FINISHED')
