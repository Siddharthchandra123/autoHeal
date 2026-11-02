"""One mock trial per Katib Job; NOT a real training benchmark."""
import argparse
from pathlib import Path
import tempfile
import yaml
from .mock_trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate',type=float,required=True)
    parser.add_argument('--lora-dropout',type=float,required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path('config/baseline.yaml').read_text())
    config['train']['learning_rate'] = args.learning_rate
    config['train']['lora_dropout'] = args.lora_dropout
    # Katib controls cross-trial optimization; no nested controller retry here.
    with tempfile.TemporaryDirectory(prefix='autotune-katib-') as folder:
        result = train(config,Path(folder),1)
        print(f'val_loss={result.best_val_loss:.8f}',flush=True)
        print(f'best_step={result.best_step}',flush=True)

if __name__ == '__main__':
    main()
