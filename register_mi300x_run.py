import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

OUTPUT = Path("outputs/gpt2-mi300x-01")
CONFIG = Path("config/lora_mi300x.yaml")
TRACKING_URI = "http://127.0.0.1:5001"

with open(OUTPUT / "result_receipt.json") as f:
    receipt = json.load(f)

with open(OUTPUT / "metrics.jsonl") as f:
    metrics = json.loads(f.readlines()[-1])

assert receipt["ok"], "Training receipt is not successful"
assert (OUTPUT / "final_adapter").is_dir()
assert (OUTPUT / "checkpoints/checkpoint-2").is_dir()

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment_name = "AutoTune-AI-MI300X"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="tiny-gpt2-mi300x-lora-01"
) as run:

    mlflow.set_tags({
        "project": "AutoTune AI",
        "hardware": "AMD Instinct MI300X VF",
        "gpu_architecture": "gfx942",
        "backend": "ROCm",
        "rocm_version": "10.1.0",
        "hip_version": "7.16.26385",
        "pytorch_version": "2.14.0+rocm10.1.0",
        "training_framework": "Hugging Face PEFT LoRA",
        "execution_status": "completed",
        "gpu_validation": "passed",
        "fault_injected": "false",
    })

    mlflow.log_params({
        "model": receipt["model"],
        "device": "gpu",
        "quantization": "none",
        "learning_rate": 0.0005,
        "epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 16,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "trainable_parameters": receipt["trainable_parameters"],
        "dataset_train_examples": 4,
        "dataset_validation_examples": 2,
    })

    mlflow.log_metrics({
        "train_loss": float(metrics["train_loss"]),
        "validation_loss": float(metrics["val_loss"]),
        "train_runtime_seconds": 11.5685,
        "training_steps": int(metrics["step"]),
        "trainable_parameters": int(receipt["trainable_parameters"]),
    })

    # Upload training outputs, adapter and resumable checkpoint.
    mlflow.log_artifacts(str(OUTPUT), artifact_path="training_output")

    # Preserve the exact training configuration.
    mlflow.log_artifact(str(CONFIG), artifact_path="configuration")

    print("\nMLFLOW REGISTRATION SUCCESSFUL")
    print("Experiment:", experiment_name)
    print("Experiment ID:", experiment_id)
    print("Run ID:", run.info.run_id)
    print("Tracking URI:", TRACKING_URI)
    print("Artifacts: uploaded")
