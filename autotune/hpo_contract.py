from copy import deepcopy
from .config import validate_config
from .lora_contract import make_plan

IMMUTABLE = ("model", "revision", "device", "quantization", "target_modules",
             "train_sha256", "validation_sha256", "data_format", "loss_mode")

def validate_trial_config(config, baseline_plan, config_path):
    cfg = validate_config(deepcopy(config), mode="hf")
    plan = make_plan(config_path)
    for key in IMMUTABLE:
        if plan.get(key) != baseline_plan.get(key):
            raise ValueError(f"HPO trial changed immutable field {key}")
    if plan["device"] != "gpu":
        raise ValueError("HPO is GPU-only; refusing CPU trial")
    if plan["quantization"] != "none":
        raise ValueError("This pilot supports LoRA without quantization")
    return cfg, plan
