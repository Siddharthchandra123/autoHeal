from copy import deepcopy
from .policy import validate_patch

def build_candidates(base_config):
    train = base_config["train"]
    lr, rank, dropout = float(train["learning_rate"]), int(train["lora_r"]), float(train["lora_dropout"])
    raw = [
        {"learning_rate": lr * 0.5},
        {"learning_rate": lr, "lora_r": 16 if rank != 16 else 8},
        {"learning_rate": lr * 0.8, "lora_dropout": min(0.35, dropout + 0.05)},
    ]
    result = []
    for i, patch in enumerate(raw, 1):
        patch = {k: v for k, v in patch.items() if v != train[k]}
        validated = validate_patch(patch, train)
        cfg = deepcopy(base_config)
        cfg["train"].update(validated)
        cfg["project"] = f'{base_config["project"]}-HPO-{i:02d}'
        result.append({"index": i, "patch": validated, "config": cfg})
    return result
