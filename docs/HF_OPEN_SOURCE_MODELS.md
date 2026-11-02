# Hugging Face Open-Source Training in AutoTune AI (v0.8.0)

## Scope

The new **unified model CLI** currently supports two real training adapters:

- `causal-lm`: the existing validated-config PEFT LoRA adapter (sample `sshleifer/tiny-gpt2`). Produces adapter weights and HF Trainer checkpoints. CPU training is allowed only when the config requests it; 4-bit QLoRA still requires a supported GPU.
- `text-classification`: new CPU full fine-tuning adapter (sample `google/bert_uncased_L-2_H-128_A-2`). Produces model weights, training/evaluation metrics, full Trainer checkpoints, label mapping, run provenance and SHA-256 manifests.

Vision, speech, sequence-to-sequence and token classification are **future plugins**, not currently supported.

## Train a model (PowerShell, from project root)

```powershell
py -3.11 -m venv .venv-hf
.\.venv-hf\Scripts\python.exe -m pip install -r requirements-lora.txt
.\.venv-hf\Scripts\python.exe -m pytest -q

# Offline: validate BERT training plan, data labels and resource caps
.\.venv-hf\Scripts\python.exe -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml
# Real CPU model download + training; requires Internet
.\.venv-hf\Scripts\python.exe -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml --execute --output outputs/bert-tiny-cpu-01
# Verify final artifact without downloads
.\.venv-hf\Scripts\python.exe -m autotune.hf_models_cli --config config/hf_bert_tiny_cpu.yaml --verify outputs/bert-tiny-cpu-01/best_model
```

For causal language modeling, substitute `--config config/lora_public_cpu.yaml` in those commands; the existing LoRA adapter is invoked. The model repository ID and `hf.target_modules` must match the architecture; not all model IDs are compatible with the same PEFT configuration.

## Configure your own HF model

1. Browse the Hugging Face Hub and select a model **whose license permits your use**. Verify architecture, tokenizer, size, trust requirements, and any access conditions. A public repo is not automatically open-source.
2. For BERT-like sequence classification, change `model.name` in `config/hf_bert_tiny_cpu.yaml`; retain `task: text-classification`. Use two JSONL files with `text` and `label` fields. Training labels must cover every validation label. Use genuinely separate examples, and preserve an independent test set for final promotion.
3. For GPT-like causal LMs, change `model.name` and `hf.target_modules` in `config/lora_public_cpu.yaml`. Inspect target layer names before picking; CPU sizes should be tiny. Initial remote download is required unless a model was cached previously.
4. Run preflight first, then `--execute` and inspect outputs. Re-run with new output directory for a new experiment; never overwrite a run in place.

## Observability, safety and next steps

The classification adapter writes `training_logs.jsonl`, `metrics.jsonl`, `decisions.jsonl`, `monitor_decisions.jsonl` and a `best_model/autotune_manifest.json`. It logs *paired* training/validation losses to the existing monitoring engine and can request an early stop. It does **not yet** call the Meta diagnostic agent or launch a second real HF trial automatically. No MLflow/Kubeflow production integration is claimed. The LoRA adapter remains separate. The immediate next step is to run the small public model on Windows, verify a full checkpoint resume, then bridge HF jobs into the durable controller and MLflow.

Checkpoint manifests detect accidental changes, not deliberate forgery by someone able to rewrite both artifacts and manifests. `revision: main` is not immutable; prefer a pinned commit hash for reproducible deployments. Use trusted model repositories and `trust_remote_code=False`. Keep model/token IDs out of Git. Never use held-out test metrics for hyperparameter search.
