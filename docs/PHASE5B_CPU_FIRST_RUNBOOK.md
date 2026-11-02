# Phase 5B CPU-first runbook - AutoTune AI v0.7.2

**Decision (2026-11-01):** User has no dedicated GPU. CPU is the default execution target; GPU stages are optional. The existing Phase 5A GRU is genuinely Windows-verified (95 tests). Phase 5B Hugging Face LoRA/PEFT is an unverified online smoke path until run. Meta Muse diagnostic API can be called from CPU; remote service performs inference. Do not claim that hosted Meta calls have been tested.

## What this changes

- Keep CPU training, diagnostics, checkpoint/recovery contracts and policy engine as primary deliverables.
- Use `config/lora_public_cpu.yaml` (small public `sshleifer/tiny-gpt2`, `hf.device: cpu`, `quantization: none`), not `config/lora_meta_gpu.yaml`.
- No 4-bit QLoRA on CPU in this pilot. Do not try licensed Meta Llama on local CPU using this configuration.
- Kubeflow/Katib, MLflow and the dashboard can be developed on CPU, although Kubernetes resource overhead should be measured. GPU benchmarking is optional and postponed.
- Model-card and validation metrics from a tiny test model only demonstrate pipeline correctness, not model quality or scaling.

## Windows PowerShell commands (from extracted project root)

First, with existing venv for offline tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml
```

For genuine HF PEFT CPU LoRA, create a fresh Python 3.11 virtual environment so working Phase 5A packages remain intact:

```powershell
py -3.11 -m venv .venv-hf
.\.venv-hf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-hf\Scripts\python.exe -m pip install torch
.\.venv-hf\Scripts\python.exe -m pip install -r requirements-lora.txt
.\.venv-hf\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml
.\.venv-hf\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml --execute --allow-cpu --output outputs/public-cpu-01
```

This requires internet access once to download the public model and dependencies, and CPU training could be slow. Do not reuse `outputs/public-cpu-01` when it already has output; choose another run directory.

After real training **if it completes successfully**, verify the LoRA adapter can reload:

```powershell
.\.venv-hf\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml --verify-adapter outputs/public-cpu-01/final_adapter --allow-cpu
```

Inspect `outputs/public-cpu-01/metrics.jsonl`, `monitor_decisions.jsonl`, `result_receipt.json`, `checkpoints/`, and `final_adapter/`. Full resume from a Trainer checkpoint and separate autonomous retry are distinct features and require their own live tests.

## Verification and next milestones

1. Existing baseline: 95/95 user Windows verified for Phase 5A; 134/134 build-env verified for v0.7 Phase 5B offline tests. v0.7.2 adds a CPU routing/CLI test; package tests should be rerun.
2. CPU PEFT first live gate: actual adapter train, nonempty validation metrics, checksum manifest, adapter reloaded, safe operator review.
3. CPU checkpoint/resume gate: interrupt a run at a safe point, inspect full trainer state/manifest, restart and compare reproducible state.
4. CPU MLflow -> agent proposal -> deterministic guard -> second trial -> best validation comparison -> audit log. This is the real self-healing integration target without GPU.
5. CPU Kubernetes/Kubeflow/Katib optional after standalone loop verified; GPU is not a blocker.

## Known limitations

- Live Hugging Face/PEFT training has NOT been tested during this documentation/CPU-route update. Versions of transformers, peft and PyTorch may require compatibility adjustments.
- Monitoring of real HF training requires matched train/eval metrics; current pilot records averaged training windows.
- `final_adapter` is not a complete optimizer/scheduler state; only a verified Trainer checkpoint may resume optimization.
- Tests on synthetic data do not prove the model generalizes or that an autonomous tuning loop improves performance.

## Acceptance evidence requested from Windows

Send test count; preflight plan (`device: cpu`); actual train output; best eval loss; checkpoint manifest/adapter reload result; any traceback. Do not share access tokens.
