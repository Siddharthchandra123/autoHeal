# Phase 5B - Hugging Face LoRA / QLoRA GPU Pilot Runbook

**Checkpoint:** AutoTune AI v0.7, November 1, 2026. Stage 5B code and offline contracts are tested. Full Hugging Face downloads, Meta gated access, live GPU fine-tuning and optimizer resume are NOT verified. Phase 5A CPU GRU remains separately verified on Windows (95 tests, real `.pt` weights).

## What this stage implements

- Offline model/data/parameter preflight (`autotune/lora_contract.py`, `autotune/lora_cli.py`). Checks SHA256, train/val duplication, positive steps, safe config values; no network call in preflight.
- Optional single-GPU Hugging Face Trainer + PEFT LoRA code path (`autotune/lora_gpu.py`). Optional `hf.quantization: 4bit` for compatible bitsandbytes backends (QLoRA).
- Training-window loss aligned to evaluation interval; the monitor requests graceful stop and checkpoint, rather than trusting an LLM to terminate processes.
- Hash manifests on full checkpoints (adapter + optimizer + scheduler + trainer state + RNG), separate final adapter files, explicit verify and training resume path.
- Default command is dry-run, and training requires explicit `--execute`. Dataset texts are never forwarded to the diagnostic agent by the new pilot.
- **Not yet connected:** automated HF real-weight retry with Phase 4 advisor; hosted Meta Muse endpoint; GPU budget controller; Kubeflow/MLflow cluster.

## A. Continue locally with Windows VS Code (safe, zero download)

```powershell
# From the extracted AutoTune_AI_Phase5B folder
.\.venv\Scripts\python.exe -m pip install numpy
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.lora_cli --config config/lora_meta_gpu.yaml
.\.venv\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml
```

The expected response is `OFFLINE PLAN ONLY; no download or training`. It reports model, data SHA256, train/val counts, estimated steps and evaluation count. It does **not** verify that Meta has granted model access, or that GPU kernels work.

## B. Optional public CPU LoRA smoke (requires online access)

A separate Python 3.11/3.12 venv is recommended if your current Python 3.13 package combination has dependency issues. Do not overwrite the working Phase 5A venv just to install HF dependencies.

```powershell
py -3.11 -m venv .venv-hf
.\.venv-hf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-hf\Scripts\python.exe -m pip install torch
.\.venv-hf\Scripts\python.exe -m pip install -r requirements-lora.txt
.\.venv-hf\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml
.\.venv-hf\Scripts\python.exe -m autotune.lora_cli --config config/lora_public_cpu.yaml --execute --allow-cpu --output outputs/public-cpu-01
```

This downloads a *small public GPT-2-derived* test model. It is not Meta Llama. This is an optional external execution not performed in the build environment. Use the output run files and tests to inspect failures before trying a larger model. CPU fine-tuning may still be slow on some systems.

## C. Actual Meta Llama fine-tune on a supported GPU

1. Request/accept the relevant model license at https://huggingface.co/meta-llama and verify your account has repository access. Never share HF tokens in chat or commit them.
2. Prepare a supported GPU environment. Prefer the remote AMD MI300X Linux/ROCm environment, with a compatible ROCm PyTorch release. On NVIDIA use compatible CUDA PyTorch. Verify `torch.cuda.is_available()` and supported precision; ROCm PyTorch commonly reports the AMD device through the `torch.cuda` API.
3. Install matching `torch` for the platform first, followed by `requirements-lora.txt`. For QLoRA install a supported `bitsandbytes` release and validate hardware/ROCm backend separately.
4. Make sure GPU RAM and storage can handle the selected model; 1B is a pilot size, not a guarantee that every accelerator can run it.
5. Pin `hf.revision` to the specific base-model commit for reproducibility once verified.
6. Run the offline plan, then explicitly execute, save and verify the adapter. Use a fresh output per experiment.

Example command **on the compatible GPU host's terminal**, from the project root:

```bash
python -c "import torch; print(torch.__version__,torch.cuda.is_available())"
python -m autotune.lora_cli --config config/lora_meta_gpu.yaml
python -m autotune.lora_cli --config config/lora_meta_gpu.yaml --execute --output outputs/llama-run-01
python -m autotune.lora_cli --config config/lora_meta_gpu.yaml --verify-adapter outputs/llama-run-01/final_adapter
```

For Windows to remote Linux, connect via `ssh <username>@<GPU-HOST>` from PowerShell, copy the project securely, and run the host-compatible commands *in the Linux shell*. Never install Linux ROCm packages into a native Windows CPU venv.

## D. Explicit checkpoint resume

A **Trainer checkpoint** can include adapter, optimizer, scheduler and RNG state. A `final_adapter` contains inference/trainable adapter weights but cannot guarantee exact optimizer resume.

```bash
python -m autotune.lora_cli --config config/lora_meta_gpu.yaml --execute \
  --output outputs/llama-run-01 \
  --resume-checkpoint outputs/llama-run-01/checkpoints/checkpoint-4
```

Before resume, verify that the prior process is no longer running, the lock file has been appropriately investigated, the dataset/config has not changed, and the manifest is valid. The command rejects tampered files and altered inputs. Checkpoint verification of trusted local files is not an authenticity proof: optimizer pickle state from untrusted checkpoints can be unsafe; never resume artifacts from unknown sources.

## E. Artifacts and expected checks

`outputs/<run>/plan.json`, `run_state.json`, `training_logs.jsonl`, `metrics.jsonl`, `monitor_decisions.jsonl`, `events.jsonl`, `checkpoints/checkpoint-*/autotune_manifest.json`, `final_adapter/adapter_model.safetensors`, `final_adapter/autotune_manifest.json`, `result_receipt.json`.

**Stop/Review conditions:** no license or missing GPU, unsupported quantization, missing/overlapping split, no planned validation, corrupt adapter/optimizer, invalid resume provenance, nonfinite metric, or insufficient logging. If any occurs, do not claim Phase 5B's live GPU gate passed.

## F. Failure-resolution table

| Symptom | Likely cause | Safe action |
|---|---|---|
| `No module named numpy` warning in v0.6 | NumPy not installed | `python -m pip install numpy` |
| `No CUDA/ROCm device visible` | CPU PyTorch or absent GPU | Run dry-run or choose supported GPU host |
| 401/403 gated model | Meta license/account not approved | Request access; do not bypass terms |
| `bitsandbytes` backend error | Incompatible QLoRA stack | Set quantization `none` until backend verified |
| `No validation would run` | Too few updates / high eval_steps | Increase data/epochs, lower eval_steps |
| Missing optimizer/RNG file | Checkpoint incomplete | Block resume; inspect and rerun from clean baseline |
| `.lora_runner.lock` exists | Active or abruptly interrupted process | Verify no process still running before manual cleanup |
| Mismatched dataset/config hash | Input changed after checkpoint | New experiment; do not resume old optimizer state |

## Next integration gate (Phase 6)

After a successful supported-GPU run with reload, connect actual HF loss history and trainer checkpoints to the durable recovery controller and MLflow tracking server; preserve train/eval alignment and operator-approved retry safeguards. Do not present the current isolated HF pilot as a fully autonomous GPU self-healing loop.
