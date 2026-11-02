# Phase 5C - Auditing the first successful real Hugging Face LoRA CPU experiment

## Evidence received from the Windows user terminal (2026-11-01)

The user ran the v0.8.0 public causal-LM LoRA training adapter, not an offline mock:

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_models_cli --config config/lora_public_cpu.yaml --execute --output outputs/gpt2-cpu-01
```

Observed: `ok:true`, `sshleifer/tiny-gpt2`, one epoch, four train / two validation examples, two optimization steps, training loss ~10.4, validation loss **10.365182876586914**, **128 trainable LoRA parameters**, final adapter in `outputs/gpt2-cpu-01/final_adapter`, best full Trainer checkpoint in `outputs/gpt2-cpu-01/checkpoints/checkpoint-2`, and adapter weights SHA-256 `d7aabd74886d97b0d83136441891a2442dc8f99266177e7ff05a52c55570e0d1`. `stop_diagnosis:null` indicates no stop diagnosis; it does not imply the monitor had sufficient repeated observations to assess overfitting.

This is a working training/integration smoke test, **not evidence of useful model quality** or guaranteed adapter reload/optimizer resume. The tiny split is far too small for meaningful generalization conclusions. No GPU was required.

## Step 1 - Read-only audit (no model download)

Keep your existing `outputs/gpt2-cpu-01` folder, YAML and original JSONL data unchanged.

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_audit_cli --config config/lora_public_cpu.yaml --run outputs/gpt2-cpu-01
```

On success, this reports `ok:true`, the verified LoRA file count and checkpoint file count, and `checkpoint_has_training_state:true`. **This means expected state files and hashes are present, not that optimizer resume has been executed.** The audit validates `plan.json`, `run_state.json`, `result_receipt.json`, final adapter and best checkpoint under `checkpoints/`, including the model/config/data provenance. Changed configuration or datasets intentionally invalidate verification.

Do not edit or delete manifest files just to pass the audit. Keep generated artifacts private if data/model licensing requires it.

## Step 2 - Fresh CPU adapter reload and inference smoke

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_audit_cli --config config/lora_public_cpu.yaml --run outputs/gpt2-cpu-01 --reload
```

This first performs the same audit, then loads base `sshleifer/tiny-gpt2`, applies the saved PEFT adapter, and generates eight deterministic tokens for a short prompt. It can access the Hugging Face cache/Hub for model weights. Expected `adapter_reload_executed:true` and `generated_token_count:8` if successful. The actual generated text is deliberately omitted from the output; this is a loading/inference contract check, not a quality benchmark.

**If you see HF warnings:** an optional HF token improves request limits; GPT-2's Conv1D LoRA fan-in/fan-out adjustment and token-ID alignment are common. Warnings should still be investigated if the command fails or gives mismatched architecture errors.

## Important: Do NOT attempt to resume the already-completed run

The completed run has `final_adapter` already saved. An attempt to `--resume-checkpoint` into the same completed output folder is not a safe demonstration and may fail due to overwrite protection. A **separately staged interrupted-run test** is needed to show actual optimizer/scheduler/RNG restoration and the correct continuation of training. The audit deliberately never invokes `trainer.train()`.

## What comes next

- Build a controlled interrupt/resume harness with an isolated output folder and deterministic max-step objective, run on CPU.
- Wire genuinely paired live HF metrics and verified full checkpoint artifact identity into the Phase 3 recovery state journal.
- Permit the Meta advisor only to propose bounded config changes; audit new configs before launching separate attempts.
- Then implement persistent MLflow tracking; move to Kubeflow/Katib after the core real-model recovery path is verified.

## Verification status

- User terminal: **real CPU LoRA training succeeded** (reported by user, not reproduced in build environment).
- Build environment: new `hf_audit_cli` tests verify hash/provenance guards, unsafe-path rejection and mocked reload contract **without downloading model weights**.
- Still pending on user machine: read-only artifact audit, fresh adapter reload, real optimizer-state resume, real automatic HF recovery loop.
