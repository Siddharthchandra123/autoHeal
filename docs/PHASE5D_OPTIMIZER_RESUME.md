# Phase 5D - Controlled CPU optimizer/scheduler resume test (v0.8.2)

## Why this test exists

The user independently verified (2026-11-01) that an existing Tiny GPT-2 LoRA adapter is loadable in a fresh model instance. That confirms inference reload, not that an interrupted HF Trainer can resume its optimizer and LR scheduler. The original completed output `outputs/gpt2-cpu-01` MUST NOT be edited or resumed. This exercise creates a new output with the same tiny model but a two-epoch, four-step immutable plan.

## Prerequisites

- Windows 11 + VS Code PowerShell; current project `.venv` contains `torch`, `transformers`, `datasets`, `peft`, `accelerate`, and `pyyaml`.
- Internet only if model is not cached. Public `sshleifer/tiny-gpt2` needs no HF token.
- Open the project root (`autoHeal_finetune`, or the extracted v0.8.2 package).
- The destination `outputs/gpt2-resume-probe-01` must NOT already exist or must be empty; never remove production/training results just to make this work.

## Step 1 - Verify 185 offline tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.hf_models_cli --config config/lora_cpu_resume_probe.yaml
```

## Step 2 - Fresh run, intentionally pause at training step 2

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_models_cli --config config/lora_cpu_resume_probe.yaml --execute --pause-after-step 2 --output outputs/gpt2-resume-probe-01
```

Expected: JSON `status: paused`, `global_step: 2`, `target_step: 4`. Verify `outputs/gpt2-resume-probe-01/checkpoints/checkpoint-2/autotune_manifest.json`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, and `rng_state*.pth`. It intentionally does NOT write `final_adapter` or `result_receipt.json` at this point.

## Step 3 - Resume that exact paused run, with unchanged config

```powershell
.\.venv\Scripts\python.exe -m autotune.hf_models_cli --config config/lora_cpu_resume_probe.yaml --execute --resume-checkpoint outputs/gpt2-resume-probe-01/checkpoints/checkpoint-2 --output outputs/gpt2-resume-probe-01
```

Expected: checkpoint-2 hash and run-state validation -> Trainer resumes -> step increases to 4 -> new final adapter and result receipt created -> `outputs/gpt2-resume-probe-01/resume_proof.json` written. The resume callback checks at `on_train_begin` that the trainer observed checkpoint step 2, nonempty optimizer state, and a scheduler step at least 2 before more gradient updates are taken. After completion it checks the final step is greater than the checkpoint step.

## Step 4 - Inspect proof (do not submit private data or API keys)

```powershell
Get-Content outputs\gpt2-resume-probe-01\resume_proof.json
Get-Content outputs\gpt2-resume-probe-01\run_state.json
```

`trainer_begin_step: 2`, `trainer_end_step: 4`, `optimizer_state_nonempty_at_resume: true`, and `scheduler_last_epoch_at_resume` at least 2 are the main acceptance checks. This is a practical Trainer-resume smoke test, **not** proof of bitwise replay equivalence across all frameworks/hardware.

## Safety, recovery and limitations

- This workflow requires an intentional `paused` state and refuses an already completed run, a modified config/data set, an outside-run checkpoint, or tampered manifest.
- The resume CLI is intentionally a one-shot operator-initiated action; it does not yet implement agent-led real Hugging Face retries.
- If stage 1 fails, do NOT run stage 2. Share redacted output for diagnosis.
- If stage 2 fails, the run records `status: blocked`. Inspect error messages; do not silently delete lock files or overwrite output. Use a new output name for a clean retry if needed.
- On each training run, HF warnings about GPT-2 `masked_bias`, pad tokens or missing public HF_TOKEN can be informational; exceptions and `ok:false` are not.
- Local checksums detect accidental corruption, not tampering by an attacker who can replace both files and manifest.
- Existing `gpt2-cpu-01` is a completed reference and cannot prove resume. Use it only for read-only audit / adapter reload.
- Model quality is not inferred from four training and two validation examples.

## Next

After the user confirms a genuine Windows result and `resume_proof.json`, integrate the HF training driver into the durable agentic controller with bounded fresh-attempt configuration patches. Only then add local MLflow tracking, Kubeflow CPU orchestration and Katib trials.
