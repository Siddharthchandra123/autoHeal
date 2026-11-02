# Phase 5A operator runbook - v0.6

## Goal

Confirm **real model weights** can be trained, hashed, loaded and selected safely, without cloud spend. The CPU byte-level GRU is NOT Meta Llama and not PEFT LoRA.

## Installation (Windows 11 / VS Code PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-real-cpu.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.real_cli --config config/real_cpu.yaml --workspace runs_real
```

Work from repository root, not `autotune/`. First installation uses network access to download dependencies; training itself is offline. Check PyTorch Windows build compatibility. Do not paste credentials into your terminal logs or configuration files.

## Success checks

- Test suite: 95 passed in the reference environment; counts can change after edits.
- CLI: `mode` is `REAL PyTorch CPU` with `ok: true`, `status: completed` or a bounded recovery status.
- `trial-01/best_checkpoint.pt` exists and contains actual model and optimizer tensors.
- `trial-01/result_receipt.json` checkpoint SHA256 matches bytes on disk.
- Run contains `recovery_state.json`, `recovery_events.jsonl`, `active.yaml`, and training/diagnostic JSONL.

## Resume safely

```powershell
.\.venv\Scripts\python.exe -m autotune.real_cli --config config/real_cpu.yaml --workspace runs_real --resume runs_real/<experiment-folder>
```

For a completed attempt, resume verifies the checkpoint again, without running the trainer a second time. For RUNNING with no validated receipt, controller BLOCKS. Check that no process is training before any manual intervention; do not delete state journals to force continuation. If the input dataset or config has changed, resume is rejected. Do not manually unpickled untrusted checkpoints; loader uses `weights_only=True`.

## Limitations

- Genuine small CPU model learning, not pretrained Meta Llama fine-tuning.
- A forced-threshold retry test proves controller transitions, **not** diagnosed genuine underfitting.
- No partial-epoch crash recovery, no external GPU scheduler cancellation, no promotion/deployment action.
- Final untouched holdout evaluation and real LoRA training come later.

## Next: Stage 5B

On a compatible GPU, replace the tiny baseline with a licensed, pinned Meta Llama checkpoint using Transformers + PEFT and verified data handling. Validate adapter weights can be independently reloaded, optimizer/scheduler state can resume safely, and run id + model revision + data hashes are recorded. Start with a manual single-run smoke test; only then connect to the autonomous durable controller.
