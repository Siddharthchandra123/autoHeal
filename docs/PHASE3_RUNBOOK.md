# Phase 3 recovery runbook (Windows PowerShell)

Phase 3 is a **CPU-only mock** recovery controller. The checkpoint marker is **not trainable weights**.

## Start a new experiment

```powershell
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/baseline.yaml --workspace runs
```

Inspect `runs\EXPERIMENT\recovery_state.json` (source of truth), `recovery_events.jsonl` (audit), `trial-01\result_receipt.json` and `trial-01\best_checkpoint.json` (simulated marker). The controller refuses to write `config\optimized.yaml` unless `--writeback` is explicitly requested.

## Resume an interrupted experiment

```powershell
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/baseline.yaml --resume 'runs\EXPERIMENT'
```

If a prior attempt has an integrity-verified result receipt, reconciliation commits the exact attempt once and continues without rerunning it. If the journal records `RUNNING` and no receipt, the controller **blocks**. Check that no local or external training process remains active before deciding how to recover.

```powershell
# ONLY after confirming the earlier job is stopped:
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/baseline.yaml --resume 'runs\EXPERIMENT' --abandon-interrupted
```

This counts the abandoned attempt against `max_trials`; no training or metrics from an uncertain attempt are promoted. A stale `.controller.lock` from a killed process may block access; manually verify no controller is still running before including `--clear-stale-lock`. Never clear a live lock. If a receipt is corrupt, the experiment stays blocked pending investigation; do not bypass checkpoint verification.

## Failure boundaries

- Atomic JSON state writing uses a temporary file, fsync and replacement. It does not claim durability across all network filesystems or power-loss scenarios.
- This is a single-host lock; it is not distributed coordination for Kubeflow. Production needs Kubernetes ownership and external transactional storage.
- SHA256 verifies marker integrity, **not provenance/authenticity or real model loadability**. Changes to the receipt and checkpoint together require stronger future trust controls.
- The Meta agent and `hf` adapter are NOT integrated into this durable CLI; this is the verified deterministic local mock path.
- `active.yaml` is selected via validation loss, **not promoted to production**. No untouched-test evaluation is implemented.
