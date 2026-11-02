# Phase 4 runbook - Meta diagnostic advisor (v0.5)

## Supported scope

Windows 11, Python 3.11-3.13, VS Code PowerShell. Offline CPU demonstration requires no Meta API key. Synthetic checkpoints are JSON metadata, **not trained model weights**.

## Start from a new extracted folder

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m autotune.doctor --mode mock
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.advisor_cli --provider mock --diagnosis overfit
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/agent_mock.yaml --workspace runs
```

Alternatively, run `run_phase4.ps1` if PowerShell allows project scripts. On a machine that blocks unsigned scripts, run the commands above. Avoid permanently weakening an organizational execution policy.

## Check artifacts

The new experiment directory contains `recovery_state.json`, `recovery_events.jsonl`, `trial-01/advisor_decision.json`, `trial-01/result_receipt.json`, `trial-02/config.yaml`, and `active.yaml`. The advisor decision binds input configuration hash, checkpoint digest, patch, output configuration hash, and a JSON record digest. An intact prior decision is replayed without sending the same request again.

## Optional hosted Meta test (may incur costs)

Install the optional client, then set `MODEL_API_KEY` in the local terminal or secret store. Do not paste the key in chat or write it into YAML.

```powershell
.\.venv\Scripts\python.exe -m pip install 'openai>=1.60,<3'
$env:MODEL_API_KEY = Read-Host 'Meta API key'
.\.venv\Scripts\python.exe -m autotune.advisor_cli --provider meta --diagnosis overfit
```

To use hosted advice inside *simulated* durable recovery:

```powershell
.\.venv\Scripts\python.exe -m autotune.recovery_cli --config config/agent_meta.example.yaml --workspace runs
```

No live call is made by default. If the key or API is missing, policy-controlled fallback is logged. `MODEL_API_KEY` can be removed with `Remove-Item Env:MODEL_API_KEY` after the test.

## Expected safeguards

Only a fixed set of finite numeric metrics and numeric hyperparameters may be supplied to Meta. Strict model reply schema (no code, commands, paths, or extra keys); check independent policy for direction, bounds, and max change. API timeout: 20 seconds, no automatic client retries. Invalid or unavailable response -> deterministic fallback. Unsupported failure states -> block. Never regenerate or silently replace a saved advisor record after interruption.

## Trouble and limitations

- Missing API key: `status=fallback` (normal in offline verification).
- Invalid proposed patch: `status=fallback`; inspect `advisor_decision.json` without exposing secret error text.
- Hash mismatch on advisor replay: block for human review and verify backups; do not rewrite record manually.
- Real model/HF fine-tuning, orchestration in Kubeflow, remote Meta success, GPU recovery and model deployment are **not** validated in this phase.
