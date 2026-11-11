# AutoTune AI v0.14.2 — FastAPI / UI fine-tuning bridge patch

This is an overlay patch for the v0.14.1 source tree. It does not replace the
trainer or the adaptive dataset pipeline. It changes the Knowledge Engine job
worker to launch the same `python -m autotune.knowledge_orchestrator` entrypoint
used from PowerShell, using the FastAPI process' exact `sys.executable` and the
repository root as its working directory.

## What changes

- FastAPI jobs execute the CLI module in a child process using the same Python
  interpreter as the API server.
- Captures stdout and stderr into `knowledge_store/jobs/<job-id>.log`.
- Persists a bounded live log tail in the job JSON for dashboard polling.
- Returns execution metadata (Python executable, working directory, output
  directory, candidate count and process exit code) for diagnosis.
- Dashboard defaults to one candidate on CPU, lets the operator select 1–3,
  and displays the latest training process logs.
- Existing adaptive preparation, training, validation and held-out test logic
  remains in `knowledge_orchestrator.py` and is not duplicated.

## Apply on Windows PowerShell

1. Stop the FastAPI and Next.js processes before copying files.
2. Extract this ZIP into the AutoTune project root, preserving folders, and
   choose **overwrite** when prompted.
3. Activate the project's `.venv` from the repository root.
4. Start FastAPI from the repository root with the same interpreter:

   ```powershell
   python -m uvicorn autotune.knowledge_engine:app --host 127.0.0.1 --port 8010
   ```

5. In another PowerShell window, start the dashboard:

   ```powershell
   Set-Location .\web
   $env:NEXT_PUBLIC_KNOWLEDGE_API = 'http://127.0.0.1:8010'
   npm run dev
   ```

6. Verify API health:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8010/api/knowledge/health
   ```

7. In the UI, select a dataset, run Preflight first, then select **1 candidate**
   for the first CPU fine-tuning test. The candidate count means model candidates;
   the orchestrator still runs the configured learning-rate trials for that
   candidate.

## Where to inspect a job

- `knowledge_store/jobs/<job-id>.json` — persistent job status and latest log tail.
- `knowledge_store/jobs/<job-id>.log` — complete combined stdout/stderr.
- `knowledge_store/training_runs/<job-id>/` — configs, events, trial outputs and
  final experiment result.

The API is unauthenticated local-development software. Keep it bound to
`127.0.0.1`; do not expose it to a LAN or public network.

## Verification boundary

The patch's Python modules should be syntax-checked and the focused tests run in
the user's Windows `.venv`. This package does not claim that the user's FastAPI
server, npm build, browser workflow, or a full UI-triggered training run has
already been executed. The decisive acceptance test is to launch a one-candidate
job from the UI and confirm the same interpreter, CLI command, logs, and output
artifacts as the working PowerShell path.
