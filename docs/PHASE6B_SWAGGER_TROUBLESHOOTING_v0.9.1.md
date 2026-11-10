# AutoTune AI v0.9.1 - Optional Swagger API Docs (Windows CPU)

## Why `/docs` returned 404

In v0.9.0, `autotune.viewer_api` intentionally set `docs_url=None`, `redoc_url=None` and `openapi_url=None` to minimize the read-only viewer's HTTP surface. There is no `/` route. A 404 on these paths does not indicate a broken server.

## Supported endpoints

- `GET http://127.0.0.1:8000/api/health` - local viewer health, does **not** probe MLflow
- `GET http://127.0.0.1:8000/api/experiments` - experiments from the local MLflow server
- `GET http://127.0.0.1:8000/api/experiments/1/runs` - bounded runs
- `GET http://127.0.0.1:8000/api/runs/{32-hex-run-id}` - sanitized run detail
- `GET http://127.0.0.1:8000/api/runs/{32-hex-run-id}/metrics/eval_loss` - bounded series

## v0.9.1 recommended opt-in for development

Update the viewer with the v0.9.1 patch. Stop Uvicorn with Ctrl+C. In the same PowerShell window:

```powershell
$env:AUTOTUNE_ENABLE_DOCS = '1'
.\.venv\Scripts\python.exe -m uvicorn autotune.viewer_api:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000/docs`. You may need a working internet connection for the standard Swagger UI static assets used by FastAPI.

To disable docs on next restart, use `Remove-Item Env:AUTOTUNE_ENABLE_DOCS` or ` $env:AUTOTUNE_ENABLE_DOCS = '0'`. Docs stay disabled by default.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/openapi.json | Select-Object openapi, info
```

## Security boundary

Do not bind the local viewer or MLflow to `0.0.0.0` or forward these ports to public networks. Swagger is for local development and does not add authentication. Keep the API read-only; permissions and auth are required before shared deployment. MLflow remains a separate server on loopback port 5000.
