# AutoTune AI Docker Compose stack

Place these files at the repository root, alongside `requirements.txt`, `autotune/`, `config/`, `pipelines/`, and `web/`.

## Endpoints
- Next.js: http://127.0.0.1:3000
- API gateway: http://127.0.0.1:8000
- MLflow: http://127.0.0.1:5000

The gateway routes `/api/experiments`, `/api/runs`, and their subpaths to `autotune.viewer_api`; other paths go to `autotune.knowledge_engine`.

## Preflight
- `requirements.txt` must contain `uvicorn`, FastAPI dependencies and `mlflow` if imported by viewer code.
- `web/package.json` must have `build` and `start` scripts.
- `autotune.viewer_api` must export a zero-argument `create_app()` factory.
- Add any other project files required at runtime to `Dockerfile.api`.

## Start from repository root (PowerShell)
```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

## Verify
```powershell
docker compose logs --tail 100 mlflow knowledge-api viewer-api api-gateway web
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:8000/api/experiments
```
The Knowledge Engine health route may differ; inspect its `/openapi.json`.

## Stop
```powershell
docker compose down
```
This preserves MLflow data. `docker compose down -v` deletes the persistent MLflow volume; do not use unless you intend to erase it.

## Notes
- APIs bind to `0.0.0.0` inside containers.
- Containers reach MLflow at `http://mlflow:5000`; the browser reaches host-published ports.
- Frontend `AUTOTUNE_VIEWER_API` is set to the existing `http://127.0.0.1:8000` gateway default. If Next.js inlines this variable at build time, use a `NEXT_PUBLIC_` variable in `web/lib/api.ts` or add the appropriate Next.js build argument.
- Development-only: ports are bound to host loopback. Add authentication/TLS before any external exposure.
