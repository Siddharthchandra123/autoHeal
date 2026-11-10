# AutoTune AI UI redesign (Phase 8 visual refresh)

## What changed
- Reworked the Next.js app shell into a dark industrial-tech mission-control layout with graphite surfaces, fine borders, and electric-yellow accents.
- Added a five-agent startup sequence (Optimizer, Monitor, Recovery, Knowledge, Deployment) using animated SVG motion paths.
- Startup sequence appears once per browser session and can be skipped. It respects reduced-motion preferences.
- Refreshed the overview page while preserving the existing MLflow `getExperiments()` integration and experiment detail links.
- Preserved the Knowledge Hub route and its existing API calls/styles; only its visual tokens were restyled.
- Marked not-yet-implemented sidebar destinations as `SOON` rather than routing them to placeholder pages.

## Run on Windows PowerShell
```powershell
cd .\web
npm install
npm run typecheck
npm run build
npm run dev
```
Open http://127.0.0.1:3000.

## Notes
- No backend API routes or data models were changed.
- Overview values are sourced from the existing MLflow API; offline state is explicitly shown when the API is unavailable.
- The five moving pointers are an intro animation only. They are not presented as real live agent activity.
- If you have made local changes after the v0.14.0 Phase 8 archive, merge the `web` folder changes into that working copy rather than replacing your entire project.
