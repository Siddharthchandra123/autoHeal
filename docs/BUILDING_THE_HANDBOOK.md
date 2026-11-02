# Building the AutoTune AI Technical Handbook

The PDF is generated from `docs/HANDBOOK_SOURCE.md` with SVG diagrams drawn from version-controlled Graphviz `.dot` files. Editable Mermaid alternatives are stored beside them as `.mmd`. Source-of-truth diagrams should be updated in BOTH formats if architecture changes.

## Tooling

- Python >=3.11 + `python -m pip install -r requirements-docs.txt`
- Pandoc executable in `PATH`
- Graphviz executable `dot` in `PATH`

## Generate from the project root

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe scripts\build_handbook.py
```

This creates `AutoTune_AI_Handbook_v0.5.pdf` at the project root, using the current checked-in source and diagrams. For the next edition, update the version in the source, stylesheet footer and builder filename. The standalone PDF is also copied to the conversation's deliverable workspace when possible.

To avoid overclaiming, keep a verified checkpoint in `AGENTS.md`, update `docs/MASTER_EXECUTION_PLAN.md`, regenerate this PDF, and cite test results only from observed executions. The handbook regeneration requires active tooling; it is not a scheduled or background job.
