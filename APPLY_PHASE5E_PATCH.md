# Applying Phase 5E patch v0.8.6 to a working Windows project

1. Copy/backup existing source and `AGENTS.md`. Never delete `.venv` or `outputs`.
2. **Extract this patch into the `autoHeal_finetune` PROJECT ROOT**, preserving directories such as `autotune/`, `config/` and `docs/` (not a new subfolder).
3. Run `python -m pytest -q`, then `python -m autotune.hf_selfheal --config config/lora_selfheal_cpu.yaml`.
4. To exercise actual HF CPU training and the controller, choose a **fresh** workspace `outputs/hf-selfheal-injection-01` and use explicit `--execute --fault-inject-overfit`.
5. No change is made to the existing `outputs/gpt2-cpu-01` or `outputs/gpt2-resume-probe-01` by these commands.

This project is source version v0.8.6. Test suite in build environment: 208/208 passed; **live two-attempt HF CPU run is not yet verified**.
