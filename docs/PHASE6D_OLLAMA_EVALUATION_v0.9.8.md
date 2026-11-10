# AutoTune AI — v0.9.8 / Phase 6D — SmolLM2 baseline versus fine-tuned evaluation

**Date:** 2026-11-03 | **Platform:** Windows 11, VS Code, PowerShell, local Ollama, CPU-first

## Verified and pending

User explicitly reported that all steps of the v0.9.7 GGUF workflow worked. Therefore the staging fix, tokenizer verification, GGUF file, `ollama create`, and an inference smoke test are **user-reported successful**, without attached terminal logs or independent reproduction here. Earlier Tiny GPT-2 real LoRA/MLflow results remain verified separately. Nothing here proves SmolLM2 quality improvement or production readiness.

This release adds a **read-only comparison utility**. The 13 dedicated offline unit tests and an 8-item offline preflight passed in the build environment. **Live comparison against Ollama remains to be executed on the Windows machine.** No model weights, LoRA artifacts, GGUF files, or Ollama registrations are modified.

## Why this is necessary

A model that can be trained, converted and run has passed a deployment compatibility test. Model quality is a different question. A toy 16-train / 4-validation dataset is too small to show generalized improvement. Hold out a separate evaluation set, use identical prompts and generation options, record the answers, and conduct blind or rubric-based human grading before promotion. Even then, eight items are a smoke test, not a statistically reliable benchmark.

## Comparison setup

- Baseline: `smollm2:135m-instruct-fp16` (Ollama's published FP16 instruction-tuned 135M model), chosen to better match the custom FP16 export. Confirm the actual baseline model digest and template; different model revisions can confound conclusions.
- Candidate: `autotune-smollm2:gguf-v095` (replace with the actual tag shown by `ollama list`).
- Evaluation: `sample_data/ollama_heldout_eval.jsonl` (8 questions on training operations/security), disjoint by exact question matching from the bundled SmolLM2 train and validation sets.
- Both use local `POST http://127.0.0.1:11434/api/generate` with `stream:false`, `raw:true`, fixed explicit ChatML formatting and system instruction, `temperature:0`, `seed:42`, `num_predict:128`, `num_ctx:2048`. This reduces prompt-template and decoding differences; cross-build and hardware variation can remain.
- Results: `evaluation_results.json` includes model digests, options, raw responses, token counts and durations. `human_review.csv` leaves verdict cells blank; no automatic pass/fail score or promotion.

## Windows PowerShell commands

Run from `C:\Users\Lenovo\Downloads\autoHeal_finetune` after overlaying the v0.9.8 update patch (back up source files first). Keep `.venv`, `.venv-gguf`, `outputs`, `mlflow.db` intact.

```powershell
ollama list
ollama pull smollm2:135m-instruct-fp16

.\.venv\Scripts\python.exe -m pytest -q tests\test_ollama_compare_v098.py

.\.venv\Scripts\python.exe -m autotune.ollama_compare

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv\Scripts\python.exe -m autotune.ollama_compare --execute --output "outputs/ollama-comparison-$stamp"

Invoke-Item "outputs\ollama-comparison-$stamp\human_review.csv"
```

For a different custom tag, append `--candidate your-model:tag`. To check additional training sets for exact question overlap, supply `--known-data <path>` for each relevant JSONL. Supplying any `--known-data` overrides the two bundled defaults, so include them explicitly if desired. Do not submit secrets or private data as prompts; evaluations are saved in plaintext on your own disk.

## Understanding results

1. Check that all 8 prompts completed for both local model tags, with matching generation options.
2. Review `human_review.csv` and grade both answers for factual correctness, directness, safety and completeness using the rubric. Add notes and a manual winner; **do not infer improvement from response length or token speed**.
3. Compare inference timing only after accounting for load time, warmed-up runs, CPU background activity and multiple repeated trials.
4. Compare the model digest / size / architecture / prompt template from local Ollama records. A baseline on a different revision or format can confound results.
5. Keep `model_promotion_approved: false`; require a larger independent test set, regression checks, license review, and security gates before promotion.

## Safety properties

- The tool only calls a hard-coded localhost address and two allowlisted API endpoints; it has no external URL input and no model mutation endpoints.
- Model name validation and `GET /api/tags` reject missing tags before generation.
- It rejects symlink evaluation files, oversized input, duplicate prompts, exact held-out overlap with known data, and nonempty output directories.
- It performs no arbitrary command execution and uses only the Python standard library plus pytest for tests.
- `raw:true` uses a single explicit ChatML wrapper for both models. It cannot guarantee that different underlying model tokenizer or revisions are identical.

## Next milestones

1. **User acceptance:** complete baseline and candidate inference and inspect CSV. Document paired manual grades and hashes in `AGENTS.md` and the next handbook.
2. **Independent model-quality and safety gate:** larger held-out corpus, leakage review, prompt-injection, privacy-leakage and harmful-output checks.
3. **Dashboard acceptance:** live Next.js-to-FastAPI-to-MLflow check, read-only on localhost.
4. **Security Sprint S0:** dependency audit, immutable model revisions, trusted artifact loading, dataset PII/poisoning triage and audit policy gates.
5. **Kubeflow/Katib:** CPU-first orchestration after locally reproducible safeguards; Unsloth remains an optional GPU path.

## References

- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Ollama 135M model tags: https://ollama.com/library/smollm2/tags
