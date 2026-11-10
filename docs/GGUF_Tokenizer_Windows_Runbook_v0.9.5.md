# AutoTune AI v0.9.5 - Safe GGUF tokenizer correction (Windows CPU)

## What happened
The v0.9.4 staging helper rejected a legitimate model-specific special token because it was not duplicated in `additional_special_tokens`. SmolLM2 tokenizers may serialize extra model-specific tokens separately. A failed stage is a safe failure: canonical merged weights and manifest remain unchanged.

## What changed in v0.9.5
The helper now reads local `tokenizer_config.json` and, where available, `tokenizer.json`. A token missing from `additional_special_tokens` may be migrated **only if** the tokenizer's serialized added-token inventory identifies it as a special token. Existing chat template, token definitions, and all weight files are preserved. The staging directory receives a new provenance record and no longer carries the original (invalidated) merge manifest. Conflicting, unknown, or non-special token definitions still fail closed. It also re-hashes all non-tokenizer files after copying them.

## PowerShell commands
Open `C:\Users\Lenovo\Downloads\autoHeal_finetune` in VS Code, and unzip the patch *into that directory* without touching `.venv`, `.venv-gguf`, `outputs`, or `mlflow.db`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.ollama_bridge verify-merge --merged outputs/smollm2-ollama-merged-01
.\.venv\Scripts\python.exe -m autotune.gguf_tokenizer_fix --merged outputs/smollm2-ollama-merged-01 --staging outputs/smollm2-gguf-stage-02
```

Continue **only if** the staging command reports `ok: true`, `non_tokenizer_files_unchanged: true`, and `weights_untouched: true`. Check the actual names in `special_tokens_migrated`, and investigate anything unexpected.

```powershell
.\.venv-gguf\Scripts\python.exe -c 'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained("outputs/smollm2-gguf-stage-02", local_files_only=True); print(type(t).__name__,len(t),t.eos_token_id)'
.\.venv-gguf\Scripts\python.exe .\external\llama.cpp\convert_hf_to_gguf.py .\outputs\smollm2-gguf-stage-02 --outfile .\outputs\smollm2-finetuned-v095-f16.gguf --outtype f16
Get-Item .\outputs\smollm2-finetuned-v095-f16.gguf
.\.venv\Scripts\python.exe -m autotune.ollama_bridge modelfile --gguf outputs/smollm2-finetuned-v095-f16.gguf --output outputs/smollm2-v095.Modelfile
ollama create autotune-smollm2:gguf-v095 -f .\outputs\smollm2-v095.Modelfile
ollama run autotune-smollm2:gguf-v095 "Explain LoRA in two sentences."
```

## If the safe staging guard still rejects a token
Inspect the token metadata; do not delete token entries or bypass the guard:

```powershell
.\.venv\Scripts\python.exe -c 'import json; from pathlib import Path; d=json.loads(Path("outputs/smollm2-ollama-merged-01/tokenizer_config.json").read_text(encoding="utf-8")); print("extra:", d.get("extra_special_tokens")); print("additional:", d.get("additional_special_tokens")); print("decoder entries:", len(d.get("added_tokens_decoder", {})))'
```

Check that `tokenizer.json` and `added_tokens_decoder` contain appropriate `special: true` definitions for every extra token. If not, stop and investigate tokenizer provenance before continuing.

## Testing and status
- Local build: 272/272 Python tests passed (v0.9.4 full source + v0.9.5 correction).
- Earlier user result: failed v0.9.4 stage with a mismatch error; no evidence of weight corruption.
- **Not yet verified on Windows:** v0.9.5 stage, tokenizer load, GGUF conversion, Ollama import, or model inference.
- The project remains CPU-first; no retraining or GPU is needed for these steps.
