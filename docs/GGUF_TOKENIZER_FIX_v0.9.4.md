# AutoTune AI v0.9.4 - Windows GGUF Tokenizer Compatibility Recovery

## Why conversion failed

The SmolLM2 model uses a GPT-2-style BPE tokenizer. The absence of `tokenizer.model` is not, by itself, an error. The converter eventually called `AutoTokenizer.from_pretrained` and failed because `extra_special_tokens` was represented as a JSON list, whereas the installed Transformers expected a dictionary. This is a known upstream issue (llama.cpp #24460).

**Do not retrain. Do not edit files in the verified merged directory.** The original `autotune_ollama_export.json` manifest covers their hashes.

## Apply the patch

Extract `AutoTune_AI_GGUF_Tokenizer_v0.9.4_Update_Patch.zip` into your existing `C:\Users\Lenovo\Downloads\autoHeal_finetune` project directory. Preserve `.venv`, `.venv-gguf`, `outputs`, and `mlflow.db`. The patch adds `autotune/gguf_tokenizer_fix.py` and tests.

## PowerShell commands (project root)

```powershell
.\.venv\Scripts\python.exe -m autotune.ollama_bridge verify-merge --merged outputs/smollm2-ollama-merged-01

.\.venv\Scripts\python.exe -m autotune.gguf_tokenizer_fix --merged outputs/smollm2-ollama-merged-01 --staging outputs/smollm2-gguf-stage-01

.\.venv-gguf\Scripts\python.exe -c 'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained("outputs/smollm2-gguf-stage-01", local_files_only=True); print(type(t).__name__,len(t))'

.\.venv-gguf\Scripts\python.exe .\external\llama.cpp\convert_hf_to_gguf.py .\outputs\smollm2-gguf-stage-01 --outfile .\outputs\smollm2-finetuned-f16.gguf --outtype f16

Get-Item .\outputs\smollm2-finetuned-f16.gguf

.\.venv\Scripts\python.exe -m autotune.ollama_bridge modelfile --gguf outputs/smollm2-finetuned-f16.gguf --output outputs/smollm2-gguf-v094.Modelfile

ollama create autotune-smollm2:gguf-v094 -f .\outputs\smollm2-gguf-v094.Modelfile

ollama run autotune-smollm2:gguf-v094 "Explain LoRA in two sentences."
```

If the stage already exists, use a new name (for example, `smollm2-gguf-stage-02`) rather than deleting an earlier output, and update the later paths accordingly.

## Verify the fix

- `gguf_tokenizer_fix` prints `ok: true` and `tokenizer_fix_applied: true` if a list was corrected.
- The fresh tokenizer load prints a tokenizer class and vocabulary size.
- Converter produces a nonempty `.gguf` file.
- `ollama create` succeeds and `ollama list` shows the new model.
- `ollama run` returns text. This verifies runtime import, not model accuracy.

## What is still unverified

The above commands have not been executed on your Windows installation. In the isolated build environment, 7 targeted tests and 265 cumulative tests passed. Do not claim production-ready conversion until the actual GGUF conversion and Ollama import succeed.

## References

- Upstream issue: https://github.com/ggml-org/llama.cpp/issues/24460
- HF model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
