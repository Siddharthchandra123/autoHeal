"""
AutoTune AI: compare a fine-tuned causal-LM LoRA adapter with its base model.

Expected JSONL format (one example per line):
{"text":"Instruction: Explain gradient descent. Response: Gradient descent updates parameters to reduce a loss."}

IMPORTANT:
- Use a genuinely held-out test file that was not used for training or validation.
- This script compares greedy generations on identical prompts.
- Token F1 is lexical overlap, not a complete measure of answer quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def token_f1(prediction: str, reference: str) -> float:
    p, r = Counter(tokens(prediction)), Counter(tokens(reference))
    if not p or not r:
        return 0.0
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def load_examples(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text")
        if not isinstance(text, str) or "Instruction:" not in text or "Response:" not in text:
            raise ValueError(
                f"{path}:{line_no}: expected {{'text':'Instruction: ... Response: ...'}}"
            )
        prompt, reference = text.split("Response:", 1)
        prompt = prompt.strip() + " Response:"
        reference = reference.strip()
        if not reference:
            raise ValueError(f"{path}:{line_no}: empty reference response")
        rows.append({"prompt": prompt, "reference": reference})
    if not rows:
        raise ValueError("Held-out test file has no examples")
    return rows


def load_model(base_model: str, adapter_dir: Path | None, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=False
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.to("cpu")
    model.eval()
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output[0, encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def evaluate(label: str, model, tokenizer, examples: list[dict], max_new_tokens: int):
    results = []
    for index, row in enumerate(examples, 1):
        prediction, seconds = generate(model, tokenizer, row["prompt"], max_new_tokens)
        reference = row["reference"]
        results.append({
            "index": index,
            "prompt": row["prompt"],
            "reference": reference,
            "prediction": prediction,
            "token_f1": round(token_f1(prediction, reference), 6),
            "exact_match": normalize(prediction) == normalize(reference),
            "latency_seconds": round(seconds, 4),
        })
    summary = {
        "examples": len(results),
        "mean_token_f1": round(sum(x["token_f1"] for x in results) / len(results), 6),
        "exact_match_rate": round(sum(x["exact_match"] for x in results) / len(results), 6),
        "mean_latency_seconds": round(sum(x["latency_seconds"] for x in results) / len(results), 4),
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter", required=True, help="Final PEFT adapter directory")
    parser.add_argument("--test-file", required=True, help="Fresh held-out JSONL")
    parser.add_argument("--output", default="outputs/base-vs-finetuned")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter).resolve()
    test_file = Path(args.test_file).resolve()
    output_dir = Path(args.output).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not test_file.is_file():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(test_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}", flush=True)
    base = load_model(args.base_model, None, tokenizer)
    base_summary, base_rows = evaluate("base", base, tokenizer, examples, args.max_new_tokens)
    del base
    import gc
    gc.collect()

    print(f"Loading LoRA adapter: {adapter_dir}", flush=True)
    tuned = load_model(args.base_model, adapter_dir, tokenizer)
    tuned_summary, tuned_rows = evaluate("fine_tuned", tuned, tokenizer, examples, args.max_new_tokens)
    del tuned
    gc.collect()

    delta = round(tuned_summary["mean_token_f1"] - base_summary["mean_token_f1"], 6)
    report = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "test_file": str(test_file),
        "test_examples": len(examples),
        "metric_note": "Token F1 is lexical overlap only; manually review answers for correctness and usefulness.",
        "base": base_summary,
        "fine_tuned": tuned_summary,
        "token_f1_delta": delta,
        "interpretation": (
            "candidate lexical-overlap score improved" if delta > 0 else
            "candidate lexical-overlap score did not improve" if delta < 0 else
            "no lexical-overlap change"
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "comparison_details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["index", "prompt", "reference", "base_prediction", "fine_tuned_prediction",
                  "base_token_f1", "fine_tuned_token_f1", "base_exact_match",
                  "fine_tuned_exact_match", "base_latency_seconds", "fine_tuned_latency_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for b, t in zip(base_rows, tuned_rows):
            writer.writerow({
                "index": b["index"], "prompt": b["prompt"], "reference": b["reference"],
                "base_prediction": b["prediction"], "fine_tuned_prediction": t["prediction"],
                "base_token_f1": b["token_f1"], "fine_tuned_token_f1": t["token_f1"],
                "base_exact_match": b["exact_match"], "fine_tuned_exact_match": t["exact_match"],
                "base_latency_seconds": b["latency_seconds"],
                "fine_tuned_latency_seconds": t["latency_seconds"],
            })

    print(json.dumps(report, indent=2), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
