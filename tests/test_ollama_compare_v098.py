import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autotune.ollama_compare import (
    EvaluationError, chatml_prompt, generation_payload, read_prompts,
    registered_models, run_evaluation, valid_model_name, verify_no_exact_leakage,
)


def dataset(tmp_path, questions=None):
    values = questions or [{"id": "a", "prompt": "First unique question?"}, {"id": "b", "prompt": "Another unique question?"}]
    path = tmp_path / "heldout.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in values), encoding="utf-8")
    return path


def test_model_names():
    assert valid_model_name("smollm2:135m-instruct-fp16") == "smollm2:135m-instruct-fp16"
    for name in ("", "../../etc/passwd", "foo bar", "https://remote", "a?token=1"):
        with pytest.raises(EvaluationError):
            valid_model_name(name)


def test_reject_duplicate_prompts(tmp_path):
    with pytest.raises(EvaluationError, match="Duplicate"):
        read_prompts(dataset(tmp_path, [{"id": "a", "prompt": "Same question here"}, {"id": "b", "prompt": " same question here "}]))


def test_reject_short_dataset(tmp_path):
    with pytest.raises(EvaluationError):
        read_prompts(dataset(tmp_path, [{"id": "a", "prompt": "Only one question here"}]))


def test_reject_oversized_dataset(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text("x" * 64001, encoding="utf-8")
    with pytest.raises(EvaluationError):
        read_prompts(path)


def test_reject_symlink(tmp_path):
    path = dataset(tmp_path)
    link = tmp_path / "linked.jsonl"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlink creation requires Developer Mode or elevated permission")
    with pytest.raises(EvaluationError):
        read_prompts(link)


def test_leakage_checks_user_messages(tmp_path):
    known = tmp_path / "train.jsonl"
    known.write_text(json.dumps({"text": "<|im_start|>user\nFirst unique question?<|im_end|>"}), encoding="utf-8")
    with pytest.raises(EvaluationError, match="overlap"):
        verify_no_exact_leakage(read_prompts(dataset(tmp_path)), [known])


def test_preflight_has_no_network_and_no_output(tmp_path):
    path = dataset(tmp_path)
    dest = tmp_path / "out"
    with patch("autotune.ollama_compare.local_api", side_effect=AssertionError("should not connect")):
        result = run_evaluation(baseline="base:1", candidate="new:1", dataset=path, output=dest)
    assert result["mode"].startswith("READ-ONLY")
    assert result["model_promotion_approved"] is False
    assert not dest.exists()


def test_refuse_nonempty_output(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "existing.json").write_text("{}")
    with pytest.raises(EvaluationError, match="not empty"):
        run_evaluation(baseline="base:1", candidate="new:1", dataset=dataset(tmp_path), output=dest)


def test_fixed_chatml_and_options():
    payload = generation_payload("base:1", "Define gradient.")
    assert payload["stream"] is False and payload["raw"] is True
    assert payload["options"]["temperature"] == 0
    assert "<|im_start|>user\nDefine gradient." in payload["prompt"]
    assert chatml_prompt("X").endswith("<|im_start|>assistant\n")


def test_registered_models():
    response = {"models": [{"name": "base:1", "digest": "abc", "details": {"format": "gguf"}}]}
    assert registered_models(response)["base:1"]["digest"] == "abc"
    with pytest.raises(EvaluationError):
        registered_models({})


def test_complete_evaluation_mock(tmp_path):
    def fake_api(path, payload=None):
        if path == "/api/tags":
            return {"models": [{"name": "base:1", "digest": "a"}, {"name": "new:1", "digest": "b"}]}
        return {"response": "Sample " + payload["model"], "done": True,
                "eval_count": 4, "total_duration": 1_000_000_000}
    out = tmp_path / "result"
    with patch("autotune.ollama_compare.local_api", side_effect=fake_api):
        result = run_evaluation(baseline="base:1", candidate="new:1", dataset=dataset(tmp_path), output=out, execute=True)
    assert result["prompts_completed"] == 2
    body = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    assert body["results"][0]["candidate"]["response"] == "Sample new:1"
    assert body["model_promotion_approved"] is False
    assert (out / "human_review.csv").is_file()


def test_missing_ollama_tag_blocks(tmp_path):
    with patch("autotune.ollama_compare.local_api", return_value={"models": [{"name": "base:1"}]}):
        with pytest.raises(EvaluationError, match="Missing"):
            run_evaluation(baseline="base:1", candidate="new:1", dataset=dataset(tmp_path), output=tmp_path / "out", execute=True)


def test_bad_response_blocks(tmp_path):
    def fake_api(path, payload=None):
        if path == "/api/tags":
            return {"models": [{"name": "base:1"}, {"name": "new:1"}]}
        return {"done": False, "response": "partial"}
    with patch("autotune.ollama_compare.local_api", side_effect=fake_api):
        with pytest.raises(EvaluationError, match="incomplete"):
            run_evaluation(baseline="base:1", candidate="new:1", dataset=dataset(tmp_path), output=tmp_path / "out", execute=True)
