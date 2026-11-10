import math
import pytest
import yaml
from autotune.katib import (KatibContractError, build_experiment, parse_metrics,
                            summarize_trials, validate_spec)

BASE = {
    "name": "autotune-test", "namespace": "kubeflow", "algorithm": "random",
    "objective": "val_loss", "direction": "minimize", "max_trials": 6,
    "parallel_trials": 2, "max_failed_trials": 1, "backend": "mock",
    "image": "registry.example.com/autotune-ai:0.14.0",
    "config_path": "/app/config/lora_public_cpu.yaml",
    "parameters": {"learning_rate": {"min": 0.00001, "max": 0.001},
                   "lora_dropout": {"min": 0.01, "max": 0.3}},
}


def test_build_experiment_has_expected_katib_contract():
    doc = build_experiment(BASE)
    assert doc["apiVersion"] == "kubeflow.org/v1beta1"
    assert doc["kind"] == "Experiment"
    assert doc["spec"]["maxTrialCount"] == 6
    assert doc["spec"]["objective"]["objectiveMetricName"] == "val_loss"
    assert doc["spec"]["trialTemplate"]["trialSpec"]["spec"]["backoffLimit"] == 0
    assert "--backend" in doc["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]["args"]


@pytest.mark.parametrize("mutate", [
    lambda x: x["parameters"].update({"shell_command": {"min": 0, "max": 1}}),
    lambda x: x.update({"max_trials": 500}),
    lambda x: x.update({"parallel_trials": 7}),
    lambda x: x.update({"config_path": "/app/config/../../etc/passwd"}),
    lambda x: x.update({"algorithm": "arbitrary-code"}),
    lambda x: x["parameters"]["learning_rate"].update({"min": float("nan")}),
])
def test_invalid_specs_fail_closed(mutate):
    import copy
    spec = copy.deepcopy(BASE)
    mutate(spec)
    with pytest.raises(KatibContractError):
        validate_spec(spec)


def test_parameter_bounds_are_bounded_and_typed():
    spec = validate_spec(BASE)
    assert spec["parameters"]["learning_rate"]["type"] == "double"
    assert spec["parameters"]["lora_dropout"]["max"] == 0.3


def test_stdout_metric_parser_requires_exactly_one_finite_value():
    assert parse_metrics("epoch=1\nval_loss=1.234\nbest_step=8\n") == 1.234
    for bad in ("no metric", "val_loss=1\nval_loss=2", "val_loss=nan", "val_loss=inf"):
        with pytest.raises((KatibContractError, ValueError)):
            parse_metrics(bad)


def test_trial_summary_ranks_successes_and_never_promotes():
    result = summarize_trials([
        {"trial_id": "t1", "status": "Succeeded", "val_loss": 1.2},
        {"trial_id": "t2", "status": "Succeeded", "val_loss": 0.8},
        {"trial_id": "t3", "status": "Failed"},
    ])
    assert result["best_trial"]["trial_id"] == "t2"
    assert result["failed_or_incomplete_trials"] == 1
    assert result["promotion_approved"] is False
    assert result["release_status"] == "HUMAN_REVIEW_REQUIRED"


def test_token_f1_is_maximized_and_range_checked():
    result = summarize_trials([
        {"trial_id": "a", "status": "Succeeded", "token_f1": 0.71},
        {"trial_id": "b", "status": "Succeeded", "token_f1": 0.84},
    ], "token_f1", "maximize")
    assert result["best_trial"]["trial_id"] == "b"
    with pytest.raises(KatibContractError):
        summarize_trials([{"trial_id": "x", "status": "Succeeded", "token_f1": 1.4}], "token_f1", "maximize")


def test_duplicate_trial_ids_rejected():
    rows = [{"trial_id": "same", "status": "Failed"}, {"trial_id": "same", "status": "Failed"}]
    with pytest.raises(KatibContractError):
        summarize_trials(rows)


def test_hf_worker_respects_model_specific_lora_targets():
    from autotune.hf_trainer import _target_modules
    assert _target_modules({"hf": {"target_modules": ["c_attn"]}}) == ["c_attn"]
    assert _target_modules({}) == ["q_proj", "v_proj"]
    with pytest.raises(ValueError):
        _target_modules({"hf": {"target_modules": ["bad module"]}})


def test_hf_backend_requires_persistent_workspace_claim():
    import copy
    spec = copy.deepcopy(BASE)
    spec["backend"] = "hf"
    with pytest.raises(KatibContractError):
        validate_spec(spec)
    spec["workspace_pvc"] = "autotune-workspace"
    assert validate_spec(spec)["workspace_pvc"] == "autotune-workspace"


def test_hf_manifest_mounts_persistent_workspace():
    import copy
    spec = copy.deepcopy(BASE)
    spec["backend"] = "hf"
    spec["workspace_pvc"] = "autotune-workspace"
    doc = build_experiment(spec)
    pod = doc["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "autotune-workspace"
    container = pod["containers"][0]
    assert {x["name"] for x in container["env"]} >= {"AUTOTUNE_TRAIN_FILE", "AUTOTUNE_VALIDATION_FILE"}
    assert container["volumeMounts"][0]["mountPath"] == "/workspace"


def test_rejects_windows_path_traversal_and_fractional_integer_bounds():
    import copy
    spec = copy.deepcopy(BASE)
    spec["config_path"] = "/app/config/..\\..\\secret.yaml"
    with pytest.raises(KatibContractError):
        validate_spec(spec)
    spec = copy.deepcopy(BASE)
    spec["parameters"]["lora_r"] = {"min": 4.5, "max": 64.5}
    with pytest.raises(KatibContractError):
        validate_spec(spec)
