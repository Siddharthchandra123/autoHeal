# Phase 8 — Katib Hyperparameter Optimization (v0.14.0)

**Date:** 2026-11-09  
**Status:** Source implementation + offline contract tests. Kubernetes/Katib cluster execution is NOT verified.

## Goal

Use Kubeflow Katib as the sole cross-trial hyperparameter optimizer. Each Katib Trial creates one isolated Kubernetes Job. AutoTune's existing self-healing controller is not nested inside a Katib trial: Katib explores across trials; deterministic monitoring may stop a single training run. Ray Tune must not optimize the same experiment concurrently.

## Implemented in this checkpoint

- `autotune/katib.py`: strict experiment contract, allowlisted search parameters, numeric bounds, trial/parallelism budgets, objective validation, Katib v1beta1 Experiment manifest generation, metric parsing and fail-closed trial ranking.
- `autotune/katib_cli.py`: offline `render` and `summarize` commands; no cluster connection is attempted.
- `autotune/katib_trial.py`: one-trial worker with explicit `mock` or `hf` backend. Default is synthetic mock. Real HF requires explicit `--backend hf`, a prebuilt image with training dependencies, an immutable model revision, authorized model access and mounted train/validation data. The included `Dockerfile.katib-cpu` is the starting CPU worker image; review and pin its dependencies before production use.
- `config/katib_cpu.yaml`: CPU-first starter experiment, 8-trial cap, 2-way parallelism, max 2 failed trials, random search and allowlisted LR/dropout/weight-decay parameters.
- `tests/test_katib_phase8.py`: validation, manifest, metric parsing, objective ranking and fail-closed promotion tests.

## Trust and execution boundaries

1. `mock` results are synthetic control-plane smoke tests, not model benchmarks.
2. The generated manifest defaults to `backend: mock`; it does not silently download models or data.
3. `hf` mode must be selected explicitly. Use only an approved immutable model revision and data mounted into the worker. Do not bake credentials into images or manifests.
4. Trial count is capped at 50; parallel trials at 8; failed-trial allowance at 10. Starter config is more conservative (8/2/2).
5. Search keys are limited to `learning_rate`, `lora_dropout`, `weight_decay`, and `lora_r`, with fixed numeric bounds. No shell commands or arbitrary config paths are accepted.
6. Worker uses `backoffLimit: 0`; Katib owns failed-trial accounting. There is no nested retry controller.
7. Summary output always sets `promotion_approved: false` and `HUMAN_REVIEW_REQUIRED`. A winning Katib objective is not independent holdout quality evidence.
8. Do not run Katib and Ray Tune on the same experiment.
9. The API/dashboard remains local prototype scope; do not expose unauthenticated services outside loopback.

## Windows PowerShell — offline validation

From the extracted project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autotune.katib_cli render --spec config/katib_cpu.yaml --output pipelines/autotune_katib_experiment.yaml
Get-Content pipelines\autotune_katib_experiment.yaml
```

This renders and validates YAML only. It does not need Docker, Kubernetes, a GPU, a model download, or cluster credentials.

## Cluster acceptance (not yet executed)

Prerequisites: a supported Kubernetes cluster, Kubeflow Katib controller/webhook and CRDs installed, namespace and service account configured, a pushed image containing this project, and a persistent volume strategy if model artifacts must survive pod exit.

```powershell
kubectl get crd experiments.kubeflow.org
kubectl get pods -n kubeflow
kubectl apply -f deploy/katib-workspace-pvc.yaml
# For a real HF run only: set backend: hf and workspace_pvc: autotune-workspace in the spec,
# then render a separate manifest. Populate the claim with data/train.jsonl and data/validation.jsonl.
kubectl apply -f pipelines/autotune_katib_experiment.yaml
kubectl get experiments -n kubeflow
kubectl get trials -n kubeflow
kubectl describe experiment autotune-ai-cpu-hpo -n kubeflow
```

For the starter mock experiment, inspect each Trial and its Job logs. For HF execution, set `backend: hf` and `workspace_pvc: autotune-workspace`, and use the `Dockerfile.katib-cpu` image. The image contains the sample config; the mounted PVC must provide the source-disjoint train and validation JSONL files. Confirm the configured objective appears once per worker and Katib records trial completion. Do not interpret the synthetic loss as model quality. Before `hf` mode, build and scan an image, pin the model revision, mount source-disjoint train/validation data read-only, configure a persistent output volume at `/workspace/runs`, and test one trial before increasing concurrency.

## Exit criteria

- [x] Strict allowlist and finite/bounded parameter validation.
- [x] Katib Experiment v1beta1 manifest rendering.
- [x] Trial budgets and no nested retry policy.
- [x] Single-line objective metric contract and offline result ranking.
- [x] Promotion remains false; human review required.
- [ ] Full offline regression suite passes in the user's Windows environment.
- [ ] Container image built and pushed to a registry.
- [ ] Katib CRDs/controller installed and manifest applied to a real cluster.
- [ ] One synthetic CPU Trial Job completes and Katib records its metric.
- [ ] One real HF CPU Trial completes with pinned model/data hashes and retained artifacts.
- [ ] Multi-trial results reconcile to MLflow lineage and the existing dashboard.

## Honest verification boundary

No Kubernetes cluster, Katib controller, container registry, or live training worker was available in the build environment. Therefore Phase 8 is an implementation checkpoint, not a claim of distributed execution. GPU scheduling, multi-node training, cost accounting, remote artifact registry, production auth and automatic release are out of scope.
