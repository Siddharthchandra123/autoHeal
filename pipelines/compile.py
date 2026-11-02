"""Compile KFP v2 YAML after installing kfp[kubernetes].

The pipeline runs the bounded controller inside one KFP container task.
Each internal attempt is individually logged to MLflow and the PVC; splitting
individual attempts into KFP DAG tasks is a subsequent production milestone.
"""
from kfp import dsl, compiler, kubernetes

@dsl.container_component
def adaptive_finetune(image: str, config_path: str, mode: str):
    return dsl.ContainerSpec(
        image=image,
        command=['python','-m','autotune.orchestrator'],
        args=['--config',config_path,'--mode',mode,'--workspace','/workspace/runs'],
    )

@dsl.pipeline(name='autotune-ai-adaptive-finetuning')
def pipeline(image: str='registry.example.com/autotune-ai:0.1.0',
             pvc_name: str='autotune-workspace',
             config_path: str='/workspace/config/baseline.yaml',
             mode: str='hf'):
    task = adaptive_finetune(image=image, config_path=config_path, mode=mode)
    task.set_caching_options(False)
    kubernetes.mount_pvc(task,pvc_name=pvc_name,mount_path='/workspace')
    # GPU resource requests are cluster-specific (NVIDIA/AMD device plugins).
    # Configure them for your installed driver/plugin before scheduling real jobs.

if __name__=='__main__':
    compiler.Compiler().compile(pipeline,package_path='pipelines/autotune_pipeline.yaml')
    print('Wrote pipelines/autotune_pipeline.yaml')
