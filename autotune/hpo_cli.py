import argparse, json
from .hpo_engine import run_hpo

def main():
    p = argparse.ArgumentParser(description="AutoTune AI Phase 2 LoRA HPO")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    p.add_argument("--experiment", default="AutoTune-AI-MI300X-HPO")
    a = p.parse_args()
    print(json.dumps(run_hpo(a.config, a.output, a.tracking_uri, a.experiment), indent=2))

if __name__ == "__main__":
    main()
