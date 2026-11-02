from pathlib import Path
import yaml
from autotune.orchestrator import run

def test_simulated_recovery(tmp_path):
    cfg = yaml.safe_load(Path('config/baseline.yaml').read_text())
    cfg['meta_agent']['enabled']=False
    cfgfile = tmp_path/'baseline.yaml'
    cfgfile.write_text(yaml.safe_dump(cfg))
    result=run(str(cfgfile),'mock',str(tmp_path/'runs'))
    assert len(result['attempts']) == 2
    assert result['attempts'][0]['state'] == 'overfit'
    assert result['attempts'][1]['state'] == 'completed'
    assert result['best']['trial'] == 2
    exp=Path(result['experiment'])
    assert (exp/'active.yaml').exists()
    assert (tmp_path/'optimized.yaml').exists()
    assert (exp/'trial-01'/'best_checkpoint.json').exists()

def test_simulated_underfit_recovery(tmp_path):
    cfg = yaml.safe_load(Path('config/baseline.yaml').read_text())
    cfg['simulator']['scenario']='underfit'
    cfg['meta_agent']['enabled']=False
    cfgfile = tmp_path/'baseline.yaml'
    cfgfile.write_text(yaml.safe_dump(cfg))
    result=run(str(cfgfile),'mock',str(tmp_path/'runs'))
    assert result['attempts'][0]['state']=='underfit'
    assert result['attempts'][1]['state']=='completed'
