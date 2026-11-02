from autotune.metrics import diagnose
from autotune.policy import validate_patch
import pytest

SETTINGS = dict(min_evaluations=4,patience=2,min_delta=.01,generalization_gap=.2,underfit_loss_floor=1.3,flat_window=4,flat_threshold=.04)

def test_overfit_detection():
    h=[{'train_loss': t, 'val_loss': v} for t,v in [(1.8,1.7),(1.5,1.6),(1.2,1.7),(.9,1.85)]]
    assert diagnose(h, SETTINGS).state=='overfit'

def test_underfit_detection():
    h=[{'train_loss':1.7-.005*i,'val_loss':1.8-.005*i} for i in range(4)]
    assert diagnose(h,SETTINGS).state=='underfit'

def test_no_early_alarm():
    assert diagnose([{'train_loss':1.5,'val_loss':1.5}],SETTINGS).state=='none'

def test_bad_patch_rejected():
    with pytest.raises(ValueError):
        validate_patch({'system_command':123},{'learning_rate':.0001})

def test_excess_lr_change_rejected():
    with pytest.raises(ValueError):
        validate_patch({'learning_rate':.0004},{'learning_rate':.0001})

def test_bad_overfit_agent_proposal_rejected():
    with pytest.raises(ValueError):
        validate_patch({'lora_dropout': .01},{'lora_dropout':.05}, 'overfit')
