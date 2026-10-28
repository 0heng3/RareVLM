"""Narrow regressions for the real historical serialization gap; no model work."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from vlm_object_behavior_recovery_v1.compat import content_sha,legacy_anchor_view

BASE=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
ROOT=BASE/'vlm_object_behavior_replay_v1'


def actual():
    old=BASE/'vlm_object_replay_coverage_v1/runs/wide32/seed42/train'
    new=ROOT/'runs/teacher_ce32/seed42/train'
    required=[old/'TRAIN_RESULT.json',old/'SCHEDULE.json',new/'SCHEDULE.json',
        ROOT/'STUDY_INPUT_LOCK.json',ROOT/'prepared/seed42/PLAN.json',new/'TRAIN_RESULT.json']
    if not all(file.exists() for file in required):
        pytest.skip('Historical integration artifacts are not bundled with the public source release')
    load=lambda file:json.loads(file.read_bytes())
    anchor=load(old/'TRAIN_RESULT.json');schedule=load(old/'SCHEDULE.json')
    assert 'replay_microbatch_order_sha256' not in anchor
    locked=load(ROOT/'STUDY_INPUT_LOCK.json')['files_sha256'][str(old/'TRAIN_RESULT.json')]
    import hashlib
    assert hashlib.sha256((old/'TRAIN_RESULT.json').read_bytes()).hexdigest()==locked
    return anchor,schedule,load(new/'SCHEDULE.json'),load(ROOT/'prepared/seed42/PLAN.json')['schedule'],load(new/'TRAIN_RESULT.json')


def test_actual_missing_historical_key_has_exact_locked_content_chain():
    anchor,old,new,plan,train=actual();before=deepcopy(anchor)
    view,proof=legacy_anchor_view(anchor,old,new,plan,train)
    assert anchor==before and view['replay_microbatch_order_sha256']==content_sha(old['replay_microbatches_by_update'])
    assert proof['historical_field_was_absent'] and proof['schedule_updates']==181
    from vlm_object_behavior_replay.contracts import validate_legacy_stream_alignment
    validate_legacy_stream_alignment(train,view,'teacher_ce32',42)


@pytest.mark.parametrize('changed',['question','boundary','learning_rate','source_order_digest','new_micro_digest','declared_old_digest','schedule_chain'])
def test_any_substantive_or_other_declared_identity_change_is_rejected(changed):
    anchor,old,new,plan,train=deepcopy(actual())
    if changed=='question':new['replay_question_ids_by_update'][0][0]='different'
    elif changed=='boundary':new['replay_microbatches_by_update'][0][0],new['replay_microbatches_by_update'][0][1]=new['replay_microbatches_by_update'][0][1],new['replay_microbatches_by_update'][0][0]
    elif changed=='learning_rate':new['learning_rates_by_update'][0]+=1e-12
    elif changed=='source_order_digest':anchor['replay_order_sha256']='0'*64
    elif changed=='new_micro_digest':train['replay_microbatch_order_sha256']='0'*64
    elif changed=='declared_old_digest':anchor['replay_microbatch_order_sha256']='0'*64
    else:anchor['schedule_sha256']='0'*64
    with pytest.raises(ValueError):legacy_anchor_view(anchor,old,new,plan,train)


def test_explicit_new_namespace_does_not_expand_original89_source_enumeration():
    from vlm_object_behavior_replay.study import study_sources
    from rarevlm_paths import PROJECT_ROOT
    if not (ROOT/'STUDY_INPUT_LOCK.json').exists():
        pytest.skip('Historical source lock is a separately supplied integration artifact')
    frozen=json.loads((ROOT/'STUDY_INPUT_LOCK.json').read_bytes())['source_sha256']
    anchor=next(Path(name) for name in frozen if name.endswith('/src/rarevlm_paths.py'))
    historical_root=anchor.parents[1]
    locked={str(Path(name).relative_to(historical_root)):digest for name,digest in frozen.items()}
    current={str(Path(name).relative_to(PROJECT_ROOT)):digest for name,digest in study_sources().items()}
    assert len(locked)==89 and set(current)==set(locked)
    export=PROJECT_ROOT/'metadata/PUBLIC_EXPORT.json'
    variants=json.loads(export.read_text())['public_variants'] if export.exists() else {}
    for name,digest in current.items():
        if digest==locked[name]:continue
        variant=variants[name]
        assert variant['local_source_sha256']==locked[name]
        assert variant['public_sha256']==digest


def test_no_training_command_or_hidden_old_global_override_in_recovery_controller():
    import ast
    repo=Path(__file__).resolve().parents[1]
    for file in (repo/'src/vlm_object_behavior_recovery_v1').glob('*.py'):
        tree=ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                assert not any(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                    and target.value.id=='frozen' for target in node.targets)
    code=(repo/'src/vlm_object_behavior_recovery_v1/study.py').read_text()
    assert "'-m','vlm_object_behavior_replay.train'" not in code
    assert "'-m','vlm_object_behavior_replay.rollout'" not in code
