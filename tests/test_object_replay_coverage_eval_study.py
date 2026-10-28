"""Coverage-only CPU contracts; no new baseline, model, or GPU run."""
from copy import deepcopy
from pathlib import Path
import pytest

from vlm_object_replay_coverage import contracts as c,evaluate as ev,analysis as a,study as s
from vlm_probe.evaluate import content_sha,read_json,sha_file,write_json


def gate_rows():
    obj=[{'slice':'overall','metrics':{'accuracy':{'delta_pp':1.1,'one_sided_98_75_lower_pp':.1,'seed_delta_pp':[.1,1.1,2.1]},
        'recall':{'delta_pp':0.},'specificity':{'delta_pp':-.5}}}]
    txt=[{'slice':label,'delta_pp':0.,'one_sided_98_75_lower_pp':-.49,'seed_delta_pp':[-.5,0,.5]}
         for label in ('overall','rare_1_5','frequent_gt20')]
    balanced={'contrasts':[{'slice':'overall','delta_pp':.01}]}
    return obj,txt,balanced


def test_all_eight_original_pilot_guards_stay_strict():
    obj,txt,balanced=gate_rows()
    assert len(a.screen_gate(obj,txt,balanced,replicated=False)['checks'])==8
    assert a.screen_gate(obj,txt,balanced,replicated=False)['passed']
    for field in ('rare_1_5','frequent_gt20'):
        changed=deepcopy(txt);next(r for r in changed if r['slice']==field)['delta_pp']=-.001
        assert not a.screen_gate(obj,changed,balanced,replicated=False)['passed']
    obj[0]['metrics']['specificity']['delta_pp']=-.50001
    assert not a.screen_gate(obj,txt,balanced,replicated=False)['passed']
    obj,txt,balanced=gate_rows();balanced['contrasts'][0]['delta_pp']=0.
    assert not a.screen_gate(obj,txt,balanced,replicated=False)['passed']


@pytest.mark.parametrize('seed,task',[(0,None),(200,'object'),(42,'external'),(42,'pope')])
def test_repeated_control_is_never_replication_or_confirmation_candidate(seed,task):
    with pytest.raises(ValueError,match='pilot42'):c.validate_condition(c.CONTROL,seed,task)


def test_role_paths_and_cli_use_actual_condition_seed_and_separate_engineering_root(tmp_path):
    name=s.role('preflight',c.CONTROL,42)
    assert s.worker_dir(tmp_path,name)==tmp_path/'workers'/name
    with pytest.raises(ValueError):s.worker_dir(tmp_path,'../wrong')
    args=s.train_arguments(tmp_path/'cfg.json',tmp_path/'engineering_runs',c.CONTROL,42,True)
    assert args[args.index('--condition')+1]=='repeat8x4' and args[args.index('--seed')+1]=='42'
    assert '--seed42' not in args and '--preflight' in args
    assert args[args.index('--output-root')+1]==str(tmp_path/'engineering_runs')
    file=tmp_path/'runs/repeat8x4/seed42/train/LATEST.json';write_json(file,{})
    assert '--resume' in s.train_arguments(tmp_path/'cfg.json',tmp_path,c.CONTROL,42)


def test_coverage_expectations_keep_object2888_and_actor_replay5792(tmp_path):
    manifest=tmp_path/'TRAIN.json';write_json(manifest,{'rows':[{}]*2888})
    model=tmp_path/'model';model.mkdir();write_json(tmp_path/'model_provenance.json',{'original':True})
    cfg={'train_manifest':str(manifest),'initial_adapter':str(tmp_path/'initial')}
    cf=tmp_path/'cfg.json';write_json(cf,cfg)
    for arm in c.TRAINED_CONDITIONS:
        value=c.training_expectations(cfg,arm,42,{}, {},model,cf)
        assert value['completed_updates']==181 and value['actual_object_exposures']==2888
        assert value['actual_replay_exposures']==5792 and value['actual_questions_seen']==8680
        assert value['replay_questions_per_update']==32 and value['replay_lambda']==1
        assert value['training_microbatch_questions']*value['training_accumulation']==16


def test_teacher_costs_distinguish_fresh5792_and_immutable_original1448():
    for arm,unique,new,old in ((c.CANDIDATE,5792,5792,0),(c.CONTROL,1448,0,1448)):
        value={'schema_version':'rarevlm.object_replay_coverage.teacher_source.v1',
            'kind':'fresh_full_fp32' if arm==c.CANDIDATE else 'immutable_repeated_reference',
            'scope':'formal_full_stream','actual_new_teacher_forward_questions':new,
            'historical_teacher_bank_questions':old,'actor_replay_questions_planned':5792,
            'actor_replay_unique_questions_planned':unique,'source_teacher_unique_questions_referenced':unique}
        c.validate_teacher_accounting(value,arm,5792,unique)
        value['actual_new_teacher_forward_questions']=5792 if arm==c.CONTROL else 0
        with pytest.raises(ValueError):c.validate_teacher_accounting(value,arm,5792,unique)


def test_first32_engineering_source_cannot_bind_a_final_adapter():
    value={'schema_version':'rarevlm.object_replay_coverage.teacher_source.v1','kind':'fresh_full_fp32',
        'scope':'engineering_first_update_only','actual_new_teacher_forward_questions':32,
        'historical_teacher_bank_questions':0,'actor_replay_questions_planned':32,
        'actor_replay_unique_questions_planned':32,'source_teacher_unique_questions_referenced':32}
    with pytest.raises(ValueError):c.validate_teacher_accounting(value,c.CANDIDATE,5792,5792)


def test_pilot_analysis_waits_for_both_full_arms(tmp_path,monkeypatch):
    cf=tmp_path/'cfg.json';write_json(cf,{})
    monkeypatch.setattr(a,'source_hashes',lambda:{'source':'locked'})
    def cell(root,condition,seed,task,*args):return {'complete':not(condition==c.CONTROL and task=='text')}
    monkeypatch.setattr(a,'read_cell',cell)
    result,_=a.analyze(cf,tmp_path,[42],'screen')
    assert result['status']=='waiting_predictions' and result['pending_cells']==['repeat8x4_42_text']
    assert 'gate' not in result


def fake_pipeline(tmp_path,monkeypatch,*,pilot=True,screen=True,external=True,confirm=True):
    root=tmp_path/'study';root.mkdir();cf=tmp_path/'cfg.json';write_json(cf,{'output_root':str(root)})
    write_json(root/'STUDY_INPUT_LOCK.json',{'fixed':True})
    monkeypatch.setattr(s,'prepare',lambda *_:({},root,{'fixed':True}))
    monkeypatch.setattr(s,'verify_study',lambda *_:{})
    calls=[];analysis_calls=[]
    def queue(root,name,jobs,identity):
        calls.append((name,deepcopy(jobs)))
        return [{'outcome':'passed','role':job['key']} for job in jobs]
    monkeypatch.setattr(s,'queue_stage',queue)
    def analyze(cf,root,seeds,phase,**kwargs):
        analysis_calls.append((list(seeds),phase))
        passed=pilot if list(seeds)==[42] else screen if phase=='screen' else external if phase=='external' else confirm
        file=root/'analysis'/str(len(analysis_calls))/'ANALYSIS.json'
        value={'status':'complete','gate':{'passed':passed},'exploratory_control_gate':{'passed':True}}
        write_json(file,value);return value,file
    monkeypatch.setattr(s,'analyze',analyze)
    monkeypatch.setattr(s,'finish',lambda root,status,phase,*args:{'status':status,'last_phase':phase,'goal_complete':False})
    return cf,calls,analysis_calls


def test_both_pilot_arms_but_only_wide32_expands_and_no_external_initial_gpu_job(tmp_path,monkeypatch):
    cf,calls,analysis_calls=fake_pipeline(tmp_path,monkeypatch)
    result=s.run_study(cf)
    assert [name for name,_ in calls]==['engineering_preflight','pilot42','replication','external_confirmation','pope_confirmation']
    assert {j['key'] for j in calls[0][1]}=={'preflight_wide32_seed42','preflight_repeat8x4_seed42'}
    assert {j['key'] for j in calls[1][1]}=={'train_eval_wide32_seed42','train_eval_repeat8x4_seed42'}
    assert all('repeat8x4' not in j['key'] and 'initial' not in j['key'] for _,jobs in calls[2:] for j in jobs)
    assert len(calls[2][1])==2 and len(calls[3][1])==len(calls[4][1])==3
    assert result['status']=='confirmation_passed_pending_independent_review' and result['goal_complete'] is False


@pytest.mark.parametrize('failed,phases',[('pilot',['engineering_preflight','pilot42']),
    ('screen',['engineering_preflight','pilot42','replication']),
    ('external',['engineering_preflight','pilot42','replication','external_confirmation'])])
def test_failed_wide32_is_not_replaced_by_successful_control(tmp_path,monkeypatch,failed,phases):
    cf,calls,_=fake_pipeline(tmp_path,monkeypatch,**{failed:False})
    result=s.run_study(cf)
    assert [name for name,_ in calls]==phases and result['status']=='completed_negative'
    assert result['goal_complete'] is False


def test_new_gpu_admission_is_uuid_and_real_config_bytes_locked(tmp_path):
    cf=tmp_path/'cfg.json';write_json(cf,{})
    file=s.task_admission(tmp_path/'worker',cf,c.CANDIDATE,42,'text',{'uuid':'GPU-a'})
    assert read_json(file)['schema_version']=='rarevlm.object_replay_coverage.gpu_admission.v1'
    assert read_json(file)['config_sha256']==sha_file(cf)
    with pytest.raises(ValueError):s.task_admission(tmp_path/'worker',cf,c.CANDIDATE,42,'text',{'uuid':'GPU-b'})


def test_text_empty_aliases_only_in_memory_and_binary_categories_unchanged():
    old=[{'category':'','score':.6,'input_ids':[1,2]}];before=deepcopy(old)
    result,proof=c.canonical_rows(old,'text')
    assert old==before and 'category' not in result[0] and proof['scores_and_inputs_unchanged']
    with pytest.raises(ValueError):c.canonical_rows([{'category':'rare'}],'text')
    binary=[{'category':'dog','score':1.}]
    assert c.canonical_rows(binary,'external')[0] is binary


def test_registry_and_evaluator_reject_control_selection_and_confirmation(tmp_path):
    from types import SimpleNamespace
    cfg={'candidates':[c.CONTROL],'controls':[c.CANDIDATE]}
    with pytest.raises(ValueError):ev.validate_config(cfg)
    args=SimpleNamespace(condition=c.CONTROL,seed=42,task='external')
    with pytest.raises(ValueError,match='pilot42'):ev.progression(args,{}, {})
