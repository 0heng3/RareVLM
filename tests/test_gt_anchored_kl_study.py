from copy import deepcopy
import json
from pathlib import Path
import time
import pytest
from vlm_gt_anchored_kl import CANDIDATE,CONTROL,SCHEMA
from vlm_gt_anchored_kl import study,inputs,contracts,train

def cfg():return json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_gt_anchored_kl_v1.json').read_text())

def test_actual_fixed_registry_and_new_control():
    c=cfg();study.validate_registry(c);train.validate_config(c)
    assert c['schema_version']==SCHEMA and c['control_reference_only'] is False
    assert study.role('train_eval',CONTROL,42)=='train_eval_gt_kl32_seed42'
    for op,cond,seed in [('teacher','initial',0),('external',CONTROL,42),('train_eval',CONTROL,0),('train_eval','teacher_ce32',42)]:
        with pytest.raises(ValueError):study.role(op,cond,seed)
    bad=deepcopy(c);bad['control_can_be_selected']=True
    with pytest.raises(ValueError):study.validate_registry(bad)

@pytest.mark.parametrize('bad',[{'gt_kl32':0.,'gt_ce_kl32':.5},{'gt_kl32':1.,'gt_ce_kl32':1.}])
def test_gamma_has_no_sweep_or_control_CE(bad):
    c=cfg();c['replay_gt_ce_weights']=bad
    with pytest.raises(ValueError):train.validate_config(c)

def test_command_cli_seed_and_control_worker(monkeypatch,tmp_path):
    c=tmp_path/'config.json';c.write_text('{}')
    cmd=study.train_arguments(c,tmp_path,CONTROL,42)
    assert cmd[cmd.index('--seed')+1]=='42' and '--preflight-only' not in cmd
    seen=[]
    monkeypatch.setattr(study,'worker_run',lambda args:seen.append(args.condition) or {'status':'mock'})
    monkeypatch.setattr('sys.argv',['study','--config',str(c),'--worker','--output-root',str(tmp_path),'--study-lock',str(tmp_path/'lock'),'--operation','train_eval','--condition',CONTROL,'--seed','42'])
    study.main();assert seen==[CONTROL]

def scenario(tmp_path,monkeypatch,passed):
    root=tmp_path/'run';root.mkdir();file=tmp_path/'config.json';c={'output_root':str(root)};file.write_text(json.dumps(c));events=[]
    monkeypatch.setattr(study,'prepare',lambda f:(c,root,{}));monkeypatch.setattr(study,'verify_study',lambda *a,**k:{})
    monkeypatch.setattr(inputs,'prepare_training_inputs',lambda c,s,k:events.append(('inputs',s,k)))
    monkeypatch.setattr(inputs,'verify_prepared_training_inputs',lambda *a:{})
    def queue(r,name,jobs,identity):
        tasks=[]
        for job in jobs:
            cmd=job['command'];get=lambda k:cmd[cmd.index(k)+1]
            tasks.append((get('--operation'),get('--condition'),int(get('--seed'))))
        events.append(('queue',name,tasks));return [{'role':j['key'],'outcome':'passed'} for j in jobs]
    monkeypatch.setattr(study,'queue_stage',queue)
    def analyze(f,r,seeds,phase,**kw):
        events.append(('analysis',list(seeds),phase));return {'status':'complete','gate':{'passed':passed if seeds==[42] else True}},root/f'analysis_{phase}_{len(seeds)}.json'
    monkeypatch.setattr(study,'analyze',analyze)
    monkeypatch.setattr(study,'finish',lambda r,status,phase,a,receipts,started:{'status':status,'goal_complete':False})
    return study.run_study(file),events

def test_actual_controller_negative_stops_expansion_without_selecting_control(tmp_path,monkeypatch):
    result,events=scenario(tmp_path,monkeypatch,False)
    assert result['status']=='completed_negative' and result['goal_complete'] is False
    queues=[e for e in events if e[0]=='queue'];assert len(queues)==1
    assert queues[0][2]==[('train_eval',CANDIDATE,42),('train_eval',CONTROL,42)]
    assert [e for e in events if e[0]=='inputs']==[('inputs',42,CANDIDATE),('inputs',42,CONTROL)]

def test_actual_controller_success_has_full_replication_external_POPE_without_rollout(tmp_path,monkeypatch):
    result,events=scenario(tmp_path,monkeypatch,True)
    assert result['status']=='confirmation_passed_pending_independent_review' and result['goal_complete'] is False
    queues=[e for e in events if e[0]=='queue'];assert [e[1] for e in queues]==['pilot42','replication','external_confirmation','pope_confirmation']
    assert queues[1][2]==[('train_eval',CANDIDATE,s) for s in (0,200)]
    assert queues[2][2]==[('external',CANDIDATE,s) for s in (0,42,200)]
    assert queues[3][2]==[('pope',CANDIDATE,s) for s in (0,42,200)]
    assert all(op!='teacher' for e in queues for op,_,_ in e[2])

def test_actual_finish_resolves_accounting_function_and_counts_both_arms(tmp_path):
    (tmp_path/'STUDY_INPUT_LOCK.json').write_text('{}')
    for condition in (CANDIDATE,CONTROL):
        directory=tmp_path/'runs'/condition/'seed42/train';directory.mkdir(parents=True)
        (directory/'TRAIN_RESULT.json').write_text(json.dumps({'completed_updates':181,'formal_reference_forward_questions':5792,'reference_forward_seconds':12.}))
        (directory/'PREFLIGHT_HISTORY.jsonl').write_text(json.dumps({'discarded_engineering_optimizer_updates':1})+'\n')
    out=study.finish(tmp_path,'completed_negative','pilot_internal_screen',{},[],time.monotonic())
    assert out['formal_training_updates']==362 and out['formal_reference_forward_questions']==11584
    assert out['preflight_accounting']['actual_discarded_engineering_optimizer_updates_all_recorded_attempts']==2
    assert out['teacher_generation_accounting']['new_generation_questions_all_seeds']==0
    assert out['goal_complete'] is False and (tmp_path/'COMPLETION.json').exists()

def test_static_declaration_plus8_runtime_fields_only():
    c=cfg()['teacher_reference'];runtime={**c,**{k:'0'*64 for k in ('initial_lora_tensor_sha256','initial_base_parameter_sha256','initial_all_parameter_sha256')},'dtypes':{'lora_elements_by_dtype':{'torch.float32':1843200}},'parameter_count':3000000,'parameter_storage_shared_with_actor':False,'actor_rng_restored_after_construction':True,'execution_path_identity_claim':False}
    assert contracts.validate_reference_binding(c,runtime)
    for mutate in ('extra','declaration','evidence','badbool'):
        bad=deepcopy(runtime)
        if mutate=='extra':bad['unknown']=0
        elif mutate=='declaration':bad['temperature']=2.
        elif mutate=='evidence':bad['parameter_storage_shared_with_actor']=True
        else:bad['parameter_count']=True
        with pytest.raises(ValueError):contracts.validate_reference_binding(c,bad)

def test_negative_pilot_cannot_admit_replication(tmp_path):
    file=tmp_path/'analysis/pilot_seed42/screen/ANALYSIS.json';file.parent.mkdir(parents=True);file.write_text(json.dumps({'status':'complete','gate':{'passed':False}}))
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    with pytest.raises(ValueError):study.worker_prior(cfg,tmp_path,'train_eval',0)
