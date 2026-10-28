"""New export identity and actual conditional controller transitions; CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from vlm_lora_coordinate_shrink import CANDIDATE,CONTROL,CONFIG_SCHEMA,SCHEMA
from vlm_lora_coordinate_shrink import contracts,analysis,study,inputs,lock
from vlm_probe.evaluate import sha_file
from test_lora_coordinate_shrink_export import synthetic_case


def dump(file,value):
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value));return file


def test_actual_configuration_has_distinct_producer_schema_and_fixed_coordinate_method():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_lora_coordinate_shrink_v1.json').read_bytes())
    study.validate_registry(config)
    assert config['schema_version']==CONFIG_SCHEMA and CONFIG_SCHEMA!=SCHEMA
    assert config['coordinate_shrink']['alpha']==.5 and config['all_seeds']==[0,42,200]
    assert config['parent_output_root'].endswith('vlm_object_behavior_replay_v1')
    assert config['producer_reference_root'].endswith('vlm_generated_prefix_kl_v1')
    assert config['seed42_input_reference']['new_training_condition']==CANDIDATE
    from vlm_lora_coordinate_shrink.train import validate_config
    validate_config(config)
    for key,value in [('control_can_be_selected',True),('seed42_new_training_allowed',True)]:
        bad=deepcopy(config);bad[key]=value
        with pytest.raises(ValueError):study.validate_registry(bad)
    bad=deepcopy(config);bad['coordinate_shrink']['alpha']=.25
    with pytest.raises(ValueError):study.validate_registry(bad)


@pytest.mark.parametrize('name',['teacher_ce32','gt_ce32','teacher_kl32','initial','object_kl'])
def test_historical_conditions_cannot_be_new_candidates(name):
    with pytest.raises(ValueError):contracts.validate_condition(name,42,'text')


def test_seed42_has_no_new_training_or_teacher_role_and_raw_adapter_is_not_candidate(tmp_path):
    assert study.role('eval',CANDIDATE,42)=='eval_lora_half32_seed42'
    assert study.role('producer',CANDIDATE,0)=='producer_lora_half32_seed0'
    assert study.role('teacher','initial',200,1)=='teacher_seed200_shard1'
    for operation,condition,seed,shard in [('producer',CANDIDATE,42,None),('teacher','initial',42,0),('train_eval',CANDIDATE,42,None)]:
        with pytest.raises(ValueError):study.role(operation,condition,seed,shard)
    with pytest.raises(ValueError):study.train_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,42)
    config={'initial_adapter':str(tmp_path/'initial'),'output_root':str(tmp_path)}
    with pytest.raises(ValueError,match='never raw producer'):
        contracts.adapter_binding(config,CANDIDATE,42,tmp_path/'producer/adapter',tmp_path/'model',config_file=tmp_path/'config.json')


def test_worker_commands_preserve_raw_producer_and_full_confirmation_priors(tmp_path):
    command=study.train_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,0)
    assert command[command.index('--seed')+1]=='0' and command[command.index('-m')+1]=='vlm_lora_coordinate_shrink.train'
    assert '--preflight-only' not in command
    with pytest.raises(ValueError):study.eval_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,42,'external',tmp_path/'a',tmp_path/'admission')
    for task,flag in [('external','--screen-analysis'),('pope','--external-analysis')]:
        command=study.eval_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,42,task,tmp_path/'a',tmp_path/'admission',tmp_path/'prior.json')
        assert flag in command


@pytest.mark.parametrize('role_name',['eval_lora_half32_seed42','teacher_seed0_shard1'])
def test_worker_directory_rejects_unsafe_role_names(tmp_path,role_name):
    assert study.worker_dir(tmp_path,role_name)==tmp_path/'workers'/role_name
    with pytest.raises(ValueError):study.worker_dir(tmp_path,'../'+role_name)


@pytest.mark.parametrize('task',['object','text'])
def test_actual_admission_writer_matches_new_eval_role_consumer(tmp_path,monkeypatch,task):
    root=tmp_path/'output';config_file=dump(tmp_path/'config.json',{'output_root':str(root)})
    dump(root/'STUDY_INPUT_LOCK.json',{'synthetic':True})
    monkeypatch.setattr(contracts,'verify_study',lambda *a,**k:{})
    file=study.task_admission(root/'workers/eval_lora_half32_seed42',config_file,CANDIDATE,42,task,{'uuid':'synthetic-uuid'})
    saved=contracts.validate_gpu_admission(config_file,file,condition=CANDIDATE,seed=42,task=task,gpu_uuid='synthetic-uuid')
    assert saved['role']=='eval_lora_half32_seed42'
    with pytest.raises(ValueError,match='gpu_uuid'):
        contracts.validate_gpu_admission(config_file,file,condition=CANDIDATE,seed=42,task=task,gpu_uuid='wrong')


def test_future_producer_reference_is_accepted_by_actual_exporter_identity_consumer(synthetic_case,monkeypatch):
    from vlm_lora_coordinate_shrink import export
    cfg,old_result,old_adapter,_,historical,_,_=synthetic_case
    config=json.loads(cfg.read_text());root=Path(config['output_root'])
    raw=root/'producer_runs'/CANDIDATE/'seed0/train';raw.mkdir(parents=True)
    shutil.copytree(old_adapter,raw/'adapter')
    initial=Path(config['initial_adapter']);initial_files=export._file_map(initial)
    result={**json.loads(old_result.read_text()),'schema_version':SCHEMA,'condition':CANDIDATE,'seed':0,
        'artifact_role':'unshrunk_producer','coordinate_shrink_applied':False,
        'final_master_sha256':historical['expected_final_master_sha256'],
        'initial_adapter_dir':str(initial),'initial_adapter_files_sha256':initial_files}
    result_file=dump(raw/'TRAIN_RESULT.json',result)
    for file in (root/'STUDY_INPUT_LOCK.json',*(raw/name for name in ('TRAIN_INPUT_LOCK.json','SCHEDULE.json','PREFLIGHT.json','TRAINED_INPUT_CONTRACTS.json'))):dump(file,{'synthetic':True})
    calls=[]
    monkeypatch.setattr(inputs,'verify_producer_training_result',lambda *a,**k:calls.append((a,k)) or {})
    reference=contracts.producer_reference_for_seed(config,0)
    assert len(calls)==1 and calls[0][1]=={'condition':CANDIDATE,'seed':0}
    output=root/'runs'/CANDIDATE/'seed0/shrink'
    _,binding,actual_initial,actual_producer=export._sources(cfg,0,result_file,raw/'adapter',output,reference)
    assert actual_initial==initial and actual_producer==raw/'adapter'
    assert binding['producer_reference']==reference
    assert all(reference['source_binding_files_sha256'].get(k)==v for k,v in config['producer_reference_files_sha256'].items())
    for key in ('initial_canonical_master_sha256','initial_adapter_dir','initial_adapter_files_sha256'):
        bad=deepcopy(reference);bad.pop(key)
        with pytest.raises(ValueError,match='initial adapter'):
            export._sources(cfg,0,result_file,raw/'adapter',output,bad)
    bad=deepcopy(result);bad['initial_adapter_files_sha256']={}
    dump(result_file,bad)
    with pytest.raises(ValueError,match='initial canonical'):
        contracts.producer_reference_for_seed(config,0)


def test_actual_analysis_output_names_match_historical_KL_and_new_half_conditions(tmp_path,monkeypatch):
    config_file=dump(tmp_path/'config.json',{'asset_root':str(tmp_path)})
    monkeypatch.setattr(lock,'verify_study',lambda *a,**k:{})
    monkeypatch.setattr(analysis,'validate_settings',lambda cfg:{'synthetic':True})
    monkeypatch.setattr(analysis,'source_hashes',lambda:{})
    pairs=[]
    def read(root,condition,seed,task,file,config):
        return {'complete':True,'provenance':{},'canonicalization':{},'rows':[{'condition':condition,'task':task}]}
    monkeypatch.setattr(analysis,'read_cell',read)
    monkeypatch.setattr(analysis,'assert_pair',lambda *a:None)
    monkeypatch.setattr(analysis,'prepare_slices',lambda *a:{'features':[],'slice_lock_sha256':'slices'})
    def contrast(left,rights,*args):
        pairs.append((left[0]['condition'],rights[0][0]['condition']));return []
    monkeypatch.setattr(analysis,'binary_contrast',lambda left,rights:(contrast(left,rights),{}))
    monkeypatch.setattr(analysis,'text_contrasts',contrast)
    monkeypatch.setattr(analysis,'balanced_contrast',lambda *a:{})
    monkeypatch.setattr(analysis,'screen_gate',lambda *a,**k:{'passed':False,'checks':{}})
    result,file=analysis.analyze(config_file,tmp_path/'output',[42],'screen')
    contrasts=result['exploratory_control_contrasts']['contrasts']
    assert set(contrasts)=={'teacher_kl32_minus_initial','lora_half32_minus_teacher_kl32'}
    assert all(set(value)=={'object','text'} for value in contrasts.values())
    assert pairs.count(('initial',CONTROL))==2 and pairs.count((CONTROL,CANDIDATE))==2
    assert result['candidate']==CANDIDATE and result['exploratory_control']==CONTROL
    assert result['eligible_candidates']==[] and result['control_can_be_selected'] is False
    assert json.loads(file.read_text())==result


def controller_scenario(tmp_path,monkeypatch,*,failed_phase=None,engineering_phase=None):
    root=tmp_path/'output';root.mkdir();config={'output_root':str(root)};config_file=dump(tmp_path/'config.json',config)
    dump(root/'STUDY_INPUT_LOCK.json',{'synthetic':True});events=[]
    monkeypatch.setattr(study,'prepare',lambda file:(config,root,{'synthetic':True}))
    monkeypatch.setattr(study,'verify_study',lambda *a,**k:{})
    monkeypatch.setattr(study,'prepare_generation_inputs',lambda cfg,seed:events.append(('prepare_prompt',seed)))
    monkeypatch.setattr(study,'finalize_teacher_replies',lambda cfg,seed:events.append(('merge_teacher',seed)))
    monkeypatch.setattr(inputs,'prepare_training_inputs',lambda cfg,seed,condition:events.append(('prepare_train_inputs',seed,condition)))
    monkeypatch.setattr(inputs,'verify_prepared_training_inputs',lambda *a:{})
    monkeypatch.setattr(study,'ensure_export',lambda file,cfg,seed:events.append(('export',seed)))
    def queue(output,name,jobs,identity):
        items=[]
        for job in jobs:
            cmd=job['command'];get=lambda key:cmd[cmd.index(key)+1]
            items.append({'operation':get('--operation'),'condition':get('--condition'),'seed':int(get('--seed')),
                'shard':int(get('--shard')) if '--shard' in cmd else None})
        events.append(('queue',name,items))
        return [{'outcome':'engineering_rejected' if name==engineering_phase else 'passed','role':job['key']} for job in jobs]
    monkeypatch.setattr(study,'queue_stage',queue)
    def analyze(file,output,seeds,phase,**kwargs):
        label='pilot' if list(seeds)==[42] else phase
        events.append(('analyze',list(seeds),phase))
        return {'status':'complete','gate':{'passed':label!=failed_phase}},root/'analysis'/(label+'.json')
    monkeypatch.setattr(study,'analyze',analyze)
    monkeypatch.setattr(study,'finish',lambda output,status,phase,analyses,receipts,started:
        {'status':status,'goal_complete':False,'last_phase':phase,'analyses':list(analyses)})
    return config_file,root,events


def test_actual_controller_negative_half_pilot_never_retrains_seed42_or_dispatches_later_phases(tmp_path,monkeypatch):
    file,root,events=controller_scenario(tmp_path,monkeypatch,failed_phase='pilot')
    result=study.run_study(file)
    assert result['status']=='completed_negative' and result['goal_complete'] is False
    queues=[e for e in events if e[0]=='queue']
    assert [e[1] for e in queues]==['pilot42']
    assert queues[0][2]==[{'operation':'eval','condition':CANDIDATE,'seed':42,'shard':None}]
    assert not any(e[0] in ('prepare_prompt','export','prepare_train_inputs') for e in events)


def test_actual_controller_success_executes_new_0_200_producers_then_full_external_and_pope(tmp_path,monkeypatch):
    file,root,events=controller_scenario(tmp_path,monkeypatch)
    result=study.run_study(file)
    assert result['status']=='confirmation_passed_pending_independent_review' and result['goal_complete'] is False
    queues=[e for e in events if e[0]=='queue'];by_name={e[1]:e[2] for e in queues}
    assert [e[1] for e in queues]==['pilot42','teacher_generation_seeds0_200','producer_replication','replication_evaluation','external_confirmation','pope_confirmation']
    assert [(x['seed'],x['shard']) for x in by_name['teacher_generation_seeds0_200']]==[(0,0),(0,1),(200,0),(200,1)]
    assert all(x['condition']=='initial' for x in by_name['teacher_generation_seeds0_200'])
    assert by_name['producer_replication']==[{'operation':'producer','condition':CANDIDATE,'seed':s,'shard':None} for s in (0,200)]
    assert [e for e in events if e[0]=='export']==[('export',0),('export',200)]
    assert [e for e in events if e[0]=='prepare_prompt']==[('prepare_prompt',0),('prepare_prompt',200)]
    assert not any(e[0]=='prepare_train_inputs' and e[1]==42 for e in events)
    for name,operation in [('external_confirmation','external'),('pope_confirmation','pope')]:
        assert by_name[name]==[{'operation':operation,'condition':CANDIDATE,'seed':s,'shard':None} for s in (0,42,200)]
    assert [e for e in events if e[0]=='analyze']==[('analyze',[42],'screen'),('analyze',[0,42,200],'screen'),('analyze',[0,42,200],'external'),('analyze',[0,42,200],'confirm')]


@pytest.mark.parametrize('failed,expected_queues',[('screen',4),('external',5)])
def test_actual_controller_later_scientific_failure_stops_before_next_confirmation(tmp_path,monkeypatch,failed,expected_queues):
    file,root,events=controller_scenario(tmp_path,monkeypatch,failed_phase=failed)
    result=study.run_study(file)
    assert result['status']=='completed_negative' and result['goal_complete'] is False
    assert len([e for e in events if e[0]=='queue'])==expected_queues


def test_actual_controller_engineering_rejection_is_not_accuracy_failure(tmp_path,monkeypatch):
    file,root,events=controller_scenario(tmp_path,monkeypatch,engineering_phase='pilot42')
    result=study.run_study(file)
    assert result['status']=='engineering_rejected' and not any(e[0]=='analyze' for e in events)


def test_actual_completed_controller_resume_verifies_receipts_without_dispatch(tmp_path,monkeypatch):
    file,root,events=controller_scenario(tmp_path,monkeypatch)
    existing={'status':'completed_negative','goal_complete':False,'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'analysis_files_sha256':{},'worker_receipts_sha256':{},'dynamic_input_files_sha256':{},'export_manifest_files_sha256':{}}
    dump(root/'COMPLETION.json',existing)
    assert study.run_study(file)==existing and events==[]


def test_completed_analysis_resume_retains_original_bytes_and_rejects_changed_export_identity(tmp_path,monkeypatch):
    source={}
    for kind in ('predictions','evaluation','input_lock'):
        file=dump(tmp_path/(kind+'.json'),{'synthetic':True})
        source[kind+'_file']=str(file);source[kind+'_sha256']=sha_file(file)
    destination=tmp_path/'ANALYSIS.json';current={'schema_version':'rarevlm.lora_coordinate_shrink.screen.v1',
        'stage':'pilot','config_sha256':'config','source_sha256':{},'seeds':[42],'candidate':CANDIDATE,
        'exploratory_control':CONTROL,'settings':{},'source_data':{'initial_42_object':source},'canonicalization':{},
        'control_can_be_selected':False,'control_reference_only':True}
    gate={'passed':False};saved={**current,'status':'complete','contrasts':{'object':[],'text':[]},'class_balanced':{},'gate':gate}
    dump(destination,saved);before=destination.read_bytes();monkeypatch.setattr(analysis,'screen_gate',lambda *a,**k:gate)
    assert analysis.reuse_complete_analysis(destination,current,'screen',None,{})==saved and destination.read_bytes()==before
    changed=deepcopy(current);changed['source_data']={'different_export':{}}
    with pytest.raises(ValueError):analysis.reuse_complete_analysis(destination,changed,'screen',None,{})


@pytest.mark.parametrize('condition,seed,task',[(CONTROL,0,'object'),(CONTROL,42,'pope'),('initial',200,'external')])
def test_historical_control_cannot_be_selected_or_expanded(condition,seed,task):
    with pytest.raises(ValueError):contracts.validate_historical_request(condition,seed,task)


def test_new_replication_cannot_use_old_stage32_failed_or_wrong_candidate_pilot(tmp_path):
    file=dump(tmp_path/'ANALYSIS.json',{'schema_version':'rarevlm.generated_prefix_kl.screen.v1','status':'complete',
        'seeds':[42],'candidate':CONTROL,'gate':{'passed':False},'eligible_candidates':[]})
    config=dump(tmp_path/'config.json',{})
    with pytest.raises(ValueError):analysis.passing_pilot(file,config)
