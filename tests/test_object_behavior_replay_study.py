"""New CPU-only stage, gate, provenance and current-attempt contracts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import pytest
from vlm_object_behavior_replay import analysis,contracts,study

REPO=Path(__file__).resolve().parents[1]


def actual_config():
    return json.loads((REPO/'configs/vlm_object_behavior_replay_v1.json').read_text())


def gate_inputs():
    obj=[{'slice':'overall','metrics':{
        'accuracy':{'delta_pp':1.,'one_sided_98_75_lower_pp':.01,'seed_delta_pp':[1.]},
        'recall':{'delta_pp':0.},'specificity':{'delta_pp':-.5}}}]
    text=[{'slice':name,'delta_pp':0.,'one_sided_98_75_lower_pp':-.499,'seed_delta_pp':[0.]}
          for name in ('overall','unseen_0','rare_1_5','medium_6_20','frequent_gt20')]
    balanced={'contrasts':[{'slice':'overall','delta_pp':.01}]}
    return obj,text,balanced


def test_actual_registry_and_eval_precision_are_fixed():
    cfg=actual_config();assert study.validate_registry(cfg)==cfg
    study.validate_evaluation(cfg);analysis.validate_settings(cfg)
    bad=copy.deepcopy(cfg);bad['control_can_be_selected']=True
    with pytest.raises(ValueError):study.validate_registry(bad)
    bad=copy.deepcopy(cfg);bad['teacher_generation']['retokenize_teacher_answers']=True
    with pytest.raises(ValueError):study.validate_registry(bad)
    bad=copy.deepcopy(cfg);bad['evaluation']['do_sample']=True
    with pytest.raises(ValueError):study.validate_evaluation(bad)


@pytest.mark.parametrize('broken',list(range(8)))
def test_each_of_original_eight_pilot_guards_is_binding(broken):
    obj,text,balanced=gate_inputs()
    assert analysis.screen_gate(obj,text,balanced,replicated=False)['passed']
    if broken==0:obj[0]['metrics']['accuracy']['delta_pp']=.999
    elif broken==1:obj[0]['metrics']['accuracy']['one_sided_98_75_lower_pp']=0.
    elif broken==2:text[0]['one_sided_98_75_lower_pp']=-.5
    elif broken==3:obj[0]['metrics']['recall']['delta_pp']=-.0001
    elif broken==4:obj[0]['metrics']['specificity']['delta_pp']=-.5001
    elif broken==5:balanced['contrasts'][0]['delta_pp']=0.
    elif broken==6:text[2]['delta_pp']=-.0001
    else:text[4]['delta_pp']=-.0001
    result=analysis.screen_gate(obj,text,balanced,replicated=False)
    assert not result['passed'] and sum(result['checks'].values())==7


def test_control_cannot_expand_or_confirm_and_roles_are_safe(tmp_path):
    assert study.role('teacher','initial',42,0)=='teacher_seed42_shard0'
    assert study.role('teacher','initial',200,1)=='teacher_seed200_shard1'
    assert study.role('train_eval','gt_ce32',42)=='train_eval_gt_ce32_seed42'
    for args in [('train_eval','gt_ce32',0),('external','gt_ce32',42),('preflight','teacher_ce32',200),('teacher','teacher_ce32',42,0),('teacher','initial',42,2)]:
        with pytest.raises(ValueError):study.role(*args)
    assert study.worker_dir(tmp_path,'teacher_seed42_shard0')==tmp_path/'workers/teacher_seed42_shard0'
    for name in ('../teacher','x/y','UpperCase'):
        with pytest.raises(ValueError):study.worker_dir(tmp_path,name)


def test_all_new_clis_use_seed_and_explicit_engineering_root(tmp_path):
    cfg=tmp_path/'config.json';root=tmp_path/'outputs'
    args=study.train_arguments(cfg,root/'engineering_runs','teacher_ce32',42,True)
    assert args[args.index('--seed')+1]=='42' and '--preflight-only' in args and '--seed42' not in args
    assert args[args.index('--output-root')+1]==str(root/'engineering_runs')
    admission=root/'admission.json';teacher=study.teacher_arguments(cfg,root,42,1,admission)
    assert teacher[teacher.index('--shard')+1]=='1' and teacher[teacher.index('--admission')+1]==str(admission)
    descriptor=study.job(cfg,root,'teacher','initial',42,shard=1)
    assert descriptor['key']=='teacher_seed42_shard1' and '--worker' in descriptor['command']
    with pytest.raises(ValueError):study.eval_arguments(cfg,root,'teacher_ce32',42,'external',root/'adapter',admission)


def setup_fake_run(monkeypatch,tmp_path,passes):
    root=tmp_path/'new';root.mkdir();cfg={'output_root':str(root)};config_file=tmp_path/'config.json';config_file.write_text(json.dumps(cfg))
    events=[];seen_jobs=[]
    monkeypatch.setattr(study,'prepare',lambda f:(cfg,root,{}))
    monkeypatch.setattr(study,'verify_study',lambda *a:{})
    monkeypatch.setattr(study,'prepare_generation_inputs',lambda c,s:events.append(('prompts',s)))
    monkeypatch.setattr(study,'finalize_teacher_replies',lambda c,s:events.append(('merge',s)))
    fake=types.ModuleType('vlm_object_behavior_replay.inputs')
    fake.prepare_training_inputs=lambda c,s,k:events.append(('target_inputs',s,k))
    fake.verify_prepared_training_inputs=lambda *a,**k:{}
    monkeypatch.setitem(sys.modules,'vlm_object_behavior_replay.inputs',fake)
    def queue(root,name,jobs,identity):
        events.append(('queue',name));seen_jobs.extend(jobs)
        return [{'outcome':'passed','role':j['key']} for j in jobs]
    monkeypatch.setattr(study,'queue_stage',queue)
    def analyze(config,root,seeds,phase,**kwargs):
        events.append(('analysis',tuple(seeds),phase))
        return {'status':'complete','gate':{'passed':passes}},root/f'analysis_{phase}_{len(seeds)}.json'
    monkeypatch.setattr(study,'analyze',analyze)
    monkeypatch.setattr(study,'finish',lambda root,status,phase,analyses,receipts,started:{'status':status,'last_phase':phase,'goal_complete':False})
    return config_file,events,seen_jobs


def test_prepare_only_does_not_prepare_5792_or_start_queue(monkeypatch,tmp_path):
    config,events,jobs=setup_fake_run(monkeypatch,tmp_path,False)
    result=study.run_study(config,prepare_only=True)
    assert result['status']=='prepared' and not result['full_prompt_preparation_run']
    assert events==[] and jobs==[] and result['goal_complete'] is False


def test_negative_pilot_stops_before_any_seed_expansion(monkeypatch,tmp_path):
    config,events,jobs=setup_fake_run(monkeypatch,tmp_path,False)
    result=study.run_study(config)
    assert result['status']=='completed_negative' and result['goal_complete'] is False
    assert events[:5]==[('prompts',42),('queue','teacher_generation_seed42'),('merge',42),('target_inputs',42,'teacher_ce32'),('target_inputs',42,'gt_ce32')]
    assert [x['key'] for x in jobs]==['teacher_seed42_shard0','teacher_seed42_shard1','preflight_teacher_ce32_seed42','preflight_gt_ce32_seed42','train_eval_teacher_ce32_seed42','train_eval_gt_ce32_seed42']
    assert not any(event[0]=='prompts' and event[1]!=42 for event in events)


def test_full_stages_keep_only_teacher_candidate_and_final_goal_false(monkeypatch,tmp_path):
    config,events,jobs=setup_fake_run(monkeypatch,tmp_path,True)
    result=study.run_study(config)
    assert result['status']=='confirmation_passed_pending_independent_review' and result['goal_complete'] is False
    pilot=events.index(('analysis',(42,),'screen'))
    assert events.index(('prompts',0))>pilot and events.index(('prompts',200))>pilot
    assert ('target_inputs',0,'gt_ce32') not in events and ('target_inputs',200,'gt_ce32') not in events
    for job in jobs:
        if 'gt_ce32' in job['key']:assert job['key'] in ('preflight_gt_ce32_seed42','train_eval_gt_ce32_seed42')
    assert events.index(('queue','external_confirmation'))>events.index(('analysis',(0,42,200),'screen'))
    assert events.index(('queue','pope_confirmation'))>events.index(('analysis',(0,42,200),'external'))


def test_current_contract_failure_ignores_stale_oom_log_and_receipt(tmp_path):
    place=tmp_path/'worker';log=place/'logs/training.log';log.parent.mkdir(parents=True)
    log.write_text('RuntimeError: CUDA out of memory\n')
    preflight=tmp_path/'PREFLIGHT.json';preflight.write_text(json.dumps({'status':'failed','peak_allocated_gib':66,'peak_reserved_gib':66,'memory_cap_gib':65}))
    command=[sys.executable,'-c',"import sys; print('RuntimeError: unrelated contract failure');sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError):study.subprocess_call(place,'role',{},'training',command,preflight)
    record=json.loads(next((place/'attempts').glob('*.call.json')).read_text())
    assert record['fresh_failure_receipts']=={} and record['log_start_byte']>0


def test_actual_teacher_memory_cap_is_engineering_not_scientific(tmp_path):
    place=tmp_path/'worker'
    command=[sys.executable,'-c',"import sys; print('RuntimeError: teacher_generation_memory_cap_exceeded');sys.exit(1)"]
    with pytest.raises(study.EngineeringRejected) as caught:study.subprocess_call(place,'role',{},'teacher_rollout',command)
    assert caught.value.reason=='teacher_generation_memory_cap_exceeded'
    record_file=next((place/'attempts').glob('*.call.json'));record=json.loads(record_file.read_text())
    assert study.call_reason(record,caught.value.evidence)=='teacher_generation_memory_cap_exceeded'


def test_source_registry_binds_new_namespaces_and_launchers():
    sources=contracts.source_hashes()
    for name in ('vlm_object_behavior_replay/contracts.py','vlm_object_behavior_replay/evaluate.py',
        'vlm_object_behavior_replay/analysis.py','vlm_object_behavior_replay/study.py',
        'tools/run_object_behavior_replay_study.py','tools/check_object_behavior_replay_contracts.py'):
        p=REPO/name if name.startswith('tools/') else REPO/'src'/name
        assert sources[name]==hashlib.sha256(p.read_bytes()).hexdigest()



def test_new_full_rows_and_old_legacy_stream_are_distinct_contracts():
    keys=('object_order_sha256','object_microbatch_order_sha256','replay_order_sha256','replay_microbatch_order_sha256')
    anchor={key:'fixed_'+key for key in keys}
    anchor.update(object_input_contract_sha256='old_object_standard',replay_input_contract_sha256='old_GT_standard')
    train={key:anchor[key] for key in keys}
    train.update(object_input_contract_sha256='new_object_full_with_roles',object_legacy_input_contract_sha256='old_object_standard',
        replay_input_contract_sha256='new_replay_full_with_roles',replay_legacy_input_contract_sha256='new_teacher_standard')
    contracts.validate_legacy_stream_alignment(train,anchor,'teacher_ce32',42)
    with pytest.raises(ValueError):contracts.validate_legacy_stream_alignment(train,anchor,'gt_ce32',42)
    train['replay_legacy_input_contract_sha256']='old_GT_standard'
    contracts.validate_legacy_stream_alignment(train,anchor,'gt_ce32',42)
    train['object_legacy_input_contract_sha256']='wrong'
    with pytest.raises(ValueError):contracts.validate_legacy_stream_alignment(train,anchor,'teacher_ce32',42)


def test_gpu_admission_binds_actual_uuid_and_config_bytes(monkeypatch,tmp_path):
    root=tmp_path/'output';root.mkdir();cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'output_root':str(root)}))
    (root/'STUDY_INPUT_LOCK.json').write_text('{}')
    monkeypatch.setattr(contracts,'verify_study',lambda *a:{})
    file=study.task_admission(root/'workers/train_eval_teacher_ce32_seed42',cfg,'teacher_ce32',42,'object',{'uuid':'GPU-fixture'})
    contracts.validate_gpu_admission(cfg,file,condition='teacher_ce32',seed=42,task='object',gpu_uuid='GPU-fixture')
    with pytest.raises(ValueError):contracts.validate_gpu_admission(cfg,file,condition='teacher_ce32',seed=42,task='object',gpu_uuid='GPU-other')
    cfg.write_text(json.dumps({'output_root':str(root),'changed':True}))
    with pytest.raises(ValueError):contracts.validate_gpu_admission(cfg,file,condition='teacher_ce32',seed=42,task='object',gpu_uuid='GPU-fixture')


def test_inherited_admission_requires_original_bytes_and_qualified_fields(tmp_path):
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    train=tmp_path/'TRAIN.json';train.write_text('{}');protocol=tmp_path/'protocol.md';protocol.write_text('fixed')
    old_train=tmp_path/'old_train.json';old_eval=tmp_path/'old_eval.json'
    original={'status':'admitted','quality_gate_passed':True,'scope':'annotation_supervised_object_sft',
        'automatic_diagnostic_bypass_allowed':False,'train_manifest_sha256':digest(train),'dataset_protocol_sha':digest(protocol)}
    old_train.write_text(json.dumps(original))
    evaluation={'schema_version':'rarevlm.object_external.evaluation_admission.v1','status':'admitted','quality_gate_passed':True,
        'scope':'object_external_confirmation','automatic_diagnostic_bypass_allowed':False}
    for key in ('external_manifest_sha256','external_class_balanced_manifest_sha256','external_dataset_protocol_sha256','frozen_audit_sha256','semantic_review_sha256','cpu_completion_sha256'):evaluation[key]='f'*64
    old_eval.write_text(json.dumps(evaluation))
    new_train=tmp_path/'new_train.json';new_eval=tmp_path/'new_eval.json'
    new_train.write_text(json.dumps({**original,'source_admission_path':str(old_train),'source_admission_sha256':digest(old_train)}))
    new_eval.write_text(json.dumps({**evaluation,'source_admission_path':str(old_eval),'source_admission_sha256':digest(old_eval)}))
    cfg={'data_admission':str(new_train),'reference_data_admission':str(old_train),'evaluation_data_admission':str(new_eval),
        'reference_evaluation_data_admission':str(old_eval),'train_manifest':str(train),'dataset_protocol':str(protocol)}
    result=contracts.validate_inherited_admissions(cfg);assert result['data_admission_sha256']==digest(new_train)
    original['additional_evidence']='changed';old_train.write_text(json.dumps(original))
    with pytest.raises(ValueError):contracts.validate_inherited_admissions(cfg)



def test_historical_jsonl_requires_real_complete_newline():
    payload=b'{"question_id":"1"}'
    assert contracts.parse_complete_history(payload+bytes([10]))==[{'question_id':'1'}]
    for raw in (payload,payload+bytes([92,110]),b'{}'+bytes([10]),b'[]'+bytes([10])):
        with pytest.raises(ValueError):contracts.parse_complete_history(raw)


def test_teacher_accounting_separates_started_completed_and_engineering(tmp_path):
    directory=tmp_path/'teacher_replies/seed42/shard0';directory.mkdir(parents=True)
    rows=[{'event':'generation_started','phase':'formal','questions':4},
        {'event':'generation_completed','phase':'formal','questions':4,'generation_seconds':1.},
        {'event':'generation_started','phase':'preflight','questions':4},
        {'event':'generation_completed','phase':'preflight','questions':4,'generation_seconds':2.},
        {'event':'generation_started','phase':'formal','questions':4}]
    (directory/'EXECUTION.jsonl').write_text(''.join(json.dumps(row)+chr(10) for row in rows))
    value=study.teacher_generation_accounting(tmp_path)
    assert value['actual_generation_started_QA_all_sessions']==12
    assert value['actual_generation_completed_QA_all_sessions']==8
    assert value['actual_formal_generation_questions']==4 and value['actual_preflight_generation_questions']==4
    assert value['unfinished_generation_call_attempts']==1 and value['actual_total_generation_questions']==8



def test_teacher_completion_binds_full_scope_uuid_and_initial_parameters():
    args={'seed':42,'shard':0,'gpu_uuid':'GPU-fixture','config_file_sha256':'c','study_input_lock_sha256':'s','initial_master_sha256':'m'}
    value={'schema_version':'rarevlm.object_behavior_replay.teacher_shard.complete.v1','status':'completed','seed':42,'shard':0,
        'role':'teacher_seed42_shard0','producer_uuid':'GPU-fixture','config_file_sha256':'c','study_input_lock_sha256':'s',
        'formal_saved_generation_questions':2896,'formal_batches_completed':724,'preflight_saved_generation_questions':8,
        'teacher_initial_master_sha256_before':'m','teacher_initial_master_sha256_after':'m','optimizer_updates':0,
        'new_probability_cache':False,'frozen_base_sha256_before':'b','frozen_base_sha256_after':'b',
        'teacher_native_lora_sha256_before':'native','teacher_native_lora_sha256_after':'native',
        'all_frozen_parameters_sha256_before':'all','all_frozen_parameters_sha256_after':'all',
        'teacher_inference':{'numerical_path':'original_bf16','autocast':False,'native_lora_dtype':'torch.bfloat16',
            'adapter_autocast_dtype':False,'explicit_lora_bf16_conversion':True,'optimizer_updates':0}}
    study.validate_teacher_completion(value,**args)
    for key,bad in [('formal_batches_completed',723),('producer_uuid','GPU-other'),('teacher_initial_master_sha256_after','changed'),('new_probability_cache',True)]:
        changed={**value,key:bad}
        with pytest.raises(ValueError):study.validate_teacher_completion(changed,**args)



def test_passed_cpu_receipt_cannot_hide_skips_zero_tests_or_source_changes():
    sources={'new.py':'sha'}
    receipt={'schema_version':'rarevlm.object_behavior_replay.cpu_contracts.v1','status':'passed',
        'source_unchanged_during_tests':True,'new_GPU_operations':0,'new_pretrained_model_forwards':0,'return_code':0,
        'counts':{'tests':3,'failures':0,'errors':0,'skipped':0},'scientific_source_sha256':sources,
        'scientific_source_sha256_before':sources,'scientific_source_sha256_after':sources}
    study.validate_cpu_contract_receipt(receipt,sources)
    for key,value in [('tests',0),('skipped',1),('failures',1),('errors',1)]:
        bad=copy.deepcopy(receipt);bad['counts'][key]=value
        with pytest.raises(ValueError):study.validate_cpu_contract_receipt(bad,sources)
    with pytest.raises(ValueError):study.validate_cpu_contract_receipt(receipt,{'changed.py':'sha'})



def test_historical_official_metrics_are_exact_with_additional_new_diagnostics():
    old={'overall':{'n':9000,'TP':4000,'accuracy':.9},'parser_contract':{'otherwise':'yes'}}
    new=copy.deepcopy(old);new['overall']['first_yesno_invalid_count']=0;new['first_yesno_contract']={'diagnostic_only':True}
    contracts.validate_legacy_metric_subset(old,new)
    new['overall']['TP']=3999
    with pytest.raises(ValueError):contracts.validate_legacy_metric_subset(old,new)



def test_clean_prepare_creates_qualified_aliases_before_validation_and_never_overwrites(tmp_path):
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=actual_config();output=tmp_path/'new';external=tmp_path/'external';external.mkdir()
    train=put(tmp_path/'TRAIN.json',{'rows':[]});protocol=tmp_path/'object_protocol.md';protocol.write_text('fixed object')
    full=put(external/'MANIFEST.json',{'rows':[{},{}]});balanced=put(external/'BALANCED.json',{})
    ext_protocol=tmp_path/'external_protocol.md';ext_protocol.write_text('fixed external')
    frozen=put(external/'FROZEN64.json',{'fixed':'audit'})
    review=put(external/'REVIEW.json',{'schema_version':'rarevlm.object_external.semantic_review.v1','gate':{'passed':True},
        'source_audit_sha256':digest(frozen),'external_manifest_sha256':digest(full)})
    journal=external/'ENCODING_CHECKS.jsonl';journal.write_text('{}'+chr(10))
    complete=put(external/'COMPLETION.json',{'complete':True,'data_only':True,'optimizer_updates':0,'training_allowed':False,
        'encoding':{'model_forward_or_gpu_used':False,'questions':2,'fresh_actual_cpu_encodes':2,'old_encoding_records_reused':0,
            'input_ids_labels_and_pixel_hash_saved':True,'truncation':False,'journal_sha256':digest(journal)}})
    old_train=put(tmp_path/'old_train_admission.json',{'status':'admitted','quality_gate_passed':True,
        'scope':'annotation_supervised_object_sft','automatic_diagnostic_bypass_allowed':False,
        'train_manifest_sha256':digest(train),'dataset_protocol_sha':digest(protocol)})
    old_external=put(tmp_path/'old_external_admission.json',{'schema_version':'rarevlm.object_external.evaluation_admission.v1',
        'status':'admitted','quality_gate_passed':True,'scope':'object_external_confirmation','automatic_diagnostic_bypass_allowed':False,
        'external_manifest_sha256':digest(full),'external_class_balanced_manifest_sha256':digest(balanced),
        'external_dataset_protocol_sha256':digest(ext_protocol),'frozen_audit_sha256':digest(frozen),
        'semantic_review_sha256':digest(review),'cpu_completion_sha256':digest(complete)})
    cfg.update(output_root=str(output),teacher_reply_root=str(output/'teacher_replies'),train_manifest=str(train),
        dataset_protocol=str(protocol),external_data_root=str(external),external_manifest=str(full),
        external_class_balanced_manifest=str(balanced),external_dataset_protocol=str(ext_protocol),
        external_frozen_audit=str(frozen),external_semantic_review=str(review),external_cpu_completion=str(complete),
        data_admission=str(output/'DATA_ADMISSION.json'),evaluation_data_admission=str(output/'EVALUATION_DATA_ADMISSION.json'),
        reference_data_admission=str(old_train),reference_evaluation_data_admission=str(old_external),cpu_contracts=str(output/'CPU_CONTRACTS.json'))
    config=put(tmp_path/'config.json',cfg)
    # Real early prepare reaches the absent CPU receipt only after creating and validating both aliases.
    with pytest.raises(FileNotFoundError) as error:study.prepare(config)
    assert Path(error.value.filename)==output/'CPU_CONTRACTS.json'
    aliases=[output/'DATA_ADMISSION.json',output/'EVALUATION_DATA_ADMISSION.json']
    before={str(path):digest(path) for path in aliases}
    contracts.create_inherited_admissions(cfg)
    assert before=={str(path):digest(path) for path in aliases}
    for path,original in zip(aliases,[old_train,old_external]):
        value=json.loads(path.read_text());source=json.loads(original.read_text())
        assert value['source_admission_path']==str(original) and value['source_admission_sha256']==digest(original)
        assert all(value[key]==item for key,item in source.items())
        assert value['automatic_diagnostic_bypass_allowed'] is False
    tampered=json.loads(aliases[0].read_text());tampered['quality_gate_passed']=False;aliases[0].write_text(json.dumps(tampered))
    tampered_sha=digest(aliases[0])
    with pytest.raises(ValueError):contracts.create_inherited_admissions(cfg)
    assert digest(aliases[0])==tampered_sha



def test_historical_cohort_uses_selected_images_not_global_lookup_and_rejects_wrong_scope():
    refs=[{'question_id':'1','image_id':'a'},{'question_id':'2','image_id':'b'}]
    records=copy.deepcopy(refs);lookup={'a':{},'b':{},'unused_other_split':{}}
    metadata={'questions':2,'unique_images':2,'n':2,'expected_questions':2,'expected_images':2}
    evaluation={'n':2,'unique_images':2,'expected_questions':2,'expected_images':2}
    lock={'n':2,'expected_questions':2,'expected_images':2}
    selected=contracts.validate_historical_cohort_scope(records,refs,lookup,metadata,evaluation,lock,'text')
    assert selected=={'a','b'} and len(lookup)>len(selected)
    for task in ('object','external','pope'):
        scope={'questions':2,'unique_images':2}
        assert contracts.validate_historical_cohort_scope(records,refs,lookup,scope,scope,scope,task)==selected
    with pytest.raises(ValueError):contracts.validate_historical_cohort_scope(records,refs,{'a':{}},metadata,evaluation,lock,'text')
    wrong=[records[0],{**records[1],'image_id':'unused_other_split'}]
    with pytest.raises(ValueError):contracts.validate_historical_cohort_scope(wrong,refs,lookup,metadata,evaluation,lock,'text')
    for source,key in [('metadata','questions'),('metadata','unique_images'),('evaluation','n'),('evaluation','unique_images'),
        ('evaluation','expected_questions'),('evaluation','expected_images'),('input_lock','n'),('input_lock','expected_images')]:
        values={'metadata':copy.deepcopy(metadata),'evaluation':copy.deepcopy(evaluation),'input_lock':copy.deepcopy(lock)}
        values[source][key]=3
        with pytest.raises(ValueError):contracts.validate_historical_cohort_scope(records,refs,lookup,values['metadata'],values['evaluation'],values['input_lock'],'text')


def test_original_pope_percent_is_derived_in_memory_without_changing_raw_metadata():
    original={'questions':9000,'unique_images':500,'metrics':{'overall':{'n':9000,'accuracy':.871}}}
    preserved=copy.deepcopy(original);view,proof=contracts.historical_evaluation_view(original,'pope')
    assert math.isclose(view['score_percent'],87.1) and original==preserved and 'score_percent' not in original
    assert proof['changed_fields']==['score_percent'] and proof['score_percent_source']=='original.metrics.overall.accuracy * 100'
    assert proof['in_memory_only'] and proof['original_fields_unchanged']
    assert proof['original_evaluation_content_sha256']==proof['all_original_fields_sha256_after']
    for task in ('text','object','external'):
        with pytest.raises(ValueError):contracts.historical_evaluation_view(original,task)
    for invalid in ({**original,'score_percent':None},{**original,'metrics':{'overall':{'n':8999,'accuracy':.871}}}):
        with pytest.raises(ValueError):contracts.historical_evaluation_view(invalid,'pope')
