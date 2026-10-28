"""New schema, exact historical aliases, and mandatory full-stage branch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from vlm_visual_merger_kl import CANDIDATE,SCHEMA
from vlm_visual_merger_kl.contracts import (validate_condition,validate_legacy_microbatch_digest,
    validate_historical_cohort_scope,historical_evaluation_view,canonical_rows,content_sha,phase_admission,validate_historical_request)
from vlm_visual_merger_kl.lock import verify_parent_proof
from vlm_visual_merger_kl.study import role,eval_arguments,train_arguments,validate_registry,worker_prior


def schedule():
    objects=[[f'o{i*16+j}' for j in range(8 if i==180 else 16)] for i in range(181)]
    replay=[[f'r{i*32+j}' for j in range(32)] for i in range(181)]
    return {'object_question_ids_by_update':objects,'replay_question_ids_by_update':replay,
        'learning_rates_by_update':[5e-6]*181,'training_microbatch_questions':4,
        'object_microbatches_by_update':[[q[j:j+4] for j in range(0,len(q),4)] for q in objects],
        'replay_microbatches_by_update':[[q[j:j+4] for j in range(0,len(q),4)] for q in replay]}


def test_missing_historical_microbatch_field_is_content_derived_without_mutation():
    value=schedule();anchor={'schedule_sha256':content_sha(value)};before=deepcopy(anchor)
    assert validate_legacy_microbatch_digest(anchor,value)==content_sha(value['replay_microbatches_by_update'])
    assert anchor==before


@pytest.mark.parametrize('bad',['declared','chain','boundary','incomplete'])
def test_legacy_missing_field_rule_rejects_actual_differences(bad):
    value=schedule();anchor={'schedule_sha256':content_sha(value)}
    if bad=='declared':anchor['replay_microbatch_order_sha256']='0'*64
    elif bad=='chain':anchor['schedule_sha256']='0'*64
    elif bad=='boundary':
        value['replay_microbatches_by_update'][0][0].reverse();anchor['schedule_sha256']=content_sha(value)
    else:
        value['learning_rates_by_update'].pop();anchor['schedule_sha256']=content_sha(value)
    with pytest.raises(ValueError):validate_legacy_microbatch_digest(anchor,value)


@pytest.mark.parametrize('evaluation,lock',[
    ({'n':1,'expected_questions':1,'unique_images':1},{'n':1,'expected_images':1}),
    ({'questions':1,'unique_images':1},{'n':1,'questions':1,'unique_images':1,'expected_images':1})])
def test_historical_text_full_lookup_is_not_the_selected_image_denominator(evaluation,lock):
    rows=[{'question_id':'q','image_id':'selected'}]
    assert validate_historical_cohort_scope(rows,rows,{'selected':{},'not_in_dev':{}},
        {'questions':1,'unique_images':1},evaluation,lock,'text')=={'selected'}


@pytest.mark.parametrize('bad',['image','missing_lookup','declared_count'])
def test_historical_selected_cohort_coverage_remains_exact(bad):
    rows=[{'question_id':'q','image_id':'selected'}];refs=deepcopy(rows);lookup={'selected':{}}
    evaluation={'questions':1,'unique_images':1};lock={'questions':1,'unique_images':1}
    if bad=='image':rows[0]['image_id']='wrong'
    elif bad=='missing_lookup':lookup={}
    else:evaluation['expected_questions']=2
    with pytest.raises(ValueError):validate_historical_cohort_scope(rows,refs,lookup,{'questions':1,'unique_images':1},evaluation,lock,'object')


def test_only_legacy_pope_can_derive_missing_percent_and_only_in_memory():
    original={'questions':9000,'metrics':{'overall':{'accuracy':.9,'n':9000}}};before=deepcopy(original)
    view,proof=historical_evaluation_view(original,'pope')
    assert view['score_percent']==90 and original==before and proof['in_memory_only']
    with pytest.raises(ValueError):historical_evaluation_view(original,'text')


def test_category_canonicalization_is_only_empty_text_metadata():
    rows=[{'question_id':1,'category':'','score':.6,'input_ids':[1]}]
    cleaned,proof=canonical_rows(rows,'text')
    assert cleaned==[{'question_id':1,'score':.6,'input_ids':[1]}] and rows[0]['category']==''
    assert proof['noncategory_fields_sha256_before']==proof['noncategory_fields_sha256_after']
    assert canonical_rows(rows,'object')[0] is rows
    with pytest.raises(ValueError):canonical_rows([{'category':'not_empty'}],'text')


@pytest.mark.parametrize('condition',['teacher_kl32','teacher_ce32','gt_ce32','initial','object_kl'])
def test_historical_conditions_cannot_be_trained_or_generated(condition):
    with pytest.raises(ValueError):validate_condition(condition,42,'text')


def test_roles_allow_only_new_candidate_and_only_fresh_replication_rollouts():
    assert role('train_eval',CANDIDATE,42)=='train_eval_merger_kl32_seed42'
    assert role('teacher','initial',0,1)=='teacher_seed0_shard1'
    with pytest.raises(ValueError):role('teacher','initial',42,0)
    with pytest.raises(ValueError):role('preflight',CANDIDATE,42)


def test_worker_cli_has_integer_seed_no_standalone_training_pref_and_strict_confirmation_prior(tmp_path):
    command=train_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,42)
    assert command[command.index('--seed')+1]=='42' and '--preflight-only' not in command
    with pytest.raises(ValueError):eval_arguments(tmp_path/'config.json',tmp_path,CANDIDATE,42,'external',tmp_path/'adapter',tmp_path/'admission')


def test_incomplete_parent_proof_is_rejected_before_payload_read():
    with pytest.raises(ValueError):verify_parent_proof({'parent_output_root':'/opt/data/private/synthetic_parent',
        'parent_config_file':'/opt/data/private/synthetic_parent_config.json','parent_proof_files_sha256':{}})


def valid_phase_fixture(tmp_path,monkeypatch,seeds,gate):
    from vlm_visual_merger_kl import contracts
    from vlm_probe.evaluate import sha_file
    config=tmp_path/'config.json';config.write_text('{}');source={}
    for kind in ('predictions','evaluation','input_lock'):
        file=tmp_path/(kind+'.json');file.write_text('{}')
        source[kind+'_file']=str(file);source[kind+'_sha256']=sha_file(file)
    value={'schema_version':'rarevlm.visual_merger_kl.screen.v1','status':'complete','candidate':CANDIDATE,
        'seeds':seeds,'config_sha256':sha_file(config),'source_sha256':{},'gate':{'passed':gate},
        'eligible_candidates':[CANDIDATE] if gate else [],'source_data':{'candidate_42_text':source}}
    file=tmp_path/'analysis.json';file.write_text(json.dumps(value));monkeypatch.setattr(contracts,'source_hashes',lambda:{})
    return file,config,value


@pytest.mark.parametrize('seeds,gate',[([42],True),([0,42,200],False)])
def test_confirmation_cannot_accept_pilot_or_failed_three_seed_gate(tmp_path,monkeypatch,seeds,gate):
    file,config,_=valid_phase_fixture(tmp_path,monkeypatch,seeds,gate)
    with pytest.raises(ValueError):phase_admission(file,config,'screen')


def test_same_complete_real_schema_and_source_bindings_admit_passing_three_seed_phase(tmp_path,monkeypatch):
    file,config,value=valid_phase_fixture(tmp_path,monkeypatch,[0,42,200],True)
    assert phase_admission(file,config,'screen')==value


def test_actual_configuration_preserves_unique_candidate_and_whole_success_scope():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_visual_merger_kl_v1.json').read_bytes())
    validate_registry(config)
    assert config['schema_version']==SCHEMA and config['all_seeds']==[0,42,200]
    assert config['experiment']=='vlm_visual_merger_kl_v1' and CANDIDATE=='merger_kl32'
    assert config['output_root'].endswith('/vlm_visual_merger_kl_v1')
    assert config['study_protocol'].endswith('/VLM_VISUAL_MERGER_KL_V1_PROTOCOL.md')
    assert config['visual_adaptation']['new_trainable_elements']==1843200
    assert config['visual_adaptation']['total_adapter_elements']==3686400
    from vlm_visual_merger_kl.model import architecture
    assert architecture()==config['visual_adaptation'] | {'scaling':2.,'new_initial_B_zero':True,
        'base_name_mapping_scope':'only visual.merger.mlp.0/2.base_layer to original paths'}
    altered=deepcopy(config);altered['control_reference_only']=False
    with pytest.raises(ValueError):validate_registry(altered)


def test_reference_reader_cannot_turn_shared_initial_into_new_seed_or_confirm_ce():
    validate_historical_request('initial',42,'pope')
    with pytest.raises(ValueError):validate_historical_request('initial',0,'object')
    with pytest.raises(ValueError):validate_historical_request('teacher_kl32',42,'external')


def test_replication_worker_cannot_skip_negative_pilot(tmp_path,monkeypatch):
    from vlm_visual_merger_kl import analysis
    from vlm_probe.evaluate import sha_file
    config=tmp_path/'config.json';config.write_text('{}')
    file=tmp_path/'analysis/pilot_seed42/screen/ANALYSIS.json';file.parent.mkdir(parents=True)
    file.write_text(json.dumps({'schema_version':'rarevlm.visual_merger_kl.screen.v1','status':'complete',
        'candidate':CANDIDATE,'seeds':[42],'eligible_candidates':[],'gate':{'passed':False},
        'config_sha256':sha_file(config),'source_sha256':{}}))
    monkeypatch.setattr(analysis,'source_hashes',lambda:{})
    with pytest.raises(ValueError):worker_prior(config,tmp_path,'train_eval',0)


def run_controller_scenario(tmp_path,monkeypatch,pilot_passed):
    from vlm_visual_merger_kl import study,inputs
    root=tmp_path/'output';root.mkdir();config_file=tmp_path/'config.json'
    config={'output_root':str(root)};config_file.write_text(json.dumps(config));events=[]
    monkeypatch.setattr(study,'prepare',lambda file:(config,root,{'mock_input_lock':True}))
    monkeypatch.setattr(study,'verify_study',lambda *args,**kwargs:{})
    monkeypatch.setattr(study,'prepare_generation_inputs',lambda cfg,seed:events.append(('prepare_prompt',seed)))
    monkeypatch.setattr(study,'finalize_teacher_replies',lambda cfg,seed:events.append(('merge_teacher',seed)))
    monkeypatch.setattr(inputs,'prepare_training_inputs',lambda cfg,seed,condition:events.append(('prepare_train_inputs',seed,condition)))
    monkeypatch.setattr(inputs,'verify_prepared_training_inputs',lambda *args:{})
    def queue(root,name,jobs,identity):
        items=[]
        for job in jobs:
            cmd=job['command'];get=lambda key:cmd[cmd.index(key)+1]
            items.append({'operation':get('--operation'),'condition':get('--condition'),'seed':int(get('--seed')),
                'shard':int(get('--shard')) if '--shard' in cmd else None})
        events.append(('queue',name,items))
        return [{'outcome':'passed','role':job['key']} for job in jobs]
    monkeypatch.setattr(study,'queue_stage',queue)
    def analyze(file,output,seeds,phase,**kwargs):
        events.append(('analyze',list(seeds),phase))
        return {'status':'complete','gate':{'passed':pilot_passed if seeds==[42] else True}},root/'analysis'/('_'.join(map(str,seeds))+'_'+phase+'.json')
    monkeypatch.setattr(study,'analyze',analyze)
    monkeypatch.setattr(study,'finish',lambda output,status,phase,analyses,receipts,started:
        {'status':status,'goal_complete':False,'last_phase':phase,'analyses':list(analyses)})
    return study.run_study(config_file),events


def test_actual_controller_negative_pilot_never_dispatches_replication_or_confirmation(tmp_path,monkeypatch):
    result,events=run_controller_scenario(tmp_path,monkeypatch,False)
    assert result['status']=='completed_negative' and result['goal_complete'] is False
    queues=[event for event in events if event[0]=='queue']
    assert [event[1] for event in queues]==['pilot42']
    assert queues[0][2]==[{'operation':'train_eval','condition':CANDIDATE,'seed':42,'shard':None}]
    assert not any(event[0]=='prepare_prompt' for event in events)
    assert [event for event in events if event[0]=='analyze']==[('analyze',[42],'screen')]


def test_actual_controller_success_dispatches_full_registered_chain_and_requires_root_acceptance(tmp_path,monkeypatch):
    result,events=run_controller_scenario(tmp_path,monkeypatch,True)
    assert result['status']=='confirmation_passed_pending_independent_review' and result['goal_complete'] is False
    queues=[event for event in events if event[0]=='queue'];by_name={event[1]:event[2] for event in queues}
    assert [event[1] for event in queues]==['pilot42','teacher_generation_seeds0_200','replication','external_confirmation','pope_confirmation']
    assert [(x['seed'],x['shard']) for x in by_name['teacher_generation_seeds0_200']]==[(0,0),(0,1),(200,0),(200,1)]
    assert all(x['condition']=='initial' and x['operation']=='teacher' for x in by_name['teacher_generation_seeds0_200'])
    assert [(x['condition'],x['seed']) for x in by_name['replication']]==[(CANDIDATE,0),(CANDIDATE,200)]
    for name,operation in [('external_confirmation','external'),('pope_confirmation','pope')]:
        assert [(x['operation'],x['condition'],x['seed']) for x in by_name[name]]==[(operation,CANDIDATE,s) for s in (0,42,200)]
    assert [event for event in events if event[0]=='analyze']==[
        ('analyze',[42],'screen'),('analyze',[0,42,200],'screen'),('analyze',[0,42,200],'external'),('analyze',[0,42,200],'confirm')]
    assert [event for event in events if event[0]=='prepare_prompt']==[('prepare_prompt',0),('prepare_prompt',200)]


def test_completed_analysis_resume_keeps_original_bytes_and_refuses_changed_identity(tmp_path,monkeypatch):
    from vlm_visual_merger_kl import analysis
    destination=tmp_path/'ANALYSIS.json';data={}
    for kind in ('predictions','evaluation','input_lock'):
        file=tmp_path/(kind+'.json');file.write_text('{}');data[kind+'_file']=str(file)
        from vlm_probe.evaluate import sha_file
        data[kind+'_sha256']=sha_file(file)
    current={'schema_version':'rarevlm.visual_merger_kl.screen.v1','stage':'pilot','config_sha256':'config',
        'source_sha256':{},'seeds':[42],'candidate':CANDIDATE,'exploratory_control':'teacher_kl32','settings':{},
        'source_data':{'initial_42_object':data},'canonicalization':{},'control_can_be_selected':False,'control_reference_only':True}
    gate={'passed':True,'checks':{'mock_existing_gate':True}}
    saved={**current,'status':'complete','updated_at':'original','contrasts':{'object':[],'text':[]},'class_balanced':{},'gate':gate}
    destination.write_text(json.dumps(saved));before=destination.read_bytes()
    monkeypatch.setattr(analysis,'screen_gate',lambda *args,**kwargs:gate)
    assert analysis.reuse_complete_analysis(destination,current,'screen',None,{})==saved
    assert destination.read_bytes()==before
    changed=deepcopy(current);changed['config_sha256']='changed'
    with pytest.raises(ValueError):analysis.reuse_complete_analysis(destination,changed,'screen',None,{})


def native_dtypes(composite=False):
    return {'parameter_tensors_by_dtype':{'bfloat16':972 if composite else 968},
        'parameter_stored_elements_by_dtype':{'bfloat16':3758309376 if composite else 3756466176},
        'lora_tensors_by_dtype':{'bfloat16':148 if composite else 144},
        'lora_elements_by_dtype':{'bfloat16':3686400 if composite else 1843200},
        'buffer_tensors_by_dtype':{'float32':38},'params4bit_tensors':0,'trainable_parameters':0,
        'note':'NF4 uint8 stored elements are packed and are not an original parameter count'}


@pytest.mark.parametrize('changed',['count','dtype','base_count','buffer','trainable','extra_field'])
def test_only_exact_declared_four_tensor_dtype_increment_is_accepted(changed):
    from vlm_visual_merger_kl.contracts import validate_evaluation_dtypes
    before,after=native_dtypes(),native_dtypes(True)
    validate_evaluation_dtypes(before,composite=False);validate_evaluation_dtypes(after,composite=True)
    if changed=='count':after['lora_tensors_by_dtype']['bfloat16']=147
    elif changed=='dtype':after['lora_tensors_by_dtype']={'float32':148}
    elif changed=='base_count':after['parameter_stored_elements_by_dtype']['bfloat16']+=1
    elif changed=='buffer':after['buffer_tensors_by_dtype']['float32']+=1
    elif changed=='trainable':after['trainable_parameters']=1843200
    else:after['unregistered']=True
    with pytest.raises(ValueError):validate_evaluation_dtypes(after,composite=True)


def test_actual_pairing_accepts_declared_inventory_but_rejects_any_actual_input_change():
    from vlm_visual_merger_kl.contracts import assert_pair
    from vlm_object_kl_eval.analysis import SCIENCE_FIELDS
    from vlm_posttrain_pope.analysis import AUDIT_FIELDS
    row={key:'same' for key in ('question_id','image_id',*AUDIT_FIELDS)};row.update(label='yes',category='car')
    before={'condition':'initial','lock':{key:'same' for key in SCIENCE_FIELDS},'rows':[row]}
    after=deepcopy(before);after['condition']=CANDIDATE
    before['lock']['actual_parameter_dtypes']=native_dtypes();after['lock']['actual_parameter_dtypes']=native_dtypes(True)
    assert_pair(before,after,'object')
    after['rows'][0]['pixel_values']='different'
    with pytest.raises(ValueError,match='actual input'):assert_pair(before,after,'object')


def test_actual_model_binding_separates_fresh_verification_seconds_and_reuses_input_identity(tmp_path,monkeypatch):
    from vlm_visual_merger_kl import contracts,inputs,model,initialization
    from vlm_probe.evaluate import sha_file,lock_file
    config={'output_root':str(tmp_path),'initial_adapter':str(tmp_path/'initial'),'model_root':str(tmp_path/'model')}
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    place=tmp_path/'runs'/CANDIDATE/'seed42/train';adapter=place/'adapter';adapter.mkdir(parents=True)
    adapter_config={'peft_type':'LORA','task_type':'CAUSAL_LM','r':8,'lora_alpha':16,'lora_dropout':0.,'bias':'none',
        'target_modules':[f'model.layers.{i}.self_attn.{projection}_proj' for i in range(36) for projection in ('q','v')]
            +['visual.merger.mlp.0','visual.merger.mlp.2'],
        'rank_pattern':{'visual.merger.mlp.0':96,'visual.merger.mlp.2':120},
        'alpha_pattern':{'visual.merger.mlp.0':192,'visual.merger.mlp.2':240},'modules_to_save':None,
        'use_dora':False,'use_rslora':False,'fan_in_fan_out':False,'lora_bias':False,'init_lora_weights':True,
        'base_model_name_or_path':str(tmp_path/'model')}
    (adapter/'adapter_config.json').write_text(json.dumps(adapter_config));(adapter/'adapter_model.safetensors').write_bytes(b'not actual weights')
    pref=place/'PREFLIGHT.json';pref.write_text(json.dumps({'status':'passed'}))
    (place/'TRAIN_INPUT_LOCK.json').write_text('{}')
    result={'schema_version':SCHEMA,'status':'completed','condition':CANDIDATE,'seed':42,'completed_updates':181,
        'actual_object_exposures':2888,'actual_replay_exposures':5792,'trainable_parameters':1843200,
        'replay_hard_ce_backward':False,'replay_objective':'forward_kl_full_vocabulary','reference_parameters_unchanged':True,
        'formal_reference_forward_questions':5792,'preflight_file_sha256':sha_file(pref),'preflight':{'status':'passed'},
        'adapter_files_sha256':{file.name:sha_file(file) for file in adapter.iterdir()}}
    (place/'TRAIN_RESULT.json').write_text(json.dumps(result))
    monkeypatch.setattr(inputs,'verify_training_result',lambda *a,**k:{'all_actual_inputs_verified':True})
    monkeypatch.setattr(initialization,'common_initializer_manifest',lambda *a,**k:{'identity':{'visual4_initial':'fixed'}})
    identity={'evaluation_identity':{'native148':'expected','frozen_language':'canonical'},'trainable_elements':1843200}
    times=iter((.1,.8,.3))
    monkeypatch.setattr(model,'verify_training_model_metadata',lambda *a:{'identity':deepcopy(identity),'telemetry':{'verify_CPU_seconds':next(times)}})
    _,first=contracts.adapter_binding(config,CANDIDATE,42,adapter,tmp_path/'model',config_file=cfg)
    cache=tmp_path/'INPUT_LOCK.json';lock_file(cache,{'training_binding':first});raw=cache.read_bytes()
    _,second=contracts.adapter_binding(config,CANDIDATE,42,adapter,tmp_path/'model',config_file=cfg)
    assert first==second and 'verify_CPU_seconds' not in json.dumps(second)
    lock_file(cache,{'training_binding':second});assert cache.read_bytes()==raw
    identity['evaluation_identity']['native148']='different'
    _,changed=contracts.adapter_binding(config,CANDIDATE,42,adapter,tmp_path/'model',config_file=cfg)
    with pytest.raises(ValueError):lock_file(cache,{'training_binding':changed})
    assert cache.read_bytes()==raw


@pytest.mark.parametrize('bad',[{'identity':{'verify_CPU_seconds':.1},'telemetry':{}},
    {'identity':{'native':'sha'},'telemetry':{'verify_CPU_seconds':float('nan')}},
    {'identity':{'native':'sha'},'telemetry':{},'undeclared_scientific':'sha'}])
def test_dynamic_fields_or_hidden_scientific_domains_in_model_verification_are_rejected(bad):
    from vlm_visual_merger_kl.contracts import verified_model_identity
    with pytest.raises(ValueError):verified_model_identity(bad)
