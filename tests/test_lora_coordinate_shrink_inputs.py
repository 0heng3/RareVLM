"""New producer admission/plan/reference contracts; no real weights or encodes."""
import copy
import json
import sys
import types
from pathlib import Path
import pytest

from vlm_lora_coordinate_shrink import CANDIDATE
from vlm_lora_coordinate_shrink import data,inputs,lock,rollout,train
from vlm_deep.common import content_sha


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_lora_coordinate_shrink_v1.json').read_text())


def test_replication_admission_uses_only_new_own_passing_pilot(tmp_path,monkeypatch):
    cfg=config();cfg['output_root']=str(tmp_path)
    cfg_file=tmp_path/'config.json';cfg_file.write_text(json.dumps(cfg))
    (tmp_path/'STUDY_INPUT_LOCK.json').write_text(json.dumps({'config_path':str(cfg_file)}))
    called=[];module=types.ModuleType('vlm_lora_coordinate_shrink.analysis')
    def passing(file,current):
        called.append((file,current));return {'status':'new_half_passed'}
    module.passing_pilot=passing;monkeypatch.setitem(sys.modules,module.__name__,module)
    result=data.require_replication_admission(cfg)
    assert result['status']=='new_half_passed'
    assert called==[(tmp_path/'analysis/pilot_seed42/screen/ANALYSIS.json',cfg_file)]
    def rejected(*args):raise ValueError('new_half_not_passed')
    module.passing_pilot=rejected
    with pytest.raises(ValueError,match='new_half_not_passed'):data.require_replication_admission(cfg)


@pytest.mark.parametrize('seed',[42,1,True])
def test_seed42_or_unknown_cannot_start_new_prefix_or_producer_inputs(seed,monkeypatch):
    cfg=config();calls=[]
    monkeypatch.setattr(data,'require_replication_admission',lambda *a:calls.append('admission'))
    for call in [lambda:data.build_plan(cfg,seed),lambda:data.load_teacher_replies(cfg,seed),
            lambda:inputs.prepare_training_inputs(cfg,seed,CANDIDATE),lambda:inputs.verify_prepared_training_inputs(cfg,seed,CANDIDATE)]:
        with pytest.raises(ValueError):call()
    assert calls==[]
    with pytest.raises(ValueError):data.prepare_generation_inputs(cfg,seed)
    with pytest.raises(ValueError):data.finalize_teacher_replies(cfg,seed)


@pytest.mark.parametrize('seed',[0,200])
def test_full181_fresh_plan_matches_frozen_stage32_schedule(seed,monkeypatch):
    cfg=config()
    objects=[{'question_id':f'o{i}'} for i in range(2888)]
    replays=[{'question_id':f'r{i}'} for i in range(32985)]
    calls=[];monkeypatch.setattr(data,'require_replication_admission',lambda *a:calls.append('own_pass'))
    monkeypatch.setattr(data,'load_rows',lambda *a:(objects,replays))
    plan=data.build_plan(cfg,seed);blocks=data.build_schedule(objects,replays,seed,1,16,32)
    assert plan['blocks']==blocks and len(blocks)==181
    assert [len(b['object_indices']) for b in blocks]==[16]*180+[8]
    assert [len(b['replay_indices']) for b in blocks]==[32]*181
    ids=[q for block in plan['schedule']['replay_question_ids_by_update'] for q in block]
    assert len(ids)==len(set(ids))==5792
    for source in ('object','replay'):
        q=plan['schedule'][source+'_question_ids_by_update']
        assert plan['schedule'][source+'_microbatches_by_update']==[[v[i:i+4] for i in range(0,len(v),4)] for v in q]
    assert plan['artifact_role']=='unshrunk_producer' and plan['coordinate_shrink_applied'] is False
    assert plan['condition']==CANDIDATE and plan==data.build_plan(cfg,seed)
    assert calls==['own_pass','own_pass']


def producer_fixture(tmp_path,monkeypatch):
    cfg=config();n=tmp_path/'accepted';cfg_file=tmp_path/'producer_config.json';initial=tmp_path/'initial'
    cfg.update(producer_reference_root=str(n),producer_reference_config=str(cfg_file),initial_adapter=str(initial),output_root=str(tmp_path/'new'))
    locations=[cfg_file,*[n/name for name in ('STUDY_INPUT_LOCK.json','COMPLETION.json',
        'analysis/pilot_seed42/screen/ANALYSIS.json','independent_review/PILOT_PREDICTION_REVIEW.json',
        'independent_review/TRAINING_COMPLETION_REVIEW.json','independent_review/ROOT_PILOT_ACCEPTANCE.json',
        'runs/teacher_kl32/seed42/train/TRAIN_RESULT.json')]]
    mapping={str(p):f'{i+1:064x}' for i,p in enumerate(locations)};cfg['producer_reference_files_sha256']=mapping
    old_config={'frozen':'numeric'};result_file=locations[-1]
    study={'schema_version':'rarevlm.generated_prefix_kl.study_input.v1','config_path':str(cfg_file),
        'config_file_sha256':mapping[str(cfg_file)],'config_content_sha256':content_sha(old_config),
        'parent_proof_files_sha256':cfg['parent_proof_files_sha256'],'source_sha256':{str(i):'a'*64 for i in range(84)}}
    result={'schema_version':'rarevlm.generated_prefix_kl.train.v1','status':'completed','condition':'teacher_kl32','seed':42,
        'completed_updates':181,'actual_object_exposures':2888,'actual_replay_exposures':5792,'actual_questions_seen':8680,
        'trainable_parameters':1843200,'initial_adapter_tensor_sha256':lock.INITIAL,'final_lora_tensor_sha256':lock.PRODUCER_FINAL,
        'frozen_parameter_sha256_before':lock.FROZEN,'frozen_parameter_sha256_after':lock.FROZEN,
        'frozen_parameters_unchanged':True,'reference_parameters_unchanged':True,'formal_reference_forward_questions':5792,
        'replay_hard_ce_backward':False,'replay_objective':'forward_kl_full_vocabulary','initial_adapter_dir':str(initial),
        'adapter_files_sha256':{'adapter_model.safetensors':'a'*64,'adapter_config.json':'b'*64},
        'initial_adapter_files_sha256':{'adapter_model.safetensors':'c'*64},'lineage_sha256':'d'*64}
    trained={'status':'passed_training_engineering_only','formal_updates':181,'actual_Adam_step_values':[181],
        'master_tensors':144,'master_elements':1843200,'study_input_lock_sha256':mapping[str(n/'STUDY_INPUT_LOCK.json')],
        'final_master_sha256':lock.PRODUCER_FINAL,'initial_canonical_master_sha256':lock.INITIAL,
        'saved_frozen_base_vision_before_after_sha256':lock.FROZEN,
        'evidence_immutable_files_sha256':{str(result_file):mapping[str(result_file)]}}
    docs={str(cfg_file):old_config,str(n/'STUDY_INPUT_LOCK.json'):study,
        str(n/'COMPLETION.json'):{'status':'completed_negative','goal_complete':False},
        str(n/'analysis/pilot_seed42/screen/ANALYSIS.json'):{'status':'complete','candidate':'teacher_kl32','seeds':[42],'gate':{'passed':False},'eligible_candidates':[]},
        str(n/'independent_review/PILOT_PREDICTION_REVIEW.json'):{'status':'passed_independent_prediction_review','new_score_mismatches':0,
            'independent_new_questions_rescored':1959,'official_analysis_sha256':mapping[str(n/'analysis/pilot_seed42/screen/ANALYSIS.json')],
            'study_input_lock_sha256':mapping[str(n/'STUDY_INPUT_LOCK.json')]},
        str(n/'independent_review/TRAINING_COMPLETION_REVIEW.json'):trained,
        str(n/'independent_review/ROOT_PILOT_ACCEPTANCE.json'):{'status':'accepted_complete_negative_pilot_only','goal_complete':False,
            'independent_score_mismatches':0,'files_sha256':dict(mapping)},str(result_file):result}
    checked=[];monkeypatch.setattr(lock,'verify_files',lambda m:checked.extend(m))
    monkeypatch.setattr(lock,'read_json',lambda p:docs[str(p)])
    return cfg,docs,checked,result_file


def test_metadata_reference_never_loads_weights_and_does_not_make_negative_pass(tmp_path,monkeypatch):
    cfg,docs,checked,result_file=producer_fixture(tmp_path,monkeypatch)
    reference=lock.verify_producer_reference(cfg)
    assert reference['seed']==42 and reference['expected_final_master_sha256']==lock.PRODUCER_FINAL
    assert reference['artifact_role']=='unshrunk_producer' and reference['historical_training_updates']==181
    assert reference['new_training_optimizer_updates']==0 and reference['producer_was_shrunk'] is False
    assert not any(name.endswith(('.safetensors','.pt','.bin')) for name in checked)
    assert not Path(cfg['output_root']).exists()  # A verifier cannot create a fake training run.
    n=Path(cfg['producer_reference_root'])
    docs[str(n/'analysis/pilot_seed42/screen/ANALYSIS.json')]['gate']['passed']=True
    with pytest.raises(ValueError,match='accepted negative'):lock.verify_producer_reference(cfg)


def test_explicit_prepare_checks_only_direct_small_adapters_and_initial(tmp_path,monkeypatch):
    cfg,docs,checked,result_file=producer_fixture(tmp_path,monkeypatch)
    lock.verify_producer_reference(cfg,check_weights=True)
    tensor_paths=[p for p in checked if p.endswith('.safetensors')]
    assert tensor_paths==[str(Path(cfg['producer_reference_root'])/'runs/teacher_kl32/seed42/train/adapter/adapter_model.safetensors'),
        str(Path(cfg['initial_adapter'])/'adapter_model.safetensors')]
    assert not any(p.endswith('.pt') for p in checked)
    docs[str(result_file)]['final_lora_tensor_sha256']='wrong'
    with pytest.raises(ValueError,match='Final invariant'):lock.verify_producer_reference(cfg)


def test_historical_producer_href_is_separate_and_never_fake_train_result(tmp_path,monkeypatch):
    cfg=config();cfg['output_root']=str(tmp_path)
    ref={'schema_version':'rarevlm.lora_coordinate_shrink.producer_reference.v1','seed':42}
    monkeypatch.setattr(lock,'verify_producer_reference',lambda *a,**k:ref)
    result=lock.prepare_historical_producer_reference(cfg)
    file=tmp_path/'producer_runs/lora_half32/seed42/HISTORICAL_PRODUCER_REFERENCE.json'
    assert result==ref and json.loads(file.read_text())==ref
    assert not list(tmp_path.rglob('TRAIN_RESULT.json')) and not (file.parent/'train').exists()


def test_result_to_complete_cpu_recovery_rejects_uuid_before_model_loading(tmp_path,monkeypatch):
    cfg=config();cfg['output_root']=str(tmp_path);cfg['teacher_reply_root']=str(tmp_path/'replies')
    cfg_file=tmp_path/'config.json';cfg_file.write_text(json.dumps(cfg))
    admission_file=tmp_path/'admission.json';admission_file.write_text('{}')
    manifest={'plan_file_sha256':'a'*64,'prompt_inputs_file_sha256':'b'*64}
    directory=Path(cfg['teacher_reply_root'])/'seed0/shard0';directory.mkdir(parents=True)
    raw_result=b'{"original_result":"must_not_be_overwritten"}\n';(directory/'RESULT.json').write_bytes(raw_result)
    binding=data.sealed({'producer_uuid':'registeredGPU','config_file_sha256':data.sha_file(cfg_file),
        'admission_file_sha256':data.sha_file(admission_file),**manifest},'input_lock_sha256')
    (directory/'INPUT_LOCK.json').write_text(json.dumps(binding))
    monkeypatch.setattr(data,'require_replication_admission',lambda *a:True)
    monkeypatch.setattr(rollout,'load_prompt_inputs',lambda *a:(manifest,{},[]))
    monkeypatch.setattr(rollout,'visible_gpu_binding',lambda:{'uuid':'registeredGPU'})
    def forbidden(*a,**k):raise AssertionError('CUDA/model must not be reached in completedRESULT recovery')
    monkeypatch.setattr(rollout.torch.cuda,'is_available',forbidden);monkeypatch.setattr(rollout,'load_path',forbidden)
    module=types.ModuleType('vlm_lora_coordinate_shrink.contracts')
    module.validate_gpu_admission=lambda *a,**k:{}
    monkeypatch.setitem(sys.modules,module.__name__,module)
    calls=[];monkeypatch.setattr(rollout,'commit_saved_result',lambda c,s,h,**k:calls.append((s,h,k)) or {'done':True})
    assert rollout.run_teacher_shard(cfg_file,0,0,tmp_path,admission_file)=={'done':True}
    assert calls==[(0,0,{'recovery':True})] and (directory/'RESULT.json').read_bytes()==raw_result
    monkeypatch.setattr(rollout,'visible_gpu_binding',lambda:{'uuid':'differentGPU'})
    with pytest.raises(ValueError,match='UUID'):rollout.run_teacher_shard(cfg_file,0,0,tmp_path,admission_file)
    assert len(calls)==1
