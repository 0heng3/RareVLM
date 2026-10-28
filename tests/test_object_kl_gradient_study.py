"""CPU-only contracts for fixed training-gradient registration and queue wiring."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import read_json, sha_file, write_json

spec=importlib.util.spec_from_file_location('gradient_launcher',PROJECT_ROOT/'tools/run_object_kl_gradient_study.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config_for(tmp_path):
    root=tmp_path/'gradient'
    value={'schema_version':runner.SCHEMA,'source_config':str(tmp_path/'source.json'),
        'source_study_root':str(tmp_path/'source'),'output_root':str(root),
        'study_protocol':str(tmp_path/'protocol.md'),'cpu_contracts':str(root/'CPU_CONTRACTS.json'),
        'checkpoint_paths':{str(step):str(tmp_path/'source'/f'checkpoint_{step:06d}_1') for step in runner.STEPS},
        'groups':deepcopy(runner.GROUPS),'parity_kl_abs_tolerance':1e-5,'memory_cap_gib':65,
        'training_microbatch':4,'optimizer_updates':0,'generation_questions':0,
        'expected_new_training_qa_forward_backward':96,'expected_main_qa_forward_backward':72,
        'expected_initial_parity_qa_forward_backward':24,'fixed_teacher_producer_gpu_uuid':'GPU-producer',
        'goal_complete':False}
    return value,root


def test_fixed_points_zero_based_blocks_and_whole_study96_count(tmp_path):
    config,_=config_for(tmp_path)
    runner.validate_config(config)
    config['groups']['middle'][0]['block_index']=99
    with pytest.raises(ValueError,match='zero-based'):
        runner.validate_config(config)


@pytest.mark.parametrize('field,value',[('memory_cap_gib',64),('optimizer_updates',1),
    ('generation_questions',1),('expected_new_training_qa_forward_backward',192),('goal_complete',True)])
def test_no_runtime_fallback_updates_generation_or_goal_success(tmp_path,field,value):
    config,_=config_for(tmp_path);config[field]=value
    with pytest.raises(ValueError):runner.validate_config(config)


def test_all_persistent_paths_absolute_and_new_output_independent(tmp_path):
    config,_=config_for(tmp_path);config['source_config']='relative.json'
    with pytest.raises(ValueError,match='absolute'):runner.validate_config(config)
    config,_=config_for(tmp_path);config['output_root']=str(Path(config['source_study_root'])/'new')
    with pytest.raises(ValueError,match='independent'):runner.validate_config(config)


def test_actual_checkpoint_master_lineage_schedule_and_boundary(tmp_path):
    from vlm_object_train.train import tensor_hash
    directory=tmp_path/'checkpoint_000025_1';(directory/'adapter').mkdir(parents=True)
    (directory/'adapter/adapter_model.safetensors').write_bytes(b'locked adapter')
    tensors={f'p{index}':torch.zeros(12800,dtype=torch.float32) for index in range(144)}
    state={'lineage_sha256':'lineage','schedule_sha256':'schedule','completed_updates':25,'next_update_index':25,
        'accumulated_questions':0,'trainable_parameters':tensors,'trainable_tensor_sha256':tensor_hash(tensors.items()),
        'optimizer':{'state':{0:{'step':25}}},'rng':{'torch':'retained'}}
    torch.save(state,directory/'state.pt')
    train={'trainable_names':{name:t.numel() for name,t in tensors.items()}}
    result=runner.checkpoint_contract(directory,25,'lineage','schedule',train,181)
    assert result['state_sha256']==sha_file(directory/'state.pt') and result['optimizer_boundary']
    state['accumulated_questions']=1;torch.save(state,directory/'state.pt')
    with pytest.raises(ValueError,match='optimizer boundary'):
        runner.checkpoint_contract(directory,25,'lineage','schedule',train,181)


def test_negative_screen_binds_exact_source_training_result_bytes(tmp_path):
    file=tmp_path/'TRAIN_RESULT.json';write_json(file,{'original':'completed training'})
    screen={'source_data':{f'object_kl_42_{task}':{'training_result_sha256':sha_file(file)} for task in ('object','text')}}
    runner.assert_training_binding(screen,file)
    write_json(file,{'modified':'training result'})
    with pytest.raises(ValueError,match='TRAIN_RESULT bytes differ'):runner.assert_training_binding(screen,file)


def test_cpu_receipt_binds_tests_complete_log_and_new_source(tmp_path):
    config,root=config_for(tmp_path);root.mkdir()
    log=root/'cpu.log';log.write_text('all relevant CPU contracts passed\n')
    test=PROJECT_ROOT/'tests/test_object_kl_gradient_study.py'
    source=Path(runner.__file__).resolve()
    receipt={'status':'passed','files_sha256':{str(log):sha_file(log),str(test):sha_file(test)},
             'scientific_source_sha256':{str(source):sha_file(source)},'new_model_forwards':0}
    write_json(Path(config['cpu_contracts']),receipt)
    assert runner.cpu_receipt(config,{str(source):sha_file(source)})[1]['status']=='passed'
    receipt['scientific_source_sha256']={};write_json(Path(config['cpu_contracts']),receipt)
    with pytest.raises(ValueError,match='current new gradient source'):
        runner.cpu_receipt(config,{str(source):sha_file(source)})


def test_prepare_only_includes_cpu_artifacts_and_never_calls_queue(tmp_path,monkeypatch):
    config,root=config_for(tmp_path)
    config_file=tmp_path/'gradient.json';write_json(config_file,config)
    protocol=Path(config['study_protocol']);protocol.write_text('registered fixed diagnostic')
    source=tmp_path/'source.py';source.write_text('locked CPU source')
    log=tmp_path/'cpu.log';log.write_text('passed')
    receipt_file=Path(config['cpu_contracts']);write_json(receipt_file,{'status':'passed'})
    receipt={'files_sha256':{str(log):sha_file(log)},'scientific_source_sha256':{str(source):sha_file(source)}}
    monkeypatch.setattr(runner,'source_hashes',lambda:{str(source):sha_file(source)})
    monkeypatch.setattr(runner,'cpu_receipt',lambda *_:(receipt_file,receipt))
    monkeypatch.setattr(runner,'source_contract',lambda _:({str(source):sha_file(source)},{'source_status':'completed_negative'}))
    monkeypatch.setattr(runner,'run_queue',lambda *_:pytest.fail('prepare cannot use GPU queue'))
    assert runner.run(config_file,prepare_only=True)['status']=='prepared'
    lock=read_json(root/'STUDY_INPUT_LOCK.json')
    assert lock['schema_version']==runner.LOCK_SCHEMA and lock['files_sha256'][str(log)]==sha_file(log)
    assert lock['optimizer_updates']==lock['generation_questions']==0 and lock['goal_complete'] is False


def test_group_jobs_use_exact_probe_cli_and_two_dynamic_queue_roles(tmp_path):
    config,root=config_for(tmp_path);file=tmp_path/'config.json'
    tasks=runner.jobs(file,root)
    assert [task['key'] for task in tasks]==['early_late','middle']
    for task in tasks:
        command=task['command']
        assert command[1:3]==['-m','vlm_object_kl_gradient.probe']
        assert command[command.index('--group')+1]==task['key'] and '--study-lock' in command
        assert '--seed42' not in command and '--seed' not in command
        assert Path(task['completion_marker'])==root/'groups'/task['key']/'COMPLETE.json'


def group_receipt(root,group,config_file,lock_file,config):
    directory=root/'groups'/group
    write_json(directory/'INPUT_LOCK.json',{'consumer_gpu_uuid':'GPU-consumer'})
    write_json(directory/'RESULT.json',{'status':'completed'})
    tensor=directory/'GRADIENTS.pt';tensor.write_bytes(b'full gradient values')
    files={str(f):sha_file(f) for f in (directory/'INPUT_LOCK.json',directory/'RESULT.json',tensor)}
    value={'schema_version':'rarevlm.object_kl.gradient.completion.v1','status':'completed','group':group,
        'config_file_sha256':sha_file(config_file),'study_lock_sha256':sha_file(lock_file),
        'producer_gpu_uuid':config['fixed_teacher_producer_gpu_uuid'],'consumer_gpu_uuid':'GPU-consumer',
        'optimizer_updates':0,'generation_questions':0,'artifact_files_sha256':files}
    write_json(directory/'COMPLETE.json',value)
    return value


def test_group_resume_revalidates_exact_gradient_artifacts(tmp_path):
    config,root=config_for(tmp_path);root.mkdir()
    cf=tmp_path/'cfg.json';write_json(cf,config)
    lf=root/'STUDY_INPUT_LOCK.json';write_json(lf,{'registered':True})
    group_receipt(root,'middle',cf,lf,config)
    file=root/'groups/middle/COMPLETE.json'
    assert runner.read_completion(file,cf,lf,config)['consumer_gpu_uuid']=='GPU-consumer'
    (file.parent/'GRADIENTS.pt').write_bytes(b'changed values')
    with pytest.raises(ValueError,match='locked file changed'):runner.read_completion(file,cf,lf,config)


def test_cpu_aggregation_uses_saved_arrays_three_points_and_never_performance(tmp_path,monkeypatch):
    config,root=config_for(tmp_path);root.mkdir()
    cf=tmp_path/'cfg.json';write_json(cf,config)
    lf=root/'STUDY_INPUT_LOCK.json';write_json(lf,{'registered':True})
    calls=[]
    def review(config_file,output,group,study_lock):
        calls.append(group);n=len(runner.GROUPS[group])
        return {'group':group,'points':deepcopy(runner.GROUPS[group]),'optimizer_updates':0,'generation_questions':0,
            'new_training_qa_forward_backward':n*32,'main_qa_forward_backward':n*24,
            'initial_parity_qa_forward_backward':n*8,'artifact_files_sha256':{}}
    for group in runner.GROUPS:group_receipt(root,group,cf,lf,config)
    monkeypatch.setattr(runner,'probe_module',lambda:SimpleNamespace(review_group=review))
    value=runner.aggregate(cf,root,config)
    assert calls==['early_late','middle'] and [p['checkpoint_step'] for p in value['points']]==[25,100,175]
    assert value['new_training_qa_forward_backward']==96 and value['goal_complete'] is False
    assert value['performance_evaluation'] is False and value['cpu_aggregation_new_model_forwards']==0


def test_unprepared_run_cannot_enter_gpu_queue(tmp_path,monkeypatch):
    config,_=config_for(tmp_path);file=tmp_path/'cfg.json';write_json(file,config)
    monkeypatch.setattr(runner,'run_queue',lambda *_:pytest.fail('unprepared GPU work'))
    with pytest.raises(FileNotFoundError):runner.run(file)


def test_queue_error_is_execution_or_engineering_failure_not_negative_accuracy(tmp_path,monkeypatch):
    config,root=config_for(tmp_path);root.mkdir()
    cf=tmp_path/'cfg.json';write_json(cf,config)
    write_json(root/'STUDY_INPUT_LOCK.json',{'registered':True})
    monkeypatch.setattr(runner,'verify_lock',lambda _:(config,root,{'locked':True}))
    def fail(*_):raise RuntimeError('worker CUDA out of memory; existing tasks retained')
    monkeypatch.setattr(runner,'run_queue',fail)
    with pytest.raises(RuntimeError):runner.run(cf)
    status=read_json(root/'STATUS.json')
    assert status['status']=='execution_or_engineering_failure' and status['scientific_failure'] is False
    assert status['performance_evaluation'] is False and status['goal_complete'] is False
    assert not (root/'COMPLETION.json').exists()
