#!/usr/bin/env python
"""Register, queue, and CPU-review a fixed training-gradient diagnostic."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import re
import sys
import time
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, read_json, sha_file, write_json

SCHEMA = 'rarevlm.object_kl.gradient.v1'
LOCK_SCHEMA = 'rarevlm.object_kl.gradient.study.v1'
GROUPS = {'early_late': [{'checkpoint_step': 25, 'block_index': 25},
                         {'checkpoint_step': 175, 'block_index': 175}],
          'middle': [{'checkpoint_step': 100, 'block_index': 100}]}
STEPS = (25, 100, 175)


def probe_module():
    return importlib.import_module('vlm_object_kl_gradient.probe')


def absolute(value):
    file = Path(value)
    if not file.is_absolute() or Path('/opt/data/private') not in file.resolve().parents:
        raise ValueError(f'absolute persistent path required: {value}')
    return file.resolve()


def verify_files(files):
    for name, digest in files.items():
        file = absolute(name)
        if not file.is_file() or sha_file(file) != digest:
            raise ValueError(f'gradient diagnostic locked file changed: {name}')


def validate_config(config):
    if config.get('schema_version') != SCHEMA or config.get('groups') != GROUPS:
        raise ValueError('fixed gradient schema/groups/zero-based blocks differ')
    for key in ('source_config', 'source_study_root', 'output_root', 'study_protocol', 'cpu_contracts'):
        absolute(config[key])
    if (absolute(config['output_root']) == absolute(config['source_study_root'])
        or absolute(config['source_study_root']) in absolute(config['output_root']).parents):
        raise ValueError('new gradient output must be independent of frozen source output')
    if set(config['checkpoint_paths']) != {str(step) for step in STEPS}:
        raise ValueError('exactly three predeclared checkpoint directories required')
    for value in config['checkpoint_paths'].values(): absolute(value)
    fixed = {'parity_kl_abs_tolerance': 1e-5, 'memory_cap_gib': 65, 'training_microbatch': 4,
             'optimizer_updates': 0, 'generation_questions': 0,
             'expected_new_training_qa_forward_backward': 96,
             'expected_main_qa_forward_backward': 72, 'expected_initial_parity_qa_forward_backward': 24}
    if any(config.get(key) != value for key, value in fixed.items()) or config.get('goal_complete') is not False:
        raise ValueError('fixed zero-update/96-QA/parity/65GiB diagnostic contract differs')
    if not isinstance(config.get('fixed_teacher_producer_gpu_uuid'), str) or not config['fixed_teacher_producer_gpu_uuid'].startswith('GPU-'):
        raise ValueError('original teacher producer UUID must be explicit')


def source_hashes():
    result = dict(probe_module().source_hashes())
    for file in (Path(__file__).resolve(), SOURCE_ROOT/'research_runtime/queue.py',
                 SOURCE_ROOT/'vlm_joint/pipeline.py', SOURCE_ROOT/'vlm_precision/pipeline.py'):
        result[str(file)] = sha_file(file)
    verify_files(result)
    return dict(sorted(result.items()))


def cpu_receipt(config, sources):
    file = absolute(config['cpu_contracts'])
    receipt = read_json(file)
    if receipt.get('status') != 'passed':
        raise ValueError('new CPU contracts must pass before registration')
    files = receipt.get('files_sha256', {})
    scientific = receipt.get('scientific_source_sha256', {})
    verify_files(files); verify_files(scientific)
    test_file = PROJECT_ROOT/'tests/test_object_kl_gradient_study.py'
    if files.get(str(test_file)) != sha_file(test_file) or not any(Path(name).suffix == '.log' for name in files):
        raise ValueError('CPU receipt must bind launcher contracts and complete pytest log')
    for name, digest in sources.items():
        if (Path(name).parent == SOURCE_ROOT/'vlm_object_kl_gradient' or Path(name) == Path(__file__).resolve()) and scientific.get(name) != digest:
            raise ValueError('CPU receipt does not bind current new gradient source')
    if any(receipt.get(key, 0) != 0 for key in ('new_gpu_operations', 'new_optimizer_updates', 'new_model_forwards')):
        raise ValueError('CPU receipt cannot claim GPU/model operations')
    return file, receipt


def checkpoint_contract(directory, step, lineage, schedule_sha, train, planned):
    import torch
    from vlm_object_train.train import tensor_hash, validate_checkpoint_lineage
    directory = absolute(directory)
    if not re.fullmatch(rf'checkpoint_{step:06d}_[0-9]+', directory.name):
        raise ValueError('checkpoint directory does not name the registered completed step')
    state_file = directory/'state.pt'
    state = torch.load(state_file, map_location='cpu', weights_only=False)
    if (validate_checkpoint_lineage(state, lineage, schedule_sha, planned) != step
        or state.get('next_update_index') != step):
        raise ValueError('checkpoint is not the specified next optimizer boundary')
    tensors = state['trainable_parameters']
    if (set(tensors) != set(train['trainable_names']) or len(tensors) != 144
        or sum(t.numel() for t in tensors.values()) != 1843200
        or any(t.dtype != torch.float32 or t.numel() != train['trainable_names'][name]
               or not bool(torch.isfinite(t).all()) for name, t in tensors.items())
        or tensor_hash(tensors.items()) != state['trainable_tensor_sha256']):
        raise ValueError('checkpoint complete FP32 master values/names/hash differ')
    if not state.get('optimizer', {}).get('state') or 'rng' not in state:
        raise ValueError('checkpoint must preserve Adam and RNG state')
    files = {str(state_file): sha_file(state_file)}
    for file in (directory/'adapter').glob('*'):
        if file.is_file(): files[str(file)] = sha_file(file)
    if not any(Path(name).name == 'adapter_model.safetensors' for name in files):
        raise ValueError('checkpoint adapter is incomplete')
    return {'checkpoint_step': step, 'block_index': step, 'directory': str(directory),
            'state_sha256': files[str(state_file)], 'lineage_sha256': lineage,
            'schedule_sha256': schedule_sha, 'trainable_tensor_sha256': state['trainable_tensor_sha256'],
            'optimizer_boundary': True, 'files_sha256': files}


def assert_training_binding(screen, train_file):
    digest=sha_file(train_file)
    if any(screen['source_data'][f'object_kl_42_{task}']['training_result_sha256'] != digest
           for task in ('object','text')):
        raise ValueError('source TRAIN_RESULT bytes differ from the completed-negative screen')


def source_contract(config):
    """CPU-only fixed ancestry, train rows, three states, and six teacher batches."""
    from vlm_object_kl_study.common import verify_study
    from vlm_object_kl_eval.read import verify_source_data
    from vlm_object_train.data import read_training_rows
    from vlm_object_kl.cache import load_entry
    source_config = absolute(config['source_config'])
    source_root = absolute(config['source_study_root'])
    parent_config = read_json(source_config)
    if absolute(parent_config['output_root']) != source_root:
        raise ValueError('parent configuration/root differ')
    parent = verify_study(source_root/'STUDY_INPUT_LOCK.json', source_config)
    completed = read_json(source_root/'COMPLETION.json')
    if (completed.get('schema_version') != 'rarevlm.object_kl.trial.v1'
        or completed.get('status') != 'completed_negative' or completed.get('trial_status') != 'completed_negative'
        or completed.get('goal_complete') is not False or completed.get('confirmation_passed') is not False
        or completed.get('study_input_lock_sha256') != sha_file(source_root/'STUDY_INPUT_LOCK.json')):
        raise ValueError('source must be the intact completed-negative KL pilot')
    verify_files(completed['analysis_files_sha256'])
    pilot = source_root/'analysis/pilot_seed42/screen/ANALYSIS.json'
    old_screen = read_json(pilot)
    if (completed['analysis_files_sha256'].get(str(pilot)) != sha_file(pilot)
        or old_screen.get('status') != 'complete' or old_screen.get('gate', {}).get('passed') is not False
        or old_screen.get('eligible_candidates') != []):
        raise ValueError('original failed screen must remain unchanged')
    verify_source_data(old_screen['source_data'])
    objects, replays, data_binding = read_training_rows(parent_config)
    admission = read_json(absolute(parent_config['data_admission']))
    if (admission.get('quality_gate_passed') is not True or admission.get('scope') != 'annotation_supervised_object_sft'
        or admission.get('automatic_diagnostic_bypass_allowed') is not False):
        raise ValueError('qualified training data required; no diagnostic bypass')
    directory = source_root/'runs/object_kl/seed42/train'
    train = read_json(directory/'TRAIN_RESULT.json')
    assert_training_binding(old_screen,directory/'TRAIN_RESULT.json')
    input_lock = read_json(directory/'TRAIN_INPUT_LOCK.json')
    schedule = read_json(directory/'SCHEDULE.json')
    schedule_sha = content_sha(schedule)
    planned = (len(objects)+15)//16
    if (train.get('schema_version') != 'rarevlm.object_kl.train.v1' or train.get('status') != 'completed'
        or train.get('completed_updates') != planned or train.get('planned_updates') != planned
        or train.get('skipped_updates') != 0 or train.get('config_file_sha256') != sha_file(source_config)
        or train.get('lineage_sha256') != content_sha(input_lock)
        or train.get('actual_object_exposures') != len(objects) or train.get('actual_replay_exposures') != planned*8
        or train.get('frozen_parameters_unchanged') is not True
        or train.get('frozen_parameter_sha256_before') != train.get('frozen_parameter_sha256_after')):
        raise ValueError('source final training/initialization/lineage is incomplete or changed')
    actual_file = directory/'TRAINED_INPUT_CONTRACTS.json'
    actual = read_json(actual_file)
    if (sha_file(actual_file) != train['trained_input_contracts_file_sha256']
        or content_sha(actual['object']) != train['object_input_contract_sha256']
        or content_sha(actual['replay']) != train['replay_input_contract_sha256']):
        raise ValueError('source actual training input contracts changed')
    cache = directory/'teacher_cache'
    manifest_file = cache/'MANIFEST.json'
    manifest, binding = read_json(manifest_file), read_json(cache/'INPUT_LOCK.json')
    if (train.get('teacher_cache_unchanged') is not True
        or sha_file(manifest_file) != train['teacher_cache_manifest_sha256']
        or sha_file(manifest_file) != train['teacher_cache_manifest_sha256_after']
        or manifest.get('schema_version') != 'rarevlm.object_kl.teacher_cache.v1' or manifest.get('status') != 'frozen'
        or manifest.get('binding_sha256') != content_sha(binding)
        or content_sha(binding) != train['teacher_cache_binding_sha256']
        or manifest.get('teacher_parameters_sha256') != train['initial_adapter_tensor_sha256']
        or manifest.get('frozen_parameters_sha256') != train['frozen_parameter_sha256_before']
        or binding.get('hardware', {}).get('uuid') != config['fixed_teacher_producer_gpu_uuid']
        or manifest.get('full_vocabulary') is not True or manifest.get('logprob_dtype') != 'float32'
        or manifest.get('temperature') != 1. or manifest.get('direction') != 'initial_to_actor'):
        raise ValueError('fixed original teacher/full-FP32/GPU0 binding changed')
    files = {}
    for key in ('source_sha256', 'protocol_files_sha256', 'data_files_sha256', 'model_files_sha256',
                'initializer_files_sha256', 'historical_files_sha256'):
        files.update(parent[key])
    for file in (source_config, source_root/'STUDY_INPUT_LOCK.json', source_root/'STUDY.json',
                 source_root/'COMPLETION.json', pilot, directory/'TRAIN_RESULT.json', directory/'TRAIN_INPUT_LOCK.json',
                 directory/'STATIC_INPUT_LOCK.json', directory/'SCHEDULE.json', actual_file, manifest_file, cache/'INPUT_LOCK.json'):
        files[str(file)] = sha_file(file)
    points, selected = [], []
    object_index = {str(r['question_id']):r for r in objects}
    replay_index = {str(r['question_id']):r for r in replays}
    for step in STEPS:
        point = checkpoint_contract(config['checkpoint_paths'][str(step)], step, train['lineage_sha256'], schedule_sha, train, planned)
        expected_dir = directory/Path(config['checkpoint_paths'][str(step)]).name
        if absolute(point['directory']) != expected_dir:
            raise ValueError('checkpoint must be a local source-training checkpoint')
        files.update(point['files_sha256']); points.append(point)
        oq = schedule['object_question_ids_by_update'][step]
        rq = schedule['replay_question_ids_by_update'][step]
        if len(oq) != 16 or len(rq) != 8:
            raise ValueError('registered next update must contain object16/replay8')
        cache_keys=[]
        for micro in range(2):
            key=f'update_{step:06d}_replay_{micro:03d}'
            entry=manifest['entries'][key]
            identity=entry['identity']
            if (identity['binding_sha256'] != manifest['binding_sha256']
                or identity['update_index'] != step or identity['replay_microbatch_index'] != micro
                or [str(q) for q in identity['question_ids']] != [str(q) for q in rq[micro*4:(micro+1)*4]]
                or identity['input_contracts'] != actual['replay'][step*8+micro*4:step*8+(micro+1)*4]):
                raise ValueError('selected teacher batch differs from actual actor schedule/inputs')
            file=cache/(key+'.pt')
            if sha_file(file) != entry['file_sha256']:
                raise ValueError('selected teacher batch file changed')
            load_entry(file,identity,entry)
            files[str(file)]=entry['file_sha256']; cache_keys.append(key)
        for q in oq:
            row=object_index[str(q)]; file=absolute(row['image_path']); files[str(file)]=row['file_sha256']
        for q in rq:
            row=replay_index[str(q)]; file=absolute(row['image_path']); files[str(file)]=row['file_sha256']
        selected.append({'checkpoint_step':step,'block_index':step,'next_update_number':step+1,
                         'object_question_ids':oq,'replay_question_ids':rq,'teacher_cache_keys':cache_keys})
    verify_files(files)
    return files, {'source_study_input_lock_sha256':sha_file(source_root/'STUDY_INPUT_LOCK.json'),
        'source_completion_sha256':sha_file(source_root/'COMPLETION.json'), 'source_status':'completed_negative',
        'qualified_training_data_binding':data_binding,'checkpoint_contracts':points,'selected_training_batches':selected,
        'teacher_cache_manifest_sha256':sha_file(manifest_file),'teacher_producer_gpu_uuid':config['fixed_teacher_producer_gpu_uuid']}


def prepare(config_file):
    config_file=absolute(config_file)
    config=read_json(config_file); validate_config(config)
    sources=source_hashes()
    cpu_file,cpu=cpu_receipt(config,sources)
    files,ancestry=source_contract(config)
    files.update(cpu['files_sha256']); files.update(cpu['scientific_source_sha256'])
    for file in (config_file,absolute(config['study_protocol']),cpu_file,*[absolute(v) for v in config.get('extra_lock_files',[])]):
        files[str(file)]=sha_file(file)
    verify_files(files); verify_files(sources)
    identity={'schema_version':LOCK_SCHEMA,'config_file':str(config_file),'config_file_sha256':sha_file(config_file),
        'config_content_sha256':content_sha(config),'source_sha256':sources,'files_sha256':dict(sorted(files.items())),
        'packages':package_versions(),'groups':GROUPS,'source_contract':ancestry,
        'scope':'descriptive training geometry only; no heldout performance or automatic method selection',
        'expected_new_training_qa_forward_backward':96,'optimizer_updates':0,'generation_questions':0,'goal_complete':False}
    root=absolute(config['output_root'])
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    lock_file(root/'STUDY.json',{'schema_version':'rarevlm.object_kl.gradient.registration.v1',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'groups':GROUPS,
        'new_main_qa_forward_backward':72,'new_initial_parity_qa_forward_backward':24,'new_training_qa_forward_backward':96,
        'original_cached_teacher_reference_questions':24,'new_model_loads':2,'optimizer_updates':0,'generation_questions':0,
        'memory_cap_gib':65,'automatic_runtime_fallback':False,'existing_tasks_untouched':True,
        'goal_complete':False,'scope':'diagnostic; no performance claim or automatic next training'})
    return config,root,identity


def verify_lock(config_file):
    config_file=absolute(config_file); config=read_json(config_file); validate_config(config)
    root=absolute(config['output_root']); lock=read_json(root/'STUDY_INPUT_LOCK.json')
    if (lock.get('schema_version') != LOCK_SCHEMA or lock.get('config_file_sha256') != sha_file(config_file)
        or lock.get('config_content_sha256') != content_sha(config) or lock.get('source_sha256') != source_hashes()
        or lock.get('packages') != package_versions() or lock.get('groups') != GROUPS):
        raise ValueError('gradient registration/config/source/runtime changed')
    verify_files(lock['source_sha256']); verify_files(lock['files_sha256'])
    registration=read_json(root/'STUDY.json')
    if registration['study_input_lock_sha256'] != sha_file(root/'STUDY_INPUT_LOCK.json') or registration['goal_complete'] is not False:
        raise ValueError('gradient registration pointer changed')
    return config,root,lock


def jobs(config_file,root):
    return [{'key':group,'command':[sys.executable,'-m','vlm_object_kl_gradient.probe','--config',str(config_file),
        '--output-root',str(root),'--group',group,'--study-lock',str(root/'STUDY_INPUT_LOCK.json')],
        'completion_marker':str(root/'groups'/group/'COMPLETE.json')} for group in GROUPS]


def read_completion(file,config_file,lock_file_value,config):
    value=read_json(file)
    if (value.get('schema_version') != 'rarevlm.object_kl.gradient.completion.v1' or value.get('status') != 'completed'
        or value.get('group') != file.parent.name or value.get('config_file_sha256') != sha_file(config_file)
        or value.get('study_lock_sha256') != sha_file(lock_file_value) or value.get('optimizer_updates') != 0
        or value.get('generation_questions') != 0 or value.get('producer_gpu_uuid') != config['fixed_teacher_producer_gpu_uuid']
        or not str(value.get('consumer_gpu_uuid','')).startswith('GPU-')):
        raise ValueError('gradient group completion/config/UUID/zero-update identity differs')
    files=value['artifact_files_sha256']; verify_files(files)
    for name in ('RESULT.json','INPUT_LOCK.json'):
        if str(file.parent/name) not in files:
            raise ValueError('group completion lacks result/input artifact binding')
    return value


def aggregate(config_file,root,config):
    results=[]; points=[]
    for group, specifications in GROUPS.items():
        read_completion(root/'groups'/group/'COMPLETE.json',config_file,root/'STUDY_INPUT_LOCK.json',config)
        result=probe_module().review_group(config_file,root,group,root/'STUDY_INPUT_LOCK.json')
        if (result['group'] != group or result['optimizer_updates'] != 0 or result['generation_questions'] != 0
            or result['new_training_qa_forward_backward'] != len(specifications)*32
            or result['main_qa_forward_backward'] != len(specifications)*24
            or result['initial_parity_qa_forward_backward'] != len(specifications)*8
            or [p['checkpoint_step'] for p in result['points']] != [p['checkpoint_step'] for p in specifications]
            or any(p['block_index'] != p['checkpoint_step'] for p in result['points'])):
            raise ValueError('CPU reviewed gradient points/counts differ')
        verify_files(result['artifact_files_sha256'])
        results.append(result); points.extend(result['points'])
    points.sort(key=lambda p:p['checkpoint_step'])
    if [p['checkpoint_step'] for p in points] != list(STEPS):
        raise ValueError('three unique gradient points required')
    return {'schema_version':'rarevlm.object_kl.gradient.analysis.v1','status':'completed','completed_at':now(),
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'group_reviews':results,'points':points,
        'new_training_qa_forward_backward':96,'main_qa_forward_backward':72,'initial_parity_qa_forward_backward':24,
        'optimizer_updates':0,'generation_questions':0,'cpu_aggregation_new_model_forwards':0,
        'performance_evaluation':False,'goal_complete':False,
        'limitations':['three fixed training batches are descriptive, not independent replications or a causal forgetting test',
            'initial-gradient noise controls limit interpretation of small KL directions',
            'CPU Adam shadow is a conditional first-order calculation, not an executed counterfactual trajectory',
            'no heldout performance, automatic lambda selection, or next experiment is authorized by this diagnostic']}


def run(config_file,prepare_only=False):
    config_file=absolute(config_file)
    if prepare_only:
        _,_,_=prepare(config_file)
        return {'status':'prepared','goal_complete':False,'gpu_operations':0}
    config,root,identity=verify_lock(config_file)
    owner=(root/'CONTROLLER.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    try:
        if (root/'COMPLETION.json').exists():
            value=read_json(root/'COMPLETION.json')
            if (value.get('schema_version') != 'rarevlm.object_kl.gradient.trial.v1'
                or value.get('status') != 'completed_diagnostic'
                or value.get('study_input_lock_sha256') != sha_file(root/'STUDY_INPUT_LOCK.json')
                or value.get('goal_complete') is not False or value.get('performance_evaluation') is not False
                or value.get('optimizer_updates') != 0 or value.get('generation_questions') != 0):
                raise ValueError('completed diagnostic identity changed')
            verify_files(value['artifact_files_sha256'])
            for group in GROUPS:
                read_completion(root/'groups'/group/'COMPLETE.json',config_file,root/'STUDY_INPUT_LOCK.json',config)
            return value
        task_list=jobs(config_file,root); queue=root/'queue'
        previous=read_json(queue/'STATUS.json') if (queue/'STATUS.json').exists() else {}
        for task in task_list:
            marker=Path(task['completion_marker'])
            if marker.exists():read_completion(marker,config_file,root/'STUDY_INPUT_LOCK.json',config)
            elif previous.get('jobs',{}).get(task['key'],{}).get('state')=='completed':
                raise ValueError('queue completed marker missing')
            if (marker.parent/'INPUT_LOCK.json').exists() and not previous:
                raise ValueError('cannot recover same-group GPU affinity without original queue status')
        if not all(Path(task['completion_marker']).exists() for task in task_list):
            write_json(root/'STATUS.json',{'status':'queued_training_gradient_diagnostic','updated_at':now(),
                'controller_pid':__import__('os').getpid(),'goal_complete':False,'existing_tasks_untouched':True})
            run_queue(queue,task_list,identity)
        result=aggregate(config_file,root,config)
        verify_lock(config_file)
        write_json(root/'ANALYSIS.json',result)
        complete={'schema_version':'rarevlm.object_kl.gradient.trial.v1','status':'completed_diagnostic',
            'completed_at':now(),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
            'artifact_files_sha256':{str(root/'ANALYSIS.json'):sha_file(root/'ANALYSIS.json'),
                **{str(Path(t['completion_marker'])):sha_file(Path(t['completion_marker'])) for t in task_list}},
            'current_controller_wall_seconds':time.monotonic()-started,'optimizer_updates':0,'generation_questions':0,
            'performance_evaluation':False,'scientific_performance_success':False,'goal_complete':False,
            'automatic_runtime_fallback':False,'existing_tasks_untouched':True}
        write_json(root/'COMPLETION.json',complete); write_json(root/'STATUS.json',complete)
        return complete
    except BaseException as error:
        write_json(root/'STATUS.json',{'status':'execution_or_engineering_failure','updated_at':now(),
            'error_type':type(error).__name__,'error':str(error),'scientific_failure':False,
            'performance_evaluation':False,'goal_complete':False,'automatic_runtime_fallback':False,
            'existing_tasks_untouched':True})
        raise
    finally:
        owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    print(run(args.config,args.prepare_only)['status'],flush=True)


if __name__=='__main__':
    main()
