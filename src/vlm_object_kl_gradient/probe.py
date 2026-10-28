"""Fixed-checkpoint source gradients: zero optimizer updates and generation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import torch

from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, to_device, write_json
from vlm_object_kl import train as training
from vlm_object_kl.cache import TeacherCache
from vlm_object_kl.inputs import encode_microbatch
from vlm_object_kl.loss import assistant_logprobs, per_question_kl
from vlm_object_train.data import private_path, read_training_rows
from vlm_object_train.schedule import build_schedule
from vlm_object_train.train import (autocast, cpu_state, frozen_hash, load_actor, lock_json,
    now, package_lock, restore_rng, restore_runtime, rng_hash, rng_state, runtime_state,
    sync, tensor_hash, trainable_hash)
from .math import adam_shadow_summary, gradient_geometry, norm, vector

SCHEMA = 'rarevlm.object_kl.gradient.v1'
GROUPS = {'early_late': [{'checkpoint_step': 25, 'block_index': 25}, {'checkpoint_step': 175, 'block_index': 175}],
          'middle': [{'checkpoint_step': 100, 'block_index': 100}]}


def path(value):
    if not Path(value).is_absolute():
        raise ValueError('gradient diagnostic requires explicit absolute persistent paths')
    return private_path(value)


def source_hashes():
    result = {str(SOURCE_ROOT / key): value for key, value in training.source_hashes().items()}
    for file in Path(__file__).parent.glob('*.py'):
        result[str(file.resolve())] = sha_file(file)
    return dict(sorted(result.items()))


def validate_config(config):
    fixed = {'schema_version': SCHEMA, 'groups': GROUPS, 'parity_kl_abs_tolerance': 1e-5,
        'memory_cap_gib': 65, 'training_microbatch': 4, 'optimizer_updates': 0,
        'generation_questions': 0, 'expected_new_training_qa_forward_backward': 96,
        'expected_main_qa_forward_backward': 72, 'expected_initial_parity_qa_forward_backward': 24}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'fixed zero-update diagnostic contract differs: {key}')
    for name in ('source_config', 'source_study_root', 'output_root', 'study_protocol', 'cpu_contracts'):
        path(config[name])
    if set(config['checkpoint_paths']) != {'25', '100', '175'}:
        raise ValueError('three unique explicit checkpoint paths required')
    for step, value in config['checkpoint_paths'].items():
        directory = path(value)
        expected = path(config['source_study_root']) / 'runs/object_kl/seed42/train'
        if directory.parent != expected or not directory.name.startswith(f'checkpoint_{int(step):06d}_'):
            raise ValueError('checkpoint path must identify the fixed saved source update')
    return config


def verify_study(config_file, study_lock):
    config_file, study_lock = path(config_file), path(study_lock)
    config = validate_config(json.loads(config_file.read_text()))
    lock = json.loads(study_lock.read_text())
    if (lock.get('schema_version') != 'rarevlm.object_kl.gradient.study.v1'
            or lock.get('config_file_sha256') != sha_file(config_file)):
        raise ValueError('gradient study config/schema binding differs')
    declared = lock.get('source_sha256', {})
    for name, digest in source_hashes().items():
        if declared.get(name) != digest:
            raise ValueError(f'gradient study does not bind current required source: {name}')
    for field in ('source_sha256', 'files_sha256'):
        if not isinstance(lock.get(field), dict) or not lock[field]:
            raise ValueError('nonempty source and original file mappings required')
        for name, digest in lock[field].items():
            file = path(name)
            if not file.is_file() or sha_file(file) != digest:
                raise ValueError(f'locked diagnostic source/input file changed: {name}')
    return config, lock


def atomic_json(file, value):
    temporary = file.with_suffix(file.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(file)


def atomic_torch(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.tmp')
    torch.save(value, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(file)


def event(directory, kind, **fields):
    with (directory / 'EXECUTION.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': kind, 'at': now(), 'pid': os.getpid(), **fields}, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def memory_cap(device):
    allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.
    reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.
    if max(allocated, reserved) > 65.:
        raise RuntimeError(f'gradient_memory_cap_exceeded: allocated={allocated}GiB reserved={reserved}GiB cap=65GiB; no runtime fallback')
    return allocated, reserved


def parameter_layout(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def capture_gradient(model, callback, device):
    """Capture one source's actual backward gradient and restore mutable state."""
    before = trainable_hash(model)
    random_before, runtime_before = rng_state(), runtime_state(model)
    result, gradient, error = None, None, None
    model.zero_grad(set_to_none=True)
    sync(device)
    begun = time.monotonic()
    try:
        result = callback()
        values = []
        for name, parameter in parameter_layout(model):
            if parameter.dtype != torch.float32 or parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f'absent/nonfinite/non-FP32 diagnostic gradient: {name}')
            values.append(parameter.grad.detach().cpu().reshape(-1).clone())
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('frozen parameter received diagnostic gradient')
        gradient = torch.cat(values)
        memory_cap(device)
    except Exception as exc:
        error = exc
    finally:
        model.zero_grad(set_to_none=True)
        restore_runtime(runtime_before)
        restore_rng(random_before)
        sync(device)
    if trainable_hash(model) != before or rng_hash(rng_state()) != rng_hash(random_before):
        raise ValueError('source gradient probe changed model parameters or failed RNG restoration')
    if error:
        raise error
    return vector(gradient), result, time.monotonic() - begun


def teaching_records(batch, contracts, samples, rows, indices, source):
    records = []
    for position, (contract, sample, index) in enumerate(zip(contracts, samples, indices)):
        row = rows[index]
        records.append({'source': source, **contract, 'question': row['question'], 'target': row['target'],
            'prompt_suffix': row['prompt_suffix'], 'image_path': row['image_path'],
            'input_ids': batch['input_ids'][position].tolist(), 'labels': batch['labels'][position].tolist(),
            'image_grid_thw': sample['image_grid_thw'].tolist(),
            'pixel_values_shape': list(sample['pixel_values'].shape)})
    return records


def initial_parity(model, processor, replays, block, block_index, source_config, teacher, device):
    """Exactly8 replay QA F/B, also obtaining each QA's KL in that same pass."""
    def calculate():
        values, records = [], []
        for micro, begin in enumerate(range(0, 8, 4)):
            selected = block['replay_indices'][begin:begin + 4]
            batch, contracts, samples = encode_microbatch(processor, replays, selected, source_config)
            records.extend(teaching_records(batch, contracts, samples, replays, selected, 'initial_replay'))
            cached, cached_counts = teacher.load(block_index, micro, batch, contracts, device)
            with torch.enable_grad(), autocast(device):
                actor, counts, _ = assistant_logprobs(model, to_device(batch, device))
                if actor.shape[1] != 151936 or not torch.equal(counts, cached_counts):
                    raise ValueError('initial parity full vocabulary or assistant counts differ')
                kl = per_question_kl(actor, cached, counts)
                if not bool(torch.isfinite(kl).all()) or float(kl.detach().abs().max()) > 1e-5:
                    raise ValueError('initial same-input consumer-to-fixed-producer KL exceeds1e-5; no tolerance fallback')
                loss = kl.sum() / 8
            loss.backward()
            values.extend(kl.detach().cpu().tolist())
            memory_cap(device)
        return {'per_question_KL': values, 'actual_QA': len(values), 'records': records,
            'max_absolute_KL': max(abs(value) for value in values)}
    return capture_gradient(model, calculate, device)


def load_context(config):
    source_file = path(config['source_config'])
    source_config = training.validate_config(json.loads(source_file.read_text()))
    source_root = path(config['source_study_root'])
    if source_root != path(source_config['output_root']):
        raise ValueError('diagnostic source config/root mismatch')
    train_dir = source_root / 'runs/object_kl/seed42/train'
    train = json.loads((train_dir / 'TRAIN_RESULT.json').read_text())
    if (train.get('status') != 'completed' or train.get('completed_updates') != 181
            or train.get('config_file_sha256') != sha_file(source_file)):
        raise ValueError('requires exact completed source training')
    schedule = json.loads((train_dir / 'SCHEDULE.json').read_text())
    trained_inputs_file = train_dir / 'TRAINED_INPUT_CONTRACTS.json'
    trained_inputs = json.loads(trained_inputs_file.read_text())
    if content_sha(schedule) != train['schedule_sha256'] or sha_file(trained_inputs_file) != train['trained_input_contracts_file_sha256']:
        raise ValueError('source fixed schedule or actual training inputs changed')
    cache_dir = train_dir / 'teacher_cache'
    if sha_file(cache_dir / 'MANIFEST.json') != train['teacher_cache_manifest_sha256_after'] or train['teacher_cache_unchanged'] is not True:
        raise ValueError('source teacher cache final binding differs')
    binding = json.loads((cache_dir / 'INPUT_LOCK.json').read_text())
    if (content_sha(binding) != train['teacher_cache_binding_sha256']
            or binding['hardware']['uuid'] != config['fixed_teacher_producer_gpu_uuid']):
        raise ValueError('source teacher producer/initial identity differs')
    return source_config, train_dir, train, schedule, trained_inputs, cache_dir, binding


def checkpoint_state(config, step, train):
    directory = path(config['checkpoint_paths'][str(step)])
    state_file = directory / 'state.pt'
    state = torch.load(state_file, map_location='cpu', weights_only=False)
    if (state['completed_updates'] != step or state['next_update_index'] != step
            or state['accumulated_questions'] != 0 or state['lineage_sha256'] != train['lineage_sha256']
            or state['schedule_sha256'] != train['schedule_sha256']
            or tensor_hash(state['trainable_parameters'].items()) != state['trainable_tensor_sha256']):
        raise ValueError('saved checkpoint step/lineage/schedule/master SHA differs')
    if (sum(p.numel() for p in state['trainable_parameters'].values()) != 1843200
            or len(state['trainable_parameters']) != 144
            or any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in state['trainable_parameters'].values())):
        raise ValueError('checkpoint must contain144 finite FP32 language LoRA tensors')
    return state


def copy_masters(model, state):
    layout = parameter_layout(model)
    if [name for name, _ in layout] != list(state['trainable_parameters']):
        raise ValueError('loaded model/Adam checkpoint parameter order differs')
    with torch.no_grad():
        for name, parameter in layout:
            parameter.copy_(state['trainable_parameters'][name])
    if trainable_hash(model) != state['trainable_tensor_sha256']:
        raise ValueError('copied FP32 checkpoint master tensors differ')


def expected_contracts(trained_inputs, block_index):
    return {'object': trained_inputs['object'][block_index * 16:block_index * 16 + 16],
            'replay': trained_inputs['replay'][block_index * 8:block_index * 8 + 8]}


def require_records_match(records, expected):
    if len(records) != len(expected):
        raise ValueError('actual diagnostic QA count differs from source block')
    for actual, contract in zip(records, expected):
        if any(actual.get(key) != value for key, value in contract.items()):
            raise ValueError('diagnostic actual IDs/pixels/grid/masks/padding differ from original trained block')


def point_directory(group_dir, point):
    return group_dir / 'points' / f"step_{point['checkpoint_step']:06d}_block_{point['block_index']:06d}"


def save_parity(directory, gradient, parity, seconds, identity_sha, initial_sha):
    tensor_file = directory / 'PARITY.pt'
    atomic_torch(tensor_file, {'initial_KL_gradient': gradient,
        'gradient_sha256': tensor_hash([('initial_KL_gradient', gradient)]), 'group_input_sha256': identity_sha})
    receipt = {'schema_version': 'rarevlm.object_kl.gradient.parity.v1', 'group_input_sha256': identity_sha,
        'initial_master_sha256': initial_sha, 'tensor_file_sha256': sha_file(tensor_file),
        'actual_QA': 8, 'new_qa_forward_backward': 8, 'optimizer_updates': 0,
        'max_absolute_KL': parity['max_absolute_KL'], 'per_question_KL': parity['per_question_KL'],
        'records': parity['records'], 'gradient_l2': norm(gradient), 'active_seconds': seconds}
    atomic_json(directory / 'PARITY.json', receipt)
    return receipt


def read_parity(directory, identity_sha, initial_sha):
    parity = json.loads((directory / 'PARITY.json').read_text())
    file = directory / 'PARITY.pt'
    if (parity['group_input_sha256'] != identity_sha or parity['initial_master_sha256'] != initial_sha
            or parity['tensor_file_sha256'] != sha_file(file) or parity['actual_QA'] != 8
            or len(parity['per_question_KL']) != 8
            or any(not math.isfinite(v) or abs(v) > 1e-5 for v in parity['per_question_KL'])):
        raise ValueError('saved initial parity source/input/precision differs')
    payload = torch.load(file, map_location='cpu', weights_only=False)
    gradient = vector(payload['initial_KL_gradient'])
    if payload['group_input_sha256'] != identity_sha or payload['gradient_sha256'] != tensor_hash([('initial_KL_gradient', gradient)]):
        raise ValueError('initial gradient floor tensor SHA differs')
    return gradient, parity


def point_artifact_files(directory):
    return {str(directory / name): sha_file(directory / name)
        for name in ('PARITY.pt', 'PARITY.json', 'GRADIENTS.pt', 'ACTUAL_INPUTS.json', 'POINT.json')}


def recompute_point(config, group_dir, point, identity_sha, train, schedule, trained_inputs):
    directory = point_directory(group_dir, point)
    saved = json.loads((directory / 'POINT.json').read_text())
    if saved['group_input_sha256'] != identity_sha:
        raise ValueError('completed diagnostic point belongs to a different group identity')
    for name, digest in saved['input_artifact_files_sha256'].items():
        if sha_file(path(name)) != digest:
            raise ValueError('completed diagnostic input/gradient artifact changed')
    state = checkpoint_state(config, point['checkpoint_step'], train)
    payload = torch.load(directory / 'GRADIENTS.pt', map_location='cpu', weights_only=False)
    if (payload['parameter_names'] != list(state['trainable_parameters'])
            or payload['group_input_sha256'] != identity_sha
            or payload['checkpoint_tensor_sha256'] != state['trainable_tensor_sha256']):
        raise ValueError('saved gradient layout/checkpoint binding differs')
    object_gradient, kl_gradient = vector(payload['object_gradient']), vector(payload['KL_gradient'])
    if payload['tensor_sha256'] != tensor_hash([('object_gradient', object_gradient), ('KL_gradient', kl_gradient)]):
        raise ValueError('source gradient tensor SHA changed')
    initial_gradient, parity = read_parity(directory, identity_sha, train['initial_adapter_tensor_sha256'])
    records = json.loads((directory / 'ACTUAL_INPUTS.json').read_text())
    expected = expected_contracts(trained_inputs, point['block_index'])
    for origin in ('object', 'replay'):
        require_records_match(records[origin], expected[origin])
    require_records_match(parity['records'], expected['replay'])
    geometry = gradient_geometry(object_gradient, kl_gradient, initial_gradient)
    shadow = adam_shadow_summary(object_gradient, kl_gradient, state, payload['parameter_names'],
                                 schedule['learning_rates_by_update'][point['block_index']])
    if saved['geometry'] != geometry or saved['adam_shadow'] != shadow:
        raise ValueError('CPU independent norm/cosine/shadow recomputation differs')
    if (saved['master_sha256_before'] != state['trainable_tensor_sha256']
            or saved['master_sha256_after'] != state['trainable_tensor_sha256']
            or saved['frozen_base_sha256_after'] != train['frozen_parameter_sha256_before']
            or saved['teacher_manifest_sha256_after'] != train['teacher_cache_manifest_sha256_after']):
        raise ValueError('completed zero-update parameter/frozen/teacher invariant differs')
    return {**point, 'next_update_one_based': point['block_index'] + 1, 'geometry': geometry,
        'adam_shadow': shadow, 'object_CE': saved['object_CE'], 'replay_KL': saved['replay_KL'],
        'initial_parity_max_absolute_KL': parity['max_absolute_KL'],
        'costs': saved['costs'], 'cpu_recomputed_from_saved_FP32_gradients': True,
        'artifact_files_sha256': point_artifact_files(directory)}


def review_group(config_file, output_root, group, study_lock):
    config, _ = verify_study(config_file, study_lock)
    root = path(output_root)
    if root != path(config['output_root']) or group not in GROUPS:
        raise ValueError('group review root/group differs')
    directory = root / 'groups' / group
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    if (complete.get('schema_version') != 'rarevlm.object_kl.gradient.completion.v1'
            or complete.get('status') != 'completed' or complete.get('group') != group
            or complete.get('config_file_sha256') != sha_file(path(config_file))
            or complete.get('study_lock_sha256') != sha_file(path(study_lock))):
        raise ValueError('gradient group completion identity differs')
    for name, digest in complete['artifact_files_sha256'].items():
        if sha_file(path(name)) != digest:
            raise ValueError('gradient group artifact SHA changed')
    input_file = directory / 'INPUT_LOCK.json'
    identity = json.loads(input_file.read_text())
    if (identity.get('group') != group or identity.get('config_file_sha256') != complete['config_file_sha256']
            or identity.get('study_lock_sha256') != complete['study_lock_sha256']
            or identity.get('producer_gpu_uuid') != complete['producer_gpu_uuid']
            or identity.get('consumer_hardware', {}).get('uuid') != complete['consumer_gpu_uuid']):
        raise ValueError('group input and completion identity/UUID differ')
    source_config, train_dir, train, schedule, trained_inputs, _, _ = load_context(config)
    points = [recompute_point(config, directory, point, content_sha(identity), train, schedule, trained_inputs)
              for point in GROUPS[group]]
    n = len(points)
    expected_counts = {'new_training_qa_forward_backward': n * 32, 'main_qa_forward_backward': n * 24,
        'initial_parity_qa_forward_backward': n * 8, 'optimizer_updates': 0, 'generation_questions': 0}
    for key, value in expected_counts.items():
        if complete.get(key) != value:
            raise ValueError('gradient completed group QA/zero-update accounting differs')
    result = json.loads((directory / 'RESULT.json').read_text())
    if (result.get('points') != points or any(result.get(k) != v for k, v in expected_counts.items())
            or result.get('group') != group or result.get('performance_evaluation') is not False
            or result.get('initial_parameters_rng_buffers_restored') is not True
            or result.get('frozen_parameters_unchanged') is not True
            or result.get('teacher_cache_manifest_unchanged') is not True
            or max(result['peak_allocated_gib'], result['peak_reserved_gib']) > 65.):
        raise ValueError('group result independent points/counts/restoration/memory contract differs')
    return {'group': group, 'points': points, **expected_counts,
        'consumer_gpu_uuid': identity['consumer_hardware']['uuid'],
        'producer_gpu_uuid': identity['producer_gpu_uuid'],
        'artifact_files_sha256': complete['artifact_files_sha256'],
        'scope': 'CPU independent recomputation; local offline training geometry, not heldout performance or forgetting causality'}


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    config_file, study_lock = path(args.config), path(args.study_lock)
    config, _ = verify_study(config_file, study_lock)
    root = path(args.output_root)
    if root != path(config['output_root']) or args.group not in GROUPS:
        raise ValueError('probe root/group differs from declared configuration')
    directory = root / 'groups' / args.group
    directory.mkdir(parents=True, exist_ok=True)
    source_config, train_dir, train, schedule, trained_inputs, cache_dir, binding = load_context(config)
    hardware = visible_gpu_binding()
    identity = {'schema_version': 'rarevlm.object_kl.gradient.group_input.v1', 'group': args.group,
        'config_file_sha256': sha_file(config_file), 'study_lock_sha256': sha_file(study_lock),
        'source_config_file_sha256': sha_file(path(config['source_config'])),
        'source_train_result_sha256': sha_file(train_dir / 'TRAIN_RESULT.json'),
        'source_teacher_manifest_sha256': sha_file(cache_dir / 'MANIFEST.json'),
        'source_teacher_binding_sha256': content_sha(binding), 'producer_gpu_uuid': binding['hardware']['uuid'],
        'consumer_hardware': hardware, 'packages': package_lock(), 'points': GROUPS[args.group],
        'source_checkpoint_file_sha256': {str(path(config['checkpoint_paths'][str(p['checkpoint_step'])]) / 'state.pt'):
            sha_file(path(config['checkpoint_paths'][str(p['checkpoint_step'])]) / 'state.pt') for p in GROUPS[args.group]}}
    lock_json(directory / 'INPUT_LOCK.json', identity)
    identity_sha = content_sha(identity)
    if (directory / 'COMPLETE.json').exists():
        return review_group(config_file, root, args.group, study_lock)
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or int(os.environ.get('WORLD_SIZE', '1')) != 1):
        raise ValueError('diagnostic requires exactly one assigned BF16-capable CUDA device')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    objects, replays, _ = read_training_rows(source_config)
    blocks = build_schedule(objects, replays, 42, 1, 16, 8)
    teacher = TeacherCache(cache_dir, binding, train['teacher_cache_manifest_sha256_after'])
    reference = json.loads(path(source_config['reference_config']).read_text())
    begun = time.monotonic()
    loading = time.monotonic()
    event(directory, 'model_load_started')
    model, _, dtypes = load_actor(source_config, reference, device)
    processor = processor_for(path(source_config['model_root']), reference, 1024)
    sync(device)
    model_load_seconds = time.monotonic() - loading
    event(directory, 'model_load_completed', model_load_seconds=model_load_seconds)
    initial = {name: cpu_state(parameter) for name, parameter in parameter_layout(model)}
    initial_sha = trainable_hash(model)
    initial_rng, initial_runtime = rng_state(), runtime_state(model)
    frozen_before = frozen_hash(model)
    if initial_sha != train['initial_adapter_tensor_sha256'] or frozen_before != train['frozen_parameter_sha256_before']:
        raise ValueError('consumer loaded initial master/frozen base differs from original numerical path')
    memory_cap(device)
    torch.cuda.reset_peak_memory_stats(device)
    resumed_points, parity_data, summaries = [], {}, []
    try:
        # Initial consumption of the original producer cache, never regenerated.
        for point in GROUPS[args.group]:
            place = point_directory(directory, point)
            place.mkdir(parents=True, exist_ok=True)
            expected = expected_contracts(trained_inputs, point['block_index'])
            if (place / 'POINT.json').exists():
                summaries.append(recompute_point(config, directory, point, identity_sha, train, schedule, trained_inputs))
                resumed_points.append(point['checkpoint_step'])
                continue
            if (place / 'PARITY.json').exists():
                initial_gradient, parity = read_parity(place, identity_sha, initial_sha)
            else:
                event(directory, 'source_started', checkpoint_step=point['checkpoint_step'], source='initial_parity', QA=8)
                initial_gradient, values, seconds = initial_parity(model, processor, replays,
                    blocks[point['block_index']], point['block_index'], source_config, teacher, device)
                event(directory, 'source_completed', checkpoint_step=point['checkpoint_step'], source='initial_parity', QA=8, phase_wall_seconds=seconds)
                require_records_match(values['records'], expected['replay'])
                parity = save_parity(place, initial_gradient, values, seconds, identity_sha, initial_sha)
            require_records_match(parity['records'], expected['replay'])
            parity_data[point['checkpoint_step']] = (initial_gradient, parity)
        # All new initial forwards end before any checkpoint master is copied.
        if trainable_hash(model) != initial_sha or rng_hash(rng_state()) != rng_hash(initial_rng):
            raise ValueError('initial parity failed exact parameter/RNG restoration')
        for point in GROUPS[args.group]:
            if point['checkpoint_step'] in resumed_points:
                continue
            place = point_directory(directory, point)
            state = checkpoint_state(config, point['checkpoint_step'], train)
            copy_masters(model, state)
            restore_runtime(initial_runtime)
            restore_rng(state['rng'])
            before = trainable_hash(model)
            gradients, records, stats_by_source, times = {}, {}, {}, {}
            for origin in ('object', 'replay'):
                current_stats = training.initial_stats()
                current_records = []
                def calculate(origin=origin):
                    training.backward_update(model, processor, objects, replays, blocks[point['block_index']],
                        'object_kl', source_config, device, current_stats, current_records,
                        sources=(origin,), teacher_cache=teacher, update=point['block_index'])
                    return current_stats
                event(directory, 'source_started', checkpoint_step=point['checkpoint_step'], source=origin, QA=16 if origin == 'object' else 8)
                gradients[origin], _, times[origin] = capture_gradient(model, calculate, device)
                event(directory, 'source_completed', checkpoint_step=point['checkpoint_step'], source=origin, QA=16 if origin == 'object' else 8, phase_wall_seconds=times[origin])
                records[origin], stats_by_source[origin] = current_records, current_stats
                require_records_match(current_records, expected_contracts(trained_inputs, point['block_index'])[origin])
            initial_gradient, parity = parity_data[point['checkpoint_step']]
            names = [name for name, _ in parameter_layout(model)]
            cpu_started = time.monotonic()
            geometry = gradient_geometry(gradients['object'], gradients['replay'], initial_gradient)
            shadow = adam_shadow_summary(gradients['object'], gradients['replay'], state, names,
                                        schedule['learning_rates_by_update'][point['block_index']])
            cpu_math_seconds = time.monotonic() - cpu_started
            if trainable_hash(model) != before:
                raise ValueError('diagnostic changed copied checkpoint parameters')
            frozen_after = frozen_hash(model)
            if frozen_after != frozen_before:
                raise ValueError('diagnostic changed frozen base/vision parameters')
            tensor_file = place / 'GRADIENTS.pt'
            atomic_torch(tensor_file, {'group_input_sha256': identity_sha, 'parameter_names': names,
                'parameter_shapes': [list(p.shape) for _, p in parameter_layout(model)],
                'checkpoint_tensor_sha256': state['trainable_tensor_sha256'],
                'object_gradient': gradients['object'], 'KL_gradient': gradients['replay'],
                'tensor_sha256': tensor_hash([('object_gradient', gradients['object']), ('KL_gradient', gradients['replay'])])})
            atomic_json(place / 'ACTUAL_INPUTS.json', records)
            point_result = {'schema_version': 'rarevlm.object_kl.gradient.point.v1', **point,
                'group_input_sha256': identity_sha, 'next_update_one_based': point['block_index'] + 1,
                'geometry': geometry, 'adam_shadow': shadow,
                'object_CE': stats_by_source['object']['object_ce_sum'] / 16,
                'replay_KL': stats_by_source['replay']['replay_kl_sum'] / 8,
                'replay_hard_CE_diagnostic_only': stats_by_source['replay']['replay_ce_sum'] / 8,
                'master_sha256_before': before, 'master_sha256_after': trainable_hash(model),
                'frozen_base_sha256_after': frozen_after,
                'teacher_manifest_sha256_after': sha_file(cache_dir / 'MANIFEST.json'),
                'optimizer_updates': 0, 'generation_questions': 0,
                'new_training_qa_forward_backward': 32, 'main_qa_forward_backward': 24,
                'initial_parity_qa_forward_backward': 8,
                'costs': {'object_forward_backward_seconds': times['object'], 'KL_forward_backward_seconds': times['replay'],
                    'initial_parity_forward_backward_seconds': parity['active_seconds'],
                    'CPU_geometry_and_Adam_shadow_seconds': cpu_math_seconds,
                    'timing_scope': 'Source phase wall includes encoding/H2D/forward/backward/FP32 gradient copy/runtime restore; not pure GPU kernel timing.',
                    'gradient_FP32_tensor_bytes': 3 * gradients['object'].numel() * 4},
                'input_artifact_files_sha256': {str(place / name): sha_file(place / name)
                    for name in ('PARITY.pt', 'PARITY.json', 'GRADIENTS.pt', 'ACTUAL_INPUTS.json')}}
            atomic_json(place / 'POINT.json', point_result)
            summaries.append(recompute_point(config, directory, point, identity_sha, train, schedule, trained_inputs))
            print('GRADIENT_POINT_COMPLETE', json.dumps({'group': args.group, **point, 'geometry': geometry}), flush=True)
    except Exception as exc:
        allocated, reserved = torch.cuda.max_memory_allocated(device) / 2**30, torch.cuda.max_memory_reserved(device) / 2**30
        atomic_json(directory / 'FAILURE.json', {'schema_version': 'rarevlm.object_kl.gradient.failure.v1',
            'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'optimizer_updates': 0,
            'peak_allocated_gib': allocated, 'peak_reserved_gib': reserved, 'memory_cap_gib': 65,
            'completed_point_steps': [value['checkpoint_step'] for value in summaries]})
        raise
    finally:
        with torch.no_grad():
            for name, parameter in parameter_layout(model):
                parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        restore_runtime(initial_runtime)
        restore_rng(initial_rng)
    if trainable_hash(model) != initial_sha or frozen_hash(model) != frozen_before:
        raise ValueError('group exit failed initial/frozen exact restoration')
    teacher.verify()
    allocated, reserved = memory_cap(device)
    n = len(GROUPS[args.group])
    execution = [json.loads(line) for line in (directory / 'EXECUTION.jsonl').read_text().splitlines() if line.strip()]
    completed_sources = [item for item in execution if item['event'] == 'source_completed']
    main_qa = sum(item['QA'] for item in completed_sources if item['source'] != 'initial_parity')
    parity_qa = sum(item['QA'] for item in completed_sources if item['source'] == 'initial_parity')
    if main_qa != n * 24 or parity_qa != n * 8:
        raise ValueError('actual completed execution QA differs from unique planned blocks; recovery attempts must not be hidden')
    counts = {'new_training_qa_forward_backward': n * 32, 'main_qa_forward_backward': n * 24,
        'initial_parity_qa_forward_backward': n * 8, 'optimizer_updates': 0, 'generation_questions': 0}
    result = {'schema_version': 'rarevlm.object_kl.gradient.group.v1', 'status': 'completed', 'group': args.group,
        'config_file_sha256': sha_file(config_file), 'study_lock_sha256': sha_file(study_lock),
        'producer_gpu_uuid': binding['hardware']['uuid'], 'consumer_gpu_uuid': hardware['uuid'], **counts,
        'points': sorted(summaries, key=lambda value: value['checkpoint_step']), 'dtypes': dtypes,
        'model_loads_current_session': 1, 'model_load_seconds_current_session': model_load_seconds,
        'actual_model_load_starts_all_sessions': sum(item['event'] == 'model_load_started' for item in execution),
        'actual_model_load_completions_all_sessions': sum(item['event'] == 'model_load_completed' for item in execution),
        'actual_completed_main_QA_all_sessions': main_qa, 'actual_completed_parity_QA_all_sessions': parity_qa,
        'source_call_starts_all_sessions': sum(item['event'] == 'source_started' for item in execution),
        'source_call_completions_all_sessions': len(completed_sources),
        'session_wall_seconds': time.monotonic() - begun, 'resumed_completed_points': resumed_points,
        'peak_allocated_gib': allocated, 'peak_reserved_gib': reserved, 'memory_cap_gib': 65,
        'initial_parameters_rng_buffers_restored': True, 'frozen_parameters_unchanged': True,
        'teacher_cache_manifest_unchanged': True, 'performance_evaluation': False,
        'limitations': 'Single seed offline training gradients; local Adam approximation conditional on historical moments; not original gradient logs, heldout gains or forgetting causality.'}
    atomic_json(directory / 'RESULT.json', result)
    artifacts = {str(directory / 'INPUT_LOCK.json'): sha_file(directory / 'INPUT_LOCK.json'),
                 str(directory / 'RESULT.json'): sha_file(directory / 'RESULT.json'),
                 str(directory / 'EXECUTION.jsonl'): sha_file(directory / 'EXECUTION.jsonl')}
    for point in GROUPS[args.group]:
        artifacts.update(point_artifact_files(point_directory(directory, point)))
    complete = {'schema_version': 'rarevlm.object_kl.gradient.completion.v1', 'status': 'completed',
        'group': args.group, 'config_file_sha256': sha_file(config_file), 'study_lock_sha256': sha_file(study_lock),
        'producer_gpu_uuid': binding['hardware']['uuid'], 'consumer_gpu_uuid': hardware['uuid'],
        **counts, 'artifact_files_sha256': artifacts}
    atomic_json(directory / 'COMPLETE.json', complete)
    return review_group(config_file, root, args.group, study_lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--group', choices=tuple(GROUPS), required=True)
    parser.add_argument('--study-lock', type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
