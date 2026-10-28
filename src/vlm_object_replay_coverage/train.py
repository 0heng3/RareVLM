"""Final-only object CE + fixed KL: unique32 versus repeated8x4 replay.

No extra hard-label replay CE, moving teacher, top-k, quantization, preference,
policy rollout or RL is used. Formal optimizer starts only after a complete
frozen full-vocabulary teacher cache and a restored discarded engineering step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from vlm_object_train import train as legacy
from vlm_object_kl import train as previous_kl
from vlm_object_train.data import OBJECT_SUFFIX, PROMPT_SUFFIX, private_path, read_training_rows, verify_image_bindings
from vlm_object_train.loss import normalized_qa_loss, per_question_ce, sequence_logps
from vlm_object_train.schedule import build_schedule, coverage
from vlm_object_train.train import (autocast, check_gradients, cpu_state, file_hashes, frozen_hash,
    learning_rate, load_actor, lock_json, now, package_lock, restore_rng, restore_runtime,
    resume_checkpoint, rng_hash, rng_state, runtime_state, save_checkpoint, sync, trainable_hash)
from .reference import create_teacher_source, schedule_for_condition
from vlm_object_kl.inputs import encode_microbatch
from vlm_object_kl.loss import assistant_logprobs, per_question_hard_ce, per_question_kl

SCHEMA = 'rarevlm.object_replay_coverage.train.v1'
CONDITIONS = ('wide32', 'repeat8x4')


def validate_config(config):
    legacy.validate_config({**config, 'schema_version': legacy.SCHEMA, 'replay_per_update': 8})
    fixed = {'schema_version': SCHEMA, 'training_microbatch': 4, 'training_accumulation': 4, 'replay_per_update': 32,
             'teacher_reference_condition': 'object_kl', 'standalone_reference_scope': 'engineering_first_update_only',
             'kl_temperature': 1., 'kl_direction': 'initial_to_actor',
             'teacher_logprob_dtype': 'float32', 'teacher_full_vocabulary': True,
             'initial_kl_abs_tolerance': 1e-5, 'teacher_cache_disk_reserve_bytes': 1073741824}
    for key, value in fixed.items():
        if config.get(key) != value or isinstance(config.get(key), bool) != isinstance(value, bool):
            raise ValueError(f'fixed-initial KL contract differs: {key}')
    private_path(config['teacher_reference_study_root'])
    return config


def source_hashes():
    result = previous_kl.source_hashes()
    for directory in (Path(__file__).parent, SOURCE_ROOT / 'vlm_object_train'):
        for path in directory.glob('*.py'):
            result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    return dict(sorted(result.items()))


def actual_indices(block, condition):
    if condition not in CONDITIONS:
        raise ValueError('only wide32 or seed42 repeat8x4 conditions are allowed')
    return block['object_indices'], block['replay_indices']


def initial_stats():
    return {**legacy.initial_stats(), 'replay_kl_sum': 0.}


def validate_exposure_cursor(stats, blocks, completed, objects, replays, condition):
    actual_indices(blocks[0], condition)
    legacy.validate_exposure_cursor(stats, blocks, completed, objects, replays, 'object_replay')
    if not math.isfinite(stats['replay_kl_sum']):
        raise ValueError('nonfinite checkpoint KL aggregate')


def enforce_memory_cap(device, config, phase):
    allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.
    reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.
    if max(allocated, reserved) > config.get('preflight_memory_cap_gib', 65.):
        raise RuntimeError(f'{phase}_memory_cap_exceeded: allocated={allocated}GiB reserved={reserved}GiB; no runtime fallback')
    return allocated, reserved


def backward_update(model, processor, objects, replays, block, condition, config, device, stats=None,
                    teaching_records=None, sources=('object', 'replay'), *, teacher_cache, update):
    """Exact object mean plus independent per-QA per-answer-token forward KL."""
    obj, rep = actual_indices(block, condition)
    for source, rows, indices in (('object', objects, obj), ('replay', replays, rep)):
        if source not in sources:
            continue
        replay_rng = rng_hash(rng_state()) if source == 'replay' else None
        for microbatch, start in enumerate(range(0, len(indices), config['training_microbatch'])):
            selected = indices[start:start + config['training_microbatch']]
            batch, contracts, samples = encode_microbatch(processor, rows, selected, config)
            if teaching_records is not None:
                for position, (index, contract, sample) in enumerate(zip(selected, contracts, samples)):
                    row = rows[index]
                    teaching_records.append({'source': source, **contract, 'question': row['question'],
                        'target': row['target'], 'prompt_suffix': row['prompt_suffix'], 'image_path': row['image_path'],
                        'input_ids': batch['input_ids'][position].tolist(), 'labels': batch['labels'][position].tolist(),
                        'image_grid_thw': sample['image_grid_thw'].tolist(),
                        'pixel_values_shape': list(sample['pixel_values'].shape)})
            padding = batch['input_ids'].numel() - int(batch['attention_mask'].sum())
            teacher, teacher_counts = (teacher_cache.load(update, microbatch, batch, contracts, device)
                                       if source == 'replay' else (None, None))
            batch = to_device(batch, device)
            with torch.enable_grad(), autocast(device):
                if source == 'object':
                    logps, counts = sequence_logps(model, batch)
                    ce = per_question_ce(logps, counts)
                    values = ce
                else:
                    logprobs, counts, targets = assistant_logprobs(model, batch)
                    if not torch.equal(counts, teacher_counts):
                        raise ValueError('teacher and actor assistant masks/counts differ')
                    values = per_question_kl(logprobs, teacher, counts)
                    # Diagnostic CE is detached and is never part of loss.
                    ce = per_question_hard_ce(logprobs.detach(), targets, counts)
                if values.numel() != len(selected) or counts.tolist() != [c['assistant_tokens'] for c in contracts]:
                    raise ValueError('microbatch assistant-mask contract differs')
                loss = normalized_qa_loss(values, source, len(obj), len(rep), config['replay_lambda'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite object CE or full vocabulary forward KL')
            loss.backward()
            if stats is not None:
                stats[f'{source}_exposures'] += len(selected)
                stats[f'{source}_input_tokens'] += sum(c['input_tokens'] for c in contracts)
                stats[f'{source}_visual_tokens'] += sum(c['visual_tokens'] for c in contracts)
                stats[f'{source}_answer_tokens'] += int(counts.sum())
                stats[f'{source}_ce_sum'] += float(ce.detach().sum())
                if source == 'replay':
                    stats['replay_kl_sum'] += float(values.detach().sum())
                stats[f'{source}_input_contracts'].extend(contracts)
                stats[f'{source}_microbatches'] += 1
                stats[f'{source}_padding_tokens'] += padding
        if replay_rng is not None:
            if rng_hash(rng_state()) != replay_rng:
                raise ValueError('KL replay advanced RNG; matched object stochastic path differs')
            if stats is not None:
                stats['replay_rng_unchanged_checks'] += 1


def replay_kl_values(model, processor, replays, block, config, device, teacher_cache, update=0):
    values = []
    for microbatch, start in enumerate(range(0, len(block['replay_indices']), config['training_microbatch'])):
        selected = block['replay_indices'][start:start + config['training_microbatch']]
        batch, contracts, _ = encode_microbatch(processor, replays, selected, config)
        teacher, expected_counts = teacher_cache.load(update, microbatch, batch, contracts, device)
        with torch.enable_grad(), autocast(device):
            actor, counts, _ = assistant_logprobs(model, to_device(batch, device))
            if not torch.equal(counts, expected_counts):
                raise ValueError('preflight teacher assistant count differs')
            values.extend(per_question_kl(actor, teacher, counts).detach().cpu().tolist())
    return values


def preflight(model, parameters, processor, objects, replays, block, config, condition, directory, device,
              *, teacher_cache, frozen_sha256=None):
    initial = {name: cpu_state(p) for name, p in model.named_parameters() if p.requires_grad}
    before, frozen_before = trainable_hash(model), frozen_sha256 or frozen_hash(model)
    rng, runtime = rng_state(), runtime_state(model)
    teacher_cache.verify()
    optimizer = torch.optim.AdamW(parameters, lr=config['preflight_lr'], weight_decay=config['weight_decay'])
    error, completed, record = None, 0, {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        before_values = replay_kl_values(model, processor, replays, block, config, device, teacher_cache)
        if not all(math.isfinite(value) for value in before_values) or max(abs(value) for value in before_values) > config['initial_kl_abs_tolerance']:
            raise ValueError('fixed teacher initial KL exceeds declared floating-point tolerance')
        optimizer.zero_grad(set_to_none=True)
        backward_update(model, processor, objects, replays, block, condition, config, device,
                        teacher_cache=teacher_cache, update=0)
        check_gradients(model, parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError('preflight has zero trainable gradient')
        optimizer.step()
        completed += 1
        after_values = replay_kl_values(model, processor, replays, block, config, device, teacher_cache)
        if not all(math.isfinite(value) for value in after_values) or trainable_hash(model) == before or sum(after_values) / len(after_values) <= 0:
            raise ValueError('temporary actor update did not yield positive fixed-teacher KL response')
        enforce_memory_cap(device, config, 'preflight')
        record = {'actual_object_questions': len(block['object_indices']), 'actual_replay_questions': len(block['replay_indices']),
            'initial_replay_kl_per_question': before_values, 'initial_kl_abs_tolerance': config['initial_kl_abs_tolerance'],
            'after_temporary_update_replay_kl_per_question': after_values,
            'after_temporary_update_mean_replay_kl': sum(after_values) / len(after_values),
            'gradient_l2_before_clip': float(norm), 'finite_gradients': True,
            'teacher_has_grad': False, 'teacher_parameters_are_fixed_initial': True,
            'trainable_sha256_after_temporary_update': trainable_hash(model)}
    except Exception as exc:
        error = exc
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in initial:
                    p.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        restore_runtime(runtime)
        restore_rng(rng)
    if trainable_hash(model) != before or frozen_hash(model) != frozen_before or rng_hash(rng_state()) != rng_hash(rng):
        raise ValueError('KL preflight failed exact master/RNG/frozen restoration')
    teacher_cache.verify()
    allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.
    reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.
    result = {'status': 'failed' if error else 'passed', 'condition': condition, 'completed_at': now(),
        'scope': 'engineering only; temporary optimizer discarded and all initial state restored',
        'reference_scope': teacher_cache.scope,
        'initial_reference_parity_questions': 32, 'after_temporary_update_reference_probe_questions': 32,
        'engineering_actor_forward_backward_questions': 48,
        'actual_new_teacher_forward_questions': teacher_cache.record['actual_new_teacher_forward_questions'],
        'historical_teacher_bank_questions': teacher_cache.record['historical_teacher_bank_questions'],
        'discarded_engineering_optimizer_updates': completed, 'formal_optimizer_updates': 0,
        'engineering_learning_rate': config['preflight_lr'], 'formal_learning_rate': config['lr'],
        'initial_parameters_restored_exactly': True, 'rng_restored': True,
        'model_runtime_buffers_and_rope_cache_restored': True, 'engineering_optimizer_discarded': True,
        'frozen_parameters_unchanged': True, 'teacher_cache_unchanged': True,
        'teacher_cache_manifest_sha256': teacher_cache.manifest_sha256,
        'frozen_parameter_sha256_before': frozen_before, 'frozen_parameter_sha256_after': frozen_hash(model),
        'trainable_sha256_before': before, 'trainable_parameters': sum(p.numel() for p in parameters),
        'training_microbatch_questions': config['training_microbatch'], 'training_accumulation': config['training_accumulation'],
        'object_effective_batch': config['object_effective_batch'], 'peak_allocated_gib': allocated,
        'peak_reserved_gib': reserved, 'memory_cap_gib': config['preflight_memory_cap_gib'],
        'memory_cap_passed': max(allocated, reserved) <= config['preflight_memory_cap_gib'], 'record': record}
    if error:
        result['error'] = f'{type(error).__name__}: {error}'
    write_json(directory / 'PREFLIGHT.json', result)
    with (directory / 'PREFLIGHT_HISTORY.jsonl').open('a') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if error:
        raise error
    return result


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    session_started = time.monotonic()
    config_path = private_path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    if args.condition not in CONDITIONS or args.seed not in (0, 42, 200) or (args.condition == 'repeat8x4' and args.seed != 42):
        raise ValueError('requires a declared condition and preregistered seed0/42/200')
    objects, replays, data_binding = read_training_rows(config)
    blocks = schedule_for_condition(objects, replays, args.seed, config['epochs'], config['object_effective_batch'],
                            config['replay_per_update'], condition=args.condition)
    schedule = {'object_question_ids_by_update': [[str(objects[i]['question_id']) for i in b['object_indices']] for b in blocks],
                'replay_question_ids_by_update': [[str(replays[i]['question_id']) for i in actual_indices(b, args.condition)[1]]
                                                  for b in blocks],
                'learning_rates_by_update': [learning_rate(step, len(blocks), config) for step in range(len(blocks))]}
    schedule['training_microbatch_questions'] = config['training_microbatch']
    for source in ('object', 'replay'):
        schedule[f'{source}_microbatches_by_update'] = [
            [qids[start:start + config['training_microbatch']] for start in range(0, len(qids), config['training_microbatch'])]
            for qids in schedule[f'{source}_question_ids_by_update']]
    schedule_sha = content_sha(schedule)
    consumed_replay = [i for block in blocks for i in actual_indices(block, args.condition)[1]]
    # Verify exactly the consumed pool prefix, not an unobserved full-pool claim.
    image_bindings = verify_image_bindings(objects + [replays[i] for i in sorted(set(consumed_replay))])
    data_binding['actual_consumed_image_bindings'] = image_bindings
    directory = private_path(args.output_root) / 'runs' / args.condition / f'seed{args.seed}' / 'train'
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume and args.preflight:
        raise ValueError('--resume and standalone --preflight are mutually exclusive')
    if (directory / 'LATEST.json').exists() and not args.resume and not args.preflight:
        raise ValueError('checkpoint exists; explicitly use --resume')
    if args.resume and not (directory / 'LATEST.json').exists():
        raise ValueError('--resume requires an existing checkpoint')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('object trainer requires a single visible device')
    hardware = visible_gpu_binding()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('object trainer requires a BF16-capable CUDA device')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reference_path = private_path(config['reference_config'])
    reference = json.loads(reference_path.read_text())
    model_root, adapter_root = private_path(config['model_root']), private_path(config['initial_adapter'])
    adapter_config = json.loads((adapter_root / 'adapter_config.json').read_text())
    if (adapter_config['r'] != 8 or adapter_config['lora_alpha'] != 16 or adapter_config['bias'] != 'none'
            or set(adapter_config['target_modules']) != {'q_proj', 'v_proj'}):
        raise ValueError('initial adapter must be the frozen language q/v rank8 alpha16 SFT adapter')
    static_lock = {'schema_version': SCHEMA, 'condition': args.condition, 'seed': args.seed,
        'config_sha256': content_sha(config), 'config_file_sha256': sha_file(config_path),
        'reference_config_sha256': sha_file(reference_path), 'source_sha256': source_hashes(),
        'packages': package_lock(), 'hardware': hardware, 'data_binding': data_binding,
        'train_manifest_sha256': data_binding['train_manifest_sha256'],
        'train_manifest_file_sha256': data_binding['train_manifest_sha256'],
        'replay_manifest_sha256': data_binding['replay_manifest_sha256'],
        'model_files_sha256': file_hashes(model_root),
        'model_provenance_sha256': sha_file(model_root.parent / 'model_provenance.json'),
        'initial_adapter_files_sha256': file_hashes(adapter_root),
        'initial_adapter_dir': str(adapter_root),
        'object_question_count': len(objects), 'question_count': len(objects),
        'unique_training_images': len({str(row['image_id']) for row in objects}),
        'planned_updates': len(blocks), 'planned_object_exposures': len(objects) * config['epochs'],
        'planned_replay_exposures': len(consumed_replay),
        'object_order_sha256': content_sha(schedule['object_question_ids_by_update']),
        'order_sha256': content_sha(schedule['object_question_ids_by_update']),
        'replay_order_sha256': content_sha(schedule['replay_question_ids_by_update']), 'schedule_sha256': schedule_sha,
        'replay_sampling': ('SHA256 complete-pool permutation unique prefix32/update' if args.condition == 'wide32' else 'original seed42 8/update SHA-prefix repeated4 times within update; matched32 exposures'),
        'training_microbatch_questions': config['training_microbatch'], 'training_accumulation': config['training_accumulation'],
        'object_effective_batch': config['object_effective_batch'],
        'object_microbatch_order_sha256': content_sha(schedule['object_microbatches_by_update']),
        'replay_questions_per_update': 32 if consumed_replay else 0,
        'replay_lambda': 1. if consumed_replay else 0., 'config_replay_lambda': config['replay_lambda'],
        'loss': 'mean_QA mean_assistant_tokens CE(object) + lambda1 mean_QA mean_assistant_tokens KL(p_initial || p_actor); full vocabulary SUM, T=1',
        'normalization': 'object/replay independently divided by actual real questions in each update, including final short object block',
        'checkpoint_selection': 'fixed Final@1 epoch; no epoch/dev/test selection',
        'numerical_path': 'original_bf16', 'base_precision': 'bf16', 'lora_parameter_precision': 'fp32',
        'lora_initialization': 'initial SFT adapter rounded to BF16 then exactly promoted to FP32',
        'autocast': 'bf16', 'dropout': 0., 'visual_token_budget': config['visual_token_budget'],
        'object_prompt_suffix': config['object_prompt_suffix'], 'replay_prompt_suffix': config['replay_prompt_suffix'],
        'max_sequence_length': config['max_sequence_length'], 'input_truncation': False,
        'equal_compute_claim': False,
        'runtime_note': 'explicit microbatch/accumulation selection; no bitwise equivalence or measured speedup claim across runtimes',
        'kl_direction': 'initial_to_actor', 'kl_temperature': 1., 'teacher_logprob_dtype': 'float32',
        'teacher_full_vocabulary': True, 'replay_hard_ce_backward': False,
        'helper_provenance': ['frozen vlm_deep.common encode/collate + vlm_sft.common.answer_labels',
                              'frozen vlm_alignment_signal.loss.sequence_logps (assistant/end-token sums only)',
                              'frozen vlm_precision.evaluate.load_path original_bf16; frozen vlm_probe.evaluate.processor_for'],
        'historical_training_comparison': 'new BF16 base/FP32-master/1024 path differs from historical NF4/FP32; not a dtype-only comparison'}
    lock_json(directory / 'STATIC_INPUT_LOCK.json', static_lock)
    lock_json(directory / 'SCHEDULE.json', schedule)
    loading_start = time.monotonic()
    model, parameters, dtypes = load_actor(config, reference, device)
    processor = processor_for(model_root, reference, config['visual_token_budget'])
    sync(device)
    loading_seconds = time.monotonic() - loading_start
    initial = {name: cpu_state(parameter) for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen_before = frozen_hash(model)
    lock = {**static_lock, 'initial_adapter_tensor_sha256': trainable_hash(model),
            'frozen_parameter_sha256_before': frozen_before, 'dtypes': dtypes,
            'trainable_parameters': sum(parameter.numel() for parameter in parameters),
            'trainable_names': {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}}
    cache_binding = {**lock, 'teacher_role': 'fixed_initial_before_optimizer',
        'replay_microbatch_order_sha256': content_sha(schedule['replay_microbatches_by_update']),
        'teacher_distribution': 'full FP32 log_softmax, all assistant positions, T=1',
        'teacher_forward_path': 'model.train; grad enabled; checkpoint enabled; dropout0; original BF16; FP32 master LoRA'}
    cache_started = time.monotonic()
    teacher_cache, teacher_source, cache_manifest = create_teacher_source(model, processor, replays,
        blocks, config, directory, cache_binding, device, condition=args.condition, preflight_only=args.preflight)
    cache_directory = teacher_cache.directory
    teacher_cache_sha = teacher_cache.manifest_sha256
    cache_seconds = time.monotonic() - cache_started
    lock.update({'teacher_cache_manifest_sha256': teacher_cache_sha,
                 'teacher_cache_binding_sha256': content_sha(teacher_cache.binding),
                 'teacher_cache_directory': str(cache_directory),
                 'teacher_source': teacher_source,
                 'teacher_source_file_sha256': sha_file(directory / 'TEACHER_SOURCE.json')})
    lock_json(directory / 'TRAIN_INPUT_LOCK.json', lock)
    lineage_sha = content_sha(lock)
    preflight_started = time.monotonic()
    preflight_result = preflight(model, parameters, processor, objects, replays, blocks[0], config, args.condition,
                                directory, device, teacher_cache=teacher_cache, frozen_sha256=frozen_before)
    sync(device)
    preflight_seconds = time.monotonic() - preflight_started
    if args.preflight:
        print('PREFLIGHT_PASSED', args.condition, args.seed, flush=True)
        return
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    stats, completed = initial_stats(), 0
    if args.resume:
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, schedule_sha, len(blocks))
        validate_exposure_cursor(stats, blocks, completed, objects, replays, args.condition)
    if (directory / 'TRAIN_RESULT.json').exists():
        result = json.loads((directory / 'TRAIN_RESULT.json').read_text())
        if result['lineage_sha256'] != lineage_sha or file_hashes(directory / 'adapter') != result['adapter_files_sha256']:
            raise ValueError('completed adapter/lineage changed')
        return
    write_json(directory / 'PROGRESS.json', {'status': 'training', 'completed_updates': completed,
        'planned_updates': len(blocks), 'actual_object_exposures': stats['object_exposures'],
        'actual_replay_exposures': stats['replay_exposures'], 'updated_at': now()})
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(completed, len(blocks)):
        sync(device)
        begun = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, len(blocks), config)
        teaching_records = [] if step == 0 else None
        # Timing separates the extra replay work. The same object blocks/order
        # remain in both conditions; replay gradients are added before one step.
        for source in ('object', 'replay'):
            phase = time.monotonic()
            backward_update(model, processor, objects, replays, blocks[step], args.condition, config, device,
                            stats, teaching_records, sources=(source,), teacher_cache=teacher_cache, update=step)
            sync(device)
            stats[f'{source}_active_seconds'] += time.monotonic() - phase
        check_gradients(model, parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        enforce_memory_cap(device, config, 'formal_training')
        stats['training_seconds'] += time.monotonic() - begun
        stats['last_gradient_l2'] = float(norm)
        if teaching_records is not None:
            write_json(directory / 'FIRST_UPDATE.json', {'scope': 'actual first formal GPU update encoded inputs',
                'completed_updates': 1, 'learning_rate': optimizer.param_groups[0]['lr'],
                'gradient_l2_before_clip': float(norm), 'object_count': len(blocks[step]['object_indices']),
                'replay_count': len(actual_indices(blocks[step], args.condition)[1]), 'records': teaching_records})
        progress = {'status': 'training', 'completed_updates': step + 1, 'planned_updates': len(blocks),
            'actual_object_exposures': stats['object_exposures'], 'actual_replay_exposures': stats['replay_exposures'],
            'training_active_seconds': stats['training_seconds'], 'gradient_l2': float(norm), 'updated_at': now()}
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == len(blocks):
            print('OBJECT_TRAIN_UPDATE', json.dumps({'condition': args.condition, **progress}), flush=True)
        if (step + 1) % config['checkpoint_every_updates'] == 0 or step + 1 == len(blocks):
            save_checkpoint(directory, model, optimizer, step + 1, stats, lineage_sha, schedule_sha)
            write_json(directory / 'PROGRESS.json', progress)
    validate_exposure_cursor(stats, blocks, len(blocks), objects, replays, args.condition)
    teacher_cache.verify()
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError('frozen base/vision weights changed')
    delta_sq, maximum, changed = 0., 0., 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().cpu()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('nonfinite final LoRA master tensors')
            delta = value - initial[name]
            delta_sq += float(delta.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            changed += int(bool((delta != 0).any()))
    if changed == 0:
        raise ValueError('final LoRA tensors did not change')
    model.save_pretrained(directory / 'adapter', safe_serialization=True)
    inputs_path = directory / 'TRAINED_INPUT_CONTRACTS.json'
    write_json(inputs_path, {'object': stats['object_input_contracts'], 'replay': stats['replay_input_contracts']})
    result = {**lock, 'lineage_sha256': lineage_sha, 'status': 'completed', 'completed_at': now(),
        'completed_updates': len(blocks), 'epochs_completed': config['epochs'],
        'actual_questions_seen': stats['object_exposures'] + stats['replay_exposures'],
        'actual_object_exposures': stats['object_exposures'], 'actual_replay_exposures': stats['replay_exposures'],
        'skipped_updates': 0, 'object_input_contract_sha256': content_sha(stats['object_input_contracts']),
        'replay_input_contract_sha256': content_sha(stats['replay_input_contracts']),
        'trained_input_contracts_file_sha256': sha_file(inputs_path),
        'replay_coverage': coverage(replays, consumed_replay),
        'replay_rng_unchanged_updates': stats['replay_rng_unchanged_checks'],
        'mean_object_ce': stats['object_ce_sum'] / stats['object_exposures'],
        'mean_replay_ce': stats['replay_ce_sum'] / stats['replay_exposures'],
        'mean_replay_ce_scope': 'hard target CE diagnostic only; never backpropagated',
        'mean_replay_kl': stats['replay_kl_sum'] / stats['replay_exposures'],
        'teacher_cache_manifest_sha256_after': sha_file(cache_directory / 'MANIFEST.json'),
        'teacher_cache_unchanged': True, 'teacher_cache_seconds_current_session': cache_seconds,
        'teacher_source_manifest_sha256_after': sha_file(cache_directory / 'MANIFEST.json'),
        'teacher_source_unchanged': True, 'teacher_reference_setup_wall_seconds': cache_seconds,
        'teacher_cache_summary': {key: value for key, value in cache_manifest.items() if key != 'entries'},
        'training_active_seconds': stats['training_seconds'], 'model_load_seconds': loading_seconds,
        'preflight_seconds_current_session': preflight_seconds, 'preflight': preflight_result,
        'reference_scope': teacher_source['scope'],
        'session_wall_seconds': time.monotonic() - session_started,
        'lora_delta_l2': math.sqrt(delta_sq), 'lora_delta_max_abs': maximum, 'changed_lora_tensors': changed,
        'final_lora_tensor_sha256': trainable_hash(model), 'frozen_parameter_sha256_after': frozen_after,
        'frozen_parameters_unchanged': True, 'resumed_at_update': completed,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'adapter_files_sha256': file_hashes(directory / 'adapter'),
        'compute_accounting': {'equal_compute_claim': False, 'measured_flops': None,
            'flops_status': 'not instrumented; separate QA/token/time counts are not equal-FLOPs evidence',
            'teacher_cache_forward_questions': teacher_source['actual_new_teacher_forward_questions'],
            'historical_teacher_bank_questions': teacher_source['historical_teacher_bank_questions'],
            'actor_replay_unique_questions': teacher_source['actor_replay_unique_questions_planned'],
            'teacher_cache_forward_seconds': teacher_source['actual_new_teacher_forward_seconds'],
            'historical_teacher_bank_forward_seconds': teacher_source['historical_teacher_bank_forward_seconds'],
            'teacher_cache_file_bytes': cache_manifest['cache_file_bytes'],
            'new_teacher_probability_file_bytes': cache_manifest['cache_file_bytes'] if args.condition == 'wide32' else 0,
            'object_forward_backward_questions': stats['object_exposures'],
            'extra_replay_forward_backward_questions': stats['replay_exposures'],
            'object_active_seconds': stats['object_active_seconds'], 'extra_replay_active_seconds': stats['replay_active_seconds']}}
    for source in ('object', 'replay'):
        for name in ('input_tokens', 'visual_tokens', 'answer_tokens', 'microbatches', 'padding_tokens'):
            result[f'training_{source}_{name}'] = stats[f'{source}_{name}']
            result['compute_accounting'][f'{source}_{name}'] = stats[f'{source}_{name}']
    write_json(directory / 'TRAIN_RESULT.json', result)
    write_json(directory / 'PROGRESS.json', {'status': 'completed', 'completed_updates': len(blocks),
        'planned_updates': len(blocks), 'actual_object_exposures': stats['object_exposures'],
        'actual_replay_exposures': stats['replay_exposures'], 'updated_at': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
