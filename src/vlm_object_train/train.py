"""Final-only annotation-supervised object SFT, optionally with TextVQA replay.

Provenance: BF16-rounded -> FP32 language LoRA loading and atomic update-boundary
checkpoint conventions follow frozen vlm_posttrain/train.py. Input encoding is
the frozen vlm_deep/common.py assistant-mask contract. This independent module
uses supervised CE only: there is no preference pair or dynamic reference.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import torch

from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, to_device, write_json
from .data import OBJECT_SUFFIX, PROMPT_SUFFIX, private_path, read_training_rows, verify_image_bindings
from .loss import normalized_qa_loss, per_question_ce, sequence_logps
from .schedule import actual_indices, build_schedule, coverage

SCHEMA = 'rarevlm.vlm_object_train.v1'
CONDITIONS = ('object_only', 'object_replay')


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    fixed = {'schema_version': SCHEMA, 'object_prompt_suffix': OBJECT_SUFFIX,
             'replay_prompt_suffix': PROMPT_SUFFIX, 'object_effective_batch': 16,
             'replay_per_update': 8, 'replay_lambda': 1.,
             'expected_trainable': 1843200, 'checkpoint_every_updates': 25,
             'epochs': 1, 'lr': 5e-6, 'preflight_lr': 1e-4, 'weight_decay': 0.,
             'warmup_ratio': .1, 'gradient_clip_norm': 1.,
             'gradient_checkpointing': True, 'visual_token_budget': 1024,
             'max_sequence_length': 2048}
    for key, value in fixed.items():
        actual = config.get(key)
        if actual != value or isinstance(actual, bool) != isinstance(value, bool):
            raise ValueError(f'object training fixed contract differs: {key}')
    microbatch = config.get('training_microbatch')
    accumulation = config.get('training_accumulation')
    if (type(microbatch) is not int or type(accumulation) is not int
            or min(microbatch, accumulation) <= 0
            or microbatch * accumulation != config['object_effective_batch']):
        raise ValueError('declare explicit positive runtime microbatch/accumulation with product16; no automatic runtime fallback')
    if config.get('preflight_memory_cap_gib', 65.) != 65.:
        raise ValueError('fixed preflight memory cap must be65GiB; runtime cannot silently expand')
    for key in ('train_manifest', 'replay_manifest', 'reference_config', 'model_root', 'initial_adapter',
                'data_admission', 'dataset_protocol'):
        if not config.get(key):
            raise ValueError(f'missing required path: {key}')
        private_path(config[key])
    return config


def learning_rate(step, total, config):
    if not 0 <= step < total:
        raise ValueError('schedule update cursor outside planned run')
    warm = max(math.ceil(total * config['warmup_ratio']), 1)
    factor = (step + 1) / warm if step < warm else (total - step) / max(total - warm, 1)
    return config['lr'] * factor


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def trainable_hash(model):
    return tensor_hash((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)


def frozen_hash(model):
    return tensor_hash((name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad)


def batch_hash(batch):
    return tensor_hash((key, batch[key]) for key in sorted(batch))


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def rng_state():
    import numpy as np
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def rng_hash(state):
    digest = hashlib.sha256(repr(state['python']).encode())
    numpy_state = state['numpy']
    digest.update(numpy_state[0].encode())
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode())
    digest.update(state['torch'].numpy().tobytes())
    if state['cuda'] is not None:
        digest.update(state['cuda'].cpu().numpy().tobytes())
    return digest.hexdigest()


RUNTIME_ATTRIBUTES = ('rope_deltas', '_rope_deltas', 'past_key_values', '_past_key_values',
                      '_cache', 'cache_position', 'max_seq_len_cached', '_seen_tokens')


def runtime_state(model):
    """Capture mutable Qwen RoPE/cache attributes and registered buffers.

    No KV cache is requested in this trainer. Qwen can still populate
    ``rope_deltas`` during a forward, so restoring LoRA/RNG alone is incomplete.
    Keep original buffer objects as well as their values/devices.
    """
    state = []
    for module in model.modules():
        buffers = {key: (value, cpu_state(value) if value is not None else None)
                   for key, value in module._buffers.items()}
        attributes = {key: (key in module.__dict__, cpu_state(module.__dict__.get(key)),
                             module.__dict__.get(key).device if isinstance(module.__dict__.get(key), torch.Tensor) else None)
                      for key in RUNTIME_ATTRIBUTES}
        state.append((module, buffers, attributes))
    return state


def restore_runtime(state):
    for module, buffers, attributes in state:
        module._buffers.clear()
        for key, (original, value) in buffers.items():
            if original is not None:
                original.copy_(value)
            module._buffers[key] = original
        for key, (existed, value, device) in attributes.items():
            if existed:
                module.__dict__[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            else:
                module.__dict__.pop(key, None)


def file_hashes(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha_file(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def source_hashes():
    paths = set(Path(__file__).parent.glob('*.py'))
    for relative in ('vlm_alignment_signal/loss.py', 'vlm_deep/common.py', 'vlm_deep/model.py',
                     'vlm_precision/evaluate.py', 'vlm_probe/evaluate.py', 'vlm_joint/evaluate.py',
                     'vlm_sft/common.py', 'rarevlm_paths.py'):
        paths.add(SOURCE_ROOT / relative)
    return {str(path.relative_to(SOURCE_ROOT)): sha_file(path) for path in sorted(paths)}


def package_lock():
    result = {'python': sys.version, 'torch_cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version()}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'Pillow', 'safetensors', 'numpy'):
        result[name] = importlib.metadata.version(name)
    return result


def lock_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'input/source/runtime lock changed: {path}')
    else:
        write_json(path, value)


def validate_checkpoint_lineage(state, lineage_sha256, schedule_sha256, completed_limit):
    if state.get('lineage_sha256') != lineage_sha256:
        raise ValueError('checkpoint lineage differs; refusing mixed data/code/config/initialization/UUID')
    if state.get('schedule_sha256') != schedule_sha256:
        raise ValueError('checkpoint schedule differs')
    step = state.get('completed_updates')
    if type(step) is not int or not 0 <= step <= completed_limit:
        raise ValueError('invalid checkpoint update cursor')
    if state.get('accumulated_questions') != 0:
        raise ValueError('checkpoint is not at an optimizer boundary')
    return step


def save_checkpoint(directory, model, optimizer, completed, stats, lineage_sha256, schedule_sha256):
    destination = directory / f'checkpoint_{completed:06d}_{time.time_ns()}'
    staging = directory / f'.{destination.name}.tmp'
    staging.mkdir()
    model.save_pretrained(staging / 'adapter', safe_serialization=True)
    # Preserve exact FP32 master tensors independently of PEFT import defaults.
    state = {'lineage_sha256': lineage_sha256, 'schedule_sha256': schedule_sha256,
             'completed_updates': completed, 'next_update_index': completed,
             'accumulated_questions': 0, 'optimizer': cpu_state(optimizer.state_dict()),
             'rng': rng_state(), 'stats': cpu_state(stats),
             'trainable_parameters': {name: cpu_state(parameter) for name, parameter in model.named_parameters()
                                      if parameter.requires_grad},
             'trainable_tensor_sha256': trainable_hash(model)}
    torch.save(state, staging / 'state.pt')
    record = {'directory': destination.name, 'lineage_sha256': lineage_sha256,
              'schedule_sha256': schedule_sha256, 'completed_updates': completed,
              'state_sha256': sha_file(staging / 'state.pt'),
              'adapter_files_sha256': file_hashes(staging / 'adapter')}
    staging.replace(destination)
    write_json(directory / 'LATEST.json', record)


def resume_checkpoint(directory, model, optimizer, lineage_sha256, schedule_sha256, planned):
    latest = json.loads((directory / 'LATEST.json').read_text())
    name = latest['directory']
    if Path(name).name != name:
        raise ValueError('checkpoint pointer must name a local checkpoint directory')
    checkpoint = directory / name
    if (latest.get('lineage_sha256') != lineage_sha256 or latest.get('schedule_sha256') != schedule_sha256
            or sha_file(checkpoint / 'state.pt') != latest['state_sha256']):
        raise ValueError('checkpoint pointer/state binding changed')
    if file_hashes(checkpoint / 'adapter') != latest['adapter_files_sha256']:
        raise ValueError('checkpoint adapter changed')
    state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    step = validate_checkpoint_lineage(state, lineage_sha256, schedule_sha256, planned)
    if step != latest['completed_updates'] or state['next_update_index'] != step:
        raise ValueError('checkpoint update cursor differs from pointer')
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = state['trainable_parameters']
    if set(parameters) != set(saved) or any(parameters[key].shape != saved[key].shape
                                          or parameters[key].dtype != saved[key].dtype for key in parameters):
        raise ValueError('checkpoint FP32 master parameter names/shapes/dtypes differ')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name])
    if trainable_hash(model) != state['trainable_tensor_sha256']:
        raise ValueError('restored LoRA master tensors differ from checkpoint')
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return step, state['stats']


def load_actor(config, reference, device):
    from vlm_precision.evaluate import load_path, parameter_dtypes
    model, _ = load_path(private_path(config['model_root']), reference, device,
                         'original_bf16', private_path(config['initial_adapter']))
    model.requires_grad_(False)
    parameters = []
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            if 'visual' in name or not any(f'.{projection}.' in name for projection in ('q_proj', 'v_proj')):
                raise ValueError(f'unexpected trainable adapter placement: {name}')
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if sum(parameter.numel() for parameter in parameters) != config['expected_trainable']:
        raise ValueError('unexpected language q/v rank8 LoRA parameter budget')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    return model, parameters, parameter_dtypes(model)


def encode_qa(processor, row, config):
    sample = encode(processor, row, row['image_path'], True)
    length = int(sample['input_ids'].shape[1])
    if length > config['max_sequence_length']:
        raise ValueError(f'QA exceeds sequence budget; truncation prohibited: {row["question_id"]}')
    grid = sample['image_grid_thw']
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError('expected exactly one static image per question')
    patches = int(grid.prod())
    if patches % 4 or not 0 < patches // 4 <= config['visual_token_budget']:
        raise ValueError('actual visual tokens exceed the frozen upper bound')
    batch = collate([sample], processor.tokenizer.pad_token_id)
    supervised = int((batch['labels'] != -100).sum())
    if supervised <= 0 or int(batch['labels'][0, 0]) != -100:
        raise ValueError('missing assistant targets or missing prompt mask')
    contract = {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
                'input_sha256': batch_hash(batch), 'input_tokens': int(batch['attention_mask'].sum()),
                'sequence_length': length, 'visual_tokens': patches // 4,
                'assistant_tokens': supervised, 'truncated': False}
    return batch, contract


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def backward_update(model, processor, objects, replays, block, condition, config, device, stats=None,
                    teaching_records=None, sources=('object', 'replay')):
    """Accumulate one real matched object block and its independent replay mean."""
    object_indices, replay_indices = actual_indices(block, condition)
    for source, rows, indices in (('object', objects, object_indices), ('replay', replays, replay_indices)):
        if source not in sources:
            continue
        replay_rng = rng_hash(rng_state()) if source == 'replay' and indices else None
        microbatch = config['training_microbatch']
        for start in range(0, len(indices), microbatch):
            selected = indices[start:start + microbatch]
            encoded = [encode_qa(processor, rows[index], config) for index in selected]
            samples, contracts = zip(*encoded)
            batch = collate(samples, processor.tokenizer.pad_token_id)
            micro_sha = batch_hash(batch)
            padded_length = int(batch['input_ids'].shape[1])
            contracts = [{**contract, 'microbatch_input_sha256': micro_sha,
                          'microbatch_question_position': position, 'microbatch_size': len(selected),
                          'microbatch_sequence_length': padded_length}
                         for position, contract in enumerate(contracts)]
            if teaching_records is not None:
                for position, (index, contract, sample) in enumerate(zip(selected, contracts, samples)):
                    row = rows[index]
                    teaching_records.append({'source': source, **contract, 'question': row['question'],
                        'target': row['target'], 'prompt_suffix': row['prompt_suffix'],
                        'image_path': row['image_path'], 'input_ids': batch['input_ids'][position].tolist(),
                        'labels': batch['labels'][position].tolist(), 'image_grid_thw': sample['image_grid_thw'].tolist(),
                        'pixel_values_shape': list(sample['pixel_values'].shape)})
            padding_tokens = padded_length * len(selected) - int(batch['attention_mask'].sum())
            batch = to_device(batch, device)
            with autocast(device):
                logps, counts = sequence_logps(model, batch)
                ce = per_question_ce(logps, counts)
                if ce.numel() != len(selected) or counts.tolist() != [contract['assistant_tokens'] for contract in contracts]:
                    raise ValueError('microbatch assistant-mask contract differs')
                loss = normalized_qa_loss(ce, source, len(object_indices), len(replay_indices), config['replay_lambda'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite supervised CE')
            loss.backward()
            if stats is not None:
                stats[f'{source}_exposures'] += len(selected)
                stats[f'{source}_input_tokens'] += sum(contract['input_tokens'] for contract in contracts)
                stats[f'{source}_visual_tokens'] += sum(contract['visual_tokens'] for contract in contracts)
                stats[f'{source}_answer_tokens'] += int(counts.sum())
                stats[f'{source}_ce_sum'] += float(ce.detach().sum())
                stats[f'{source}_input_contracts'].extend(contracts)
                stats[f'{source}_microbatches'] += 1
                stats[f'{source}_padding_tokens'] += padding_tokens
        if replay_rng is not None:
            if rng_hash(rng_state()) != replay_rng:
                raise ValueError('extra replay advanced RNG; matched object stochastic path would differ')
            if stats is not None:
                stats['replay_rng_unchanged_checks'] += 1


def check_gradients(model, parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError('absent/nonfinite trainable gradients')
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError('frozen base/vision parameter received a gradient')


def preflight(model, parameters, processor, objects, replays, block, config, condition, directory, device,
              frozen_sha256=None):
    """One actual temporary optimizer update; exact master/RNG restore in finally."""
    initial = {name: cpu_state(parameter) for name, parameter in model.named_parameters() if parameter.requires_grad}
    before = trainable_hash(model)
    frozen_before = frozen_sha256 or frozen_hash(model)
    rng_before = rng_state()
    runtime_before = runtime_state(model)
    optimizer = torch.optim.AdamW(parameters, lr=config['preflight_lr'], weight_decay=config['weight_decay'])
    error, completed, record = None, 0, {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        probe_batch, _ = encode_qa(processor, objects[block['object_indices'][0]], config)
        probe_batch = to_device(probe_batch, device)
        with torch.enable_grad(), autocast(device):
            before_logps, _ = sequence_logps(model, probe_batch)
        before_logps = before_logps.detach()
        optimizer.zero_grad(set_to_none=True)
        backward_update(model, processor, objects, replays, block, condition, config, device)
        check_gradients(model, parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError('preflight has zero trainable gradient')
        optimizer.step()
        completed += 1
        with torch.enable_grad(), autocast(device):
            after_logps, _ = sequence_logps(model, probe_batch)
        delta = float((after_logps.detach() - before_logps).abs().max())
        after = trainable_hash(model)
        if before == after or not math.isfinite(delta) or delta <= 0:
            raise ValueError('temporary update did not change live LoRA/assistant logps')
        obj, rep = actual_indices(block, condition)
        record = {'actual_object_questions': len(obj), 'actual_replay_questions': len(rep),
                  'gradient_l2_before_clip': float(norm), 'finite_gradients': True,
                  'assistant_logp_max_difference_after_temporary_update': delta,
                  'trainable_sha256_after_temporary_update': after}
    except Exception as exc:
        error = exc
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        restore_runtime(runtime_before)
        restore_rng(rng_before)
    if trainable_hash(model) != before:
        raise ValueError('preflight failed to restore exact FP32 master initialization')
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError('preflight changed frozen base/vision parameters')
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.
    cap = config.get('preflight_memory_cap_gib', 65.)
    if error is None and max(peak_allocated, peak_reserved) > cap:
        error = RuntimeError(f'preflight_memory_cap_exceeded: allocated={peak_allocated:.6f}GiB '
                             f'reserved={peak_reserved:.6f}GiB cap={cap:.6f}GiB; candidate stopped without runtime fallback')
    result = {'status': 'failed' if error else 'passed', 'condition': condition, 'completed_at': now(),
              'scope': 'engineering contract; temporary step discarded before final-only formal SFT',
              'discarded_engineering_optimizer_updates': completed, 'formal_optimizer_updates': 0,
              'engineering_learning_rate': config['preflight_lr'], 'formal_learning_rate': config['lr'],
              'initial_parameters_restored_exactly': True, 'rng_restored': True,
              'model_runtime_buffers_and_rope_cache_restored': True,
              'engineering_optimizer_discarded': True, 'frozen_parameters_unchanged': True,
              'frozen_parameter_sha256_before': frozen_before, 'frozen_parameter_sha256_after': frozen_after,
              'trainable_sha256_before': before, 'trainable_parameters': sum(p.numel() for p in parameters),
              'training_microbatch_questions': config['training_microbatch'],
              'training_accumulation': config['training_accumulation'], 'object_effective_batch': config['object_effective_batch'],
              'peak_allocated_gib': peak_allocated, 'peak_reserved_gib': peak_reserved,
              'memory_cap_gib': cap, 'memory_cap_passed': max(peak_allocated, peak_reserved) <= cap,
              'record': record}
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


def initial_stats():
    stats = {'training_seconds': 0., 'last_gradient_l2': None, 'replay_rng_unchanged_checks': 0}
    for source in ('object', 'replay'):
        stats.update({f'{source}_exposures': 0, f'{source}_input_tokens': 0,
                      f'{source}_visual_tokens': 0, f'{source}_answer_tokens': 0,
                      f'{source}_ce_sum': 0., f'{source}_input_contracts': [],
                      f'{source}_active_seconds': 0., f'{source}_microbatches': 0,
                      f'{source}_padding_tokens': 0})
    return stats


def validate_exposure_cursor(stats, blocks, completed, objects, replays, condition):
    if stats['replay_rng_unchanged_checks'] != completed * int(condition == 'object_replay'):
        raise ValueError('checkpoint replay RNG-check count differs from update cursor')
    for source, rows, position in (('object', objects, 0), ('replay', replays, 1)):
        indices = [index for block in blocks[:completed] for index in actual_indices(block, condition)[position]]
        records = stats[f'{source}_input_contracts']
        if stats[f'{source}_exposures'] != len(indices) or len(records) != len(indices):
            raise ValueError('checkpoint exposure count differs from fixed schedule')
        expected_ids = [str(rows[index]['question_id']) for index in indices]
        if [record['question_id'] for record in records] != expected_ids:
            raise ValueError('checkpoint actual question order differs from fixed schedule')


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    session_started = time.monotonic()
    config_path = private_path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    if args.condition not in CONDITIONS or args.seed not in (0, 42, 200):
        raise ValueError('requires a declared condition and preregistered seed0/42/200')
    objects, replays, data_binding = read_training_rows(config)
    blocks = build_schedule(objects, replays, args.seed, config['epochs'], config['object_effective_batch'],
                            config['replay_per_update'])
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
        'replay_sampling': 'SHA256(seed,cycle,question_id) complete-pool permutation; input-only fixed prefix',
        'training_microbatch_questions': config['training_microbatch'], 'training_accumulation': config['training_accumulation'],
        'object_effective_batch': config['object_effective_batch'],
        'object_microbatch_order_sha256': content_sha(schedule['object_microbatches_by_update']),
        'replay_questions_per_update': 8 if consumed_replay else 0,
        'replay_lambda': 1. if consumed_replay else 0., 'config_replay_lambda': config['replay_lambda'],
        'loss': ('mean per-question mean assistant-token CE(object)'
                 + (' + lambda1 mean per-question mean CE(TextVQA replay)' if consumed_replay else '')),
        'normalization': 'object/replay independently divided by actual real questions in each update, including final short object block',
        'checkpoint_selection': 'fixed Final@1 epoch; no epoch/dev/test selection',
        'numerical_path': 'original_bf16', 'base_precision': 'bf16', 'lora_parameter_precision': 'fp32',
        'lora_initialization': 'initial SFT adapter rounded to BF16 then exactly promoted to FP32',
        'autocast': 'bf16', 'dropout': 0., 'visual_token_budget': config['visual_token_budget'],
        'object_prompt_suffix': config['object_prompt_suffix'], 'replay_prompt_suffix': config['replay_prompt_suffix'],
        'max_sequence_length': config['max_sequence_length'], 'input_truncation': False,
        'equal_compute_claim': False,
        'runtime_note': 'explicit microbatch/accumulation selection; no bitwise equivalence or measured speedup claim across runtimes',
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
    lock_json(directory / 'TRAIN_INPUT_LOCK.json', lock)
    lineage_sha = content_sha(lock)
    preflight_started = time.monotonic()
    preflight_result = preflight(model, parameters, processor, objects, replays, blocks[0], config, args.condition,
                                directory, device, frozen_sha256=frozen_before)
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
            if source == 'replay' and args.condition == 'object_only':
                continue
            phase = time.monotonic()
            backward_update(model, processor, objects, replays, blocks[step], args.condition, config, device,
                            stats, teaching_records, sources=(source,))
            sync(device)
            stats[f'{source}_active_seconds'] += time.monotonic() - phase
        check_gradients(model, parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        sync(device)
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
        'mean_replay_ce': stats['replay_ce_sum'] / stats['replay_exposures'] if stats['replay_exposures'] else None,
        'training_active_seconds': stats['training_seconds'], 'model_load_seconds': loading_seconds,
        'preflight_seconds_current_session': preflight_seconds, 'preflight': preflight_result,
        'session_wall_seconds': time.monotonic() - session_started,
        'lora_delta_l2': math.sqrt(delta_sq), 'lora_delta_max_abs': maximum, 'changed_lora_tensors': changed,
        'final_lora_tensor_sha256': trainable_hash(model), 'frozen_parameter_sha256_after': frozen_after,
        'frozen_parameters_unchanged': True, 'resumed_at_update': completed,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'adapter_files_sha256': file_hashes(directory / 'adapter'),
        'compute_accounting': {'equal_compute_claim': False, 'measured_flops': None,
            'flops_status': 'not instrumented; separate QA/token/time counts are not equal-FLOPs evidence',
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
    seed = parser.add_mutually_exclusive_group(required=True)
    seed.add_argument('--seed42', dest='seed', action='store_const', const=42)
    seed.add_argument('--seed', type=int)
    parser.add_argument('--output-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
