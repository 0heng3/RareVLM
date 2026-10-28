"""Single-device matched SFT/DPO, fixed final epoch, atomic update-boundary resume.

No generation benchmark is read here. A manifest explicitly admitted by the
separate data protocol is required before loading a model. The reference cache
is produced with the initial SFT weights, before any optimizer can update them.
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

import torch

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, to_device, write_json
from .loss import StaticReference, normalize_pair_loss, pair_objective, sequence_logps

PRIVATE = Path('/opt/data/private')
SCHEMA = 'rarevlm.vlm_posttrain.v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def private_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError(f'artifact outside persistent workspace: {path}')
    return path


def validate_config(config):
    fixed = {'schema_version': SCHEMA, 'prompt_suffix': '', 'training_microbatch': 1,
             'expected_trainable': 1843200, 'checkpoint_every_updates': 25}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'posttraining fixed contract differs: {key}')
    for key in ('train_manifest', 'data_admission', 'reference_config', 'model_root', 'initial_adapter'):
        if not config.get(key):
            raise ValueError(f'missing required path: {key}')
        private_path(config[key])
    for key in ('epochs', 'training_accumulation', 'max_sequence_length', 'visual_token_budget', 'preflight_pairs'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'require positive integer: {key}')
    if config['gradient_checkpointing'] is not True:
        raise ValueError('this runner requires the declared non-reentrant gradient checkpointing')
    for key in ('lr', 'beta', 'gradient_clip_norm'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'require finite positive value: {key}')
    if not 0 <= config['warmup_ratio'] < 1 or config['weight_decay'] < 0:
        raise ValueError('invalid optimizer schedule')
    return config


def read_training_rows(config):
    """Recheck admission and actual RGB pixels; no truncate/drop/relabel path."""
    from PIL import Image
    manifest = private_path(config['train_manifest'])
    admission_path = private_path(config['data_admission'])
    admission = json.loads(admission_path.read_text())
    if admission.get('status') != 'admitted' or admission.get('train_manifest_sha256') != sha_file(manifest):
        raise ValueError('training data is not admitted, or admitted manifest changed')
    scope = admission.get('scope', 'qualified_factual_preference')
    if scope == 'noisy_public_preference_diagnostic':
        if (config.get('study_scope') != scope or admission.get('factual_quality_gate_passed') is not False
                or admission.get('automatic_promotion_allowed') is not False):
            raise ValueError('noise diagnostic must preserve failed factual gate and prohibit promotion')
    elif scope != 'qualified_factual_preference' or admission.get('factual_quality_gate_passed') is not True:
        raise ValueError('qualified factual training requires its quality gate;not a diagnostic override')
    document = json.loads(manifest.read_text())
    rows = document.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected nonempty manifest rows')
    ids, images, checked = set(), {}, {}
    for row in rows:
        for key in ('question_id', 'image_id', 'question', 'chosen', 'rejected', 'image_path', 'pixel_sha256'):
            if key not in row or not str(row[key]).strip():
                raise ValueError(f'missing preference field: {key}')
        qid = str(row['question_id'])
        if qid in ids or row['chosen'] == row['rejected']:
            raise ValueError('duplicate question or identical preference completions')
        ids.add(qid)
        if row.get('prompt_suffix', '') != config['prompt_suffix']:
            raise ValueError('row prompt suffix differs from frozen preference prompt')
        if not Path(row['image_path']).is_absolute():
            raise ValueError('manifest image_path must be absolute')
        path = private_path(row['image_path'])
        if str(path) not in checked:
            with Image.open(path) as opened:
                rgb = opened.convert('RGB')
                pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            checked[str(path)] = {'pixel_sha256': pixel, 'file_sha256': sha_file(path)}
        binding = checked[str(path)]
        if row['pixel_sha256'] != binding['pixel_sha256']:
            raise ValueError('training image RGB pixels changed')
        images[str(path)] = binding
    return rows, {'train_manifest_sha256': sha_file(manifest),
                  'data_admission_sha256': sha_file(admission_path), 'image_bindings': images}


def update_blocks(count, seed, epochs, accumulation):
    """Both conditions see every chosen QA once/epoch and have identical updates."""
    if min(count, epochs, accumulation) <= 0:
        raise ValueError('empty schedule')
    blocks = []
    for epoch in range(epochs):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        blocks.extend({'epoch': epoch, 'indices': order[start:start + accumulation]}
                      for start in range(0, count, accumulation))
    return blocks


def learning_rate(step, total, config):
    warm = max(math.ceil(total * config['warmup_ratio']), 1)
    factor = (step + 1) / warm if step < warm else (total - step) / max(total - warm, 1)
    return config['lr'] * factor


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def trainable_hash(model):
    return tensor_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)


def frozen_hash(model):
    return tensor_hash((n, p) for n, p in model.named_parameters() if not p.requires_grad)


def batch_hash(batch):
    return tensor_hash((key, batch[key]) for key in sorted(batch))


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
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


def file_hashes(directory):
    return {str(path.relative_to(directory)): sha_file(path)
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def source_hashes():
    paths = set((SOURCE_ROOT / 'vlm_posttrain').glob('*.py'))
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


def validate_checkpoint_lineage(state, lineage_sha256, completed_limit):
    if state.get('lineage_sha256') != lineage_sha256:
        raise ValueError('checkpoint lineage differs; refusing mixed initialization/data/code/hardware')
    step = state.get('completed_updates')
    if type(step) is not int or not 0 <= step <= completed_limit:
        raise ValueError('invalid checkpoint update cursor')
    if state.get('accumulated_pairs', 0) != 0:
        raise ValueError('checkpoint is not at an optimizer boundary')
    return step


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
            # Preserve precisely the frozen BF16 evaluation initialisation, but
            # update FP32 master LoRA parameters under BF16 autocast.
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if sum(p.numel() for p in parameters) != config['expected_trainable']:
        raise ValueError('unexpected language q/v rank8 LoRA parameter budget')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    return model, parameters, parameter_dtypes(model)


def encode_pair(processor, row, config):
    samples = [encode(processor, {'question': row['question'], 'target': row[key],
                                  'prompt_suffix': config['prompt_suffix']}, row['image_path'], True)
               for key in ('chosen', 'rejected')]
    for sample in samples:
        if sample['input_ids'].shape[1] > config['max_sequence_length']:
            raise ValueError(f'completion exceeds frozen sequence budget: {row["question_id"]}')
        grid = sample['image_grid_thw']
        if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError('expected one static image')
        visual = int(grid.prod()) // 4
        if int(grid.prod()) % 4 or not 0 < visual <= config['visual_token_budget']:
            raise ValueError('visual token budget differs')
    batch = collate(samples, processor.tokenizer.pad_token_id)
    return samples, batch, batch_hash(batch)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reference_cache(model, processor, rows, indices, config, directory, lineage_sha256, device):
    """Append-only completed-pair records; all forwards precede any actor update."""
    path = directory / 'reference_cache.jsonl'
    selected = [rows[index] for index in indices]
    lock = {'lineage_sha256': lineage_sha256, 'question_ids': [r['question_id'] for r in selected],
            'source': 'static_initial_SFT_BF16_rounded_FP32_LoRA',
            'pair_batch_size': 2, 'logp': 'sum of assistant token log probabilities',
            'autocast': 'bf16', 'forward_mode': 'deterministic actor training mode; graph discarded without backward'}
    lock_json(directory / 'REFERENCE_INPUT_LOCK.json', lock)
    records = []
    if path.exists():
        raw = path.read_bytes().splitlines(keepends=True)
        if raw and not raw[-1].endswith(b'\n'):
            raw.pop()
            path.write_bytes(b''.join(raw))
        records = [json.loads(line) for line in raw]
    if len(records) > len(selected):
        raise ValueError('too many cached reference pairs')
    for item, row in zip(records, selected):
        if item['question_id'] != row['question_id'] or item['reference_lock_sha256'] != content_sha(lock):
            raise ValueError('reference cache lineage/order changed')
    completion_path = directory / 'REFERENCE_RESULT.json'
    if completion_path.exists():
        completed = json.loads(completion_path.read_text())
        if completed['cache_sha256'] != sha_file(path) or completed['pairs'] != len(selected):
            raise ValueError('completed static reference cache changed')
    initial_hash = trainable_hash(model)
    # Use the actor's exact deterministic forward path (dropout is zero). This
    # avoids an inference-only kernel changing initial sequence sums. Graphs
    # are discarded immediately and no backward/optimizer exists in this phase.
    model.train()
    sync(device)
    begun = time.monotonic()
    with path.open('a') as stream, torch.enable_grad():
        for row in selected[len(records):]:
            _, batch, input_sha = encode_pair(processor, row, config)
            batch = to_device(batch, device)
            sync(device)
            start = time.monotonic()
            with autocast(device):
                values, counts = sequence_logps(model, batch)
            sync(device)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite initial-SFT reference')
            item = {'question_id': row['question_id'], 'reference_lock_sha256': content_sha(lock),
                    'input_sha256': input_sha, 'logps': values.detach().cpu().tolist(), 'counts': counts.cpu().tolist(),
                    'input_tokens': int(batch['attention_mask'].sum()),
                    'forward_seconds': time.monotonic() - start}
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            records.append(item)
            del values, counts, batch
            if len(records) == 1 or len(records) % 25 == 0 or len(records) == len(selected):
                print('STATIC_REFERENCE', len(records), '/', len(selected), flush=True)
                write_json(directory / 'PROGRESS.json', {'status': 'static_reference',
                           'completed_reference_pairs': len(records), 'planned_reference_pairs': len(selected), 'updated_at': now()})
    sync(device)
    if trainable_hash(model) != initial_hash:
        raise ValueError('initial actor changed during static reference construction')
    model.train()
    result = {'cache_sha256': sha_file(path), 'pairs': len(records),
               'input_tokens': sum(x['input_tokens'] for x in records),
               'answer_tokens': sum(sum(x['counts']) for x in records),
               'forward_seconds': sum(x['forward_seconds'] for x in records)}
    lock_json(completion_path, result)
    return StaticReference(records), {**result, 'current_session_wall_seconds': time.monotonic() - begun}


def preflight(model, parameters, processor, rows, indices, config, condition, cache, directory, device,
              reference_directory=None, frozen_sha256=None):
    """Real finite gradients, an update against an immutable reference, exact reset."""
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    before_hash = trainable_hash(model)
    frozen_before = frozen_sha256 or frozen_hash(model)
    state = rng_state()
    reference_path = (reference_directory or directory) / 'reference_cache.jsonl'
    reference_before = sha_file(reference_path)
    results = []
    steps_executed = 0
    error = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    engineering_lr = config.get('preflight_lr', config['lr'])
    optimizer = torch.optim.AdamW(parameters, lr=engineering_lr, weight_decay=config['weight_decay'])
    try:
        for index in indices:
            row = rows[index]
            samples, pair, input_sha = encode_pair(processor, row, config)
            reference, expected_counts = cache.lookup(row['question_id'], input_sha, device)
            with torch.enable_grad(), autocast(device):
                identity, identity_counts = sequence_logps(model, to_device(pair, device))
            identity = identity.detach()
            # Each preflight example begins at the exact initial SFT state.
            if not torch.equal(identity_counts, expected_counts) or not torch.allclose(identity, reference, atol=1e-5, rtol=0):
                raise ValueError('actor and static initial-SFT reference disagree at initialization')
            batch = pair if condition == 'dpo' else collate(samples[:1], processor.tokenizer.pad_token_id)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                values, counts = sequence_logps(model, batch)
                loss = pair_objective(condition, values, counts, reference if condition == 'dpo' else None, config['beta'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite preflight loss')
            loss.backward()
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
                raise FloatingPointError('preflight has absent/nonfinite LoRA gradients')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('frozen parameter received a gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
            if float(norm) <= 0:
                raise ValueError('preflight has zero gradient')
            optimizer.step()
            steps_executed += 1
            # Keep the same SDPA/checkpoint forward path as the initial cache;
            # an inference-only path must not masquerade as an optimizer effect.
            with torch.enable_grad(), autocast(device):
                updated, _ = sequence_logps(model, to_device(pair, device))
            updated = updated.detach()
            still_reference, _ = cache.lookup(row['question_id'], input_sha, device)
            if not torch.equal(reference, still_reference) or sha_file(reference_path) != reference_before:
                raise ValueError('static reference changed after actor update')
            delta = float((updated - reference).abs().max())
            if trainable_hash(model) == before_hash or not math.isfinite(delta) or delta <= 0:
                raise ValueError('preflight update did not separate actor from static reference')
            results.append({'question_id': row['question_id'], 'loss': float(loss.detach()),
                            'gradient_l2_before_clip': float(norm), 'finite_gradients': True,
                            'actor_reference_max_logp_difference_after_update': delta,
                            'static_reference_unchanged': True, 'trainable_parameters': sum(p.numel() for p in parameters)})
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in initial:
                        parameter.copy_(initial[name])
            optimizer.state.clear()
    except Exception as exc:
        error = exc
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        restore_rng(state)
    if trainable_hash(model) != before_hash:
        raise ValueError('preflight failed to restore exact initialization')
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError('preflight changed frozen base/vision weights')
    result = {'status': 'failed' if error else 'passed', 'condition': condition, 'completed_at': now(), 'records': results,
              'scope': 'engineering contract; all provisional updates and RNG restored',
              'discarded_engineering_optimizer_updates': steps_executed,
              'engineering_learning_rate': engineering_lr,
              'formal_learning_rate': config['lr'],
              'formal_optimizer_updates': 0, 'engineering_optimizer_discarded': True,
              'initial_parameters_restored_exactly': True,
              'frozen_parameter_sha256_before': frozen_before, 'frozen_parameter_sha256_after': frozen_after,
              'frozen_parameters_unchanged': True,
              'reference_cache_sha256': reference_before,
              'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0}
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


def save_checkpoint(directory, model, optimizer, completed, stats, lineage_sha256):
    # A crash after directory rename but before pointer rename may leave an
    # orphan; a unique directory permits deterministic replay from old LATEST.
    destination = directory / f'checkpoint_{completed:06d}_{time.time_ns()}'
    staging = directory / f'.{destination.name}.tmp'
    staging.mkdir()
    model.save_pretrained(staging / 'adapter', safe_serialization=True)
    state = {'lineage_sha256': lineage_sha256, 'completed_updates': completed,
             'accumulated_pairs': 0, 'optimizer': cpu_state(optimizer.state_dict()),
             'rng': rng_state(), 'stats': dict(stats), 'trainable_tensor_sha256': trainable_hash(model)}
    torch.save(state, staging / 'state.pt')
    record = {'directory': destination.name, 'lineage_sha256': lineage_sha256,
              'completed_updates': completed, 'state_sha256': sha_file(staging / 'state.pt'),
              'adapter_files_sha256': file_hashes(staging / 'adapter')}
    staging.replace(destination)
    write_json(directory / 'LATEST.json', record)


def resume_checkpoint(directory, model, optimizer, lineage_sha256, planned):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    latest = json.loads((directory / 'LATEST.json').read_text())
    checkpoint = directory / latest['directory']
    if latest.get('lineage_sha256') != lineage_sha256 or sha_file(checkpoint / 'state.pt') != latest['state_sha256']:
        raise ValueError('checkpoint pointer/state binding changed')
    if file_hashes(checkpoint / 'adapter') != latest['adapter_files_sha256']:
        raise ValueError('checkpoint adapter changed')
    state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    step = validate_checkpoint_lineage(state, lineage_sha256, planned)
    if step != latest['completed_updates']:
        raise ValueError('checkpoint update cursor differs from pointer')
    adapter_state = load_file(str(checkpoint / 'adapter/adapter_model.safetensors'))
    expected = get_peft_model_state_dict(model)
    if set(adapter_state) != set(expected) or any(adapter_state[key].shape != expected[key].shape for key in expected):
        raise ValueError('checkpoint adapter tensor keys/shapes differ')
    result = set_peft_model_state_dict(model, adapter_state)
    if result.unexpected_keys:
        raise ValueError('checkpoint has unexpected adapter tensors')
    if trainable_hash(model) != state['trainable_tensor_sha256']:
        raise ValueError('restored LoRA tensors differ from checkpoint')
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return step, state['stats']


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    session_started = time.monotonic()
    config_path = private_path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    rows, data_binding = read_training_rows(config)
    root = private_path(args.output_root)
    directory = root / 'runs' / args.condition / f'seed{args.seed}' / 'train'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'LATEST.json').exists() and not args.resume and not args.preflight:
        raise ValueError('checkpoint exists; use --resume explicitly')
    if args.resume and not (directory / 'LATEST.json').exists():
        raise ValueError('--resume requires an existing checkpoint')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('matched pair runner is single device only')
    hardware = visible_gpu_binding()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('formal runner requires a BF16-capable CUDA device')
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
    blocks = update_blocks(len(rows), args.seed, config['epochs'], config['training_accumulation'])
    model_root = private_path(config['model_root'])
    adapter_root = private_path(config['initial_adapter'])
    adapter_config = json.loads((adapter_root / 'adapter_config.json').read_text())
    if adapter_config['r'] != 8 or adapter_config['lora_alpha'] != 16 or adapter_config['bias'] != 'none':
        raise ValueError('initial SFT adapter is not the frozen q/v rank8 alpha16 adapter')
    provenance_path = model_root.parent / 'model_provenance.json'
    static_lock = {'schema_version': SCHEMA, 'condition': args.condition, 'seed': args.seed,
           'config_sha256': content_sha(config), 'config_file_sha256': sha_file(config_path),
           'reference_config_sha256': sha_file(reference_path), 'data_binding': data_binding,
           'train_manifest_sha256': data_binding['train_manifest_sha256'],
           'source_sha256': source_hashes(), 'packages': package_lock(), 'hardware': hardware,
           'model_files_sha256': file_hashes(model_root),
           'model_provenance_sha256': sha_file(provenance_path),
           'initial_adapter_files_sha256': file_hashes(adapter_root),
           'question_count': len(rows), 'unique_training_images': len({str(r['image_id']) for r in rows}),
           'planned_updates': len(blocks), 'planned_chosen_exposures': len(rows) * config['epochs'],
           'order_sha256': content_sha([[rows[i]['question_id'] for i in b['indices']] for b in blocks]),
           'global_pair_order_sha256': content_sha([[rows[i]['question_id'] for i in b['indices']] for b in blocks]),
           'training_microbatch_pairs': 1, 'training_accumulation': config['training_accumulation'],
           'normalization': 'sum of pair objectives divided by actual real pairs in each update',
           'checkpoint_selection': 'fixed final epoch; no evaluation-based choice',
           'numerical_path': 'original_bf16', 'base_precision': 'bf16', 'lora_parameter_precision': 'fp32',
           'lora_initialization': 'initial SFT adapter rounded to BF16, then exactly promoted to FP32',
           'autocast': 'bf16', 'dropout': 0., 'visual_token_budget': config['visual_token_budget']}
    lock_json(directory / 'STATIC_INPUT_LOCK.json', static_lock)
    loading_start = time.monotonic()
    model, parameters, dtypes = load_actor(config, reference, device)
    processor = processor_for(model_root, reference, config['visual_token_budget'])
    sync(device)
    loading_seconds = time.monotonic() - loading_start
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    initial_sha = trainable_hash(model)
    frozen_before = frozen_hash(model)
    lock = {**static_lock, 'initial_adapter_tensor_sha256': initial_sha,
            'frozen_parameter_sha256_before': frozen_before, 'dtypes': dtypes,
            'trainable_parameters': sum(p.numel() for p in parameters),
            'trainable_names': {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}}
    lock_json(directory / 'TRAIN_INPUT_LOCK.json', lock)
    lineage_sha = content_sha(lock)
    first_indices = [i for block in blocks for i in block['indices']][:min(config['preflight_pairs'], len(rows))]
    # A standalone preflight has its own cache, so it cannot restrict or mutate
    # the later full DPO reference namespace.
    cache_directory = directory / 'engineering_preflight' if args.preflight else directory
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_indices = list(range(len(rows))) if args.condition == 'dpo' and not args.preflight else first_indices
    cache, reference_cost = reference_cache(model, processor, rows, cache_indices, config, cache_directory, lineage_sha, device)
    preflight_start = time.monotonic()
    preflight_result = preflight(model, parameters, processor, rows, first_indices, config, args.condition, cache,
                                directory, device, reference_directory=cache_directory, frozen_sha256=frozen_before)
    sync(device)
    preflight_seconds = time.monotonic() - preflight_start
    if args.preflight:
        print('PREFLIGHT_PASSED', args.condition, args.seed, flush=True)
        return
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    stats = {'questions': 0, 'chosen_exposures': 0, 'rejected_exposures': 0, 'input_tokens': 0,
             'chosen_answer_tokens': 0, 'rejected_answer_tokens': 0, 'loss_sum': 0.,
             'training_seconds': 0., 'last_gradient_l2': None, 'chosen_input_contracts': []}
    completed = 0
    if args.resume:
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, len(blocks))
        expected_exposures = sum(len(block['indices']) for block in blocks[:completed])
        if (stats['chosen_exposures'] != expected_exposures or stats['questions'] != expected_exposures
                or len(stats['chosen_input_contracts']) != expected_exposures
                or stats['rejected_exposures'] != expected_exposures * int(args.condition == 'dpo')):
            raise ValueError('checkpoint exposure cursor differs from frozen QA/update order')
    if (directory / 'TRAIN_RESULT.json').exists():
        result = json.loads((directory / 'TRAIN_RESULT.json').read_text())
        if result['lineage_sha256'] != lineage_sha or file_hashes(directory / 'adapter') != result['adapter_files_sha256']:
            raise ValueError('completed adapter/lineage changed')
        return
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(completed, len(blocks)):
        sync(device)
        begun = time.monotonic()
        indices = blocks[step]['indices']
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, len(blocks), config)
        for index in indices:
            row = rows[index]
            samples, pair, input_sha = encode_pair(processor, row, config)
            chosen_batch = collate(samples[:1], processor.tokenizer.pad_token_id)
            stats['chosen_input_contracts'].append(content_sha({'question_id': row['question_id'],
                                                               'input_sha256': batch_hash(chosen_batch)}))
            if args.condition == 'dpo':
                values_reference, cached_counts = cache.lookup(row['question_id'], input_sha, device)
                batch = pair
            else:
                values_reference = None
                batch = chosen_batch
            batch = to_device(batch, device)
            with autocast(device):
                values, counts = sequence_logps(model, batch)
                if args.condition == 'dpo' and not torch.equal(counts, cached_counts):
                    raise ValueError('actor/reference assistant mask differs')
                pair_loss = pair_objective(args.condition, values, counts, values_reference, config['beta'])
                loss = normalize_pair_loss(pair_loss, len(indices))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite training objective')
            loss.backward()
            stats['questions'] += 1
            stats['chosen_exposures'] += 1
            stats['rejected_exposures'] += int(args.condition == 'dpo')
            stats['input_tokens'] += int(batch['attention_mask'].sum())
            stats['chosen_answer_tokens'] += int(counts[0])
            stats['rejected_answer_tokens'] += int(counts[1]) if args.condition == 'dpo' else 0
            stats['loss_sum'] += float(pair_loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        stats['training_seconds'] += time.monotonic() - begun
        stats['last_gradient_l2'] = float(norm)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == len(blocks):
            print('POSTTRAIN_UPDATE', json.dumps({'condition': args.condition, 'seed': args.seed,
                  'completed_updates': step + 1, 'planned_updates': len(blocks), 'chosen_exposures': stats['chosen_exposures'],
                  'mean_pair_objective': stats['loss_sum'] / stats['questions'], 'gradient_l2': float(norm),
                  'training_seconds': stats['training_seconds']}), flush=True)
        if (step + 1) % config['checkpoint_every_updates'] == 0 or step + 1 == len(blocks):
            save_checkpoint(directory, model, optimizer, step + 1, stats, lineage_sha)
            write_json(directory / 'PROGRESS.json', {'status': 'training', 'completed_updates': step + 1,
                       'planned_updates': len(blocks), 'actual_chosen_exposures': stats['chosen_exposures'], 'updated_at': now()})
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError('frozen base/vision weights changed')
    if stats['chosen_exposures'] != len(rows) * config['epochs']:
        raise ValueError('chosen exposure contract failed')
    delta_sq, maximum, changed = 0., 0., 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().cpu()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('nonfinite final LoRA')
            delta = value - initial[name]
            delta_sq += float(delta.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            changed += int(bool((delta != 0).any()))
    if changed == 0 or sha_file(directory / 'reference_cache.jsonl') != reference_cost['cache_sha256']:
        raise ValueError('actor unchanged or static reference modified')
    model.save_pretrained(directory / 'adapter', safe_serialization=True)
    result = {**lock, 'lineage_sha256': lineage_sha, 'status': 'completed', 'completed_at': now(),
              'completed_updates': len(blocks), 'epochs_completed': config['epochs'],
              'actual_questions_seen': stats['questions'], 'actual_chosen_exposures': stats['chosen_exposures'],
              'actual_rejected_exposures': stats['rejected_exposures'], 'skipped_updates': 0,
              'chosen_input_contract_sha256': content_sha(stats['chosen_input_contracts']),
              'training_input_tokens': stats['input_tokens'], 'training_chosen_answer_tokens': stats['chosen_answer_tokens'],
              'training_rejected_answer_tokens': stats['rejected_answer_tokens'],
              'mean_pair_objective': stats['loss_sum'] / stats['questions'], 'training_active_seconds': stats['training_seconds'],
              'model_load_seconds': loading_seconds, 'preflight_seconds_current_session': preflight_seconds,
              'reference_cost': reference_cost, 'preflight': preflight_result,
              'session_wall_seconds': time.monotonic() - session_started,
              'lora_delta_l2': math.sqrt(delta_sq), 'lora_delta_max_abs': maximum, 'changed_lora_tensors': changed,
              'final_lora_tensor_sha256': trainable_hash(model), 'frozen_parameter_sha256_after': frozen_after,
              'frozen_parameters_unchanged': True, 'resumed_at_update': completed,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
              'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
              'adapter_files_sha256': file_hashes(directory / 'adapter')}
    write_json(directory / 'TRAIN_RESULT.json', result)
    write_json(directory / 'PROGRESS.json', {'status': 'completed', 'completed_updates': len(blocks),
               'planned_updates': len(blocks), 'actual_chosen_exposures': stats['chosen_exposures'], 'updated_at': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--condition', choices=('sft', 'dpo'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
