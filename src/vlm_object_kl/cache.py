"""Atomic immutable full-assistant FP32 teacher cache with actual-input locks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import torch

from vlm_deep.common import content_sha, sha_file, to_device, write_json
from vlm_object_train.train import (autocast, cpu_state, file_hashes, frozen_hash, lock_json,
    restore_rng, restore_runtime, rng_hash, rng_state, runtime_state, sync, tensor_hash, trainable_hash)
from .inputs import cache_key, encode_microbatch
from .loss import assistant_logprobs

SCHEMA = 'rarevlm.object_kl.teacher_cache.v1'


def validate_payload(payload, identity):
    if payload.get('identity') != identity:
        raise ValueError('teacher cache input/initial/source/config/UUID identity differs')
    values, counts = payload.get('logprobs'), payload.get('counts')
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.ndim != 2
            or values.requires_grad or values.device.type != 'cpu'
            or not isinstance(counts, torch.Tensor) or counts.ndim != 1
            or counts.dtype != torch.int64 or not bool((counts > 0).all())
            or int(counts.sum()) != values.shape[0] or values.shape[1] <= 1):
        raise ValueError('teacher cache must hold detached CPU FP32 full vocabulary rows')
    if not bool(torch.isfinite(values).all()) or not torch.allclose(
            torch.logsumexp(values, -1), torch.zeros(values.shape[0]), atol=1e-5, rtol=0):
        raise ValueError('teacher cache contains nonfinite or unnormalized log probabilities')
    if payload.get('tensor_sha256') != tensor_hash([('logprobs', values), ('counts', counts)]):
        raise ValueError('teacher cache tensor SHA changed')
    return payload


def load_entry(path, identity, entry=None):
    if entry is not None and sha_file(path) != entry['file_sha256']:
        raise ValueError('immutable teacher cache file SHA changed')
    payload = validate_payload(torch.load(path, map_location='cpu', weights_only=False), identity)
    if entry is not None and payload['tensor_sha256'] != entry['tensor_sha256']:
        raise ValueError('immutable teacher cache tensor binding changed')
    return payload


def verify_frozen_cache(directory, manifest_sha=None):
    path = Path(directory) / 'MANIFEST.json'
    if manifest_sha is not None and sha_file(path) != manifest_sha:
        raise ValueError('frozen teacher cache manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != SCHEMA or manifest.get('status') != 'frozen':
        raise ValueError('teacher cache is not completely frozen')
    binding = json.loads((Path(directory) / 'INPUT_LOCK.json').read_text())
    if content_sha(binding) != manifest['binding_sha256']:
        raise ValueError('teacher cache complete input/source/initial/UUID binding changed')
    if (manifest.get('full_vocabulary') is not True or manifest.get('logprob_dtype') != 'float32'
            or manifest.get('temperature') != 1. or manifest.get('direction') != 'initial_to_actor'
            or manifest.get('teacher_parameters_sha256') != binding['initial_adapter_tensor_sha256']
            or manifest.get('frozen_parameters_sha256') != binding['frozen_parameter_sha256_before']):
        raise ValueError('teacher cache scientific precision/distribution contract differs')
    micro = binding['training_microbatch_questions']
    per_update = binding['replay_questions_per_update']
    expected_keys = {cache_key(update, sub) for update in range(binding['planned_updates'])
                     for sub in range((per_update + micro - 1) // micro)}
    if set(manifest['entries']) != expected_keys:
        raise ValueError('teacher cache occurrence schedule is incomplete')
    grouped_ids, grouped_micro_ids = [], []
    for update in range(binding['planned_updates']):
        question_ids, chunks = [], []
        for sub in range((per_update + micro - 1) // micro):
            entry = manifest['entries'][cache_key(update, sub)]
            identity = entry['identity']
            labels = torch.tensor(identity['actual_assistant_labels'], dtype=torch.int64)
            counts = (labels != -100).sum(1).tolist()
            if (identity['binding_sha256'] != manifest['binding_sha256']
                    or identity['update_index'] != update or identity['replay_microbatch_index'] != sub
                    or counts != identity['assistant_tokens']
                    or len(counts) != len(identity['question_ids'])
                    or entry['shape'][0] != sum(counts)
                    or entry['shape'][1] != manifest['vocabulary_size']):
                raise ValueError('teacher cache input/assistant positions/vocabulary shape differs')
            question_ids.extend(identity['question_ids'])
            chunks.append(identity['question_ids'])
        grouped_ids.append(question_ids)
        grouped_micro_ids.append(chunks)
    if (content_sha(grouped_ids) != binding['replay_order_sha256']
            or content_sha(grouped_micro_ids) != binding['replay_microbatch_order_sha256']
            or sum(map(len, grouped_ids)) != manifest['teacher_forward_questions']
            or manifest['teacher_forward_questions'] != binding['planned_replay_exposures']):
        raise ValueError('teacher cache complete replay QA order/exposure differs')
    for key, entry in manifest['entries'].items():
        path = Path(directory) / f'{key}.pt'
        if sha_file(path) != entry['file_sha256']:
            raise ValueError('frozen teacher cache file changed')
    return manifest


def entry_identity(binding, update, microbatch, batch, contracts):
    return {'binding_sha256': content_sha(binding), 'update_index': update,
            'replay_microbatch_index': microbatch,
            'actual_padded_input_sha256': contracts[0]['microbatch_input_sha256'],
            'question_ids': [item['question_id'] for item in contracts],
            'assistant_tokens': [item['assistant_tokens'] for item in contracts],
            'actual_input_ids': batch['input_ids'].tolist(),
            'actual_attention_mask': batch['attention_mask'].tolist(),
            'actual_assistant_labels': batch['labels'].tolist(),
            'actual_image_grid_thw': batch['image_grid_thw'].tolist(),
            'tensor_sha256_by_key': {key: tensor_hash([(key, value)]) for key, value in batch.items()},
            'tensor_shapes_by_key': {key: list(value.shape) for key, value in batch.items()},
            'input_contracts': contracts}


def build_teacher_cache(model, processor, replays, blocks, config, directory, binding, device):
    """Run grad-enabled frozen-initial forwards, then restore all mutable state.

    Each atomic batch can be reused after interruption only if its complete
    input and teacher/source/config/UUID identity agrees. A frozen cache rejects
    corruption; it is never silently regenerated against a current actor.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A caller cannot pass a current actor while retaining an old binding.
    # Validate before either generation or reuse of an existing frozen cache.
    if (trainable_hash(model) != binding['initial_adapter_tensor_sha256']
            or frozen_hash(model) != binding['frozen_parameter_sha256_before']):
        raise ValueError('teacher construction requires exact fixed initial and frozen base tensors')
    lock_json(directory / 'INPUT_LOCK.json', binding)
    frozen_manifest = directory / 'MANIFEST.json'
    if frozen_manifest.exists():
        manifest = verify_frozen_cache(directory)
        if manifest['binding_sha256'] != content_sha(binding):
            raise ValueError('frozen teacher lineage differs')
        return manifest
    before, frozen_before = trainable_hash(model), frozen_hash(model)
    initial = {name: cpu_state(p) for name, p in model.named_parameters() if p.requires_grad}
    rng, runtime = rng_state(), runtime_state(model)
    begun = time.monotonic()
    entries, estimated_bytes, teacher_seconds = {}, 0, 0.
    reserve = config['teacher_cache_disk_reserve_bytes']
    vocab = int(model.get_output_embeddings().weight.shape[0]) if hasattr(model.get_output_embeddings(), 'weight') else int(model.config.vocab_size)
    # Exact assistant-token count from original CPU encoder, before GPU cache.
    plans = []
    for update, block in enumerate(blocks):
        for microbatch, start in enumerate(range(0, len(block['replay_indices']), config['training_microbatch'])):
            selected = block['replay_indices'][start:start + config['training_microbatch']]
            batch, contracts, _ = encode_microbatch(processor, replays, selected, config)
            bytes_needed = sum(item['assistant_tokens'] for item in contracts) * vocab * 4
            estimated_bytes += bytes_needed
            plans.append((update, microbatch, selected, bytes_needed))
    existing_bytes = sum(path.stat().st_size for path in directory.glob('update_*.pt'))
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes < max(estimated_bytes - existing_bytes, 0) + reserve:
        raise RuntimeError('insufficient persistent disk for full FP32 vocabulary cache plus reserve')
    write_json(directory / 'ESTIMATE.json', {'assistant_logprob_bytes': estimated_bytes, 'vocabulary_size': vocab,
        'replay_microbatches': len(plans), 'existing_file_bytes': existing_bytes,
        'free_disk_bytes': free_bytes, 'disk_reserve_bytes': reserve, 'dtype': 'float32',
        'full_vocabulary': True, 'no_top_k': True, 'no_quantization': True})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    error = None
    try:
        for update, microbatch, selected, _ in plans:
            key = cache_key(update, microbatch)
            batch, contracts, _ = encode_microbatch(processor, replays, selected, config)
            identity = entry_identity(binding, update, microbatch, batch, contracts)
            path = directory / f'{key}.pt'
            if path.exists():
                payload = load_entry(path, identity)
            else:
                sync(device)
                forward_begun = time.monotonic()
                gpu_batch = to_device(batch, device)
                with torch.enable_grad(), autocast(device):
                    values, counts, _ = assistant_logprobs(model, gpu_batch)
                values = values.detach().cpu().contiguous()
                counts = counts.detach().cpu().to(torch.int64)
                sync(device)
                seconds = time.monotonic() - forward_begun
                teacher_seconds += seconds
                if values.shape[1] != vocab:
                    raise ValueError('teacher head did not return complete vocabulary')
                payload = {'identity': identity, 'logprobs': values, 'counts': counts,
                    'tensor_sha256': tensor_hash([('logprobs', values), ('counts', counts)]),
                    'forward_seconds': seconds}
                validate_payload(payload, identity)
                temporary = path.with_suffix('.pt.tmp')
                torch.save(payload, temporary)
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                temporary.replace(path)
                del gpu_batch
            entries[key] = {'identity': identity, 'file_sha256': sha_file(path),
                'tensor_sha256': payload['tensor_sha256'], 'shape': list(payload['logprobs'].shape),
                'file_bytes': path.stat().st_size, 'forward_seconds': payload['forward_seconds']}
            if (len(entries) % 25 == 0 or len(entries) == len(plans)):
                print('TEACHER_CACHE_BATCH', json.dumps({'completed_microbatches': len(entries),
                    'planned_microbatches': len(plans), 'new_forward_seconds': teacher_seconds}), flush=True)
                write_json(directory / 'PROGRESS.json', {'status': 'building', 'completed_microbatches': len(entries),
                    'planned_microbatches': len(plans)})
            if device.type == 'cuda' and max(torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device)) > config['preflight_memory_cap_gib'] * 2**30:
                raise RuntimeError('teacher_cache_memory_cap_exceeded; no runtime fallback')
    except Exception as exc:
        error = exc
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in initial:
                    p.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        restore_runtime(runtime)
        restore_rng(rng)
    if trainable_hash(model) != before or frozen_hash(model) != frozen_before or rng_hash(rng_state()) != rng_hash(rng):
        raise ValueError('teacher cache construction failed exact initial/RNG/frozen restoration')
    if error:
        write_json(directory / 'BUILD_FAILURE.json', {'status': 'failed',
            'error': f'{type(error).__name__}: {error}', 'completed_microbatches': len(entries),
            'planned_microbatches': len(plans), 'formal_optimizer_updates': 0,
            'initial_parameters_rng_buffers_rope_restored': True,
            'memory_cap_gib': config['preflight_memory_cap_gib'],
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.})
        raise error
    manifest = {'schema_version': SCHEMA, 'status': 'frozen', 'binding_sha256': content_sha(binding),
        'entries': entries, 'full_vocabulary': True, 'logprob_dtype': 'float32', 'temperature': 1.,
        'vocabulary_size': vocab,
        'direction': 'initial_to_actor', 'teacher_parameters_sha256': before,
        'frozen_parameters_sha256': frozen_before, 'assistant_logprob_bytes': estimated_bytes,
        'cache_file_bytes': sum(entry['file_bytes'] for entry in entries.values()),
        'teacher_forward_questions': sum(len(entry['identity']['question_ids']) for entry in entries.values()),
        'teacher_forward_assistant_tokens': sum(entry['shape'][0] for entry in entries.values()),
        'teacher_forward_seconds_all_batches': sum(entry['forward_seconds'] for entry in entries.values()),
        'new_forward_seconds_current_session': teacher_seconds, 'wall_seconds_current_session': time.monotonic() - begun,
        'initial_parameters_rng_buffers_rope_restored': True,
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.}
    write_json(frozen_manifest, manifest)
    write_json(directory / 'PROGRESS.json', {'status': 'frozen', 'completed_microbatches': len(entries),
        'planned_microbatches': len(plans)})
    return verify_frozen_cache(directory)


class TeacherCache:
    def __init__(self, directory, binding, manifest_sha256):
        self.directory, self.binding, self.manifest_sha256 = Path(directory), binding, manifest_sha256
        self.manifest = verify_frozen_cache(self.directory, manifest_sha256)
        if self.manifest['binding_sha256'] != content_sha(binding):
            raise ValueError('teacher cache binding changed')

    def load(self, update, microbatch, batch, contracts, device):
        key = cache_key(update, microbatch)
        identity = entry_identity(self.binding, update, microbatch, batch, contracts)
        entry = self.manifest['entries'][key]
        if entry['identity'] != identity:
            raise ValueError('teacher cache actual padded input or QA order changed')
        payload = load_entry(self.directory / f'{key}.pt', identity, entry)
        return payload['logprobs'].to(device), payload['counts'].to(device)

    def verify(self):
        return verify_frozen_cache(self.directory, self.manifest_sha256)
