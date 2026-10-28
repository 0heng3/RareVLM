"""Independent generated-prefix KL replication shards; native parent numerics unchanged.

Copied parent rollout SHA9325453769ea660b1dd4f11a19d537b971d4edc59d704842a884f5801260108f.
Only new namespace/admission and replication-only guard differ; never patches parent globals.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

import torch

from vlm_deep.common import collate, content_sha, encode, sha_file, to_device
from vlm_joint.evaluate import visible_gpu_binding
from vlm_object_train.data import private_path
from vlm_object_train.train import (batch_hash, package_lock, restore_rng,
    restore_runtime, rng_state, runtime_state, sync, tensor_hash)
from vlm_precision.evaluate import load_path
from .data import (SHARD_SCHEMA, atomic_json, check_seal, generation_row, generation_spec,
    immutable_json, load_prompt_inputs, parse_generated_suffix, prepared_dir, read_json,
    reply_dir, sealed, validate_reply, verify_prompt)


def generation_kwargs(spec):
    return {'max_new_tokens': spec['max_new_tokens'], 'do_sample': False,
        'use_cache': True, 'eos_token_id': spec['eos_token_ids'],
        'pad_token_id': spec['pad_token_id'], 'repetition_penalty': spec['repetition_penalty']}


def generation_policy(config, native, loaded=None):
    """Clone the native policy; preserve all processors beyond explicit old-eval overrides."""
    effective = deepcopy(native if loaded is None else loaded)
    loader_config = effective.to_dict()
    overrides = {'temperature': 1., 'top_p': 1., 'top_k': 50,
        **generation_kwargs(generation_spec(config))}
    for key, value in overrides.items():
        setattr(effective, key, value)
    native_file = private_path(config['model_root']) / 'generation_config.json'
    return effective, {'native_generation_config_file': str(native_file),
        'native_generation_config_file_sha256': sha_file(native_file),
        'native_generation_config': native.to_dict(), 'loader_generation_config': loader_config,
        'actual_overrides': overrides,
        'actual_generate_kwargs': generation_kwargs(generation_spec(config)),
        'effective_generation_config': effective.to_dict()}


def teacher_identity(model, config):
    """The canonical FP32 representation hashes BF16 values without changing stored dtype."""
    parameters = list(model.named_parameters())
    lora = [(name, p) for name, p in parameters if 'lora_' in name]
    if (not lora or sum(p.numel() for _, p in lora) != config['expected_trainable']
            or any('visual' in name or not any(f'.{projection}.' in name for projection in ('q_proj', 'v_proj'))
                or p.dtype != torch.bfloat16 for name, p in lora)
            or any(p.requires_grad for _, p in parameters)):
        raise ValueError('teacher requires frozen native BF16 language q/v LoRA1843200')
    return {'canonical_lora_fp32_representation_sha256': tensor_hash((name, p.detach().float()) for name, p in lora),
        'native_lora_sha256': tensor_hash(lora),
        'frozen_base_sha256': tensor_hash((name, p) for name, p in parameters if 'lora_' not in name),
        'all_frozen_parameters_sha256': tensor_hash(parameters)}


TEACHER_INFERENCE = {'numerical_path': 'original_bf16', 'autocast': False,
    'native_lora_dtype': 'torch.bfloat16', 'adapter_autocast_dtype': False,
    'explicit_lora_bf16_conversion': True, 'optimizer_updates': 0,
    'canonical_identity': 'FP32 representation of native BF16 LoRA values; parameters remain BF16'}


def event(directory, kind, **values):
    import json
    with (directory / 'EXECUTION.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': kind, 'unix_time': time.time(), **values}, ensure_ascii=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def enforce_memory(device, config):
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    if max(allocated, reserved) > config['preflight_memory_cap_gib']:
        raise RuntimeError(f'teacher generation exceeds65GiB: {allocated}/{reserved}; no fallback')
    return allocated, reserved


def validate_batch(value, binding, expected, config, vocab_size, phase='formal'):
    check_seal(value, 'batch_sha256')
    if (value.get('schema_version') != 'rarevlm.generated_prefix_kl.teacher_batch.v1'
            or value.get('binding_sha256') != content_sha(binding) or value.get('phase') != phase
            or value.get('producer_uuid') != binding['producer_uuid'] or len(value.get('records', [])) != len(expected)
            or value.get('global_batch_index') != expected[0]['global_batch_index']
            or value.get('actual_microbatch_input_sha256') != expected[0]['microbatch_input_sha256']
            or value.get('actual_padded_prompt_width') != expected[0]['padded_prompt_width']):
        raise ValueError('saved complete teacher batch role/source/UUID differs')
    if phase == 'preflight' and value.get('finite_generation_logits') is not True:
        raise ValueError('preflight must check actual generation logits without an extra forward')
    for row, prompt in zip(value['records'], expected):
        validate_reply(row, prompt, config, vocab_size)
        if (row.get('seed') != binding['seed'] or row.get('producer_uuid') != binding['producer_uuid']
                or row.get('teacher_initial_master_sha256') != binding['teacher_initial_master_sha256']
                or row.get('shard_binding_sha256') != content_sha(binding)):
            raise ValueError('saved teacher record producer/initial/seed differs')
    return value


def generate_batch(model, processor, prompts, config, binding, device, phase):
    """No reference target is accepted by this generation path."""
    spec = generation_spec(config)
    rows = [generation_row(row) for row in prompts]
    samples = [encode(processor, row, prompt['image_path'], False) for row, prompt in zip(rows, prompts)]
    for sample, prompt in zip(samples, prompts):
        verify_prompt(sample, prompt, prompt, config)
    batch = collate(samples, processor.tokenizer.pad_token_id)
    actual_sha = batch_hash(batch)
    if any(prompt['microbatch_input_sha256'] != actual_sha for prompt in prompts):
        raise ValueError('teacher actual padded generation inputs differ from fresh CPU contract')
    width = int(batch['input_ids'].shape[1])
    if any(prompt['padded_prompt_width'] != width for prompt in prompts):
        raise ValueError('generation padded prefix width differs')
    gpu_batch = to_device(batch, device)
    if model.generation_config.to_dict() != binding['generation_policy']['effective_generation_config']:
        raise ValueError('teacher effective generation policy changed')
    finite_checks = []
    hook = None
    if phase == 'preflight':
        def check_logits(module, args, output):
            finite = bool(torch.isfinite(output).all())
            finite_checks.append(finite)
            if not finite:
                raise ValueError('nonfinite teacher preflight generation logits')
        hook = model.get_output_embeddings().register_forward_hook(check_logits)
    sync(device); begun = time.monotonic()
    try:
        with torch.inference_mode():
            generated = model.generate(**gpu_batch, **generation_kwargs(spec))
        sync(device)
    finally:
        if hook is not None:
            hook.remove()
    generation_seconds = time.monotonic() - begun
    if phase == 'preflight' and not finite_checks:
        raise ValueError('preflight did not observe generation logits')
    if generated.ndim != 2 or generated.shape[0] != len(prompts) or not torch.equal(generated[:, :width].cpu(), batch['input_ids']):
        raise ValueError('generate changed the actual padded prompt IDs')
    records = []
    for i, prompt in enumerate(prompts):
        raw_ids = generated[i, width:].detach().cpu().tolist()
        parsed = parse_generated_suffix(raw_ids, spec, binding['vocabulary_size'])
        try:
            display = processor.tokenizer.decode(parsed['content_token_ids'], skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            decode_error = None
        except Exception as error:
            display, decode_error = None, f'{type(error).__name__}: {error}'
        records.append({'question_id': prompt['question_id'], 'image_id': prompt['image_id'],
            'global_index': prompt['global_index'], 'seed': binding['seed'], **parsed,
            'decoded_answer_for_display_only': display, 'display_decode_error': decode_error,
            'prompt_contract': prompt, 'producer_uuid': binding['producer_uuid'],
            'teacher_initial_master_sha256': binding['teacher_initial_master_sha256'],
            'shard_binding_sha256': content_sha(binding), 'actual_padded_prompt_width': width,
            'raw_generated_suffix_ids_sha256': content_sha(raw_ids)})
    value = sealed({'schema_version': 'rarevlm.generated_prefix_kl.teacher_batch.v1',
        'phase': phase, 'binding_sha256': content_sha(binding), 'producer_uuid': binding['producer_uuid'],
        'global_batch_index': prompts[0]['global_batch_index'], 'records': records,
        'actual_padded_prompt_width': width, 'actual_microbatch_input_sha256': actual_sha,
        'generation_seconds': generation_seconds, 'finite_generation_logits': all(finite_checks) if finite_checks else None}, 'batch_sha256')
    del gpu_batch, generated
    return value


def execution_accounting(directory):
    import json
    rows = [json.loads(line) for line in (directory / 'EXECUTION.jsonl').read_text().splitlines() if line.strip()]
    started = [row for row in rows if row['event'] == 'generation_started']
    completed = [row for row in rows if row['event'] == 'generation_completed']
    phase_counts = {phase: sum(row['questions'] for row in completed if row['phase'] == phase) for phase in ('formal', 'preflight')}
    return {'actual_generation_started_QA_all_sessions': sum(row['questions'] for row in started),
        'actual_generation_completed_QA_all_sessions': sum(row['questions'] for row in completed),
        'actual_formal_generation_questions': phase_counts['formal'],
        'actual_preflight_generation_questions': phase_counts['preflight'],
        'actual_total_generation_questions': sum(phase_counts.values()),
        'generation_call_starts_all_sessions': len(started), 'generation_call_completions_all_sessions': len(completed),
        'unfinished_generation_call_attempts': len(started) - len(completed),
        'actual_model_load_starts_all_sessions': sum(row['event'] == 'model_load_started' for row in rows),
        'actual_model_load_completions_all_sessions': sum(row['event'] == 'model_load_completed' for row in rows),
        'formal_generation_seconds_all_completed_calls': sum(row['generation_seconds'] for row in completed if row['phase'] == 'formal'),
        'preflight_generation_seconds_all_completed_calls': sum(row['generation_seconds'] for row in completed if row['phase'] == 'preflight')}


def verify_completed_shard(config, seed, shard, *, _completion=None):
    if shard not in (0, 1):
        raise ValueError('exactly two contiguous shards required')
    prompt_manifest, _, prompts = load_prompt_inputs(config, seed)
    directory = reply_dir(config, seed) / f'shard{shard}'
    complete = read_json(directory / 'COMPLETE.json') if _completion is None else _completion
    binding = check_seal(read_json(directory / 'INPUT_LOCK.json'), 'input_lock_sha256')
    if (complete.get('schema_version') != SHARD_SCHEMA or complete.get('status') != 'completed'
            or complete.get('seed') != seed or complete.get('shard') != shard
            or complete.get('binding_sha256') != content_sha(binding)
            or binding.get('config_sha256') != content_sha(config)
            or binding.get('prompt_inputs_file_sha256') != prompt_manifest['prompt_inputs_file_sha256']
            or binding.get('plan_file_sha256') != prompt_manifest['plan_file_sha256']
            or complete.get('formal_saved_generation_questions') != 2896
            or complete.get('formal_batches_completed') != 724
            or complete.get('preflight_saved_generation_questions') != 8
            or complete.get('optimizer_updates') != 0
            or complete.get('teacher_initial_master_sha256_before') != config['teacher_generation']['initial_runtime_master_sha256']
            or complete.get('teacher_initial_master_sha256_after') != complete.get('teacher_initial_master_sha256_before')
            or complete.get('frozen_base_sha256_before') != complete.get('frozen_base_sha256_after')
            or complete.get('teacher_native_lora_sha256_before') != binding.get('teacher_native_lora_sha256')
            or complete.get('teacher_native_lora_sha256_after') != complete.get('teacher_native_lora_sha256_before')
            or complete.get('all_frozen_parameters_sha256_before') != binding.get('all_frozen_parameters_sha256')
            or complete.get('all_frozen_parameters_sha256_after') != complete.get('all_frozen_parameters_sha256_before')
            or complete.get('teacher_inference') != TEACHER_INFERENCE or binding.get('teacher_inference') != TEACHER_INFERENCE
            or complete.get('generation_policy') != binding.get('generation_policy')):
        raise ValueError('teacher shard completion/config/initial/zero-update contract differs')
    required_files = {str(directory / name) for name in ('INPUT_LOCK.json', 'PREFLIGHT.json', 'RESULT.json', 'EXECUTION.jsonl')}
    required_files.update(str(directory / f'batches/batch_{i:06d}.json') for i in range(724))
    if set(complete['artifact_files_sha256']) != required_files:
        raise ValueError('complete shard must bind every input/preflight/result/journal/formal batch')
    for path, digest in complete['artifact_files_sha256'].items():
        if sha_file(private_path(path)) != digest:
            raise ValueError('completed teacher shard source file changed')
    result = read_json(directory / 'RESULT.json')
    expected_result = {key: value for key, value in complete.items() if key != 'artifact_files_sha256'}
    expected_result['schema_version'] = 'rarevlm.generated_prefix_kl.teacher_shard_result.v1'
    if result != expected_result or any(result.get(key) != value for key, value in execution_accounting(directory).items()):
        raise ValueError('teacher RESULT/COMPLETE/actual execution accounting differs')
    for key, path_key in (('study_input_lock_sha256', 'study_input_lock_path'), ('config_file_sha256', 'config_file'), ('admission_file_sha256', 'admission_file')):
        if sha_file(private_path(binding[path_key])) != binding[key]:
            raise ValueError('teacher STUDY/config/admission bytes changed')
    admission = read_json(binding['admission_file'])
    if (admission.get('schema_version') != 'rarevlm.generated_prefix_kl.gpu_admission.v1'
            or admission.get('status') != 'admitted' or admission.get('condition') != 'initial'
            or admission.get('seed') != seed or admission.get('shard') != shard
            or admission.get('role') != f'teacher_seed{seed}_shard{shard}'
            or admission.get('gpu_uuid') != binding['producer_uuid'] or admission.get('task') != 'teacher_rollout'
            or admission.get('config_file_sha256') != binding['config_file_sha256']
            or admission.get('config_sha256') != binding['config_file_sha256']
            or admission.get('plan_file_sha256') != binding['plan_file_sha256']
            or admission.get('prompt_inputs_file_sha256') != binding['prompt_inputs_file_sha256']):
        raise ValueError('teacher saved queue admission/role/UUID differs')
    policy = binding['generation_policy']
    if (sha_file(private_path(policy['native_generation_config_file'])) != policy['native_generation_config_file_sha256']
            or policy['actual_generate_kwargs'] != generation_kwargs(generation_spec(config))
            or any(policy['effective_generation_config'].get(key) != value for key, value in policy['actual_overrides'].items())):
        raise ValueError('teacher native/effective generation policy differs')
    expected = prompts[shard * 2896:(shard + 1) * 2896]
    preflight = check_seal(read_json(directory / 'PREFLIGHT.json'))
    if (preflight.get('status') != 'passed' or preflight.get('binding_sha256') != content_sha(binding)
            or preflight.get('generated_questions') != 8 or len(preflight.get('batches', [])) != 2
            or preflight.get('not_reused_as_formal_generation') is not True):
        raise ValueError('independent8-question preflight source differs')
    for i, value in enumerate(preflight['batches']):
        validate_batch(value, binding, expected[i * 4:(i + 1) * 4], config, binding['vocabulary_size'], 'preflight')
    records = []
    for local in range(724):
        path = directory / f'batches/batch_{local:06d}.json'
        value = validate_batch(read_json(path), binding, expected[local * 4:(local + 1) * 4], config, binding['vocabulary_size'])
        if str(path) not in complete['artifact_files_sha256']:
            raise ValueError('complete shard omitted a formal batch')
        records.extend(value['records'])
    return complete, records


def commit_saved_result(config, seed, shard, *, recovery=False):
    """Finish the RESULT→COMPLETE transaction on CPU, preserving every existing result byte."""
    directory = reply_dir(config, seed) / f'shard{shard}'
    result = read_json(directory / 'RESULT.json')
    started = time.monotonic()
    if recovery:
        event(directory, 'finalization_CPU_started', seed=seed, shard=shard,
            result_file_sha256=sha_file(directory / 'RESULT.json'))
    def completion():
        files = [directory / 'INPUT_LOCK.json', directory / 'PREFLIGHT.json',
            directory / 'RESULT.json', directory / 'EXECUTION.jsonl',
            *[directory / f'batches/batch_{i:06d}.json' for i in range(724)]]
        return {**result, 'schema_version': SHARD_SCHEMA,
            'artifact_files_sha256': {str(path): sha_file(path) for path in files}}
    candidate = completion()
    verify_completed_shard(config, seed, shard, _completion=candidate)
    if recovery:
        event(directory, 'finalization_CPU_completed', seed=seed, shard=shard,
            cpu_seconds=time.monotonic() - started, result_file_sha256=sha_file(directory / 'RESULT.json'))
        candidate = completion()
    immutable_json(directory / 'COMPLETE.json', candidate)
    return verify_completed_shard(config, seed, shard)[0]


def run_teacher_shard(config_file, seed, shard, output_root, admission_file):
    if seed == 42:
        raise ValueError('seed42 reuses accepted parent actual IDs; new teacher generation forbidden.')
    from .data import require_replication_admission
    require_replication_admission(read_json(private_path(config_file)))
    from .contracts import validate_gpu_admission
    config_file = private_path(config_file); config = read_json(config_file)
    if private_path(output_root) != private_path(config['output_root']) or shard not in (0, 1) or seed not in (0, 42, 200):
        raise ValueError('registered root/seed/shard required')
    spec = generation_spec(config)
    prompt_manifest, plan, prompts = load_prompt_inputs(config, seed)
    hardware = visible_gpu_binding()
    admission = validate_gpu_admission(config_file, private_path(admission_file), condition='initial', seed=seed,
        task='teacher_rollout', gpu_uuid=hardware['uuid'], shard=shard,
        plan_file_sha256=prompt_manifest['plan_file_sha256'],
        prompt_inputs_file_sha256=prompt_manifest['prompt_inputs_file_sha256'])
    directory = reply_dir(config, seed) / f'shard{shard}'
    directory.mkdir(parents=True, exist_ok=True)
    # Resume never swaps a producer or silently regenerates complete batches.
    if (directory / 'COMPLETE.json').exists():
        complete = verify_completed_shard(config, seed, shard)[0]
        if complete['producer_uuid'] != hardware['uuid']:
            raise ValueError('same-role UUID changed; complete output is not a new GPU recovery')
        return complete
    if (directory / 'INPUT_LOCK.json').exists():
        previous = check_seal(read_json(directory / 'INPUT_LOCK.json'), 'input_lock_sha256')
        if (previous.get('producer_uuid') != hardware['uuid'] or previous.get('config_file_sha256') != sha_file(config_file)
                or previous.get('admission_file_sha256') != sha_file(admission_file)
                or previous.get('plan_file_sha256') != prompt_manifest['plan_file_sha256']
                or previous.get('prompt_inputs_file_sha256') != prompt_manifest['prompt_inputs_file_sha256']):
            raise ValueError('resume role/UUID/config/admission/prompt binding changed before model load')
    if (directory / 'RESULT.json').exists():
        # No new model-load event can invalidate the immutable completed RESULT.
        return commit_saved_result(config, seed, shard, recovery=True)
    started = time.monotonic(); model = None; head_hook = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('admitted BF16 CUDA device required; no runtime fallback')
        torch.cuda.set_device(0); device = torch.device('cuda', 0); torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        event(directory, 'model_load_started', seed=seed, shard=shard, producer_uuid=hardware['uuid'])
        load_begun = time.monotonic()
        model, dtypes = load_path(private_path(config['model_root']), read_json(config['reference_config']),
            device, 'original_bf16', private_path(config['initial_adapter']))
        identity_before = teacher_identity(model, config)
        initial_sha = identity_before['canonical_lora_fp32_representation_sha256']
        frozen_before = identity_before['frozen_base_sha256']
        if initial_sha != spec['initial_runtime_master_sha256']:
            raise ValueError('teacher native BF16 LoRA canonical FP32 representation differs from initial SFT42')
        model.eval(); model.config.use_cache = True; model.gradient_checkpointing_disable()
        from transformers import GenerationConfig
        native_generation = GenerationConfig.from_pretrained(str(private_path(config['model_root'])), local_files_only=True)
        model.generation_config, policy = generation_policy(config, native_generation, model.generation_config)
        event(directory, 'model_load_completed', seed=seed, shard=shard, producer_uuid=hardware['uuid'])
        load_seconds = time.monotonic() - load_begun
        from vlm_probe.evaluate import processor_for
        processor = processor_for(private_path(config['model_root']), read_json(config['reference_config']), config['visual_token_budget'])
        vocab = int(model.get_output_embeddings().weight.shape[0])
        if vocab != read_json(private_path(config['model_root']) / 'config.json')['vocab_size']:
            raise ValueError('actual teacher head vocabulary differs')
        binding = sealed({'schema_version': 'rarevlm.generated_prefix_kl.teacher_shard_input.v1',
            'seed': seed, 'shard': shard, 'role': f'teacher_seed{seed}_shard{shard}',
            'config_file': str(config_file), 'config_file_sha256': sha_file(config_file), 'config_sha256': content_sha(config),
            'study_input_lock_path': str(private_path(admission['study_input_lock_path'])),
            'study_input_lock_sha256': admission['study_input_lock_sha256'],
            'admission_file': str(private_path(admission_file)), 'admission_file_sha256': sha_file(admission_file),
            'plan_file_sha256': prompt_manifest['plan_file_sha256'],
            'prompt_inputs_file_sha256': prompt_manifest['prompt_inputs_file_sha256'],
            'prompt_manifest_sha256': prompt_manifest['manifest_sha256'], 'plan_sha256': plan['plan_sha256'],
            'producer_uuid': hardware['uuid'], 'hardware': hardware, 'packages': package_lock(),
            'teacher_initial_master_sha256': initial_sha, 'frozen_base_sha256': frozen_before,
            'teacher_native_lora_sha256': identity_before['native_lora_sha256'],
            'all_frozen_parameters_sha256': identity_before['all_frozen_parameters_sha256'],
            'teacher_inference': TEACHER_INFERENCE, 'generation_policy': policy,
            'vocabulary_size': vocab, 'generation': spec, 'dtypes': dtypes}, 'input_lock_sha256')
        immutable_json(directory / 'INPUT_LOCK.json', binding)
        head_hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
        enforce_memory(device, config)
        expected = prompts[shard * 2896:(shard + 1) * 2896]
        def call(chunk, phase, batch_index):
            event(directory, 'generation_started', phase=phase, questions=len(chunk), batch_index=batch_index,
                question_ids=[r['question_id'] for r in chunk], producer_uuid=hardware['uuid'])
            value = generate_batch(model, processor, chunk, config, binding, device, phase)
            event(directory, 'generation_completed', phase=phase, questions=len(chunk), batch_index=batch_index,
                question_ids=[r['question_id'] for r in chunk], producer_uuid=hardware['uuid'],
                generation_seconds=value['generation_seconds'])
            enforce_memory(device, config)
            return value
        preflight_file = directory / 'PREFLIGHT.json'
        if preflight_file.exists():
            preflight = check_seal(read_json(preflight_file))
            if preflight.get('status') != 'passed' or preflight.get('binding_sha256') != content_sha(binding):
                raise ValueError('preflight immutable source differs')
        else:
            runtime, rng = runtime_state(model), rng_state()
            values = []
            try:
                values = [call(expected[i:i + 4], 'preflight', i // 4) for i in (0, 4)]
            finally:
                restore_runtime(runtime); restore_rng(rng); model.zero_grad(set_to_none=True)
            if teacher_identity(model, config) != identity_before:
                raise ValueError('teacher preflight changed initial parameters')
            allocated, reserved = enforce_memory(device, config)
            preflight = sealed({'schema_version': 'rarevlm.generated_prefix_kl.teacher_preflight.v1',
                'status': 'passed', 'binding_sha256': content_sha(binding), 'generated_questions': 8,
                'batches': values, 'not_reused_as_formal_generation': True, 'optimizer_updates': 0,
                'peak_allocated_gib': allocated, 'peak_reserved_gib': reserved})
            immutable_json(preflight_file, preflight)
        resumed = 0
        for local in range(724):
            file = directory / f'batches/batch_{local:06d}.json'; chunk = expected[local * 4:(local + 1) * 4]
            if file.exists():
                validate_batch(read_json(file), binding, chunk, config, vocab); resumed += 1
            else:
                immutable_json(file, call(chunk, 'formal', local))
        identity_after = teacher_identity(model, config)
        after = identity_after['canonical_lora_fp32_representation_sha256']
        frozen_after = identity_after['frozen_base_sha256']
        if identity_after != identity_before:
            raise ValueError('fixed teacher initial/base parameters changed')
        allocated, reserved = enforce_memory(device, config)
        accounting = execution_accounting(directory)
        result = {'schema_version': 'rarevlm.generated_prefix_kl.teacher_shard_result.v1', 'status': 'completed',
            'seed': seed, 'shard': shard, 'role': binding['role'], 'binding_sha256': content_sha(binding),
            'config_file_sha256': binding['config_file_sha256'], 'config_sha256': binding['config_sha256'],
            'study_input_lock_sha256': binding['study_input_lock_sha256'],
            'plan_file_sha256': binding['plan_file_sha256'], 'prompt_inputs_file_sha256': binding['prompt_inputs_file_sha256'],
            'producer_uuid': hardware['uuid'], 'teacher_initial_master_sha256_before': initial_sha,
            'teacher_initial_master_sha256_after': after, 'frozen_base_sha256_before': frozen_before,
            'teacher_native_lora_sha256_before': identity_before['native_lora_sha256'],
            'teacher_native_lora_sha256_after': identity_after['native_lora_sha256'],
            'all_frozen_parameters_sha256_before': identity_before['all_frozen_parameters_sha256'],
            'all_frozen_parameters_sha256_after': identity_after['all_frozen_parameters_sha256'],
            'teacher_inference': TEACHER_INFERENCE, 'generation_policy': policy,
            'frozen_base_sha256_after': frozen_after, 'formal_batches_completed': 724,
            'formal_saved_generation_questions': 2896, 'preflight_saved_generation_questions': 8,
            'resumed_complete_formal_batches': resumed, 'model_load_seconds_current_session': load_seconds,
            'session_wall_seconds': time.monotonic() - started, 'peak_allocated_gib': allocated,
            'peak_reserved_gib': reserved, 'optimizer_updates': 0, 'new_probability_cache': False, **accounting}
        immutable_json(directory / 'RESULT.json', result)
        return commit_saved_result(config, seed, shard)
    except Exception as error:
        event(directory, 'worker_failed', seed=seed, shard=shard, producer_uuid=hardware['uuid'],
            error=f'{type(error).__name__}: {error}')
        atomic_json(directory / 'FAILURE.json', {'status': 'failed', 'seed': seed, 'shard': shard,
            'error': f'{type(error).__name__}: {error}', 'optimizer_updates': 0, 'automatic_runtime_fallback': False})
        raise
    finally:
        if head_hook is not None:
            head_hook.remove()
        if model is not None:
            model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, choices=(0, 42, 200))
    parser.add_argument('--shard', required=True, type=int, choices=(0, 1))
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--admission', required=True, type=Path)
    args = parser.parse_args()
    run_teacher_shard(args.config, args.seed, args.shard, args.output_root, args.admission)


if __name__ == '__main__':
    main()
