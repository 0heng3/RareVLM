"""Fixed training-only plans, actual prompt contracts and ID-only supervision."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import torch

from vlm_deep.common import collate, content_sha, encode, sha_file
from vlm_object_train.data import private_path, read_training_rows
from vlm_object_train.schedule import build_schedule
from vlm_object_train.train import batch_hash, learning_rate, tensor_hash

PLAN_SCHEMA = 'rarevlm.object_behavior_replay.plan.v1'
PROMPT_SCHEMA = 'rarevlm.object_behavior_replay.prompt_inputs.v1'
REPLY_SCHEMA = 'rarevlm.object_behavior_replay.teacher_replies.v1'
SHARD_SCHEMA = 'rarevlm.object_behavior_replay.teacher_shard.complete.v1'
ROLES = {'prompt_pad': 0, 'first_content': 1, 'later_content': 2, 'actual_eos': 3, 'template_suffix': 4}
CONDITIONS = ('teacher_ce32', 'gt_ce32')


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = private_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'immutable JSON differs: {path}')
    else:
        atomic_json(path, value)
    return value


def sealed(value, key='manifest_sha256'):
    return {**value, key: content_sha(value)}


def check_seal(value, key='manifest_sha256'):
    if value.get(key) != content_sha({k: v for k, v in value.items() if k != key}):
        raise ValueError(f'changed {key}')
    return value


def generation_spec(config):
    spec = config['teacher_generation']
    fixed = {'shards': 2, 'batch_size': 4, 'max_new_tokens': 32, 'do_sample': False,
        'use_cache': True, 'repetition_penalty': 1.05,
        'preflight_questions': 8, 'plan_questions_per_seed': 5792,
        'qa_per_shard': 2896, 'microbatches_per_shard': 724,
        'partition': 'contiguous_complete_microbatches',
        'actual_generated_ids_required': True, 'retokenize_teacher_answers': False,
        'eos_token_ids': [151645, 151643], 'pad_token_id': 151643,
        'chat_template_eos_token_id': 151645, 'post_chat_eos_template_ids': [198],
        'non_chat_eos_append_template': False, 'capped_append_eos_or_template': False,
        'keep_empty_content_when_real_eos': True, 'filter_or_resample_teacher_answers': False,
        'teacher_inference_parameter_precision': 'bf16_all_floating_including_lora',
        'teacher_inference_autocast': False,
        'initial_runtime_master_sha256': '6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'}
    if any(spec.get(key) != expected or isinstance(spec.get(key), bool) != isinstance(expected, bool)
           for key, expected in fixed.items()):
        raise ValueError('fixed teacher generation/ID boundary contract differs')
    return spec


def load_rows(config):
    from .contracts import validate_inherited_admissions
    validate_inherited_admissions(config)
    objects, replays, _ = read_training_rows(config)
    if len(objects) != 2888 or len(replays) != 32985:
        raise ValueError('full frozen object and train_all pools required')
    return objects, replays


def build_plan(config, seed):
    generation_spec(config)
    if seed not in (0, 42, 200) or config['epochs'] != 1 or config['object_effective_batch'] != 16 or config['replay_per_update'] != 32:
        raise ValueError('fixed order seeds/one epoch/object16/replay32 required')
    objects, replays = load_rows(config)
    blocks = build_schedule(objects, replays, seed, 1, 16, 32)
    oq = [[str(objects[i]['question_id']) for i in b['object_indices']] for b in blocks]
    rq = [[str(replays[i]['question_id']) for i in b['replay_indices']] for b in blocks]
    schedule = {'object_question_ids_by_update': oq, 'replay_question_ids_by_update': rq,
        'learning_rates_by_update': [learning_rate(i, len(blocks), config) for i in range(len(blocks))],
        'training_microbatch_questions': 4,
        'object_microbatches_by_update': [[q[i:i + 4] for i in range(0, len(q), 4)] for q in oq],
        'replay_microbatches_by_update': [[q[i:i + 4] for i in range(0, len(q), 4)] for q in rq]}
    flat = [q for block in rq for q in block]
    if len(blocks) != 181 or len(flat) != 5792 or len(set(flat)) != 5792:
        raise ValueError('complete unique5792/181-step stream required')
    if seed == 42:
        old_file = private_path(config['historical_coverage_study_root']) / 'runs/wide32/seed42/train/SCHEDULE.json'
        if read_json(old_file) != schedule:
            raise ValueError('seed42 stream differs from completed frozen wide32 schedule')
    value = {'schema_version': PLAN_SCHEMA, 'seed': seed, 'config_sha256': content_sha(config),
        'blocks': blocks, 'schedule': schedule,
        'counts': {'objects': len(objects), 'replay_pool': len(replays), 'updates': 181,
            'replay_exposures': 5792, 'unique_replay_QA': 5792, 'shards': 2, 'QA_per_shard': 2896},
        'object_order_sha256': content_sha(oq), 'replay_order_sha256': content_sha(rq),
        'source_files_sha256': {str(private_path(config[key])): sha_file(private_path(config[key]))
            for key in ('train_manifest', 'replay_manifest', 'study_protocol')}}
    return sealed(value, 'plan_sha256')


def prepared_dir(config, seed):
    return private_path(config['output_root']) / 'prepared' / f'seed{seed}'


def reply_dir(config, seed):
    return private_path(config['teacher_reply_root']) / f'seed{seed}'


def generation_row(row):
    """Strict allowlist: no GT, reference answers, labels or model preference."""
    return {key: row[key] for key in ('question_id', 'image_id', 'question', 'prompt_suffix')}


def prompt_contract(sample, row, config):
    if 'labels' in sample or 'answer_roles' in sample:
        raise ValueError('generation prompt must be unsupervised')
    length = int(sample['input_ids'].shape[1]); grid = sample['image_grid_thw']
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1 or int(grid.prod()) % 4:
        raise ValueError('one static patch14/merge2 image required')
    visual = int(grid.prod()) // 4
    image_token = read_json(private_path(config['model_root']) / 'config.json')['image_token_id']
    if (length > config['max_sequence_length'] or not 0 < visual <= config['visual_token_budget']
            or int((sample['input_ids'] == image_token).sum()) != visual
            or sample['pixel_values'].shape[0] != int(grid.prod())
            or sample['attention_mask'].shape != sample['input_ids'].shape
            or not bool((sample['attention_mask'] == 1).all())):
        raise ValueError('prompt exceeds budget; truncation prohibited')
    return {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
        'question': row['question'], 'prompt_suffix': row['prompt_suffix'], 'image_path': str(private_path(row['image_path'])),
        'prompt_input_ids': sample['input_ids'][0].tolist(), 'prompt_attention_mask': sample['attention_mask'][0].tolist(),
        'prompt_input_sha256': batch_hash(sample), 'input_tokens': int(sample['attention_mask'].sum()),
        'sequence_length': length, 'visual_tokens': visual, 'image_grid_thw': grid.tolist(),
        'pixel_values_sha256': tensor_hash([('pixel_values', sample['pixel_values'])]),
        'pixel_values_shape': list(sample['pixel_values'].shape), 'pixel_values_dtype': str(sample['pixel_values'].dtype),
        'input_truncated': False}


def verify_prompt(sample, record, row, config):
    actual = prompt_contract(sample, row, config)
    if any(actual[key] != record.get(key) for key in actual):
        raise ValueError('actual generation/teacher-training prompt input differs')
    return actual


def prepare_generation_inputs(config, seed):
    """Fresh CPU processing only; never call this until the new source is frozen."""
    from vlm_probe.evaluate import processor_for
    plan = build_plan(config, seed); directory = prepared_dir(config, seed)
    plan_file = directory / 'PLAN.json'; inputs_file = directory / 'PROMPT_INPUTS.jsonl'
    manifest_file = directory / 'PROMPT_MANIFEST.json'
    immutable_json(plan_file, plan)
    if manifest_file.exists():
        return load_prompt_inputs(config, seed)[0]
    if inputs_file.exists():
        raise ValueError('prompt JSONL exists without manifest; no silent overwrite')
    _, replays = load_rows(config)
    processor = processor_for(private_path(config['model_root']), read_json(config['reference_config']), config['visual_token_budget'])
    indices = [i for b in plan['blocks'] for i in b['replay_indices']]
    begun = time.monotonic(); batches = []; temporary = inputs_file.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for begin in range(0, len(indices), 4):
            selected = [replays[i] for i in indices[begin:begin + 4]]
            samples = [encode(processor, generation_row(row), row['image_path'], False) for row in selected]
            batch = collate(samples, processor.tokenizer.pad_token_id); micro_sha = batch_hash(batch)
            for position, (sample, row) in enumerate(zip(samples, selected)):
                ordinal = begin + position
                record = {**prompt_contract(sample, row, config), 'global_index': ordinal,
                    'global_batch_index': begin // 4, 'microbatch_question_position': position,
                    'shard': ordinal // 2896, 'shard_batch_index': (ordinal % 2896) // 4,
                    'plan_sha256': plan['plan_sha256'], 'config_sha256': content_sha(config),
                    'microbatch_input_sha256': micro_sha, 'padded_prompt_width': int(batch['input_ids'].shape[1]),
                    'padded_input_ids_sha256': tensor_hash([('input_ids', batch['input_ids'][position])]),
                    'padded_attention_mask_sha256': tensor_hash([('attention_mask', batch['attention_mask'][position])])}
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            batches.append({'global_batch_index': begin // 4, 'question_ids': [str(row['question_id']) for row in selected],
                'microbatch_input_sha256': micro_sha, 'padded_prompt_width': int(batch['input_ids'].shape[1])})
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(inputs_file)
    value = sealed({'schema_version': PROMPT_SCHEMA, 'seed': seed, 'status': 'completed_CPU_prompts',
        'config_sha256': content_sha(config), 'plan_sha256': plan['plan_sha256'],
        'plan_file_sha256': sha_file(plan_file), 'prompt_inputs_file_sha256': sha_file(inputs_file),
        'questions': len(indices), 'microbatches': len(batches), 'batches': batches, 'fresh_CPU_prompt_encodes': len(indices),
        'generation_inputs_contain_GT': False, 'model_forwards': 0, 'GPU_operations': 0,
        'CPU_wall_seconds': time.monotonic() - begun})
    immutable_json(manifest_file, value)
    return value


def load_prompt_inputs(config, seed):
    directory = prepared_dir(config, seed)
    plan = check_seal(read_json(directory / 'PLAN.json'), 'plan_sha256')
    manifest = check_seal(read_json(directory / 'PROMPT_MANIFEST.json'))
    if (manifest.get('schema_version') != PROMPT_SCHEMA or manifest.get('seed') != seed
            or manifest.get('config_sha256') != content_sha(config) or plan.get('config_sha256') != content_sha(config)
            or manifest.get('plan_sha256') != plan['plan_sha256']
            or manifest.get('plan_file_sha256') != sha_file(directory / 'PLAN.json')
            or manifest.get('prompt_inputs_file_sha256') != sha_file(directory / 'PROMPT_INPUTS.jsonl')):
        raise ValueError('CPU prompt plan/manifest/source binding differs')
    records = [json.loads(line) for line in (directory / 'PROMPT_INPUTS.jsonl').read_text().splitlines()]
    qids = [q for block in plan['schedule']['replay_question_ids_by_update'] for q in block]
    if len(records) != 5792 or len(manifest['batches']) != 1448 or len(qids) != len(records):
        raise ValueError('full CPU prompt stream missing')
    for i, record in enumerate(records):
        if (record.get('global_index') != i or record.get('question_id') != qids[i]
                or record.get('plan_sha256') != plan['plan_sha256']
                or record.get('config_sha256') != content_sha(config)
                or any(key in record for key in ('target', 'answers', 'labels', 'answer_roles'))):
            raise ValueError('prompt order/GT-free input contract differs')
    return manifest, plan, records


def parse_generated_suffix(raw_ids, spec, vocab_size):
    """Actual IDs only; first declared EOS is retained, returned PAD is not trained."""
    if (not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > spec['max_new_tokens']
            or any(type(token) is not int or not 0 <= token < vocab_size for token in raw_ids)):
        raise ValueError('empty, overlong or illegal actual generated suffix')
    ends = [i for i, token in enumerate(raw_ids) if token in spec['eos_token_ids']]
    position = ends[0] if ends else None
    if position is None and len(raw_ids) != spec['max_new_tokens']:
        raise ValueError('short continuation without actual EOS is not a capped answer')
    if position is not None and any(token != spec['pad_token_id'] for token in raw_ids[position + 1:]):
        raise ValueError('nonpadding tokens occur after actual termination')
    content = raw_ids[:position] if position is not None else list(raw_ids)
    roles = ([1] + [2] * (len(content) - 1)) if content else []
    eos = raw_ids[position] if position is not None else None
    target = list(content)
    if eos is not None:
        target.append(eos); roles.append(3)
    suffix = list(spec['post_chat_eos_template_ids']) if eos == spec['chat_template_eos_token_id'] else []
    if any(type(token) is not int or not 0 <= token < vocab_size for token in suffix):
        raise ValueError('illegal explicit template suffix')
    target.extend(suffix); roles.extend([4] * len(suffix))
    reason = ('capped_without_eos' if eos is None else 'empty_content_chat_eos' if not content and eos == spec['chat_template_eos_token_id']
              else 'chat_eos' if eos == spec['chat_template_eos_token_id'] else 'other_eos')
    return {'raw_generated_suffix_ids': list(raw_ids), 'content_token_ids': list(content),
        'training_target_ids': target, 'training_answer_roles': roles, 'actual_eos_token_id': eos,
        'actual_eos_suffix_index': position, 'termination_reason': reason, 'empty_content': not content,
        'explicit_template_suffix_ids': suffix, 'capped_without_eos': eos is None}


def validate_reply(row, prompt, config, vocab_size):
    parsed = parse_generated_suffix(row['raw_generated_suffix_ids'], generation_spec(config), vocab_size)
    if (any(row.get(key) != expected for key, expected in parsed.items())
            or row.get('question_id') != prompt['question_id'] or row.get('image_id') != prompt['image_id']
            or row.get('global_index') != prompt['global_index'] or row.get('prompt_contract') != prompt):
        raise ValueError('teacher IDs/roles/termination/order/prompt source differs')
    return row


def load_teacher_replies(config, seed):
    manifest, plan, prompts = load_prompt_inputs(config, seed)
    directory = reply_dir(config, seed); file = directory / 'MANIFEST.json'
    lock = check_seal(read_json(directory / 'TEACHER_REPLIES_LOCK.json'), 'lock_sha256')
    value = check_seal(read_json(file))
    if (value.get('schema_version') != REPLY_SCHEMA or value.get('status') != 'completed' or value.get('seed') != seed
            or value.get('config_sha256') != content_sha(config) or value.get('prompt_manifest_sha256') != manifest['manifest_sha256']
            or value.get('plan_sha256') != plan['plan_sha256'] or lock.get('manifest_file_sha256') != sha_file(file)
            or lock.get('schema_version') != 'rarevlm.object_behavior_replay.teacher_replies_lock.v1'
            or lock.get('seed') != seed or lock.get('manifest_sha256') != value['manifest_sha256']
            or lock.get('plan_file_sha256') != manifest['plan_file_sha256']
            or lock.get('prompt_inputs_file_sha256') != manifest['prompt_inputs_file_sha256']
            or len(value.get('rows', [])) != len(prompts)):
        raise ValueError('complete teacher reply manifest/lock required')
    for path, digest in lock['artifact_files_sha256'].items():
        if sha_file(private_path(path)) != digest:
            raise ValueError('teacher source artifact changed')
    from .rollout import verify_completed_shard
    produced = []
    for shard in (0, 1):
        complete_path = directory / f'shard{shard}/COMPLETE.json'
        if lock['artifact_files_sha256'].get(str(complete_path)) != sha_file(complete_path):
            raise ValueError('reply lock must bind both complete contiguous shards')
        _, shard_rows = verify_completed_shard(config, seed, shard)
        produced.extend(shard_rows)
    if produced != value['rows']:
        raise ValueError('merged teacher rows differ from complete producer batches')
    vocab = read_json(private_path(config['model_root']) / 'config.json')['vocab_size']
    result = {}
    for prompt, row in zip(prompts, value['rows']):
        validate_reply(row, prompt, config, vocab)
        if row.get('seed') != seed or row['question_id'] in result:
            raise ValueError('teacher seed/qid differs or duplicates')
        result[row['question_id']] = row
    return result


def finalize_teacher_replies(config, seed):
    prompt_manifest, plan, prompts = load_prompt_inputs(config, seed)
    directory = reply_dir(config, seed)
    if (directory / 'MANIFEST.json').exists() and (directory / 'TEACHER_REPLIES_LOCK.json').exists():
        load_teacher_replies(config, seed)
        return read_json(directory / 'MANIFEST.json')
    from .rollout import verify_completed_shard
    rows, files, summaries = [], {}, []
    for shard in (0, 1):
        complete, records = verify_completed_shard(config, seed, shard)
        files.update(complete['artifact_files_sha256'])
        complete_path = directory / f'shard{shard}/COMPLETE.json'
        files[str(complete_path)] = sha_file(complete_path)
        rows.extend(records); summaries.append(complete)
    vocab = read_json(private_path(config['model_root']) / 'config.json')['vocab_size']
    if len(rows) != len(prompts):
        raise ValueError('both contiguous shards required; no substitute answers')
    for prompt, row in zip(prompts, rows):
        validate_reply(row, prompt, config, vocab)
        if row.get('seed') != seed:
            raise ValueError('mixed seed teacher output')
    value = sealed({'schema_version': REPLY_SCHEMA, 'seed': seed, 'status': 'completed',
        'config_sha256': content_sha(config), 'plan_sha256': plan['plan_sha256'],
        'prompt_manifest_sha256': prompt_manifest['manifest_sha256'], 'rows': rows,
        'generation_spec': generation_spec(config),
        'teacher_initial_master_sha256': config['teacher_generation']['initial_runtime_master_sha256'],
        'producer_shards': [{**{key: c[key] for key in ('seed', 'shard', 'role', 'producer_uuid',
            'teacher_initial_master_sha256_before', 'teacher_initial_master_sha256_after',
            'frozen_base_sha256_before', 'frozen_base_sha256_after', 'binding_sha256',
            'teacher_native_lora_sha256_before', 'teacher_native_lora_sha256_after',
            'all_frozen_parameters_sha256_before', 'all_frozen_parameters_sha256_after',
            'teacher_inference', 'generation_policy')},
            'complete_file': str(directory / f"shard{c['shard']}/COMPLETE.json"),
            'complete_file_sha256': sha_file(directory / f"shard{c['shard']}/COMPLETE.json")}
            for c in summaries],
        'questions': len(rows), 'teacher_replies_are_factual_GT': False,
        'termination_counts': {kind: sum(row['termination_reason'] == kind for row in rows)
            for kind in sorted({row['termination_reason'] for row in rows})},
        'empty_content_questions': sum(row['empty_content'] for row in rows),
        'formal_saved_generation_questions': len(rows),
        'preflight_saved_generation_questions': sum(c['preflight_saved_generation_questions'] for c in summaries),
        'actual_generation_started_QA_all_sessions': sum(c['actual_generation_started_QA_all_sessions'] for c in summaries),
        'actual_generation_completed_QA_all_sessions': sum(c['actual_generation_completed_QA_all_sessions'] for c in summaries),
        'actual_formal_generation_questions': sum(c['actual_formal_generation_questions'] for c in summaries),
        'actual_preflight_generation_questions': sum(c['actual_preflight_generation_questions'] for c in summaries),
        'actual_total_generation_questions': sum(c['actual_total_generation_questions'] for c in summaries),
        'optimizer_updates': 0, 'new_full_vocabulary_probability_cache': False})
    immutable_json(directory / 'MANIFEST.json', value)
    immutable_json(directory / 'TEACHER_REPLIES_LOCK.json', sealed({
        'schema_version': 'rarevlm.object_behavior_replay.teacher_replies_lock.v1', 'seed': seed,
        'manifest_file_sha256': sha_file(directory / 'MANIFEST.json'), 'manifest_sha256': value['manifest_sha256'],
        'plan_file_sha256': prompt_manifest['plan_file_sha256'],
        'prompt_inputs_file_sha256': prompt_manifest['prompt_inputs_file_sha256'],
        'artifact_files_sha256': files}, 'lock_sha256'))
    load_teacher_replies(config, seed)
    return value


def check_sample(sample, config):
    labels, roles = sample['labels'], sample['answer_roles']
    if (roles.dtype != torch.long or roles.shape != labels.shape or labels.shape != sample['input_ids'].shape
            or not torch.equal(labels != -100, roles != 0) or not bool(((roles >= 0) & (roles <= 4)).all())
            or not torch.equal(labels[labels != -100], sample['input_ids'][labels != -100])
            or not bool((labels != -100).any()) or int(labels[0, 0]) != -100
            or sample['input_ids'].shape[1] > config['max_sequence_length']):
        raise ValueError('assistant IDs/labels/roles/mask/length contract differs')
    grid = sample['image_grid_thw']
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1 or int(grid.prod()) % 4 or not 0 < int(grid.prod()) // 4 <= config['visual_token_budget']:
        raise ValueError('one full static image within fixed visual budget required')
    return sample


def encode_training_sample(processor, row, config, *, source, condition, teacher_replies=None):
    if source not in ('object', 'replay') or condition not in CONDITIONS:
        raise ValueError('declared CE source/condition required')
    spec = generation_spec(config)
    if source == 'replay' and condition == 'teacher_ce32':
        if teacher_replies is None or str(row['question_id']) not in teacher_replies:
            raise ValueError('complete immutable teacher reply required; no GT fallback')
        reply = teacher_replies[str(row['question_id'])]
        sample = encode(processor, generation_row(row), row['image_path'], False)
        verify_prompt(sample, reply['prompt_contract'], row, config)
        validate_reply(reply, reply['prompt_contract'], config,
                       int(read_json(private_path(config['model_root']) / 'config.json')['vocab_size']))
        prefix = sample['input_ids'].shape[1]
        answer = torch.tensor([reply['training_target_ids']], dtype=sample['input_ids'].dtype)
        sample['input_ids'] = torch.cat((sample['input_ids'], answer), dim=1)
        sample['attention_mask'] = torch.cat((sample['attention_mask'], torch.ones_like(answer)), dim=1)
        sample['labels'] = sample['input_ids'].clone(); sample['labels'][:, :prefix] = -100
        sample['answer_roles'] = torch.cat((torch.zeros((1, prefix), dtype=torch.long),
            torch.tensor([reply['training_answer_roles']], dtype=torch.long)), dim=1)
    else:
        sample = encode(processor, row, row['image_path'], True)
        answer = sample['labels'][sample['labels'] != -100].tolist()
        suffix = list(spec['post_chat_eos_template_ids']); eos_index = len(answer) - len(suffix) - 1
        if eos_index < 0 or answer[eos_index] != spec['chat_template_eos_token_id'] or answer[eos_index + 1:] != suffix:
            raise ValueError('original GT chat-template EOS/suffix differs')
        roles = ([1] + [2] * (eos_index - 1)) if eos_index else []
        roles += [3] + [4] * len(suffix)
        sample['answer_roles'] = torch.zeros_like(sample['labels'], dtype=torch.long)
        sample['answer_roles'][sample['labels'] != -100] = torch.tensor(roles, dtype=torch.long)
    return check_sample(sample, config)


def collate_training(samples, pad_token_id):
    if not samples:
        raise ValueError('empty training microbatch')
    batch = collate(samples, pad_token_id)
    roles = torch.zeros_like(batch['input_ids'], dtype=torch.long)
    for i, sample in enumerate(samples):
        roles[i, -sample['answer_roles'].shape[1]:] = sample['answer_roles'][0]
    batch['answer_roles'] = roles
    if not torch.equal(batch['labels'] != -100, roles != 0):
        raise ValueError('left padding changed role/label coverage')
    return batch


def sample_contract(sample, row, config):
    check_sample(sample, config)
    standard = {key: value for key, value in sample.items() if key != 'answer_roles'}
    valid = sample['labels'] != -100
    return {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
        'input_sha256': batch_hash(standard), 'input_tokens': int(sample['attention_mask'].sum()),
        'sequence_length': int(sample['input_ids'].shape[1]), 'visual_tokens': int(sample['image_grid_thw'].prod()) // 4,
        'assistant_tokens': int(valid.sum()), 'truncated': False,
        'answer_roles_sha256': tensor_hash([('answer_roles', sample['answer_roles'])]),
        'assistant_target_ids_sha256': tensor_hash([('assistant_target_ids', sample['labels'][valid])])}
