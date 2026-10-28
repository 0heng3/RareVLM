"""Accepted-parent input references and fresh replication-only generated prefixes.

No frozen module globals are patched. teacher_ce32 is an input-layout label only;
the only new optimizer identity is teacher_kl32 / full-vocabulary forward KL.
"""
from __future__ import annotations

COPY_STAGE32_SOURCE = {'file': 'src/vlm_generated_prefix_kl/data.py', 'sha256': '9271dac1973836bf31ccbcbadffdf46f72acc1e5358bc3f5cc31152d28c059c4', 'scope': 'explicit versioned stage32 copy; native numerical helpers unchanged; new study admission'}
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
from vlm_object_behavior_replay import data as parent
from . import CANDIDATE

COPY_PROVENANCE = {'parent_data_sha256': 'f01b0825d0a83a9ad0d3f6b74069b59b5434e5c8db24f46605a7b5088d628985',
    'scope': 'Pure tensor helpers reused explicitly; fresh replication metadata methods copied with independent namespace; no global patch.'}
PLAN_SCHEMA = 'rarevlm.lora_coordinate_shrink.plan.v1'
PROMPT_SCHEMA = 'rarevlm.lora_coordinate_shrink.prompt_inputs.v1'
REPLY_SCHEMA = 'rarevlm.lora_coordinate_shrink.teacher_replies.v1'
SHARD_SCHEMA = 'rarevlm.lora_coordinate_shrink.teacher_shard.complete.v1'
atomic_json, immutable_json, read_json = parent.atomic_json, parent.immutable_json, parent.read_json
sealed, check_seal = parent.sealed, parent.check_seal
generation_spec, generation_row = parent.generation_spec, parent.generation_row
prompt_contract, verify_prompt = parent.prompt_contract, parent.verify_prompt
parse_generated_suffix, validate_reply = parent.parse_generated_suffix, parent.validate_reply
prepared_dir, reply_dir = parent.prepared_dir, parent.reply_dir
collate_training, sample_contract = parent.collate_training, parent.sample_contract


def validate_parent_proofs(config):
    root = private_path(config['parent_output_root'])
    required = {str(private_path(config['parent_config_file']))}
    required.update(str(root / name) for name in ('STUDY_INPUT_LOCK.json', 'recovery_v1/RECOVERY_INPUT_LOCK.json',
        'analysis/pilot_seed42/screen/ANALYSIS.json', 'independent_review/PILOT_PREDICTION_REVIEW_RECOVERY_V1.json',
        'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json', 'independent_review/TEACHER_COMPLETION_REVIEW.json',
        'independent_review/TRAINING_COMPLETION_REVIEW.json', 'teacher_replies/seed42/MANIFEST.json',
        'teacher_replies/seed42/TEACHER_REPLIES_LOCK.json', 'independent_review/TEACHER_TARGET_INDEX.json',
        'prepared/seed42/PLAN.json', 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json',
        'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json'))
    proofs = config.get('parent_proof_files_sha256', {})
    if len(proofs) != 14 or set(proofs) != required:
        raise ValueError('All14 exact parent proofs required; no missing/substitute metadata.')
    from .lock import verify_parent_proof
    if verify_parent_proof(config) != proofs:
        raise ValueError('Accepted parent proof metadata differs.')
    reference = config['seed42_input_reference']
    expected = {'parent_condition': 'teacher_ce32', 'new_training_condition': CANDIDATE,
        'plan_file': str(root / 'prepared/seed42/PLAN.json'),
        'train_inputs_file': str(root / 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json'),
        'train_contracts_file': str(root / 'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json'),
        'teacher_target_index_file': str(root / 'independent_review/TEACHER_TARGET_INDEX.json'),
        'teacher_reply_manifest_file': str(root / 'teacher_replies/seed42/MANIFEST.json'),
        'new_generation_questions': 0, 'new_CPU_full_input_encodings': 0}
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError('Parent input layout/reference identity differs.')
    return dict(proofs)


def load_parent_bundle(config):
    """Parse compact index/contracts/prompt only; large teacher JSON is never parsed."""
    proofs = validate_parent_proofs(config)
    ref = config['seed42_input_reference']; root = private_path(config['parent_output_root'])
    header = read_json(ref['train_inputs_file']); contracts = read_json(ref['train_contracts_file'])
    plan = check_seal(read_json(ref['plan_file']), 'plan_sha256')
    index = read_json(ref['teacher_target_index_file'])
    if (header.get('condition') != 'teacher_ce32' or header.get('seed') != 42
            or header.get('replay_target_source') != 'fixed_initial_greedy_ids'
            or header.get('plan_file_sha256') != proofs[ref['plan_file']]
            or header.get('plan_content_sha256') != content_sha(plan)
            or header.get('contracts_file_sha256') != proofs[ref['train_contracts_file']]
            or index.get('questions') != 5792 or index.get('seed') != 42
            or index.get('source_manifest_file_sha256') != proofs[ref['teacher_reply_manifest_file']]
            or index.get('source_review_file_sha256') != proofs[str(root / 'independent_review/TEACHER_COMPLETION_REVIEW.json')]):
        raise ValueError('Accepted parent prepared/index source binding differs.')
    prompt_manifest_path = root / 'prepared/seed42/PROMPT_MANIFEST.json'
    prompt_inputs_path = root / 'prepared/seed42/PROMPT_INPUTS.jsonl'
    if sha_file(prompt_manifest_path) != header['prompt_manifest_file_sha256']:
        raise ValueError('Parent prompt metadata differs from accepted prepared header.')
    pm = check_seal(read_json(prompt_manifest_path))
    if sha_file(prompt_inputs_path) != pm['prompt_inputs_file_sha256']:
        raise ValueError('Parent prompt input bytes differ; no fresh encoding fallback.')
    prompts = [json.loads(line) for line in prompt_inputs_path.read_text().splitlines()]
    qids = [q for block in plan['schedule']['replay_question_ids_by_update'] for q in block]
    if (len(prompts) != 5792 or len(index['rows']) != 5792 or len(contracts['replay']) != 5792
            or len(set(qids)) != 5792 or [r['question_id'] for r in index['rows']] != qids
            or index['target_ids_stream_sha256'] != header['streams']['replay']['supervised_target_ids_stream_sha256']
            or index['answer_roles_stream_sha256'] != header['streams']['replay']['supervised_answer_roles_stream_sha256']
            or content_sha(index['rows']) != index['ordered_rows_sha256']):
        raise ValueError('Accepted compact index order/count differs.')
    replies = {}
    for ordinal, (target, prompt, contract) in enumerate(zip(index['rows'], prompts, contracts['replay'])):
        qid = str(target['question_id'])
        if (target['global_index'] != ordinal or prompt['global_index'] != ordinal
                or qid != prompt['question_id'] or qid != contract['question_id']
                or prompt['image_id'] != contract['image_id'] or prompt['plan_sha256'] != plan['plan_sha256']
                or target['training_target_ids'] != contract['supervised_target_ids']
                or target['training_answer_roles'] != contract['supervised_answer_roles']):
            raise ValueError('Accepted compact target/prefix/contract pairing differs.')
        replies[qid] = {**target, 'image_id': contract['image_id'], 'prompt_contract': prompt,
            'source_teacher_reply_row_sha256': contract['teacher_reply_row_sha256'],
            'source_layout': 'accepted_parent_teacher_ce32', 'source_target_index_sha256': proofs[ref['teacher_target_index_file']]}
    return {'proofs': proofs, 'parent_header': header, 'contracts': contracts, 'parent_plan': plan,
        'index': index, 'prompt_manifest': pm, 'prompt_manifest_path': prompt_manifest_path,
        'prompt_inputs_path': prompt_inputs_path, 'teacher_replies': replies}


def load_rows(config):
    from .contracts import validate_inherited_admissions
    validate_inherited_admissions(config)
    objects, replays, _ = read_training_rows(config)
    if len(objects) != 2888 or len(replays) != 32985:
        raise ValueError('Complete frozen2888/32985 training pools required.')
    return objects, replays


def validate_producer_seed(seed):
    if type(seed) is not int or seed not in (0,200):
        raise ValueError('seed42 is historical only; new unshrunk producers allow only admitted0/200.')


def build_plan(config, seed):
    generation_spec(config)
    validate_producer_seed(seed)
    require_replication_admission(config)
    objects,replays=load_rows(config)
    blocks=build_schedule(objects,replays,seed,1,16,32)
    oq=[[str(objects[i]['question_id']) for i in b['object_indices']] for b in blocks]
    rq=[[str(replays[i]['question_id']) for i in b['replay_indices']] for b in blocks]
    schedule={'object_question_ids_by_update':oq,'replay_question_ids_by_update':rq,
        'learning_rates_by_update':[learning_rate(i,len(blocks),config) for i in range(len(blocks))],
        'training_microbatch_questions':4,
        'object_microbatches_by_update':[[q[i:i+4] for i in range(0,len(q),4)] for q in oq],
        'replay_microbatches_by_update':[[q[i:i+4] for i in range(0,len(q),4)] for q in rq]}
    return sealed({'schema_version':PLAN_SCHEMA,'seed':seed,'condition':CANDIDATE,
        'artifact_role':'unshrunk_producer','coordinate_shrink_applied':False,
        'replay_objective':'forward_kl_full_vocabulary','config_sha256':content_sha(config),
        'blocks':blocks,'schedule':schedule,'counts':{'objects':2888,'replay_pool':32985,
            'updates':181,'replay_exposures':5792,'unique_replay_QA':5792,'shards':2,'QA_per_shard':2896},
        'object_order_sha256':content_sha(oq),'replay_order_sha256':content_sha(rq)},'plan_sha256')


def load_teacher_replies(config, seed):
    validate_producer_seed(seed)
    require_replication_admission(config)
    return _load_fresh_teacher_replies(config,seed)


def encode_training_sample(processor, row, config, *, source, condition, teacher_replies=None):
    if condition != CANDIDATE or source not in ('object', 'replay'):
        raise ValueError('Only unshrunk lora_half32 producer input layout is admitted.')
    if source == 'object':
        return parent.encode_training_sample(processor, row, config, source=source, condition='teacher_ce32')
    if teacher_replies is None or str(row['question_id']) not in teacher_replies:
        raise ValueError('Exact accepted generated-prefix target required; no GT fallback.')
    reply = teacher_replies[str(row['question_id'])]
    sample = encode(processor, generation_row(row), row['image_path'], False)
    verify_prompt(sample, reply['prompt_contract'], row, config)
    ids, roles = reply['training_target_ids'], reply['training_answer_roles']
    vocabulary = read_json(private_path(config['model_root']) / 'config.json')['vocab_size']
    if (not ids or len(ids) != len(roles)
            or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
            or any(type(role) is not int or role not in (1, 2, 3, 4) for role in roles)):
        raise ValueError('Saved generated-prefix target/role IDs are invalid.')
    prefix = sample['input_ids'].shape[1]
    target = torch.tensor([ids], dtype=sample['input_ids'].dtype)
    sample['input_ids'] = torch.cat((sample['input_ids'], target), dim=1)
    sample['attention_mask'] = torch.cat((sample['attention_mask'], torch.ones_like(target)), dim=1)
    sample['labels'] = sample['input_ids'].clone(); sample['labels'][:, :prefix] = -100
    sample['answer_roles'] = torch.cat((torch.zeros((1, prefix), dtype=torch.long), torch.tensor([roles], dtype=torch.long)), dim=1)
    return parent.check_sample(sample, config)


def _prepare_fresh_generation_inputs(config, seed):
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


def require_replication_admission(config):
    """A passing pilot42, distinct from the three-seed screen, admits replication."""
    from .analysis import passing_pilot
    root = private_path(config['output_root'])
    study = read_json(root / 'STUDY_INPUT_LOCK.json')
    return passing_pilot(root / 'analysis/pilot_seed42/screen/ANALYSIS.json', private_path(study['config_path']))


def prepare_generation_inputs(config, seed):
    if seed not in (0, 200):
        raise ValueError('Only admitted new replication seeds generate; seed42 must reuse parent IDs.')
    require_replication_admission(config)
    return _prepare_fresh_generation_inputs(config, seed)


def finalize_teacher_replies(config, seed):
    if seed not in (0, 200):
        raise ValueError('seed42 uses the accepted parent reference, never a new merged teacher stream.')
    return _finalize_fresh_teacher_replies(config, seed)


def load_prompt_inputs(config, seed):
    validate_producer_seed(seed)
    require_replication_admission(config)
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


def _load_fresh_teacher_replies(config, seed):
    manifest, plan, prompts = load_prompt_inputs(config, seed)
    directory = reply_dir(config, seed); file = directory / 'MANIFEST.json'
    lock = check_seal(read_json(directory / 'TEACHER_REPLIES_LOCK.json'), 'lock_sha256')
    value = check_seal(read_json(file))
    if (value.get('schema_version') != REPLY_SCHEMA or value.get('status') != 'completed' or value.get('seed') != seed
            or value.get('config_sha256') != content_sha(config) or value.get('prompt_manifest_sha256') != manifest['manifest_sha256']
            or value.get('plan_sha256') != plan['plan_sha256'] or lock.get('manifest_file_sha256') != sha_file(file)
            or lock.get('schema_version') != 'rarevlm.lora_coordinate_shrink.teacher_replies_lock.v1'
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


def _finalize_fresh_teacher_replies(config, seed):
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
        'schema_version': 'rarevlm.lora_coordinate_shrink.teacher_replies_lock.v1', 'seed': seed,
        'manifest_file_sha256': sha_file(directory / 'MANIFEST.json'), 'manifest_sha256': value['manifest_sha256'],
        'plan_file_sha256': prompt_manifest['plan_file_sha256'],
        'prompt_inputs_file_sha256': prompt_manifest['prompt_inputs_file_sha256'],
        'artifact_files_sha256': files}, 'lock_sha256'))
    load_teacher_replies(config, seed)
    return value
