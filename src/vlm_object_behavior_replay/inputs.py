"""CPU full-stream admission and actual-batch contracts for behavior replay."""
from __future__ import annotations

import json
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

from vlm_deep.common import content_sha, sha_file
from vlm_object_train.data import private_path
from vlm_object_train.train import batch_hash, lock_json
from . import CANDIDATE, CONTROL, SCHEMA

LEGACY_FIELDS = ('question_id', 'image_id', 'input_sha256', 'input_tokens', 'sequence_length',
                 'visual_tokens', 'assistant_tokens', 'truncated', 'microbatch_input_sha256',
                 'microbatch_question_position', 'microbatch_size', 'microbatch_sequence_length')


def paths(config, seed, condition):
    root = private_path(config['output_root']) / 'prepared' / f'seed{seed}'
    teacher = private_path(config['teacher_reply_root']) / f'seed{seed}'
    return {'plan': root/'PLAN.json', 'prompt': root/'PROMPT_MANIFEST.json',
            'manifest': root/f'TRAIN_INPUTS_{condition}.json',
            'contracts': root/f'TRAIN_CONTRACTS_{condition}.json',
            'teacher': teacher/'MANIFEST.json', 'teacher_lock': teacher/'TEACHER_REPLIES_LOCK.json'}


def legacy_contracts(rows):
    return [{key: row[key] for key in LEGACY_FIELDS} for row in rows]


def encode_group(processor, rows, selected, config, *, source, condition, teacher_replies):
    import torch
    from .data import collate_training, encode_training_sample, sample_contract
    samples = [encode_training_sample(processor, rows[index], config, source=source,
                condition=condition, teacher_replies=teacher_replies) for index in selected]
    batch = collate_training(samples, processor.tokenizer.pad_token_id)
    standard = {key: value for key, value in batch.items() if key != 'answer_roles'}
    micro_sha = batch_hash(standard)
    contracts = []
    for position, (index, sample) in enumerate(zip(selected, samples)):
        row = rows[index]
        valid = sample['labels'][0] != -100
        ids = sample['labels'][0][valid].tolist()
        roles = sample['answer_roles'][0][valid].tolist()
        if (sample['answer_roles'].dtype != torch.long or not ids
                or not torch.equal(sample['answer_roles'] != 0, sample['labels'] != -100)):
            raise ValueError('actual training sample roles/labels are invalid')
        target_source = ('object_ground_truth' if source == 'object' else
                         'fixed_initial_greedy_ids' if condition == CANDIDATE else 'original_ground_truth')
        reply_sha = None
        if source == 'replay' and condition == CANDIDATE:
            reply = teacher_replies[str(row['question_id'])]
            if ids != reply['training_target_ids'] or roles != reply['training_answer_roles']:
                raise ValueError('actual teacher labels were changed/re-tokenized')
            reply_sha = content_sha(reply)
        contract = {**sample_contract(sample, row, config),
            'microbatch_input_sha256': micro_sha, 'microbatch_question_position': position,
            'microbatch_size': len(selected), 'microbatch_sequence_length': int(batch['input_ids'].shape[1]),
            'supervised_target_ids': ids, 'supervised_answer_roles': roles,
            'supervised_target_ids_sha256': content_sha(ids),
            'supervised_answer_roles_sha256': content_sha(roles),
            'target_source': target_source, 'teacher_reply_row_sha256': reply_sha}
        contracts.append(contract)
    return batch, contracts, samples


def _binding(config, seed, condition, plan, teacher_rows):
    from .train import source_hashes
    p = paths(config, seed, condition)
    if not all(p[key].is_file() for key in ('plan', 'prompt', 'teacher', 'teacher_lock')):
        raise ValueError('prepared PLAN/prompt and complete teacher replies/lock must exist before training admission')
    saved_plan = json.loads(p['plan'].read_text())
    if saved_plan != plan:
        raise ValueError('saved prepared PLAN differs from current fixed schedule')
    historical = (private_path(config['historical_coverage_study_root']) /
                  'runs/wide32/seed42/train/TRAIN_RESULT.json')
    return {'schema_version': 'rarevlm.object_behavior_replay.prepared_training_inputs.v1',
            'condition': condition, 'seed': seed, 'config_sha256': content_sha(config),
            'source_sha256': source_hashes(), 'plan_file_sha256': sha_file(p['plan']),
            'data_admission_file_sha256': sha_file(private_path(config['data_admission'])),
            'train_manifest_file_sha256': sha_file(private_path(config['train_manifest'])),
            'replay_manifest_file_sha256': sha_file(private_path(config['replay_manifest'])),
            'plan_content_sha256': content_sha(plan), 'schedule_sha256': content_sha(plan['schedule']),
            'prompt_manifest_file_sha256': sha_file(p['prompt']),
            'teacher_replies_manifest_file_sha256': sha_file(p['teacher']),
            'teacher_replies_lock_file_sha256': sha_file(p['teacher_lock']),
            'teacher_reply_rows': len(teacher_rows),
            'historical_wide42_training_result_file_sha256': sha_file(historical),
            'replay_target_source': 'fixed_initial_greedy_ids' if condition == CANDIDATE else 'original_ground_truth',
            'teacher_probability_cache_used': False}


def _stream_summary(rows):
    return {'questions': len(rows), 'unique_questions': len({r['question_id'] for r in rows}),
            'contracts_content_sha256': content_sha(rows),
            'legacy_contracts_content_sha256': content_sha(legacy_contracts(rows)),
            'supervised_target_ids_stream_sha256': content_sha([r['supervised_target_ids'] for r in rows]),
            'supervised_answer_roles_stream_sha256': content_sha([r['supervised_answer_roles'] for r in rows])}


def prepare_training_inputs(config, seed, condition):
    """Explicit CPU stage after teacher merge, before any actor GPU preflight."""
    from vlm_probe.evaluate import processor_for
    from .data import build_plan, load_rows, load_teacher_replies
    from .train import validate_config, validate_condition
    validate_config(config)
    validate_condition(condition, seed)
    objects, replays = load_rows(config)
    plan = build_plan(config, seed)
    replies = load_teacher_replies(config, seed)
    binding = _binding(config, seed, condition, plan, replies)
    reference = json.loads(private_path(config['reference_config']).read_text())
    processor = processor_for(private_path(config['model_root']), reference, config['visual_token_budget'])
    result = {'object': [], 'replay': []}
    for block in plan['blocks']:
        for source, rows in (('object', objects), ('replay', replays)):
            indices = block[f'{source}_indices']
            for start in range(0, len(indices), config['training_microbatch']):
                _, contracts, _ = encode_group(processor, rows, indices[start:start+4], config,
                    source=source, condition=condition, teacher_replies=replies)
                result[source].extend(contracts)
    p = paths(config, seed, condition)
    p['manifest'].parent.mkdir(parents=True, exist_ok=True)
    lock_json(p['contracts'], result)
    manifest = {**binding, 'contracts_file': str(p['contracts']),
                'contracts_file_sha256': sha_file(p['contracts']),
                'streams': {source: _stream_summary(rows) for source, rows in result.items()}}
    lock_json(p['manifest'], manifest)
    verify_prepared_training_inputs(config, seed, condition)
    return manifest


def verify_prepared_training_inputs(config, seed, condition):
    """Strict CPU metadata/target checker; no processor/model/probability forward."""
    from .data import build_plan, load_rows, load_teacher_replies
    from .train import validate_config, validate_condition
    validate_config(config)
    validate_condition(condition, seed)
    objects, replays = load_rows(config)
    plan = build_plan(config, seed)
    replies = load_teacher_replies(config, seed)
    binding = _binding(config, seed, condition, plan, replies)
    p = paths(config, seed, condition)
    manifest, contracts = json.loads(p['manifest'].read_text()), json.loads(p['contracts'].read_text())
    if any(manifest.get(key) != value for key, value in binding.items()):
        raise ValueError('prepared training input config/source/teacher/plan binding differs')
    if manifest['contracts_file'] != str(p['contracts']) or manifest['contracts_file_sha256'] != sha_file(p['contracts']):
        raise ValueError('prepared complete actual training-contract file differs')
    for source, count in (('object', 2888), ('replay', 5792)):
        rows = contracts[source]
        expected_qids = [q for group in plan['schedule'][f'{source}_question_ids_by_update'] for q in group]
        if ([r['question_id'] for r in rows] != expected_qids or len(rows) != count
                or len({r['question_id'] for r in rows}) != count
                or _stream_summary(rows) != manifest['streams'][source]):
            raise ValueError('prepared actual training stream order/count/hash differs')
        for row in rows:
            ids, roles = row['supervised_target_ids'], row['supervised_answer_roles']
            if (len(ids) != len(roles) or len(ids) != row['assistant_tokens']
                    or any(type(v) is not int or v not in (1,2,3,4) for v in roles)
                    or content_sha(ids) != row['supervised_target_ids_sha256']
                    or content_sha(roles) != row['supervised_answer_roles_sha256']):
                raise ValueError('prepared actual target IDs/roles are invalid')
            if source == 'replay' and condition == CANDIDATE:
                reply = replies[row['question_id']]
                if (ids != reply['training_target_ids'] or roles != reply['training_answer_roles']
                        or row['teacher_reply_row_sha256'] != content_sha(reply)
                        or row['target_source'] != 'fixed_initial_greedy_ids'):
                    raise ValueError('prepared teacher target is not exact generated-ID target')
            elif (row['teacher_reply_row_sha256'] is not None
                  or row['target_source'] != ('object_ground_truth' if source == 'object' else 'original_ground_truth')):
                raise ValueError('ground-truth sample falsely claims teacher labels')
    # Compare full legacy contracts to the already completed matching pilot;
    # no old encoder/model, old probability bank or large checkpoint is loaded.
    if seed == 42:
        historical = json.loads((private_path(config['historical_coverage_study_root']) /
                    'runs/wide32/seed42/train/TRAIN_RESULT.json').read_text())
        for source in (('object','replay') if condition == CONTROL else ('object',)):
            if manifest['streams'][source]['legacy_contracts_content_sha256'] != historical[f'{source}_input_contract_sha256']:
                raise ValueError('object/GT actual legacy stream differs from frozen wide42 actual stream')
    return {'manifest': manifest, 'contracts': contracts, 'plan': plan,
            'objects': objects, 'replays': replays, 'teacher_replies': replies,
            'manifest_file_sha256': sha_file(p['manifest']),
            'teacher_replies_manifest': json.loads(p['teacher'].read_text()),
            'teacher_replies_lock': json.loads(p['teacher_lock'].read_text())}


def verify_prepared_files_unchanged(config, seed, condition, context):
    p = paths(config, seed, condition)
    manifest = context['manifest']
    expected = {'plan': manifest['plan_file_sha256'], 'prompt': manifest['prompt_manifest_file_sha256'],
                'teacher': manifest['teacher_replies_manifest_file_sha256'],
                'teacher_lock': manifest['teacher_replies_lock_file_sha256'],
                'contracts': manifest['contracts_file_sha256'], 'manifest': context['manifest_file_sha256']}
    for key, digest in expected.items():
        if sha_file(p[key]) != digest:
            raise ValueError(f'prepared data/teacher file changed during training: {key}')
    for field,key in (('data_admission','data_admission_file_sha256'),
                      ('train_manifest','train_manifest_file_sha256'),
                      ('replay_manifest','replay_manifest_file_sha256')):
        if sha_file(private_path(config[field])) != manifest[key]:
            raise ValueError('data admission/source manifest changed during training')
    from .train import source_hashes
    if source_hashes() != manifest['source_sha256'] or content_sha(config) != manifest['config_sha256']:
        raise ValueError('scientific source/config changed during training')
    return True


def verify_training_result(config, directory, *, condition, seed):
    """Full CPU input/target/provenance validator; adapter tensors are eval-owned."""
    context = verify_prepared_training_inputs(config, seed, condition)
    directory = private_path(directory)
    result = json.loads((directory/'TRAIN_RESULT.json').read_text())
    lock = json.loads((directory/'TRAIN_INPUT_LOCK.json').read_text())
    p = paths(config, seed, condition)
    if (result.get('schema_version') != SCHEMA or result.get('condition') != condition
            or result.get('seed') != seed or result.get('status') != 'completed'
            or result.get('lineage_sha256') != content_sha(lock)
            or any(result.get(key) != value for key,value in lock.items())):
        raise ValueError('completed training result/lineage/condition differs')
    fixed = {'completed_updates':181,'planned_updates':181,'actual_object_exposures':2888,
             'actual_replay_exposures':5792,'actual_questions_seen':8680,'skipped_updates':0,
             'epochs_completed':1,'replay_rng_unchanged_updates':181,
             'replay_hard_ce_backward':True,'teacher_probability_cache_used':False,
             'teacher_reply_generation_new_training_calls':0,'position_diagnostics_extra_model_calls':0,
             'position_diagnostics_change_loss_weights':False,
             'prepared_inputs_and_teacher_replies_unchanged':True}
    for key,value in fixed.items():
        if result.get(key) != value or isinstance(result.get(key),bool) != isinstance(value,bool):
            raise ValueError(f'completed behavior training invariant differs: {key}')
    if (result['config_sha256'] != content_sha(config)
            or result['source_sha256'] != context['manifest']['source_sha256']
            or result['prepared_training_inputs_file_sha256'] != context['manifest_file_sha256']
            or result['prepared_training_inputs_content_sha256'] != content_sha(context['manifest'])
            or result['prepared_training_contracts_file_sha256'] != context['manifest']['contracts_file_sha256']
            or result['prepared_input_streams'] != context['manifest']['streams']):
        raise ValueError('completed training prepared/source/config binding differs')
    for kind,key in (('teacher','teacher_replies_manifest_file_sha256'),('teacher_lock','teacher_replies_lock_file_sha256')):
        if not (sha_file(p[kind]) == result[key] == result[f'{key}_after']):
            raise ValueError('teacher reply manifest/lock changed before or after training')
    actual_path = directory/'TRAINED_INPUT_CONTRACTS.json'
    actual = json.loads(actual_path.read_text())
    if actual != context['contracts'] or sha_file(actual_path) != result['trained_input_contracts_file_sha256']:
        raise ValueError('complete actual training inputs differ from frozen CPU stream')
    summaries = {}
    for source,rows in actual.items():
        summary = _stream_summary(rows)
        for suffix,key in (('input_contract_sha256','contracts_content_sha256'),
                           ('legacy_input_contract_sha256','legacy_contracts_content_sha256'),
                           ('supervised_target_ids_stream_sha256','supervised_target_ids_stream_sha256'),
                           ('supervised_answer_roles_stream_sha256','supervised_answer_roles_stream_sha256')):
            if result[f'{source}_{suffix}'] != summary[key]:
                raise ValueError('actual target/role/standard stream SHA differs from result')
        role_counts = {name:0 for name in ('first_content','later_content','actual_eos','template_suffix')}
        for row in rows:
            for role in row['supervised_answer_roles']:
                role_counts[('first_content','later_content','actual_eos','template_suffix')[role-1]] += 1
        diagnostics = result['position_role_statistics'][source]
        if set(diagnostics) != set(role_counts):
            raise ValueError('unknown/missing role diagnostics')
        import math
        for name,count in role_counts.items():
            value = diagnostics[name]
            role_id = ('first_content','later_content','actual_eos','template_suffix').index(name)+1
            role_questions = sum(role_id in row['supervised_answer_roles'] for row in rows)
            if (value['token_count'] != count or not math.isfinite(value['ce_sum']) or value['ce_sum'] < 0
                    or value['questions_with_role'] != role_questions
                    or (count == 0 and value['mean_token_ce'] is not None)
                    or (count > 0 and value['mean_token_ce'] != value['ce_sum']/count)):
                raise ValueError('actual role token counts/CE diagnostics differ')
        totals = {'input_tokens':sum(r['input_tokens'] for r in rows),
                  'visual_tokens':sum(r['visual_tokens'] for r in rows),
                  'answer_tokens':sum(r['assistant_tokens'] for r in rows),
                  'microbatches':len(rows)//4,
                  'padding_tokens':sum(r['microbatch_sequence_length']-r['sequence_length'] for r in rows)}
        if any(result[f'training_{source}_{key}'] != value for key,value in totals.items()):
            raise ValueError('actual source token/microbatch accounting differs')
        summaries[source] = {**summary,**totals,'role_counts':role_counts}
    if result['replay_coverage']['unique_questions_consumed'] != 5792:
        raise ValueError('actual unique replay count differs')
    verify_prepared_files_unchanged(config,seed,condition,context)
    return {'schema_version':'rarevlm.object_behavior_replay.completed_input_verification.v1',
            'condition':condition,'seed':seed,'completed_updates':181,'streams':summaries,
            'teacher_replies_manifest_file_sha256':result['teacher_replies_manifest_file_sha256'],
            'teacher_replies_lock_file_sha256':result['teacher_replies_lock_file_sha256'],
            'prepared_training_inputs_file_sha256':context['manifest_file_sha256'],
            'teacher_probability_cache_used':False,'teacher_reply_generation_new_training_calls':0,
            'all_actual_target_ids_roles_and_input_contracts_verified':True}
