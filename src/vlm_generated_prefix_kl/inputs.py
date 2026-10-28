"""Sealed parent-layout references and exact runtime contracts for online KL."""
from __future__ import annotations
import json
import os
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

from vlm_deep.common import content_sha, sha_file
from vlm_object_train.data import private_path
from vlm_object_train.train import batch_hash
from vlm_object_behavior_replay.inputs import LEGACY_FIELDS, legacy_contracts, _stream_summary
from . import CANDIDATE, SCHEMA
from .data import (atomic_json, immutable_json, sealed, check_seal, read_json, load_parent_bundle,
    load_rows, build_plan, load_teacher_replies, validate_parent_proofs)


def paths(config, seed, condition):
    if condition != CANDIDATE:
        raise ValueError('Only teacher_kl32 owns new prepared inputs.')
    root = private_path(config['output_root']) / 'prepared' / f'seed{seed}'
    teacher = private_path(config['teacher_reply_root']) / f'seed{seed}'
    return {'plan': root / 'PLAN.json', 'prompt': root / 'PROMPT_MANIFEST.json',
        'manifest': root / f'TRAIN_INPUTS_{condition}.json', 'contracts': root / f'TRAIN_CONTRACTS_{condition}.json',
        'teacher': teacher / 'MANIFEST.json', 'teacher_lock': teacher / 'TEACHER_REPLIES_LOCK.json'}


def encode_group(processor, rows, selected, config, *, source, condition, teacher_replies):
    from .data import encode_training_sample, collate_training, sample_contract
    samples = [encode_training_sample(processor, rows[i], config, source=source,
        condition=condition, teacher_replies=teacher_replies) for i in selected]
    batch = collate_training(samples, processor.tokenizer.pad_token_id)
    micro_sha = batch_hash({k: value for k, value in batch.items() if k != 'answer_roles'})
    contracts = []
    for position, (index, sample) in enumerate(zip(selected, samples)):
        row = rows[index]; valid = sample['labels'][0] != -100
        ids, roles = sample['labels'][0][valid].tolist(), sample['answer_roles'][0][valid].tolist()
        reply_sha = None
        if source == 'replay':
            reply = teacher_replies[str(row['question_id'])]
            if ids != reply['training_target_ids'] or roles != reply['training_answer_roles']:
                raise ValueError('Runtime target/roles differ from exact fixed generated prefix.')
            # Compact accepted references are not the original full producer row.
            reply_sha = (reply['source_teacher_reply_row_sha256'] if 'source_teacher_reply_row_sha256' in reply else content_sha(reply))
        contracts.append({**sample_contract(sample, row, config),
            'microbatch_input_sha256': micro_sha, 'microbatch_question_position': position,
            'microbatch_size': len(selected), 'microbatch_sequence_length': int(batch['input_ids'].shape[1]),
            'supervised_target_ids': ids, 'supervised_answer_roles': roles,
            'supervised_target_ids_sha256': content_sha(ids), 'supervised_answer_roles_sha256': content_sha(roles),
            'target_source': 'object_ground_truth' if source == 'object' else 'fixed_initial_greedy_ids',
            'teacher_reply_row_sha256': reply_sha})
    return batch, contracts, samples


def _source_hashes():
    from .train import source_hashes
    return source_hashes()


def _plan42(config, bundle):
    parent = bundle['parent_plan']
    return sealed({'schema_version': 'rarevlm.generated_prefix_kl.plan.v1', 'seed': 42,
        'condition': CANDIDATE, 'replay_objective': 'forward_kl_full_vocabulary', 'config_sha256': content_sha(config),
        **{key: parent[key] for key in ('blocks', 'schedule', 'counts', 'object_order_sha256', 'replay_order_sha256')},
        'parent_plan_file_sha256': bundle['proofs'][config['seed42_input_reference']['plan_file']],
        'parent_input_layout_condition': 'teacher_ce32'}, 'plan_sha256')


def _reference42(config, bundle, p):
    """Create small new-study references; never copy or parse the large teacher manifest."""
    ref, proofs = config['seed42_input_reference'], bundle['proofs']
    parent_root = private_path(config['parent_output_root'])
    review = read_json(parent_root / 'independent_review/TEACHER_COMPLETION_REVIEW.json')
    common = {'seed': 42, 'condition': CANDIDATE, 'parent_input_layout_condition': 'teacher_ce32',
        'replay_objective': 'forward_kl_full_vocabulary', 'parent_proof_files_sha256': proofs,
        'new_generation_questions': 0, 'new_CPU_full_input_encodings': 0}
    immutable_json(p['plan'], _plan42(config, bundle))
    immutable_json(p['prompt'], sealed({'schema_version': 'rarevlm.generated_prefix_kl.prompt_reference.v1',
        **common, 'questions': 5792, 'source_prompt_manifest_file': str(bundle['prompt_manifest_path']),
        'source_prompt_manifest_file_sha256': sha_file(bundle['prompt_manifest_path']),
        'source_prompt_inputs_file': str(bundle['prompt_inputs_path']),
        'source_prompt_inputs_file_sha256': bundle['prompt_manifest']['prompt_inputs_file_sha256']}))
    immutable_json(p['teacher'], sealed({'schema_version': 'rarevlm.generated_prefix_kl.teacher_reply_reference.v1',
        'status': 'accepted_parent_reference', **common, 'questions': 5792,
        'source_manifest_file': ref['teacher_reply_manifest_file'],
        'source_manifest_file_sha256': proofs[ref['teacher_reply_manifest_file']],
        'source_target_index_file': ref['teacher_target_index_file'],
        'source_target_index_file_sha256': proofs[ref['teacher_target_index_file']],
        'source_completion_review_file': str(parent_root / 'independent_review/TEACHER_COMPLETION_REVIEW.json'),
        'source_completion_review_file_sha256': proofs[str(parent_root / 'independent_review/TEACHER_COMPLETION_REVIEW.json')],
        'generation_spec': config['teacher_generation'], 'producer_shards': review['shards'],
        'parent_actual_generation_accounting': review['actual_accounting'],
        'accounting_scope': 'Shared accepted parent generation; zero new generation in this study seed42.'}))
    immutable_json(p['teacher_lock'], sealed({'schema_version': 'rarevlm.generated_prefix_kl.teacher_reply_reference_lock.v1',
        **common, 'manifest_file_sha256': sha_file(p['teacher']),
        'source_teacher_lock_file': str(parent_root / 'teacher_replies/seed42/TEACHER_REPLIES_LOCK.json'),
        'source_teacher_lock_file_sha256': proofs[str(parent_root / 'teacher_replies/seed42/TEACHER_REPLIES_LOCK.json')]}, 'lock_sha256'))
    immutable_json(p['contracts'], sealed({'schema_version': 'rarevlm.generated_prefix_kl.training_contract_reference.v1',
        **common, 'source_contracts_file': ref['train_contracts_file'],
        'source_contracts_file_sha256': proofs[ref['train_contracts_file']],
        'streams': bundle['parent_header']['streams']}))


def _binding(config, seed, condition, plan, contracts, p):
    from .contracts import validate_inherited_admissions
    validate_inherited_admissions(config)
    if read_json(p['plan']) != plan:
        raise ValueError('Saved new PLAN differs from exact parent/fresh schedule.')
    teacher = read_json(p['teacher'])
    parent = read_json(config['seed42_input_reference']['train_inputs_file'])
    return {'schema_version': 'rarevlm.generated_prefix_kl.prepared_training_inputs.v1',
        'condition': condition, 'seed': seed, 'replay_objective': 'forward_kl_full_vocabulary',
        'parent_input_layout_condition': 'teacher_ce32' if seed == 42 else None,
        'config_sha256': content_sha(config), 'source_sha256': _source_hashes(),
        'parent_proof_files_sha256': dict(config['parent_proof_files_sha256']),
        'plan_file_sha256': sha_file(p['plan']), 'plan_content_sha256': content_sha(plan),
        'schedule_sha256': content_sha(plan['schedule']),
        'prompt_manifest_file_sha256': sha_file(p['prompt']),
        'teacher_replies_manifest_file': str(p['teacher']), 'teacher_replies_lock_file': str(p['teacher_lock']),
        'teacher_replies_manifest_file_sha256': sha_file(p['teacher']),
        'teacher_replies_lock_file_sha256': sha_file(p['teacher_lock']),
        'teacher_reply_rows': 5792, 'replay_target_source': 'fixed_initial_greedy_ids',
        'historical_wide42_training_result_file_sha256': parent['historical_wide42_training_result_file_sha256'],
        'data_admission_file_sha256': sha_file(private_path(config['data_admission'])),
        'train_manifest_file_sha256': sha_file(private_path(config['train_manifest'])),
        'replay_manifest_file_sha256': sha_file(private_path(config['replay_manifest'])),
        'contracts_file': str(p['contracts']), 'contracts_file_sha256': sha_file(p['contracts']),
        'contracts_storage': 'sealed_parent_reference' if seed == 42 else 'fresh_CPU_encoded_rows',
        'teacher_probability_cache_used': False,
        'teacher_reply_generation_new_training_calls': 0,
        'new_CPU_full_input_encodings': 0 if seed == 42 else 8680,
        'streams': {source: _stream_summary(rows) for source, rows in contracts.items()}}


def prepare_training_inputs(config, seed, condition):
    if condition != CANDIDATE or seed not in (0, 42, 200):
        raise ValueError('Only registered teacher_kl32 preparation is allowed.')
    p = paths(config, seed, condition)
    if p['manifest'].exists():
        return verify_prepared_training_inputs(config, seed, condition)['manifest']
    if seed == 42:
        bundle = load_parent_bundle(config)
        _reference42(config, bundle, p)
        contracts, plan = bundle['contracts'], read_json(p['plan'])
    else:
        from .data import require_replication_admission
        from vlm_probe.evaluate import processor_for
        require_replication_admission(config)
        objects, replays = load_rows(config); plan = build_plan(config, seed)
        replies = load_teacher_replies(config, seed)
        processor = processor_for(private_path(config['model_root']), read_json(config['reference_config']), config['visual_token_budget'])
        contracts = {'object': [], 'replay': []}
        for block in plan['blocks']:
            for source, rows in (('object', objects), ('replay', replays)):
                selected = block[f'{source}_indices']
                for start in range(0, len(selected), 4):
                    _, group, _ = encode_group(processor, rows, selected[start:start + 4], config,
                        source=source, condition=condition, teacher_replies=replies)
                    contracts[source].extend(group)
        immutable_json(p['contracts'], contracts)
    manifest = sealed(_binding(config, seed, condition, plan, contracts, p))
    immutable_json(p['manifest'], manifest)
    return manifest


def verify_prepared_training_inputs(config, seed, condition):
    p = paths(config, seed, condition)
    manifest = check_seal(read_json(p['manifest']))
    objects, replays = load_rows(config)
    if seed == 42:
        bundle = load_parent_bundle(config)
        plan, contracts, replies = _plan42(config, bundle), bundle['contracts'], bundle['teacher_replies']
        reference = check_seal(read_json(p['contracts']))
        teacher = check_seal(read_json(p['teacher']))
        teacher_lock = check_seal(read_json(p['teacher_lock']), 'lock_sha256')
        if (reference.get('schema_version') != 'rarevlm.generated_prefix_kl.training_contract_reference.v1'
                or reference.get('condition') != CANDIDATE or reference.get('replay_objective') != 'forward_kl_full_vocabulary'
                or reference.get('parent_input_layout_condition') != 'teacher_ce32'
                or teacher.get('schema_version') != 'rarevlm.generated_prefix_kl.teacher_reply_reference.v1'
                or teacher.get('status') != 'accepted_parent_reference'
                or teacher.get('source_manifest_file_sha256') != bundle['proofs'][config['seed42_input_reference']['teacher_reply_manifest_file']]
                or teacher_lock.get('manifest_file_sha256') != sha_file(p['teacher'])
                or reference.get('source_contracts_file') != config['seed42_input_reference']['train_contracts_file']
                or reference.get('source_contracts_file_sha256') != bundle['proofs'][reference['source_contracts_file']]
                or read_json(p['plan']) != plan):
            raise ValueError('New reference differs from accepted parent layout.')
    else:
        plan, contracts, replies = build_plan(config, seed), read_json(p['contracts']), load_teacher_replies(config, seed)
    binding = _binding(config, seed, condition, plan, contracts, p)
    if {k: v for k, v in manifest.items() if k != 'manifest_sha256'} != binding:
        raise ValueError('New prepared input identity/source/condition differs.')
    for source, expected_count in (('object', 2888), ('replay', 5792)):
        rows = contracts[source]
        expected = [q for group in plan['schedule'][f'{source}_question_ids_by_update'] for q in group]
        if len(rows) != expected_count or [r['question_id'] for r in rows] != expected:
            raise ValueError('Exact complete parent/fresh input order differs.')
        for row in rows:
            ids, roles = row['supervised_target_ids'], row['supervised_answer_roles']
            if (len(ids) != row['assistant_tokens'] or len(ids) != len(roles)
                    or content_sha(ids) != row['supervised_target_ids_sha256']
                    or content_sha(roles) != row['supervised_answer_roles_sha256']):
                raise ValueError('Prepared generated-prefix target/role digest differs.')
            if source == 'replay':
                reply = replies[row['question_id']]
                original_sha = (reply['source_teacher_reply_row_sha256'] if 'source_teacher_reply_row_sha256' in reply else content_sha(reply))
                if (ids != reply['training_target_ids'] or roles != reply['training_answer_roles']
                        or row['teacher_reply_row_sha256'] != original_sha):
                    raise ValueError('Teacher prefix/source reference differs from actual accepted IDs.')
    return {'manifest': manifest, 'contracts': contracts, 'plan': plan,
        'objects': objects, 'replays': replays, 'teacher_replies': replies,
        'manifest_file_sha256': sha_file(p['manifest']),
        'teacher_replies_manifest': read_json(p['teacher']), 'teacher_replies_lock': read_json(p['teacher_lock'])}


def verify_prepared_files_unchanged(config, seed, condition, context):
    p = paths(config, seed, condition); manifest = context['manifest']
    expected = {'plan': manifest['plan_file_sha256'], 'prompt': manifest['prompt_manifest_file_sha256'],
        'teacher': manifest['teacher_replies_manifest_file_sha256'], 'teacher_lock': manifest['teacher_replies_lock_file_sha256'],
        'contracts': manifest['contracts_file_sha256'], 'manifest': context['manifest_file_sha256']}
    for key, digest in expected.items():
        if sha_file(p[key]) != digest:
            raise ValueError(f'New prepared reference/input changed: {key}')
    validate_parent_proofs(config)
    for name, field in (('data_admission', 'data_admission_file_sha256'), ('train_manifest', 'train_manifest_file_sha256'),
            ('replay_manifest', 'replay_manifest_file_sha256')):
        if sha_file(private_path(config[name])) != manifest[field]:
            raise ValueError('Frozen training source/admission changed.')
    if _source_hashes() != manifest['source_sha256'] or content_sha(config) != manifest['config_sha256']:
        raise ValueError('New scientific source/config changed.')
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
             'replay_hard_ce_backward':False,'replay_objective':'forward_kl_full_vocabulary','teacher_probability_cache_used':False,
             'teacher_reply_generation_new_training_calls':0,'position_diagnostics_extra_model_calls':0,
             'position_diagnostics_change_loss_weights':False,
             'prepared_inputs_and_teacher_replies_unchanged':True,
             'formal_reference_forward_questions':5792,'formal_reference_forward_started_questions':5792,
             'reference_rng_unchanged_microbatches':1448,'reference_parameters_unchanged':True}
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
    import math
    kl_roles = result['replay_kl_position_role_statistics']
    replay_rows = actual['replay']
    for role_id, name in enumerate(('first_content','later_content','actual_eos','template_suffix'), 1):
        count = sum(row['supervised_answer_roles'].count(role_id) for row in replay_rows)
        questions = sum(role_id in row['supervised_answer_roles'] for row in replay_rows)
        value = kl_roles[name]
        if (value['token_count'] != count or value['questions_with_role'] != questions
                or not math.isfinite(value['kl_sum'])
                or (count == 0 and value['mean_token_kl'] is not None)
                or (count > 0 and value['mean_token_kl'] != value['kl_sum'] / count)):
            raise ValueError('Actual online-KL role counts/finite sums differ from target stream.')
    verify_prepared_files_unchanged(config,seed,condition,context)
    return {'schema_version':'rarevlm.generated_prefix_kl.completed_input_verification.v1',
            'condition':condition,'seed':seed,'completed_updates':181,'streams':summaries,
            'teacher_replies_manifest_file_sha256':result['teacher_replies_manifest_file_sha256'],
            'teacher_replies_lock_file_sha256':result['teacher_replies_lock_file_sha256'],
            'prepared_training_inputs_file_sha256':context['manifest_file_sha256'],
            'teacher_probability_cache_used':False,'teacher_reply_generation_new_training_calls':0,
            'all_actual_target_ids_roles_and_input_contracts_verified':True}
