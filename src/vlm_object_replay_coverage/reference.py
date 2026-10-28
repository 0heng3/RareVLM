"""Fresh coverage teacher or immutable old-bank reference, never a moving actor.

The old producer UUID and cache binding remain unchanged. New consumer UUID,
actor occurrences, unique coverage and actual new teacher computation are kept
separate. No local probability tensors are copied for the repeat8x4 control.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from vlm_deep.common import content_sha, sha_file, write_json
from vlm_object_train.data import private_path
from vlm_object_train.schedule import build_schedule as original_schedule
from vlm_object_train.train import lock_json, trainable_hash
from vlm_object_kl.cache import TeacherCache, build_teacher_cache, load_entry, verify_frozen_cache

CONDITIONS = ('wide32', 'repeat8x4')
SOURCE_SCHEMA = 'rarevlm.object_replay_coverage.teacher_source.v1'


def schedule_for_condition(objects, replays, seed, epochs=1, effective_batch=16, replay_per_update=32, *, condition):
    if condition not in CONDITIONS or replay_per_update != 32 or (condition == 'repeat8x4' and seed != 42):
        raise ValueError('requires declared replay32 condition; repeat8x4 only seed42')
    blocks = original_schedule(objects, replays, seed, epochs, effective_batch,
                               32 if condition == 'wide32' else 8)
    if condition == 'repeat8x4':
        for block in blocks:
            block['replay_indices'] = block['replay_indices'] * 4
    return blocks


def source_microbatch(condition, new_microbatch):
    if condition not in CONDITIONS or type(new_microbatch) is not int or not 0 <= new_microbatch < 8:
        raise ValueError('requires one of8 declared actor replay microbatches')
    return new_microbatch if condition == 'wide32' else new_microbatch % 2


def make_reference_map(condition, blocks, replays, manifest):
    entries = []
    for update, block in enumerate(blocks):
        for micro in range(8):
            bank_micro = source_microbatch(condition, micro)
            key = f'update_{update:06d}_replay_{bank_micro:03d}'
            entry = manifest['entries'][key]
            ids = [str(replays[index]['question_id']) for index in block['replay_indices'][micro * 4:micro * 4 + 4]]
            if ids != entry['identity']['question_ids']:
                raise ValueError('immutable reference QA sequence differs from actual declared replay stream')
            entries.append({'update_index': update, 'actor_microbatch_index': micro,
                'source_microbatch_index': bank_micro, 'source_key': key,
                'question_ids': ids, 'actual_padded_input_sha256': entry['identity']['actual_padded_input_sha256'],
                'source_file_sha256': entry['file_sha256'], 'source_tensor_sha256': entry['tensor_sha256']})
    return {'schema_version': 'rarevlm.object_replay_coverage.reference_map.v1',
        'condition': condition, 'entries': entries, 'actor_replay_occurrences': len(entries) * 4,
        'unique_questions_referenced': len({qid for entry in entries for qid in entry['question_ids']}),
        'unique_source_microbatches_referenced': len({entry['source_key'] for entry in entries})}


class TeacherSource:
    def __init__(self, bank, record, mapping):
        self.bank, self.record, self.mapping = bank, record, mapping
        self.directory, self.binding = bank.directory, bank.binding
        self.manifest, self.manifest_sha256 = bank.manifest, bank.manifest_sha256
        self.scope = record['scope']
        self.lookup = {(entry['update_index'], entry['actor_microbatch_index']): entry for entry in mapping['entries']}

    def load(self, update, microbatch, batch, contracts, device):
        entry = self.lookup[(update, microbatch)]
        if ([item['question_id'] for item in contracts] != entry['question_ids']
                or contracts[0]['microbatch_input_sha256'] != entry['actual_padded_input_sha256']):
            raise ValueError('new actor reference occurrence IDs/pixels/grid/labels/padding differ from frozen bank')
        # The original binding is passed intact, with the actual old micro index.
        # TeacherCache.load then checks full actual IDs/mask/labels/grid/tensor SHA.
        return self.bank.load(update, entry['source_microbatch_index'], batch, contracts, device)

    def verify(self):
        map_file = private_path(self.record['reference_map_path'])
        record_file = map_file.parent / 'TEACHER_SOURCE.json'
        if (sha_file(map_file) != self.record['reference_map_sha256']
                or json.loads(record_file.read_text()) != self.record
                or sha_file(private_path(self.record['source_binding_path'])) != self.record['source_binding_file_sha256']):
            raise ValueError('new reference mapping/record or original source binding changed')
        return self.bank.verify()


def create_teacher_source(model, processor, replays, blocks, config, directory, binding, device,
                          *, condition, preflight_only):
    directory = private_path(directory)
    selected_blocks = blocks[:1] if preflight_only else blocks
    scope = 'engineering_first_update_only' if preflight_only else 'formal_full_stream'
    if condition == 'wide32':
        qids = [[str(replays[i]['question_id']) for i in block['replay_indices']] for block in selected_blocks]
        cache_binding = {**binding, 'reference_scope': scope,
            'planned_updates': len(selected_blocks), 'planned_replay_exposures': len(selected_blocks) * 32,
            'replay_questions_per_update': 32, 'replay_order_sha256': content_sha(qids),
            'replay_microbatch_order_sha256': content_sha([[ids[i:i+4] for i in range(0,32,4)] for ids in qids])}
        bank_directory = directory / 'teacher_cache'
        manifest = build_teacher_cache(model, processor, replays, selected_blocks, config,
                                       bank_directory, cache_binding, device)
        manifest_sha = sha_file(bank_directory / 'MANIFEST.json')
        bank = TeacherCache(bank_directory, cache_binding, manifest_sha)
        original_train_file = None
        kind, new_questions, history_questions = 'fresh_full_fp32', len(selected_blocks) * 32, 0
    elif condition == 'repeat8x4':
        original_train_file = private_path(config['teacher_reference_study_root']) / 'runs/object_kl/seed42/train/TRAIN_RESULT.json'
        old = json.loads(original_train_file.read_text())
        if (old.get('status') != 'completed' or old.get('schema_version') != 'rarevlm.object_kl.train.v1'
                or old['initial_adapter_files_sha256'] != binding['initial_adapter_files_sha256']
                or old['initial_adapter_tensor_sha256'] != binding['initial_adapter_tensor_sha256']
                or old['frozen_parameter_sha256_before'] != binding['frozen_parameter_sha256_before']):
            raise ValueError('reference bank initial/base/source differs from declared frozen KL study')
        bank_directory = private_path(old['teacher_cache_directory'])
        if bank_directory != original_train_file.parent / 'teacher_cache':
            raise ValueError('historical bank must belong to original completed KL training')
        manifest_sha = sha_file(bank_directory / 'MANIFEST.json')
        if (manifest_sha != old['teacher_cache_manifest_sha256'] or manifest_sha != old['teacher_cache_manifest_sha256_after']
                or old['teacher_cache_unchanged'] is not True):
            raise ValueError('historical full teacher bank changed')
        cache_binding = json.loads((bank_directory / 'INPUT_LOCK.json').read_text())
        if content_sha(cache_binding) != old['teacher_cache_binding_sha256']:
            raise ValueError('historical initial/producer binding changed')
        bank = TeacherCache(bank_directory, cache_binding, manifest_sha)
        manifest = bank.manifest
        if manifest['teacher_forward_questions'] != 1448:
            raise ValueError('control requires original complete1448-question teacher bank')
        kind, new_questions, history_questions = 'immutable_repeated_reference', 0, 1448
    else:
        raise ValueError('unknown replay coverage condition')
    if (trainable_hash(model) != binding['initial_adapter_tensor_sha256']
            or manifest['teacher_parameters_sha256'] != binding['initial_adapter_tensor_sha256']
            or manifest['frozen_parameters_sha256'] != binding['frozen_parameter_sha256_before']):
        raise ValueError('reference must be fixed original initial master/base, never current actor')
    mapping = make_reference_map(condition, selected_blocks, replays, manifest)
    map_file = directory / 'TEACHER_REFERENCE_MAP.json'
    lock_json(map_file, mapping)
    source = {'schema_version': SOURCE_SCHEMA, 'kind': kind, 'scope': scope,
        'source_manifest_path': str(bank_directory / 'MANIFEST.json'), 'source_manifest_sha256': manifest_sha,
        'source_binding_path': str(bank_directory / 'INPUT_LOCK.json'),
        'source_binding_file_sha256': sha_file(bank_directory / 'INPUT_LOCK.json'),
        'source_binding_content_sha256': content_sha(cache_binding),
        'source_training_result_path': str(original_train_file) if original_train_file else None,
        'source_training_result_sha256': sha_file(original_train_file) if original_train_file else None,
        'producer_gpu_uuid': cache_binding['hardware']['uuid'], 'consumer_gpu_uuid': binding['hardware']['uuid'],
        'mapping': 'identity' if condition == 'wide32' else 'new_microbatch_mod2_same_update',
        'actual_new_teacher_forward_questions': new_questions,
        'actual_new_teacher_forward_seconds': manifest['teacher_forward_seconds_all_batches'] if new_questions else 0.,
        'historical_teacher_bank_questions': history_questions,
        'historical_teacher_bank_forward_seconds': manifest['teacher_forward_seconds_all_batches'] if history_questions else 0.,
        'source_teacher_questions': manifest['teacher_forward_questions'],
        'source_teacher_unique_questions_referenced': mapping['unique_questions_referenced'],
        'actor_replay_questions_planned': len(selected_blocks) * 32,
        'actor_replay_unique_questions_planned': mapping['unique_questions_referenced'],
        'reference_map_path': str(map_file), 'reference_map_sha256': sha_file(map_file)}
    lock_json(directory / 'TEACHER_SOURCE.json', source)
    return TeacherSource(bank, source, mapping), source, manifest


def verify_teacher_source(train, train_dir, config):
    """Full CPU verifier for the new adapter binding, including repeated refs."""
    train_dir = private_path(train_dir)
    source_file = train_dir / 'TEACHER_SOURCE.json'
    source = json.loads(source_file.read_text())
    if (source != train.get('teacher_source') or sha_file(source_file) != train.get('teacher_source_file_sha256')
            or source.get('schema_version') != SOURCE_SCHEMA or source.get('scope') != 'formal_full_stream'
            or train.get('teacher_source_unchanged') is not True
            or train.get('teacher_source_manifest_sha256_after') != source['source_manifest_sha256']):
        raise ValueError('complete formal reference source identity/scope/before-after differs')
    manifest_file, binding_file = private_path(source['source_manifest_path']), private_path(source['source_binding_path'])
    if (sha_file(manifest_file) != source['source_manifest_sha256']
            or sha_file(binding_file) != source['source_binding_file_sha256']):
        raise ValueError('original immutable teacher manifest/input binding changed')
    binding = json.loads(binding_file.read_text())
    if (content_sha(binding) != source['source_binding_content_sha256']
            or source['producer_gpu_uuid'] != binding['hardware']['uuid']
            or source['consumer_gpu_uuid'] != train['hardware']['uuid']):
        raise ValueError('teacher producer or new consumer UUID identity differs')
    manifest = verify_frozen_cache(manifest_file.parent, source['source_manifest_sha256'])
    if (manifest['teacher_parameters_sha256'] != train['initial_adapter_tensor_sha256']
            or manifest['frozen_parameters_sha256'] != train['frozen_parameter_sha256_before']):
        raise ValueError('teacher initial/base differs from actor initialization')
    condition = train['condition']
    if condition == 'wide32':
        expected_kind, expected_new, expected_history, expected_unique = 'fresh_full_fp32', 5792, 0, 5792
        if manifest_file.parent != train_dir / 'teacher_cache':
            raise ValueError('fresh teacher must be local to its formal run')
    elif condition == 'repeat8x4' and train['seed'] == 42:
        expected_kind, expected_new, expected_history, expected_unique = 'immutable_repeated_reference', 0, 1448, 1448
        old_file = private_path(config['teacher_reference_study_root']) / 'runs/object_kl/seed42/train/TRAIN_RESULT.json'
        old = json.loads(old_file.read_text())
        if (source['source_training_result_path'] != str(old_file)
                or source['source_training_result_sha256'] != sha_file(old_file)
                or manifest_file.parent != old_file.parent / 'teacher_cache'
                or manifest['teacher_forward_questions'] != 1448
                or old['teacher_cache_manifest_sha256_after'] != source['source_manifest_sha256']):
            raise ValueError('control must reference original complete unmodified1448 bank')
    else:
        raise ValueError('invalid coverage condition/seed')
    if (source['kind'] != expected_kind or source['actual_new_teacher_forward_questions'] != expected_new
            or source['historical_teacher_bank_questions'] != expected_history
            or source['actor_replay_questions_planned'] != 5792
            or source['actor_replay_unique_questions_planned'] != expected_unique
            or source['source_teacher_unique_questions_referenced'] != expected_unique
            or source['source_teacher_questions'] != manifest['teacher_forward_questions']
            or source['mapping'] != ('identity' if condition == 'wide32' else 'new_microbatch_mod2_same_update')
            or source['actual_new_teacher_forward_seconds'] != (manifest['teacher_forward_seconds_all_batches'] if expected_new else 0.)
            or source['historical_teacher_bank_forward_seconds'] != (manifest['teacher_forward_seconds_all_batches'] if expected_history else 0.)):
        raise ValueError('actual new teacher/old bank/actor exposure/unique accounting differs')
    parity = train['preflight']['record']['initial_replay_kl_per_question']
    if (train['preflight'].get('status') != 'passed' or train['preflight'].get('formal_optimizer_updates') != 0
            or len(parity) != 32 or any(not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value) > 1e-5 for value in parity)):
        raise ValueError('formal fixed-reference initial32QA parity missing or failed')
    map_file = private_path(source['reference_map_path'])
    if map_file != train_dir / 'TEACHER_REFERENCE_MAP.json' or sha_file(map_file) != source['reference_map_sha256']:
        raise ValueError('new occurrence-to-source mapping changed')
    mapping = json.loads(map_file.read_text())
    actual_file = train_dir / 'TRAINED_INPUT_CONTRACTS.json'
    if sha_file(actual_file) != train['trained_input_contracts_file_sha256']:
        raise ValueError('actual actor training input records changed')
    replay = json.loads(actual_file.read_text())['replay']
    if len(replay) != 5792 or content_sha(replay) != train['replay_input_contract_sha256'] or len(mapping['entries']) != 1448:
        raise ValueError('full5792 actor inputs or reference occurrence count differs')
    checked = set()
    for occurrence, entry in enumerate(mapping['entries']):
        update, micro = divmod(occurrence,8)
        bank_micro = source_microbatch(condition,micro)
        key = f'update_{update:06d}_replay_{bank_micro:03d}'
        bank_entry = manifest['entries'][key]
        identity = bank_entry['identity']
        actual = replay[occurrence*4:occurrence*4+4]
        if (entry['update_index'] != update or entry['actor_microbatch_index'] != micro
                or entry['source_microbatch_index'] != bank_micro or entry['source_key'] != key
                or entry['question_ids'] != [item['question_id'] for item in actual]
                or identity['input_contracts'] != actual
                or entry['actual_padded_input_sha256'] != actual[0]['microbatch_input_sha256']
                or entry['source_file_sha256'] != bank_entry['file_sha256']
                or entry['source_tensor_sha256'] != bank_entry['tensor_sha256']):
            raise ValueError('reference mapping does not exactly match actor IDs/masks/pixels/grid/labels/padding')
        if key not in checked:
            payload = load_entry(manifest_file.parent / f'{key}.pt',identity,bank_entry)
            if payload['logprobs'].shape[1] != 151936:
                raise ValueError('teacher must contain full151936 model vocabulary')
            checked.add(key)
    return {'kind':source['kind'],'scope':source['scope'],'source_manifest_path':str(manifest_file),
        'source_manifest_sha256':source['source_manifest_sha256'],'producer_gpu_uuid':source['producer_gpu_uuid'],
        'consumer_gpu_uuid':source['consumer_gpu_uuid'],'actual_new_teacher_forward_questions':expected_new,
        'historical_teacher_bank_questions':expected_history,'actor_replay_exposures':5792,
        'actor_replay_unique_questions':expected_unique,'source_microbatches_verified':len(checked),
        'actor_microbatch_occurrences_verified':1448,'actual_padded_actor_inputs_matched':True,
        'full_cache_files_and_tensors_verified':True,'initial32QA_parity_passed':True}
