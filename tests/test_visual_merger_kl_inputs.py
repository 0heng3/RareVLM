"""Synthetic input/source consumer contracts; no real processing or model calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vlm_visual_merger_kl import CANDIDATE
from vlm_visual_merger_kl import data, inputs, lock, initialization


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tiny-visual-input-', dir='/opt/data/private')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); (self.root / 'config.json').write_text(json.dumps({'vocab_size': 200000}))
        self.config = {'model_root': str(self.root), 'max_sequence_length': 2048, 'visual_token_budget': 1024}
        self.row = {'question_id': 'q', 'image_id': 'i', 'question': 'What is shown?', 'prompt_suffix': '\nAnswer.',
                    'target': 'SECRET_GROUND_TRUTH', 'answers': ['SECRET_GROUND_TRUTH'], 'image_path': '/unused/image.jpg'}
        self.reply = {'q': {'training_target_ids': [7, 151645, 198], 'training_answer_roles': [1, 3, 4], 'prompt_contract': {}}}

    def prompt(self, extra=False):
        ids = torch.tensor([[99, 8, 9] if extra else [99, 9]], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'pixel_values': torch.zeros((4, 2)), 'image_grid_thw': torch.tensor([[1, 2, 2]])}

    def sample(self, extra=False):
        with patch.object(data, 'encode', return_value=self.prompt(extra)) as encode, patch.object(data, 'verify_prompt'):
            result = data.encode_training_sample(None, self.row, self.config, source='replay', condition=CANDIDATE, teacher_replies=self.reply)
            row = encode.call_args.args[1]
            self.assertFalse(any(key in row for key in ('target', 'answers', 'labels', 'answer_roles')))
            self.assertIs(encode.call_args.args[3], False)
            return result

    def test_actual_teacher_ids_roles_no_retokenize_or_GT_prompt(self):
        sample = self.sample()
        self.assertEqual(sample['input_ids'].tolist(), [[99, 9, 7, 151645, 198]])
        self.assertEqual(sample['labels'].tolist(), [[-100, -100, 7, 151645, 198]])
        self.assertEqual(sample['answer_roles'].tolist(), [[0, 0, 1, 3, 4]])

    def test_left_padding_prompt_and_roles_are_zero_masked(self):
        a, b = self.sample(), self.sample(extra=True)
        batch = data.collate_training([a, b], 151643)
        self.assertEqual(int(batch['input_ids'][0, 0]), 151643)
        self.assertEqual(int(batch['labels'][0, 0]), -100)
        self.assertEqual(int(batch['answer_roles'][0, 0]), 0)
        self.assertTrue(torch.equal(batch['labels'] != -100, batch['answer_roles'] != 0))

    def test_no_missing_teacher_fallback_or_old_condition(self):
        for condition, replies in ((CANDIDATE, {}), ('teacher_kl32', self.reply)):
            with self.assertRaises(ValueError):
                data.encode_training_sample(None, self.row, self.config, source='replay', condition=condition, teacher_replies=replies)

    def test_wrong_prompt_contract_rejected(self):
        with patch.object(data, 'encode', return_value=self.prompt()), patch.object(data, 'verify_prompt', side_effect=ValueError('wrong prefix')):
            with self.assertRaisesRegex(ValueError, 'wrong prefix'):
                data.encode_training_sample(None, self.row, self.config, source='replay', condition=CANDIDATE, teacher_replies=self.reply)

    def test_seed42_never_starts_fresh_generation(self):
        with patch.object(data, '_prepare_fresh_generation_inputs', side_effect=AssertionError('fresh processing forbidden')):
            with self.assertRaisesRegex(ValueError, 'seed42 must reuse'):
                data.prepare_generation_inputs({}, 42)

    def test_replication_requires_this_study_pilot_before_processing(self):
        with patch.object(data, 'require_replication_admission', side_effect=ValueError('pilot not passed')), patch.object(data, '_prepare_fresh_generation_inputs') as fresh:
            with self.assertRaisesRegex(ValueError, 'pilot not passed'):data.prepare_generation_inputs({}, 0)
            fresh.assert_not_called()

    def test_namespace_and_prepared_paths(self):
        config = {'output_root': str(self.root), 'teacher_reply_root': str(self.root / 'teacher_replies')}
        self.assertEqual(inputs.paths(config, 42, CANDIDATE)['manifest'].name, 'TRAIN_INPUTS_merger_kl32.json')
        self.assertEqual(data.PLAN_SCHEMA, 'rarevlm.visual_merger_kl.plan.v1')
        with self.assertRaises(ValueError):inputs.paths(config, 42, 'teacher_kl32')

    def test_R14_keeps_teacher_bank_SHA_declaration_without_reading_bank(self):
        parent = self.root / 'parent'
        parent_config = self.root / 'parent_config.json'
        names = ('STUDY_INPUT_LOCK.json', 'recovery_v1/RECOVERY_INPUT_LOCK.json',
            'analysis/pilot_seed42/screen/ANALYSIS.json', 'independent_review/PILOT_PREDICTION_REVIEW_RECOVERY_V1.json',
            'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json', 'independent_review/TEACHER_COMPLETION_REVIEW.json',
            'independent_review/TRAINING_COMPLETION_REVIEW.json', 'teacher_replies/seed42/MANIFEST.json',
            'teacher_replies/seed42/TEACHER_REPLIES_LOCK.json', 'independent_review/TEACHER_TARGET_INDEX.json',
            'prepared/seed42/PLAN.json', 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json',
            'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json')
        proofs = {str(parent_config): '0' * 64, **{str(parent / name): '0' * 64 for name in names}}
        bank = parent / 'teacher_replies/seed42/MANIFEST.json'
        reference = {'parent_condition': 'teacher_ce32', 'new_training_condition': CANDIDATE,
            'plan_file': str(parent / 'prepared/seed42/PLAN.json'),
            'train_inputs_file': str(parent / 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json'),
            'train_contracts_file': str(parent / 'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json'),
            'teacher_target_index_file': str(parent / 'independent_review/TEACHER_TARGET_INDEX.json'),
            'teacher_reply_manifest_file': str(bank), 'new_generation_questions': 0, 'new_CPU_full_input_encodings': 0}
        config = {'parent_output_root': str(parent), 'parent_config_file': str(parent_config),
                  'parent_proof_files_sha256': proofs, 'seed42_input_reference': reference}
        def small_hash(file):
            if Path(file) == bank:raise AssertionError('historical large teacher bank must not be read')
            return '0' * 64
        with patch.object(data, 'sha_file', side_effect=small_hash) as hashes:
            self.assertEqual(data.validate_parent_proofs(config), proofs)
            self.assertEqual(hashes.call_count, 13)


class SourceConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tiny-visual-source-', dir='/opt/data/private')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root / 'src'; self.source.mkdir()
        for target, value in (('PROJECT_ROOT', self.root), ('SOURCE_ROOT', self.source)):
            item = patch.object(lock, target, value); item.start(); self.addCleanup(item.stop)

    def maps(self):
        names = ['pkg_%02d/common.py' % i for i in range(94)] + ['tools/prepare_posttrain_pope.py', 'tools/run_visual_merger_kl_study.py']
        relative = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        absolute = {str(self.root / name if name.startswith('tools/') else self.source / name): value for name, value in relative.items()}
        return relative, absolute

    def test_whole_relative_absolute_graph_exact_extra_missing_hash_rejected(self):
        relative, absolute = self.maps()
        self.assertEqual(lock.verify_runtime_sources(relative, absolute), absolute)
        extra = dict(relative, **{'extra.py': '0' * 64})
        missing = dict(relative); missing.pop(next(iter(missing)))
        changed = dict(relative); changed[next(iter(changed))] = '0' * 64
        for value in (extra, missing, changed):
            with self.assertRaisesRegex(ValueError, 'whole runtime/study'):lock.verify_runtime_sources(value, absolute)

    def test_path_escape_and_canonical_duplicate_rejected(self):
        for name in ('../escape.py', 'tools/../../escape.py', '/absolute.py', './alias.py'):
            with self.assertRaisesRegex(ValueError, 'escape/alias'):lock.canonical_sources({name: '0' * 64})
        (self.source / 'alias').symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'duplicate canonical'):
            lock.canonical_sources({'a.py': '0' * 64, 'alias/a.py': '0' * 64})

    def test_real_structure_study_consumer_148_inventory_and_no_elapsed_identity(self):
        relative, absolute = self.maps(); manifest_file = self.root / 'common_init_manifest.json'; manifest_file.write_text('{}')
        cfg_file = self.root / 'config.json'
        config = {'parent_proof_files_sha256': {}, 'prior_research_files_sha256': {},
                  'seed42_input_reference': {'teacher_reply_manifest_file': str(self.root / 'historical_large_bank.json')}}
        cfg_file.write_text(json.dumps(config))
        identity = {'trainable_tensors': 4, 'trainable_elements': 1843200, 'frozen_language_tensors': 144,
                    'total_adapter_tensors': 148, 'total_adapter_elements': 3686400}
        saved = {'schema_version': lock.STUDY_SCHEMA, 'config_path': str(cfg_file), 'config_file_sha256': lock.sha_file(cfg_file),
                 'packages': {'synthetic': '1'}, 'candidate': CANDIDATE, 'control_reference_only': True,
                 'source_sha256': absolute, 'parent_proof_files_sha256': {}, 'prior_research_files_sha256': {},
                 'metadata_files_sha256': {str(manifest_file): lock.sha_file(manifest_file)},
                 'weight_files_sha256': {str(self.root / 'never-read.safetensors'): '0' * 64},
                 'common_initialization_identity': copy.deepcopy(identity)}
        study_file = self.root / 'STUDY.json'; study_file.write_text(json.dumps(saved))
        with patch.object(lock, 'source_hashes', return_value=relative), patch.object(lock, 'package_versions', return_value={'synthetic': '1'}), patch.object(lock, 'verify_files') as files, patch.object(lock, 'verify_parent_proof'), patch.object(lock, 'verify_prior_research'), patch.object(lock, 'accepted_source_maps', return_value={}), patch.object(initialization, 'initialization_paths', return_value={'manifest': manifest_file}), patch.object(initialization, 'common_initializer_manifest', return_value={'identity': identity, 'telemetry': {'verify_CPU_wall_seconds': 999.0}}):
            self.assertEqual(lock.verify_study(study_file, cfg_file), saved)
            self.assertFalse(any(call.args[0] == saved['weight_files_sha256'] for call in files.call_args_list))
            identity['total_adapter_tensors'] = 144
            with self.assertRaisesRegex(ValueError, 'stable identity/manifest differs'):
                lock.verify_study(study_file, cfg_file)


if __name__ == '__main__':unittest.main()
