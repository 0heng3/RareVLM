"""Tiny synthetic CPU initialization only; never prepares the real 4-factor artifact."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vlm_visual_merger_kl import initialization as init


class ActualInventoryTests(unittest.TestCase):
    def test_protocol_inventory_without_tensor_allocation(self):
        specs = init.visual_tensor_specs()
        self.assertEqual(len(specs), 4)
        self.assertEqual(sum(math.prod(shape) for shape in specs.values()), 1843200)
        self.assertEqual(init.LANGUAGE_TENSORS + len(specs), 148)
        self.assertEqual(init.LANGUAGE_ELEMENTS + init.TRAINABLE_ELEMENTS, 3686400)
        self.assertEqual([rank for _, rank, _, _, _ in init.TARGET_SPECS], [96, 120])
        self.assertTrue(all(alpha / rank == 2 for _, rank, alpha, _, _ in init.TARGET_SPECS))


class TinyInitializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tiny-visual-init-', dir='/opt/data/private')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        specs = (('visual.merger.mlp.0', 2, 4, (2, 4), (4, 2)),
                 ('visual.merger.mlp.2', 2, 4, (2, 4), (2, 2)))
        for item in (patch.object(init, 'TARGET_SPECS', specs), patch.object(init, 'TRAINABLE_ELEMENTS', 28)):
            item.start(); self.addCleanup(item.stop)
        self.config = {'output_root': str(self.root), 'common_initialization_dir': str(self.root / 'common_initialization'),
            'common_initialization': copy.deepcopy(init.METHOD), 'expected_trainable': 28,
            'teacher_generation': {'initial_runtime_master_sha256': init.INITIAL_LANGUAGE_SHA256},
            'visual_adaptation': {'language_frozen_tensors': 144, 'language_frozen_elements': 1843200,
                'new_trainable_tensors': 4, 'new_trainable_elements': 28, 'total_adapter_tensors': 148,
                'total_adapter_elements': 1843228, 'active_adapter': 'default', 'always_active_composite': True,
                'targets': {target: {'r': rank, 'lora_alpha': alpha, 'A_shape': list(a), 'B_shape': list(b)}
                            for target, rank, alpha, a, b in specs}}}

    def test_local_generator_deterministic_B_zero_and_global_rng_unchanged(self):
        before = torch.get_rng_state().clone()
        one, two = init._create_tensors(), init._create_tensors()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key in one:
            self.assertTrue(torch.equal(one[key], two[key]))
            if '.lora_B.' in key:
                self.assertEqual(int(torch.count_nonzero(one[key])), 0)
            else:
                self.assertGreater(int(torch.count_nonzero(one[key])), 0)
                self.assertLessEqual(float(one[key].abs().max()), 1 / math.sqrt(4))

    def test_prepare_load_idempotent_identity_excludes_telemetry(self):
        manifest = init.prepare_common_initialization(self.config)
        paths = init.initialization_paths(self.config)
        saved = paths['manifest'].read_bytes()
        tensors, loaded = init.load_common_initialization(self.config)
        self.assertEqual(manifest, loaded)
        self.assertEqual(init.prepare_common_initialization(self.config), manifest)
        self.assertEqual(saved, paths['manifest'].read_bytes())
        self.assertEqual(len(tensors), 4)
        self.assertNotIn('prepare_CPU_wall_seconds', manifest['identity'])
        self.assertIn('prepare_CPU_wall_seconds', manifest['telemetry'])

    def test_metadata_accessor_never_loads_tensor(self):
        expected = init.prepare_common_initialization(self.config)
        with patch('safetensors.torch.load_file', side_effect=AssertionError('tensor load forbidden')):
            self.assertEqual(init.common_initializer_manifest(self.config, verify_file_bytes=False), expected)
            self.assertEqual(init.common_initializer_manifest(self.config, verify_file_bytes=True), expected)

    def test_missing_extra_shape_dtype_nonfinite_or_nonzero_B_rejected(self):
        original = init._create_tensors()
        first = next(iter(original)); bkey = next(key for key in original if '.lora_B.' in key)
        missing = dict(original); missing.pop(first)
        extra = dict(original); extra['unexpected'] = torch.zeros(1)
        wrong_shape = dict(original); wrong_shape[first] = torch.zeros((1, 1))
        wrong_dtype = dict(original); wrong_dtype[first] = original[first].double()
        nonfinite = {key: value.clone() for key, value in original.items()}; nonfinite[first][0, 0] = float('nan')
        bad_b = {key: value.clone() for key, value in original.items()}; bad_b[bkey][0, 0] = 1
        for values in (missing, extra, wrong_shape, wrong_dtype, nonfinite, bad_b):
            with self.subTest(keys=list(values)):
                with self.assertRaises(ValueError):init.validate_visual_tensors(values)

    def test_wrong_seed_rank_alpha_or_budget_rejected(self):
        changes = [('seed', 0), ('rank', 3), ('alpha', 8), ('budget', 144)]
        for field, value in changes:
            config = copy.deepcopy(self.config)
            if field == 'seed':config['common_initialization']['seed'] = value
            elif field == 'rank':config['visual_adaptation']['targets']['visual.merger.mlp.0']['r'] = value
            elif field == 'alpha':config['visual_adaptation']['targets']['visual.merger.mlp.0']['lora_alpha'] = value
            else:config['visual_adaptation']['total_adapter_tensors'] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):init.validate_initialization_config(config)

    def test_missing_prepare_never_implicitly_creates(self):
        with self.assertRaisesRegex(ValueError, 'explicit common initialization prepare'):
            init.load_common_initialization(self.config, require_prepared=False)
        self.assertFalse(init.initialization_paths(self.config)['directory'].exists())

    def test_corrupted_artifact_and_manifest_identity_rejected(self):
        init.prepare_common_initialization(self.config)
        paths = init.initialization_paths(self.config)
        paths['tensors'].write_bytes(paths['tensors'].read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'tensor bytes changed'):
            init.load_common_initialization(self.config)

    def test_partial_artifact_refuses_overwrite(self):
        paths = init.initialization_paths(self.config); paths['directory'].mkdir()
        paths['tensors'].write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'partial initializer'):
            init.prepare_common_initialization(self.config)
        self.assertEqual(paths['tensors'].read_bytes(), b'partial')


if __name__ == '__main__':unittest.main()
