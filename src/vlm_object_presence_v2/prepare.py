"""Freeze the predeclared v2 population and encode every actual input on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from vlm_object_presence.data import (PROMPT_SUFFIX, assert_disjoint, content_sha,
    image_identity, sha_file, write_json)
from vlm_object_presence.prepare import render_audit
from vlm_public_preference.data import load_boundaries
from .data import (AUDIT_ORDER_SALT, AUDIT_SALT, CATEGORIES, CLASS_BALANCE_SALT,
    MIN_BOX_SHORT_SIDE, MIN_SEGMENTATION_AREA_FRACTION, NEGATIVE_SALT,
    audit_exclusion_sets, class_balanced_validation, freeze_audit,
    fresh_cpu_encode, select_split, summary)

PROTOCOL_SHA256 = '9fa7cb0dea34e36a99e3936e4de51fa80c2c387442ee96c3924e6928499535e7'
V1_COMPLETION_SHA256 = '390b60d3cac3b4a76552c868ba0b14f04d23d4ddcbe46a97ade25ec18ed1f814'


def freeze_or_match(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen v2 content differs; refusing to overwrite: {path}')
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--encoding-workers', type=int, default=8)
    args = parser.parse_args()
    root, source = args.output_root.resolve(), args.source_root.resolve()
    if not root.is_relative_to('/opt/data/private') or root == source:
        raise ValueError('Use a distinct persistent output root')
    if sha_file(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('The predeclared v2 protocol SHA differs')
    if sha_file(source / 'COMPLETION.json') != V1_COMPLETION_SHA256:
        raise ValueError('Frozen v1 completion differs')
    v1_completion = json.loads((source / 'COMPLETION.json').read_text())
    if not v1_completion['complete']:
        raise ValueError('Frozen source v1 is incomplete')
    for filename, expected in v1_completion['artifacts_sha256'].items():
        if sha_file(source / filename) != expected:
            raise ValueError(f'Frozen v1 artifact differs: {filename}')
    v1_lock = json.loads((source / 'SOURCE_LOCK.json').read_text())
    for filename, expected in v1_lock['code_files'].items():
        if sha_file(filename) != expected:
            raise ValueError(f'Frozen v1 source differs: {filename}')
    old_audit = json.loads((source / 'semantic_audit/FROZEN_64.json').read_text())['rows']
    if len(old_audit) != 64 or len({row['pixel_sha256'] for row in old_audit}) != 64:
        raise ValueError('All original 64 audit images must be excluded')
    model_root = Path(v1_lock['model_root'])
    reference_path = Path(v1_lock['reference_config']['path'])
    if sha_file(reference_path) != v1_lock['reference_config']['sha256']:
        raise ValueError('Frozen reference config differs')
    for filename, expected in v1_lock['processor_files'].items():
        if sha_file(model_root / filename) != expected:
            raise ValueError('Frozen processor files differ')
    for binding in v1_lock['boundary_manifests']:
        if sha_file(binding['path']) != binding['file_sha256']:
            raise ValueError('Inherited benchmark boundary manifest differs')
    if sha_file(source / 'ENCODING_CHECKS.jsonl') != v1_completion['encoding']['journal_sha256']:
        raise ValueError('V1 CPU contract ancestor differs')
    boundaries = load_boundaries([entry['path'] for entry in v1_lock['boundary_manifests']])
    blocked_ids = set(v1_lock['blocked_coco_ids'])
    if len(blocked_ids) < 500:
        raise ValueError('Full POPE source-ID boundary is missing')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'COMPLETION.json').exists():
        raise ValueError('V2 data is complete and frozen; do not rerun or replace it')
    start = time.time()
    records, image_maps, inventory = {}, {}, []
    for split, filename in [('train', 'TRAIN_MANIFEST.json'), ('val', 'VAL_MANIFEST.json')]:
        original = json.loads((source / filename).read_text())
        rows, images, inv = select_split(original, old_audit, split)
        expected = (1211, 2888) if split == 'train' else (143, 342)
        if (len(images), len(rows)) != expected:
            raise ValueError(f'Predeclared count conflict: {split} {(len(images), len(rows))} != {expected}')
        for image in images.values():
            actual = image_identity(Path(image['path']).read_bytes())
            if (actual['sha256'] != image['sha256'] or actual['pixel_sha256'] != image['pixel_sha256']
                    or (actual['width'], actual['height']) != (image['width'], image['height'])):
                raise ValueError('Existing native image bytes/pixels/size differ')
            if (actual['sha256'] in boundaries['files'] or actual['pixel_sha256'] in boundaries['pixels']
                    or image['coco_image_id'] in blocked_ids):
                raise ValueError('Inherited benchmark image boundary overlap')
        records[split], image_maps[split] = rows, images
        inventory += inv
    assert_disjoint(records['train'], records['val'])
    excluded = audit_exclusion_sets(old_audit)
    for row in records['train'] + records['val']:
        if any(str(row[key]) in values for key, values in excluded.items()):
            raise ValueError('A v1 audit image crossed into the v2 corpus')
    sources = [Path(__file__), Path(__file__).with_name('data.py'), Path(__file__).with_name('__init__.py'),
        Path(__file__).parents[2] / 'tools/prepare_object_presence_data_v2.py',
        Path(__file__).parents[2] / 'tests/test_object_presence_data_v2.py',
        Path(__file__).parents[1] / 'vlm_object_presence/data.py',
        Path(__file__).parents[1] / 'vlm_object_presence/prepare.py',
        Path(__file__).parents[1] / 'vlm_public_preference/data.py',
        Path(__file__).parents[1] / 'vlm_deep/common.py',
        Path(__file__).parents[1] / 'vlm_probe/evaluate.py']
    source_lock = {'schema_version': 'rarevlm.object_presence.source_lock.v2',
        'protocol': {'path': str(args.protocol.resolve()), 'sha256': PROTOCOL_SHA256},
        'v1_completion': {'path': str(source / 'COMPLETION.json'), 'sha256': V1_COMPLETION_SHA256},
        'v1_artifact_bindings': v1_completion['artifacts_sha256'],
        'v1_source_lock_sha256': sha_file(source / 'SOURCE_LOCK.json'),
        'v1_cpu_contract_sha256': v1_completion['encoding']['journal_sha256'],
        'v1_encoding_records_reused': False,
        'old_audit_excluded_image_identities': {key: sorted(values) for key, values in excluded.items()},
        'inherited_annotation_lock': v1_lock['annotation_lock'],
        'boundary_manifests': v1_lock['boundary_manifests'], 'blocked_coco_ids': sorted(blocked_ids),
        'model_root': str(model_root), 'reference_config': v1_lock['reference_config'],
        'processor_files': v1_lock['processor_files'], 'audit_font': v1_lock['audit_font'],
        'code_files': {str(path): sha_file(path) for path in sources}}
    source_lock['lock_sha256'] = content_sha(source_lock)
    freeze_or_match(root / 'SOURCE_LOCK.json', source_lock)
    policy = {'schema_version': 'rarevlm.object_presence.policy.v2',
        'protocol_sha256': PROTOCOL_SHA256, 'category_scope': list(CATEGORIES),
        'not_full_80_class_training': True, 'full_eligible_population': True,
        'positive_min_current_bbox_short_side_px': MIN_BOX_SHORT_SIDE,
        'positive_min_segmentation_area_fraction': MIN_SEGMENTATION_AREA_FRACTION,
        'both_positive_requirements_must_hold_on_same_noncrowd_instance': True,
        'negative_policy': 'absent from all annotations including crowd and small instances; equal unique count to all qualified positive categories; insufficient negatives excludes entire image',
        'old_audit_exclusion': 'all 64 original images by image_id, file/pixel SHA and COCO source ID; no per-review-outcome exceptions',
        'seed': 42, 'negative_salt': NEGATIVE_SALT,
        'negative_hash_input': '[salt,seed,int(coco_image_id),pixel_sha256,category_id]',
        'audit_salt': AUDIT_SALT, 'audit_order_salt': AUDIT_ORDER_SALT,
        'audit_rule': 'yes32 then no32 by [salt,seed,qid,pixel_sha], skip seen pixel SHA across labels; final [audit_order_salt,seed,qid] ordering',
        'class_balanced_validation_salt': CLASS_BALANCE_SALT,
        'question_template': 'Is there a {COCO category} in the image?', 'prompt_suffix': PROMPT_SUFFIX,
        'model_prediction_or_validation_error_selection': False, 'model_input_crop': False,
        'v1_encoding_reused': False, 'all_actual_sequences_fresh_cpu_encoded': True,
        'max_sequence_length': 2048, 'visual_token_budget': 1024, 'truncation': False,
        'quality_gate': {'clear_fraction_at_least': .90, 'wilson_two_sided_95_lower_at_least': .80},
        'annotation_scope': 'closed-world COCO instance labels restricted to declared 18 classes and visible-size rules',
        'remaining_risks': ['missing labels may create false negatives',
            'vehicle subtype, depictions/toys, occlusion and visibility remain ambiguous',
            'public source images can be resized; original COCO byte identity/cropping not proven',
            'per-image label balance does not balance each category',
            'near duplicates and pretraining overlap unknown'],
        'training_allowed': False, 'training_requires_external_admission': True,
        'external_admission_path': str(root.parent / 'vlm_object_adapt_v2/DATA_ADMISSION.json'),
        'final_model_target': 'complete original POPE9000 across 80 categories plus TextVQA protection; this restricted internal validation does not replace it'}
    freeze_or_match(root / 'PREPARATION_POLICY.json', policy)
    freeze_or_match(root / 'INVENTORY.json', {'rows': inventory,
        'reasons': dict(Counter(reason for row in inventory for reason in row['reasons'])),
        'inherited_v1_original_source_pool_images': v1_completion['summary']['source_pool_images'],
        'inherited_v1_excluded_images': v1_completion['summary']['excluded_images']})
    for split, filename in [('train', 'TRAIN_MANIFEST.json'), ('val', 'VAL_MANIFEST.json')]:
        manifest = {'schema_version': 'rarevlm.object_presence.manifest.v2', 'split': split,
            'source_lock_sha256': source_lock['lock_sha256'], 'protocol_sha256': PROTOCOL_SHA256,
            'prompt_suffix': PROMPT_SUFFIX, 'category_scope': list(CATEGORIES),
            'training_requires_external_admission': True,
            'label_source': policy['annotation_scope'], 'rows': records[split], 'images': image_maps[split]}
        manifest['manifest_sha256'] = content_sha(manifest)
        freeze_or_match(root / filename, manifest)
    balanced, balance_counts = class_balanced_validation(records['val'])
    balanced_ids = {row['image_id'] for row in balanced}
    balanced_images = {key: value for key, value in image_maps['val'].items() if key in balanced_ids}
    balanced_manifest = {'schema_version': 'rarevlm.object_presence.manifest.v2',
        'split': 'val_class_balanced', 'source_lock_sha256': source_lock['lock_sha256'],
        'protocol_sha256': PROTOCOL_SHA256, 'prompt_suffix': PROMPT_SUFFIX,
        'training_requires_external_admission': True, 'diagnostic_only': True,
        'selection': 'per-category min(yes,no), fixed v2 SHA, exact subset of complete validation',
        'category_balance': balance_counts, 'rows': balanced, 'images': balanced_images}
    balanced_manifest['manifest_sha256'] = content_sha(balanced_manifest)
    freeze_or_match(root / 'VAL_CLASS_BALANCED_MANIFEST.json', balanced_manifest)
    counts = {split: summary(records[split], image_maps[split]) for split in ('train', 'val')}
    counts.update(val_class_balanced=summary(balanced, balanced_images),
        inherited_v1_images=len(inventory), excluded_images=sum(not row['admitted'] for row in inventory),
        excluded_old_audit_images=64, model_predictions_used=False,
        train_val_overlap_zero=['question_id','image_id','file_sha256','pixel_sha256','coco_image_id'],
        inherited_benchmark_overlap_zero=['file_sha256','pixel_sha256','known_coco_source_id'],
        category_prior_risk='Per-image balance does not balance each category; report the frozen category-balanced subset separately',
        semantic_review='pending_external_review', training_allowed=False)
    freeze_or_match(root / 'SUMMARY.json', counts)
    audit = freeze_audit(records['train'])
    audit_manifest = {'schema_version': 'rarevlm.object_presence.frozen_audit.v2',
        'source_train_manifest_sha256': sha_file(root / 'TRAIN_MANIFEST.json'),
        'previous_audit_manifest_sha256': sha_file(source / 'semantic_audit/FROZEN_64.json'),
        'protocol_sha256': PROTOCOL_SHA256, 'seed': 42, 'audit_salt': AUDIT_SALT,
        'audit_order_salt': AUDIT_ORDER_SALT, 'labels': {'yes': 32, 'no': 32},
        'unique_images_by_pixel_sha': 64, 'old_audit_image_overlap': 0,
        'review_state': 'pending_external_review', 'rows': audit}
    audit_manifest['audit_sha256'] = content_sha(audit_manifest)
    freeze_or_match(root / 'semantic_audit/FROZEN_64.json', audit_manifest)
    render_audit(root, audit, image_maps['train'])
    print('v2_manifests_and_audit_frozen', json.dumps({key: counts[key] for key in ('train','val','val_class_balanced')}), flush=True)
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(1)
    processor = processor_for(model_root, json.loads(reference_path.read_text()), 1024)
    rows = records['train'] + records['val']
    journal_path = root / 'ENCODING_CHECKS.jsonl'
    checks = {}
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            if line:
                record = json.loads(line)
                if record['encoding_origin'] != 'fresh_v2_cpu' or record['v1_encoding_reused']:
                    raise ValueError('V2 journal contains a reused v1 encoding')
                checks[record['question_id']] = record
    remaining = [row for row in rows if row['question_id'] not in checks]
    with journal_path.open('a') as journal, ThreadPoolExecutor(max_workers=args.encoding_workers) as pool:
        for index, check in enumerate(pool.map(lambda row: fresh_cpu_encode(row, processor), remaining), 1):
            checks[check['question_id']] = check
            journal.write(json.dumps(check, separators=(',', ':'))+'\n')
            if index % 128 == 0 or index == len(remaining):
                journal.flush()
                write_json(root / 'STATUS.json', {'phase': 'fresh_cpu_encoding', 'complete': len(checks),
                    'total': len(rows), 'elapsed_seconds': time.time()-start,
                    'training_allowed': False, 'v1_encoding_reused': False})
                print('fresh_cpu_encoded', len(checks), '/', len(rows), flush=True)
    if len(rows) != 3230 or set(checks) != {row['question_id'] for row in rows}:
        raise ValueError('Complete fresh CPU contract inventory differs from the predeclared corpus')
    encoding = {'questions': len(checks), 'fresh_actual_cpu_encodes': len(checks),
        'v1_encoding_records_reused': 0, 'model_forward_or_gpu_used': False,
        'sequence_length_min': min(check['sequence_length'] for check in checks.values()),
        'sequence_length_max': max(check['sequence_length'] for check in checks.values()),
        'visual_tokens_min': min(check['visual_tokens'] for check in checks.values()),
        'visual_tokens_max': max(check['visual_tokens'] for check in checks.values()),
        'assistant_tokens': dict(Counter(check['assistant_tokens'] for check in checks.values())),
        'input_ids_and_labels_saved': True, 'actual_processed_pixel_sha_saved': True,
        'visual_token_budget': 1024, 'max_sequence_length': 2048, 'truncation': False,
        'journal_sha256': sha_file(journal_path)}
    write_json(root / 'ENCODING_SUMMARY.json', encoding)
    artifacts = ['TRAIN_MANIFEST.json','VAL_MANIFEST.json','VAL_CLASS_BALANCED_MANIFEST.json',
        'SOURCE_LOCK.json','PREPARATION_POLICY.json','INVENTORY.json','SUMMARY.json',
        'semantic_audit/FROZEN_64.json','ENCODING_SUMMARY.json']
    completion = {'complete': True, 'data_only': True, 'optimizer_updates': 0,
        'elapsed_seconds': time.time()-start, 'summary': counts, 'encoding': encoding,
        'protocol_sha256': PROTOCOL_SHA256, 'source_lock_sha256': source_lock['lock_sha256'],
        'training_allowed': False, 'training_requires_external_admission': True,
        'artifacts_sha256': {name: sha_file(root/name) for name in artifacts}}
    write_json(root / 'COMPLETION.json', completion)
    write_json(root / 'STATUS.json', {'phase': 'complete', 'complete': len(checks),
        'total': len(rows), 'elapsed_seconds': time.time()-start,
        'training_allowed': False, 'v1_encoding_reused': False})
    print('COMPLETE', json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
