"""Repack frozen, image-only OCR into compact word-only native-image inputs.

No recognition is run here. The upstream detector, recognizer and complete
v1 input contract are hash-verified and remain immutable. Predicted text only
is packed; geometry is retained for fixed reading order and teaching metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import PROMPT_SUFFIX, read_manifest
from vlm_sft.common import content_sha, sha_file
from vlm_ocr.data import (CONDITIONS, _artifact_path, _check_group, _config_content_sha,
                          _config_sha, _historical_checks, _package_versions,
                          build_ocr_batch, inference_row, normalize_ocr_text,
                          private_path, processor_for, sattolo_sources, token_count,
                          validate_config, write_json)

UPSTREAM_CONFIG_SHA = '5b9e954fa2137d5b131bed8d9a96d23b847fd9af436ea2a4c668deb05af81f38'
UPSTREAM_CONTEXT_SHA = '019f612231a099b1d05eff455b7ffc24d6f1389bcc645dfe730ed1a721dd1b77'
UPSTREAM_CONTRACT_SHA = '46e35472c5ffedcdf123fa568a2a98ba3d8cd04748c61b901c150265dc8bd142'
UPSTREAM_RAW_SHA = 'b4b3f1c0c5180475463893bfcab8631f9a06b2266d815e9846207bb2ff91935f'
UPSTREAM_ANALYSIS_SHA = '05f45d9389c2b4eab9ac276ed6f0e4e5ee752e0bbb7e6e2eb66c2957e5998ba6'
PREFIX = 'OCR text:\n{context}\n\n'
V1_ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_ocr_context_v1')


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_compact_config(config):
    validate_config(config)
    if config['ocr_prefix_template'] != PREFIX:
        raise ValueError('compact v1 prefix must be the frozen OCR text template')
    if config.get('context_mode', 'word_only') != 'word_only':
        raise ValueError('compact v1 context must contain predicted words without coordinates')
    if config['ocr_position_format'] != '{predicted_text}':
        raise ValueError('compact v1 input formatter must contain text only')
    return True


def source_hashes():
    paths = [Path(__file__), Path(__file__).parent/'__init__.py',
             SOURCE_ROOT/'vlm_ocr/data.py', SOURCE_ROOT/'vlm_ocr/recognize.py',
             SOURCE_ROOT/'vlm_ocr/__init__.py', SOURCE_ROOT/'vlm_region/detect.py', SOURCE_ROOT/'rarevlm_paths.py',
             SOURCE_ROOT/'vlm_deep/common.py', SOURCE_ROOT/'vlm_sft/common.py']
    return {str(path.relative_to(SOURCE_ROOT)): sha_file(path) for path in paths}


def bind_upstream(config, manifest):
    """Verify cached recognition lineage before constructing any new context."""
    upstream_cfg = private_path(config.get('upstream_config_path', PROJECT_ROOT/'configs/vlm_ocr_context_v1.json'))
    context_path = private_path(config.get('upstream_context_lock_path', V1_ROOT/'OCR_CONTEXT_LOCK.json'))
    contract_path = private_path(config.get('upstream_input_contract_path', V1_ROOT/'INPUT_CONTRACT.json'))
    analysis_path = private_path(config.get('upstream_analysis_path', V1_ROOT/'analysis/ANALYSIS.json'))
    expected = {'config': (upstream_cfg, UPSTREAM_CONFIG_SHA),
                'context_lock': (context_path, UPSTREAM_CONTEXT_SHA),
                'input_contract': (contract_path, UPSTREAM_CONTRACT_SHA),
                'analysis': (analysis_path, UPSTREAM_ANALYSIS_SHA)}
    for name, (path, digest) in expected.items():
        declared = config.get(f'upstream_{name}_sha256', digest)
        if declared != digest or sha_file(path) != digest:
            raise ValueError(f'frozen OCR v1 upstream {name} changed')
    previous = json.loads(upstream_cfg.read_text())
    old_context = json.loads(context_path.read_text())
    old_contract = json.loads(contract_path.read_text())
    if (old_context.get('status') != 'passed' or not old_context.get('complete')
            or old_contract.get('status') != 'passed' or not old_contract.get('complete')
            or old_contract['total_records'] != 6468
            or old_contract['context_lock_sha256'] != UPSTREAM_CONTEXT_SHA
            or old_context['config_sha256'] != UPSTREAM_CONFIG_SHA
            or old_contract['config_sha256'] != UPSTREAM_CONFIG_SHA):
        raise ValueError('upstream complete input contract binding does not pass')
    unchanged = ('manifest_path', 'model_root', 'fixed_adapter_dir', 'model_revision',
                 'ocr_raw_path', 'ocr_provenance_path', 'ocr_ready_path',
                 'detector_predictions_path', 'detector_provenance_path', 'recognizer_model_dir')
    for key in unchanged:
        if config[key] != previous[key]:
            raise ValueError(f'compact study must reuse the same upstream input/model: {key}')
    if _package_versions() != old_context['package_versions']:
        raise ValueError('VLM processor package versions changed since OCR v1')
    for name, digest in old_context['source_hashes'].items():
        if sha_file(SOURCE_ROOT/name) != digest:
            raise ValueError(f'frozen upstream input source changed: {name}')
    model_metadata = {path.name: sha_file(path) for path in sorted(private_path(config['model_root']).glob('*.json'))}
    if model_metadata != old_context['model_processor_metadata_sha256']:
        raise ValueError('frozen native model/processor metadata changed')
    raw_path = private_path(config['ocr_raw_path'])
    prov_path = private_path(config['ocr_provenance_path'])
    ready_path = private_path(config['ocr_ready_path'])
    if (sha_file(raw_path) != UPSTREAM_RAW_SHA or sha_file(raw_path) != old_context['ocr_raw_file_sha256']
            or sha_file(prov_path) != old_context['ocr_provenance_sha256']
            or sha_file(ready_path) != old_context['ocr_ready_sha256']):
        raise ValueError('cached upstream raw OCR/provenance/readiness changed')
    raw, provenance, ready = [json.loads(path.read_text()) for path in (raw_path, prov_path, ready_path)]
    if (not raw['complete'] or raw['completed_count'] != 1024 or raw['expected_count'] != 1024
            or not ready.get('complete') or not ready.get('ready')
            or ready['config']['file_sha256'] != UPSTREAM_CONFIG_SHA
            or provenance['config']['file_sha256'] != UPSTREAM_CONFIG_SHA
            or ready['raw_file_sha256'] != UPSTREAM_RAW_SHA
            or raw['provenance_sha256'] != sha_file(prov_path)):
        raise ValueError('cached recognition does not bind the original completed OCR run')
    if (raw['manifest_file_sha256'] != sha_file(config['manifest_path'])
            or manifest['manifest_sha256'] != old_context['manifest_sha256']
            or raw['detections_file_sha256'] != sha_file(config['detector_predictions_path'])
            or provenance['detector_provenance_sha256'] != sha_file(config['detector_provenance_path'])):
        raise ValueError('frozen detector/source manifest bytes changed')
    for name, entry in provenance['source_locks'].items():
        if sha_file(private_path(entry['path'])) != entry['sha256']:
            raise ValueError(f'upstream recognition implementation changed: {name}')
    for name, entry in provenance['model_files'].items():
        if sha_file(private_path(entry['path'])) != entry['sha256']:
            raise ValueError(f'upstream pretrained recognition weights changed: {name}')
    if sha_file(provenance['source_path']) != provenance['source_sha256'] or ready['source_sha256'] != provenance['source_sha256']:
        raise ValueError('frozen recognizer source changed')
    binding = {'config_path': str(upstream_cfg), 'config_sha256': UPSTREAM_CONFIG_SHA,
               'context_lock_path': str(context_path), 'context_lock_sha256': UPSTREAM_CONTEXT_SHA,
               'input_contract_path': str(contract_path), 'input_contract_sha256': UPSTREAM_CONTRACT_SHA,
               'raw_path': str(raw_path), 'raw_sha256': UPSTREAM_RAW_SHA,
               'provenance_sha256': sha_file(prov_path), 'ready_sha256': sha_file(ready_path),
               'source_hashes': old_context['source_hashes'],
               'recognizer_source_sha256': provenance['source_sha256'],
               'recognizer_implementation_source_locks': provenance['source_locks'],
               'recognizer_model_files': provenance['model_files'],
               'upstream_analysis_path': str(analysis_path), 'upstream_analysis_sha256': UPSTREAM_ANALYSIS_SHA,
               'detector_sha256': raw['detections_file_sha256'],
               'detector_provenance_sha256': provenance['detector_provenance_sha256'],
               'manifest_file_sha256': sha_file(config['manifest_path']),
               'model_processor_metadata_sha256': model_metadata,
               'new_recognition_calls': 0, 'timing_scope': 'reused OCR v1 cache; its recognition/detection times are upstream costs, not new execution'}
    return raw, old_context, binding


def build_image_context(raw_record, tokenizer, config):
    """Keep image-only reading order; enter predicted text, not coordinates."""
    width, height = int(raw_record['width']), int(raw_record['height'])
    if min(width, height) <= 0:
        raise ValueError('invalid raw image dimensions')
    eligible = []
    for item in raw_record['items']:
        score = float(item['rec_score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid raw recognition confidence')
        text = normalize_ocr_text(item['rec_text'])
        if score < config['ocr_min_rec_score'] or not text:
            continue
        points = [(float(point[0]), float(point[1])) for point in item['polygon']]
        if len(points) < 3 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
            raise ValueError('invalid OCR polygon')
        left = min(max(min(x for x, _ in points), 0.0), float(width))
        right = min(max(max(x for x, _ in points), 0.0), float(width))
        top = min(max(min(y for _, y in points), 0.0), float(height))
        bottom = min(max(max(y for _, y in points), 0.0), float(height))
        if right <= left or bottom <= top:
            raise ValueError('empty clipped polygon envelope')
        input_text = text.replace('<|', '< |').replace('|>', '| >')
        eligible.append({'detector_index': int(item['detector_index']), 'polygon': item['polygon'],
                         'clipped_bbox_xyxy': [left, top, right, bottom],
                         'center_xy_normalized': [(left+right)/(2*width), (top+bottom)/(2*height)],
                         'rec_text': text, 'rec_score': score, 'det_score': float(item['det_score']),
                         'input_text': input_text, 'reserved_token_escaped': input_text != text,
                         'formatted_line': input_text, 'coordinates_enter_model': False})
    eligible.sort(key=lambda item: (item['clipped_bbox_xyxy'][1], item['clipped_bbox_xyxy'][0], item['detector_index']))
    unique, duplicate_indices, seen = [], [], set()
    for item in eligible:
        key = item['rec_text'].casefold()
        if key in seen:
            duplicate_indices.append(item['detector_index'])
        else:
            seen.add(key)
            unique.append(item)
    admitted, skipped, context = [], [], ''
    for item in unique:
        candidate = '\n'.join([*(value['formatted_line'] for value in admitted), item['formatted_line']])
        if token_count(tokenizer, candidate) > config['ocr_context_token_cap']:
            skipped.append(item['detector_index'])
            continue
        admitted.append(item)
        context = candidate
    count = token_count(tokenizer, context)
    prefix_count = token_count(tokenizer, config['ocr_prefix_template'].replace('{context}', context))
    if count > 256 or prefix_count > 320:
        raise ValueError('compact OCR budget failed')
    return {'source_image_sha256': raw_record['image_sha256'], 'raw_size_wh': [width, height],
            'raw_items_count': len(raw_record['items']), 'eligible_items_count': len(eligible),
            'retained_items_count': len(admitted), 'dropped_confidence_or_empty_count': len(raw_record['items'])-len(eligible),
            'duplicate_detector_indices': duplicate_indices, 'skipped_by_token_cap_detector_indices': skipped,
            'context_items': admitted, 'context': context, 'context_sha256': content_sha(context),
            'ocr_text_tokens': count, 'ocr_prefix_tokens': prefix_count,
            'context_mode': 'word_only', 'coordinates_enter_model': False,
            'normalization': 'NFC + whitespace collapse; literal reserved-token delimiter escape',
            'deduplication': 'normalized text casefold; first in reading order',
            'selection_uses': ['predicted recognition confidence', 'polygon reading order', 'tokenizer'],
            'answers_or_target_used': False, 'question_used': False}


def prepare_contexts(config, processor, manifest):
    raw, old_context, upstream = bind_upstream(config, manifest)
    ids = sorted({str(row['image_id']) for row in manifest['splits']['dev']})
    if set(ids) != set(raw['images']):
        raise ValueError('cached OCR coverage differs from full dev')
    contexts = {}
    for iid in ids:
        record, image = raw['images'][iid], manifest['images'][iid]
        if record['image_sha256'] != image['sha256'] or [record['width'], record['height']] != [image['width'], image['height']]:
            raise ValueError('cached OCR source image identity differs')
        contexts[iid] = build_image_context(record, processor.tokenizer, config)
    donors, groups = sattolo_sources(contexts, config['shuffle_seed'])
    old_lines = sum(item['retained_items_count'] for item in old_context['images'].values())
    new_lines = sum(item['retained_items_count'] for item in contexts.values())
    regained = lost = 0
    for iid in ids:
        old_set = {item['detector_index'] for item in old_context['images'][iid]['context_items']}
        new_set = {item['detector_index'] for item in contexts[iid]['context_items']}
        regained += len(new_set-old_set)
        lost += len(old_set-new_set)
    lock = {'schema_version': 'rarevlm.ocr_context.lock.v1', 'protocol_family': 'ocr_compact_word_only_v1',
            'status': 'passed', 'complete': True, 'created_at': now(),
            'config_sha256': _config_sha(config), 'configuration_content_sha256': _config_content_sha(config),
            'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': sha_file(config['manifest_path']),
            'ocr_raw_file_sha256': upstream['raw_sha256'], 'ocr_provenance_sha256': upstream['provenance_sha256'],
            'ocr_ready_sha256': upstream['ready_sha256'], 'upstream_v1': upstream,
            'source_hashes': source_hashes(), 'package_versions': _package_versions(),
            'model_processor_metadata_sha256': upstream['model_processor_metadata_sha256'],
            'ocr_prefix_template': PREFIX, 'prompt_suffix': PROMPT_SUFFIX,
            'image_count': 1024, 'question_count': 1617, 'images': contexts,
            'shuffle_sources': donors, 'shuffle_group_counts': groups, 'shuffle_seed': config['shuffle_seed'],
            'shuffle_rng': 'unchanged frozen v1 Sattolo implementation, separate empty/nonempty groups',
            'context_token_cap': 256, 'prefix_context_token_cap': 320,
            'recognition_confidence_threshold': 0.8, 'context_mode': 'word_only', 'coordinates_enter_model': False,
            'scope': 'one complete-dev compact followup; cached image-only OCR; no question/answer/error selection',
            'per_qa_text_length_matched': False, 'new_recognition_calls': 0,
            'coverage_comparison_v1': {'old_retained_lines': old_lines, 'new_retained_lines': new_lines,
                                       'newly_admitted_lines': regained, 'old_admitted_now_excluded_lines': lost,
                                       'note': 'shorter wrapper and coordinate removal change admitted information; not a coordinate-only causal contrast'},
            'shuffle_note': 'unique-image context multiset preserved; QA weighting, input length and compute need not match'}
    path = _artifact_path(config, 'context_lock_path', 'OCR_CONTEXT_LOCK.json')
    if path.exists():
        old = json.loads(path.read_text())
        keys = ('config_sha256', 'upstream_v1', 'source_hashes', 'package_versions', 'images', 'shuffle_sources')
        if any(old.get(key) != lock[key] for key in keys):
            raise ValueError('existing compact context lock differs; use a new output/protocol')
        return old
    write_json(path, lock)
    return lock


def prepare_inputs(config):
    validate_compact_config(config)
    started = time.monotonic()
    root = private_path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = private_path(config['manifest_path'])
    manifest = read_manifest(manifest_path)
    rows = manifest['splits']['dev']
    if len(rows) != 1617 or len({row['image_id'] for row in rows}) != 1024:
        raise ValueError('wrong complete dev coverage')
    processor = processor_for(config['model_root'], config)
    contexts = prepare_contexts(config, processor, manifest)
    lock_path = _artifact_path(config, 'context_lock_path', 'OCR_CONTEXT_LOCK.json')
    ctx_sha, sources, versions = sha_file(lock_path), source_hashes(), _package_versions()
    checks = {'native_images_pixels_grids_equal_all_conditions': True,
              'original_question_and_suffix_identical': True, 'original_baseline_prompt_preserved': True,
              'actual_visual_tokens_at_most_1024': True, 'actual_ocr_text_tokens_at_most_256': True,
              'actual_rendered_prefix_context_tokens_at_most_320': True, 'no_input_truncation': True,
              'no_answers_or_targets_used': True, 'different_image_shuffled_context': True,
              'same_image_questions_share_predicted_ocr': True, 'frozen_upstream_v1_verified': True,
              'coordinates_do_not_enter_context': True, 'equal_total_input_tokens_claimed': False}

    def encode_rows(selected, phase):
        records = []
        for begin in range(0, len(selected), 4):
            chunk = [inference_row(row) for row in selected[begin:begin+4]]
            batch_records = []
            for condition in CONDITIONS:
                batch, values, _ = build_ocr_batch(processor, chunk, manifest['images'],
                                                  manifest_path.parent, condition, config, contexts)
                del batch
                records.extend(values)
                batch_records.extend(values)
            for row in chunk:
                group = [value for value in batch_records if value['question_id'] == str(row['question_id'])]
                _check_group(group)
                baseline = next(value['input_tokens'] for value in group if value['condition'] == 'baseline')
                if any(not 0 <= value['input_tokens']-baseline <= 320 for value in group):
                    raise ValueError('actual added total input exceeds compact cap')
            if begin % 64 == 0 or begin+4 >= len(selected):
                status = {'schema_version': 'rarevlm.ocr_compact.cpu_prepare.v1', 'status': 'running',
                          'phase': phase, 'questions_completed': min(begin+4, len(selected)),
                          'questions_expected': len(selected), 'records_completed': len(records),
                          'updated_at': now(), 'elapsed_seconds': time.monotonic()-started}
                write_json(root/'CPU_PREPARE_STATUS.json', status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
        return records

    selected = sorted(rows, key=lambda row: int(row['question_id']))[:32]
    preflight = encode_rows(selected, 'first32_compact_input_only_precheck')
    header = {'protocol_family': 'ocr_compact_word_only_v1', 'status': 'passed', 'complete': True,
              'config_sha256': _config_sha(config), 'configuration_content_sha256': _config_content_sha(config),
              'context_lock_sha256': ctx_sha, 'source_hashes': sources, 'package_versions': versions,
              'conditions': list(CONDITIONS), 'checks': checks,
              'upstream_v1_input_contract_sha256': UPSTREAM_CONTRACT_SHA}
    precheck = {'schema_version': 'rarevlm.ocr_context.precheck.v1', **header, 'created_at': now(),
                'question_ids': [str(row['question_id']) for row in selected],
                'selection': 'first32 numeric dev question IDs; no outcome or answer selection',
                'questions': 32, 'processor_checks': 128, 'records': preflight,
                'historical_native_input_check': _historical_checks(preflight, config)}
    write_json(_artifact_path(config, 'input_precheck_path', 'INPUT_PRECHECK.json'), precheck)
    complete = encode_rows(rows, 'complete_internal_dev_compact_input_contract')
    if len(complete) != 6468:
        raise ValueError('incomplete compact native input contract')
    contract = {'schema_version': 'rarevlm.ocr_context.input_contract.v1', **header, 'created_at': now(),
                'manifest_sha256': manifest['manifest_sha256'],
                'question_order_sha256': content_sha([str(row['question_id']) for row in rows]),
                'questions': 1617, 'images': 1024, 'total_records': 6468, 'processor_checks': 6468,
                'records': complete, 'historical_native_input_check': _historical_checks(complete, config),
                'elapsed_seconds': time.monotonic()-started,
                'scope': 'input-only validation; reused upstream OCR; no recognition, GPU prediction or scores'}
    write_json(_artifact_path(config, 'input_contract_path', 'INPUT_CONTRACT.json'), contract)
    write_json(root/'CPU_PREPARE_STATUS.json', {'status': 'passed', 'complete': True,
               'completed_at': now(), 'elapsed_seconds': time.monotonic()-started,
               'processor_checks': 6596, 'context_lock_sha256': ctx_sha, 'new_recognition_calls': 0})
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(private_path(args.config).read_text())
    config['_config_path'] = str(private_path(args.config))
    config['_config_file_sha256'] = sha_file(args.config)
    result = prepare_inputs(config)
    print(json.dumps({'status': result['status'], 'records': result['total_records'],
                      'elapsed_seconds': result['elapsed_seconds'], 'new_recognition_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
