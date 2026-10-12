"""Pinned multi-image processor contract and CPU input preparation, no scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from vlm_deep.common import PROMPT_SUFFIX, collate, read_manifest
from .views import (CONDITIONS, build_image_views, content_sha, get_views,
                    input_rules, private_path, rgb_sha, sha_file, validate_geometry)

REPO = PROJECT_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def processor_for(model_root, config):
    from transformers import AutoProcessor
    validate_geometry(config)
    processor = AutoProcessor.from_pretrained(
        str(model_root), local_files_only=True, use_fast=False,
        min_pixels=50176, max_pixels=802816)
    processor.tokenizer.padding_side = 'left'
    if (int(processor.image_processor.patch_size) != 14
            or int(processor.image_processor.merge_size) != 2):
        raise ValueError('region v1 requires pinned patch14/merge2 processor')
    return processor


def inference_row(row):
    return {key: row[key] for key in ('question_id', 'image_id', 'question')}


def multiimage_messages(row, image_slots):
    if image_slots not in (1, 2):
        raise ValueError('region v1 requires one or two static image slots')
    suffix = row.get('prompt_suffix', PROMPT_SUFFIX)
    if suffix != PROMPT_SUFFIX:
        raise ValueError('region v1 cannot change the frozen text prompt')
    return [{'role': 'user', 'content': [{'type': 'image'} for _ in range(image_slots)]
             + [{'type': 'text', 'text': str(row['question']) + PROMPT_SUFFIX}]}]


def build_multiimage_batch(processor, rows, images_per_row, device=None):
    """Encode each QA with its exact slots, then left-pad the model batch.

    No truncation or labels are provided. Flattening across the batch is explicit
    in collate: Qwen consumes the grids/images in sample order and slot order.
    """
    if not rows or len(rows) != len(images_per_row):
        raise ValueError('rows and view-image groups must be nonempty and paired')
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if image_token_id is None or image_token_id < 0:
        raise ValueError('pinned Qwen image token absent')
    samples, details, chats = [], [], []
    for row, images in zip(rows, images_per_row):
        expected = [1024] if len(images) == 1 else [512, 512]
        if len(images) not in (1, 2):
            raise ValueError('invalid region image-slot count')
        expected_wh = [[896, 896]] if len(images) == 1 else [[896, 448], [896, 448]]
        if [list(image.size) for image in images] != expected_wh or any(image.mode != 'RGB' for image in images):
            raise ValueError('actual view dimensions or RGB mode changed')
        messages = multiimage_messages(row, len(images))
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sample = dict(processor(text=[text], images=images, return_tensors='pt', truncation=False))
        grids = [[int(value) for value in values] for values in sample['image_grid_thw'].tolist()]
        if len(grids) != len(images) or any(grid[0] != 1 for grid in grids):
            raise ValueError('processor image-grid slots or temporal dimension changed')
        per_image = [grid[0] * grid[1] * grid[2] // 4 for grid in grids]
        patches = sum(grid[0] * grid[1] * grid[2] for grid in grids)
        if any(grid[1] % 2 or grid[2] % 2 for grid in grids) or per_image != expected:
            raise ValueError(f'actual visual-token allocation failed: {row["question_id"]}: {grids}')
        if [[grid[2] * 14, grid[1] * 14] for grid in grids] != expected_wh:
            raise ValueError('processor resized the already-budgeted canvas')
        counted = int((sample['input_ids'] == image_token_id).sum())
        if counted != 1024 or patches != int(sample['pixel_values'].shape[0]):
            raise ValueError('image token expansion and pixel-values disagree')
        input_ids = sample['input_ids'][0].tolist()
        frame_hashes = [rgb_sha(image) for image in images]
        record = {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
                  'image_slots': len(images), 'image_grid_thw': grids,
                  'per_image_visual_tokens': per_image, 'visual_tokens': sum(per_image),
                  'counted_image_tokens': counted, 'premerge_patch_tokens': patches,
                  'input_tokens': len(input_ids), 'input_exceeds_training_768': len(input_ids) > 768,
                  'input_truncated': False, 'processor_resized_canvas': False,
                  'input_ids_sha256': content_sha(input_ids),
                  'text_prompt_sha256': content_sha(str(row['question']) + PROMPT_SUFFIX),
                  'chat_template_sha256': content_sha(text), 'rgb_frame_sha256': frame_hashes,
                  'encoded_input_sha256': content_sha({'ids': input_ids, 'grid': grids, 'rgb_frames': frame_hashes}),
                  'pixel_values_shape': list(sample['pixel_values'].shape),
                  'answers_or_target_used': False}
        details.append(record)
        chats.append(messages)
        samples.append(sample)
    batch = collate(samples, processor.tokenizer.pad_token_id)
    for record in details:
        record['padded_input_tokens'] = int(batch['input_ids'].shape[1])
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch, details, chats


def _file_bindings(config):
    return {'source_manifest_file_sha256': sha_file(config['manifest_path']),
            'detector_provenance_sha256': sha_file(config['detector_provenance_path']),
            'detections_sha256': sha_file(config['detections_path']),
            'model_processor_metadata_sha256': {
                path.name: sha_file(path) for path in sorted(Path(config['model_root']).glob('*.json'))}}


def prepare_inputs(config):
    """Complete real input views, 32-QA preflight and all-dev token checks."""
    started = time.monotonic()
    validate_geometry(config)
    output_root = private_path(config['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config['manifest_path'])
    manifest = read_manifest(manifest_path)
    rows = manifest['splits']['dev']
    if len(rows) != 1617 or len({row['image_id'] for row in rows}) != 1024:
        raise ValueError('region v1 expected dev1617/1024')
    detections = json.loads(Path(config['detections_path']).read_text())
    bindings = _file_bindings(config)
    if (not detections.get('complete') or detections.get('completed_count') != 1024
            or detections['provenance_sha256'] != bindings['detector_provenance_sha256']
            or detections['manifest_sha256'] != manifest['manifest_sha256']):
        raise ValueError('detector is incomplete or provenance/source manifest changed')
    ids = sorted({str(row['image_id']) for row in rows})
    if set(ids) != set(detections['images']):
        raise ValueError('detector image coverage differs from declared dev')
    source_hashes = {str(path.relative_to(SOURCE_ROOT)): sha_file(path) for path in
                     [Path(__file__), Path(__file__).parent / 'views.py', Path(__file__).parent / '__init__.py',
                      SOURCE_ROOT / 'vlm_deep/common.py', SOURCE_ROOT / 'vlm_sft/common.py',
                      SOURCE_ROOT / 'rarevlm_paths.py']}
    lock_header = {'schema_version': 'rarevlm.region_views.lock.v1',
                   'source_manifest_sha256': manifest['manifest_sha256'], **bindings,
                   'input_rules': input_rules(config), 'source_hashes': source_hashes,
                   'image_ids_sha256': content_sha(ids),
                   'question_order_sha256': content_sha([str(row['question_id']) for row in rows]),
                   'question_count': 1617, 'image_count': 1024,
                   'input_only': True, 'scope': 'actual cached VLM input views; no scores or answer-directed selection'}
    lock_path = output_root / 'VIEW_LOCK.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if {key: lock.get(key) for key in lock_header} != lock_header or set(lock['images']) != set(ids):
            raise ValueError('previous full view lock source/coverage differs')
    else:
        prepared = {}
        for index, iid in enumerate(ids):
            if detections['images'][iid]['provenance_sha256'] != bindings['detector_provenance_sha256']:
                raise ValueError(f'individual detector image provenance differs: {iid}')
            prepared[iid] = build_image_views(iid, manifest['images'][iid], manifest_path.parent,
                                             detections['images'][iid], config)
            if index % 100 == 0:
                print('PREPARED SOURCE FRAMES', index + 1, '/', len(ids), flush=True)
        lock = {**lock_header, 'images': prepared}
        write_json(lock_path, lock)
    lock_sha = sha_file(lock_path)
    preflight = sorted(rows, key=lambda row: int(row['question_id']))[:32]
    processor = processor_for(config['model_root'], config)
    records, precheck_records = [], []
    validation_begin = time.monotonic()
    for condition in CONDITIONS:
        for begin in range(0, len(preflight), 4):
            chunk = preflight[begin:begin+4]
            views, view_records = zip(*(get_views(inference_row(row), manifest['images'][row['image_id']],
                                                  manifest_path.parent, condition, config, lock) for row in chunk))
            batch, details, _ = build_multiimage_batch(processor, [inference_row(row) for row in chunk], views)
            del batch
            for view_record, detail in zip(view_records, details):
                precheck_records.append({**detail, 'condition': condition, 'view_sha256': view_record['view_sha256'],
                                         'selected_bbox_xyxy': view_record['selected_region']['bbox_xyxy'],
                                         'random_bbox_xyxy': view_record['random_region']['bbox_xyxy'],
                                         'fallback': view_record['selected_region']['fallback']})
    if len(precheck_records) != 128:
        raise ValueError('real first32 input-only precheck count differs')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'Pillow')}
    precheck_result = {'schema_version': 'rarevlm.region_input_precheck.v1', 'status': 'passed',
                       'completed_at': now(), 'view_lock_sha256': lock_sha, 'source_hashes': source_hashes,
                       'package_versions': versions, 'source_manifest_sha256': manifest['manifest_sha256'], **bindings,
                       'conditions': list(CONDITIONS), 'no_gpu_model_or_scores': True,
                       'questions': 32, 'processor_checks': 128,
                       'actual_visual_tokens_min': min(item['visual_tokens'] for item in precheck_records),
                       'actual_visual_tokens_max': max(item['visual_tokens'] for item in precheck_records),
                       'processor_resized_canvas_count': sum(item['processor_resized_canvas'] for item in precheck_records),
                       'input_truncated_count': sum(item['input_truncated'] for item in precheck_records),
                       'input_validation_seconds': time.monotonic() - validation_begin,
                       'selection': 'first32 dev question IDs in ascending numeric order; input-only',
                       'question_ids': [str(row['question_id']) for row in preflight], 'records': precheck_records}
    write_json(output_root / 'INPUT_PRECHECK.json', precheck_result)
    print('REAL INPUT PRECHECK PASSED', 128, flush=True)
    for condition in CONDITIONS:
        for begin in range(0, len(rows), 4):
            chunk = rows[begin:begin+4]
            views, view_records = zip(*(get_views(inference_row(row), manifest['images'][row['image_id']],
                                                  manifest_path.parent, condition, config, lock) for row in chunk))
            batch, details, _ = build_multiimage_batch(processor, [inference_row(row) for row in chunk], views)
            del batch
            for view_record, detail in zip(view_records, details):
                record = {**detail, 'condition': condition, 'view_sha256': view_record['view_sha256'],
                          'selected_bbox_xyxy': view_record['selected_region']['bbox_xyxy'],
                          'random_bbox_xyxy': view_record['random_region']['bbox_xyxy'],
                          'fallback': view_record['selected_region']['fallback']}
                records.append(record)
            if begin % 200 == 0:
                print('CPU INPUT CHECK', condition, min(begin+4, len(rows)), '/', len(rows), flush=True)
    common = {'status': 'passed', 'completed_at': now(), 'view_lock_sha256': lock_sha,
              'source_hashes': source_hashes, 'package_versions': versions,
              'source_manifest_sha256': manifest['manifest_sha256'], **bindings,
              'conditions': list(CONDITIONS), 'no_gpu_model_or_scores': True,
              'actual_visual_tokens_min': min(item['visual_tokens'] for item in records),
              'actual_visual_tokens_max': max(item['visual_tokens'] for item in records),
              'processor_resized_canvas_count': sum(item['processor_resized_canvas'] for item in records),
              'input_truncated_count': sum(item['input_truncated'] for item in records),
              'fallback_images': sum(item['conditions']['text_region']['selected_region']['fallback'] for item in lock['images'].values()),
              'input_validation_seconds': time.monotonic() - validation_begin,
              'full_prepare_wall_seconds': time.monotonic() - started}
    if len(precheck_records) != 128 or len(records) != 6468:
        raise ValueError('preflight/full input-contract count differs')
    if _file_bindings(config) != bindings or any(sha_file(SOURCE_ROOT / path) != digest for path, digest in source_hashes.items()):
        raise ValueError('source/model/detector binding changed while preparing inputs')
    full_result = {'schema_version': 'rarevlm.region_input_contract.v1', **common,
                   'questions': 1617, 'images': 1024, 'processor_checks': 6468,
                   'records': records}
    write_json(output_root / 'INPUT_CONTRACT.json', full_result)
    print('INPUT CONTRACT PASSED', json.dumps({key: full_result[key] for key in ('questions', 'images', 'processor_checks', 'full_prepare_wall_seconds')}), flush=True)
    return precheck_result, full_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepare_inputs(config)


if __name__ == '__main__':
    main()
