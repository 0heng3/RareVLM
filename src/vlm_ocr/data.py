"""Input-only OCR context and unchanged native-image processor contracts.

Targets, reference answers and prediction scores never enter this module's
context construction or inference messages. Only complete predicted OCR lines
are admitted under a fixed tokenizer budget. No encoded tensor cache is saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import PROMPT_SUFFIX, collate, read_manifest
from vlm_sft.common import content_sha, sha_file

PRIVATE = Path('/opt/data/private')
CONDITIONS = ('baseline', 'empty_template', 'predicted_ocr', 'shuffled_ocr')


def now():
    return datetime.now(timezone.utc).isoformat()


def private_path(path):
    value = Path(path).resolve()
    if value != PRIVATE and PRIVATE not in value.parents:
        raise ValueError(f'path must remain in persistent workspace: {value}')
    return value


def write_json(path, value):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _config_sha(config):
    path = private_path(config.get('_config_path', PROJECT_ROOT / 'configs/vlm_ocr_context_v1.json'))
    clean = {key: value for key, value in config.items() if not key.startswith('_')}
    if json.loads(path.read_text()) != clean:
        raise ValueError('runtime config differs from its frozen JSON file')
    result = sha_file(path)
    if config.get('_config_file_sha256', result) != result:
        raise ValueError('runtime config file hash differs')
    return result


def _config_content_sha(config):
    return content_sha({key: value for key, value in config.items() if not key.startswith('_')})


def validate_config(config):
    if tuple(config['conditions']) != CONDITIONS:
        raise ValueError('OCR v1 requires the four preregistered conditions in order')
    required = {'visual_token_budget': 1024, 'min_pixels': 50176,
                'max_pixels': 802816, 'batch_size': 4, 'max_new_tokens': 32,
                'ocr_context_token_cap': 256, 'ocr_prefix_token_cap': 320,
                'ocr_min_rec_score': 0.8, 'shuffle_seed': 20261013}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'OCR v1 fixed setting changed: {key}')
    if config.get('input_truncation') is not False or config.get('do_sample') is not False:
        raise ValueError('OCR v1 requires untruncated greedy generation')
    template = config['ocr_prefix_template']
    if not isinstance(template, str) or template.count('{context}') != 1:
        raise ValueError('prefix must contain exactly one literal {context} placeholder')
    # Reject accidental secondary formatting fields; replacement is literal.
    if '{' in template.replace('{context}', '') or '}' in template.replace('{context}', ''):
        raise ValueError('OCR prefix may contain only the context placeholder')
    if config.get('expected_questions', 1617) != 1617 or config.get('expected_images', 1024) != 1024:
        raise ValueError('OCR v1 is frozen to internal dev1617/images1024')
    return required


def processor_for(model_root, config):
    from transformers import AutoProcessor
    validate_config(config)
    processor = AutoProcessor.from_pretrained(
        str(private_path(model_root)), local_files_only=True, use_fast=False,
        min_pixels=50176, max_pixels=802816)
    processor.tokenizer.padding_side = 'left'
    if (int(processor.image_processor.patch_size) != 14
            or int(processor.image_processor.merge_size) != 2):
        raise ValueError('OCR v1 requires the frozen Qwen patch14/merge2 processor')
    return processor


def token_count(tokenizer, value):
    return len(tokenizer.encode(value, add_special_tokens=False))


def normalize_ocr_text(value):
    """NFC and whitespace only; no answer normalizer or semantic substitutions."""
    return ' '.join(unicodedata.normalize('NFC', str(value)).split())


def _eligible_line(item, width, height, threshold):
    score = float(item['rec_score'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('raw recognition confidence is nonfinite or outside [0,1]')
    text = normalize_ocr_text(item['rec_text'])
    if score < threshold or not text:
        return None
    points = [(float(point[0]), float(point[1])) for point in item['polygon']]
    if len(points) < 3 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError('invalid polygon in completed raw OCR')
    left = min(max(min(x for x, _ in points), 0.0), float(width))
    right = min(max(max(x for x, _ in points), 0.0), float(width))
    top = min(max(min(y for _, y in points), 0.0), float(height))
    bottom = min(max(max(y for _, y in points), 0.0), float(height))
    if right <= left or bottom <= top:
        raise ValueError('completed raw OCR contains empty clipped polygon envelope')
    x = (left + right) / (2 * width)
    y = (top + bottom) / (2 * height)
    input_text = text.replace('<|', '< |').replace('|>', '| >')
    line = f'[x={x:.2f},y={y:.2f}] {input_text}'
    return {'detector_index': int(item['detector_index']), 'polygon': item['polygon'],
            'clipped_bbox_xyxy': [left, top, right, bottom],
            'center_xy_normalized': [x, y], 'rec_text': text, 'rec_score': score,
            'det_score': float(item['det_score']), 'input_text': input_text,
            'reserved_token_escaped': input_text != text, 'formatted_line': line}


def build_image_context(raw_record, tokenizer, config):
    """Use image/recognizer output only, in fixed geometry reading order."""
    width, height = int(raw_record['width']), int(raw_record['height'])
    if min(width, height) <= 0:
        raise ValueError('invalid raw image dimensions')
    eligible = []
    for item in raw_record['items']:
        line = _eligible_line(item, width, height, config['ocr_min_rec_score'])
        if line is not None:
            eligible.append(line)
    eligible.sort(key=lambda item: (item['clipped_bbox_xyxy'][1],
                                    item['clipped_bbox_xyxy'][0], item['detector_index']))
    unique, duplicate_indices, seen = [], [], set()
    for item in eligible:
        key = item['rec_text'].casefold()
        if key in seen:
            duplicate_indices.append(item['detector_index'])
            continue
        seen.add(key)
        unique.append(item)
    admitted, skipped, context = [], [], ''
    for item in unique:
        candidate = '\n'.join([*(value['formatted_line'] for value in admitted), item['formatted_line']])
        count = token_count(tokenizer, candidate)
        if count > config['ocr_context_token_cap']:
            skipped.append(item['detector_index'])
            continue
        admitted.append(item)
        context = candidate
    count = token_count(tokenizer, context)
    prefix_count = token_count(tokenizer, config['ocr_prefix_template'].replace('{context}', context))
    if count > 256 or prefix_count > 320:
        raise ValueError('fixed OCR context/prefix tokenizer cap failed')
    return {'source_image_sha256': raw_record['image_sha256'],
            'raw_size_wh': [width, height], 'raw_items_count': len(raw_record['items']),
            'eligible_items_count': len(eligible), 'retained_items_count': len(admitted),
            'dropped_confidence_or_empty_count': len(raw_record['items']) - len(eligible),
            'duplicate_detector_indices': duplicate_indices,
            'skipped_by_token_cap_detector_indices': skipped,
            'context_items': admitted, 'context': context, 'context_sha256': content_sha(context),
            'ocr_text_tokens': count, 'ocr_prefix_tokens': prefix_count,
            'normalization': 'Unicode NFC + whitespace collapse only; literal reserved-token delimiters escaped for prompt structure',
            'deduplication': 'normalized text casefold; first in reading order',
            'selection_uses': ['predicted recognition confidence', 'polygon geometry', 'tokenizer'],
            'answers_or_target_used': False, 'question_used': False}


def sattolo_sources(contexts, seed):
    """Derange separately in empty/nonempty groups, preserving each distribution."""
    groups = {'empty': sorted(iid for iid, record in contexts.items() if not record['context']),
              'nonempty': sorted(iid for iid, record in contexts.items() if record['context'])}
    sources = {}
    for name, ids in groups.items():
        if not ids:
            continue
        if len(ids) == 1:
            raise ValueError(f'cannot construct within-group different-image control: {name}')
        digest = hashlib.sha256(f'{int(seed)}:{name}'.encode()).hexdigest()
        generator = random.Random(int(digest, 16))
        shuffled = ids.copy()
        for index in range(len(shuffled) - 1, 0, -1):
            previous = generator.randrange(index)
            shuffled[index], shuffled[previous] = shuffled[previous], shuffled[index]
        sources.update(zip(ids, shuffled))
    if (set(sources) != set(contexts) or set(sources.values()) != set(contexts)
            or any(iid == donor for iid, donor in sources.items())):
        raise ValueError('Sattolo source mapping has a fixed point or lost image')
    return sources, {key: len(values) for key, values in groups.items()}


def _package_versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'tokenizers', 'pillow', 'numpy')}


def source_hashes():
    paths = [Path(__file__), Path(__file__).parent / '__init__.py',
             SOURCE_ROOT / 'rarevlm_paths.py', SOURCE_ROOT / 'vlm_deep/common.py',
             SOURCE_ROOT / 'vlm_sft/common.py']
    return {str(path.relative_to(SOURCE_ROOT)): sha_file(path) for path in paths}


def _artifact_path(config, key, filename):
    return private_path(config.get(key, Path(config['output_root']) / filename))


def prepare_contexts(config, processor, manifest):
    raw_path = private_path(config['ocr_raw_path'])
    provenance_path = private_path(config['ocr_provenance_path'])
    ready_path = private_path(config['ocr_ready_path'])
    raw = json.loads(raw_path.read_text())
    if (raw.get('complete') is not True or raw.get('completed_count') != 1024
            or raw.get('expected_count') != 1024):
        raise ValueError('raw OCR must cover all 1024 dev images before preparation')
    if raw.get('provenance_sha256') != sha_file(provenance_path):
        raise ValueError('raw OCR recognizer provenance changed')
    manifest_path = private_path(config['manifest_path'])
    if raw.get('manifest_file_sha256') != sha_file(manifest_path):
        raise ValueError('raw OCR was not bound to the current source manifest')
    if raw.get('detections_file_sha256') != sha_file(config['detector_predictions_path']):
        raise ValueError('raw OCR detector bytes changed')
    ready = json.loads(ready_path.read_text())
    if not (ready.get('complete') is True or ready.get('status') in ('passed', 'ready', 'completed')):
        raise ValueError('recognition readiness marker does not pass')
    ids = sorted({str(row['image_id']) for row in manifest['splits']['dev']})
    if set(ids) != set(raw['images']):
        raise ValueError('raw OCR image coverage differs from internal dev')
    contexts = {}
    for iid in ids:
        record, image = raw['images'][iid], manifest['images'][iid]
        if (record['image_sha256'] != image['sha256']
                or [int(record['width']), int(record['height'])] != [int(image['width']), int(image['height'])]):
            raise ValueError(f'raw OCR image identity differs: {iid}')
        contexts[iid] = build_image_context(record, processor.tokenizer, config)
    sources, group_counts = sattolo_sources(contexts, config['shuffle_seed'])
    lock = {'schema_version': 'rarevlm.ocr_context.lock.v1', 'status': 'passed', 'complete': True,
            'created_at': now(), 'config_sha256': _config_sha(config),
            'configuration_content_sha256': _config_content_sha(config),
            'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': sha_file(manifest_path),
            'ocr_raw_file_sha256': sha_file(raw_path), 'ocr_provenance_sha256': sha_file(provenance_path),
            'ocr_ready_sha256': sha_file(ready_path), 'source_hashes': source_hashes(),
            'package_versions': _package_versions(),
            'model_processor_metadata_sha256': {path.name: sha_file(path) for path in
                                                sorted(private_path(config['model_root']).glob('*.json'))},
            'ocr_prefix_template': config['ocr_prefix_template'],
            'prompt_suffix': PROMPT_SUFFIX, 'image_count': len(ids), 'question_count': 1617,
            'images': contexts, 'shuffle_sources': sources, 'shuffle_group_counts': group_counts,
            'shuffle_seed': config['shuffle_seed'],
            'shuffle_rng': 'Python Random(SHA256(str(seed)+":"+group)); Sattolo randrange(i)',
            'context_token_cap': 256, 'prefix_context_token_cap': 320,
            'recognition_confidence_threshold': 0.8,
            'scope': 'image-only predicted OCR; no question, reference, target or score selection',
            'per_qa_text_length_matched': False,
            'shuffle_note': 'each unique image donates once in its empty/nonempty group; QA multiplicity and per-QA text lengths may differ'}
    path = _artifact_path(config, 'context_lock_path', 'OCR_CONTEXT_LOCK.json')
    if path.exists():
        old = json.loads(path.read_text())
        compared_keys = ('config_sha256', 'manifest_file_sha256', 'ocr_raw_file_sha256',
                         'ocr_provenance_sha256', 'ocr_ready_sha256', 'source_hashes',
                         'package_versions', 'model_processor_metadata_sha256', 'images', 'shuffle_sources')
        if any(old.get(key) != lock[key] for key in compared_keys):
            raise ValueError('existing OCR context lock differs; use a new protocol/output')
        return old
    write_json(path, lock)
    return lock


def inference_row(row):
    """Strip scoring/training fields before any input construction."""
    if row.get('prompt_suffix', PROMPT_SUFFIX) != PROMPT_SUFFIX:
        raise ValueError('original question suffix is frozen')
    return {key: row[key] for key in ('question_id', 'image_id', 'question')}


def context_for(row, condition, config, context_lock):
    if condition not in CONDITIONS:
        raise ValueError(f'unknown OCR condition: {condition}')
    iid = str(row['image_id'])
    donor = context_lock['shuffle_sources'][iid] if condition == 'shuffled_ocr' else iid
    record = context_lock['images'][donor]
    context = record['context'] if condition in ('predicted_ocr', 'shuffled_ocr') else ''
    prefix = '' if condition == 'baseline' else config['ocr_prefix_template'].replace('{context}', context)
    return context, prefix, (donor if condition == 'shuffled_ocr' else None), record


def _pixel_sha(tensor):
    if str(tensor.dtype) != 'torch.float32' or tensor.device.type != 'cpu':
        raise ValueError('processor pixel tensor must be CPU float32')
    array = tensor.detach().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def build_ocr_batch(processor, rows, images, image_root, condition, config, context_lock, device=None):
    """Encode full native RGB images and context; never truncate input."""
    from PIL import Image
    validate_config(config)
    if not rows:
        raise ValueError('empty input batch')
    if context_lock['config_sha256'] != _config_sha(config):
        raise ValueError('OCR context config binding differs')
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if image_token_id is None or image_token_id < 0:
        raise ValueError('missing frozen Qwen image token')
    samples, details, chats = [], [], []
    for provided in rows:
        row = inference_row(provided)
        qid, iid = str(row['question_id']), str(row['image_id'])
        context, prefix, donor, ocr_record = context_for(row, condition, config, context_lock)
        context_tokens, prefix_tokens = token_count(processor.tokenizer, context), token_count(processor.tokenizer, prefix)
        if context_tokens > 256 or prefix_tokens > 320:
            raise ValueError('actual OCR context/prefix exceeds frozen tokenizer caps')
        image_record = images[iid]
        image_path = private_path(Path(image_root) / image_record['path'])
        if sha_file(image_path) != image_record['sha256']:
            raise ValueError(f'original image changed: {iid}')
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        if list(image.size) != [int(image_record['width']), int(image_record['height'])]:
            raise ValueError('original image dimensions differ')
        if context_lock['images'][iid]['source_image_sha256'] != image_record['sha256']:
            raise ValueError('recipient image/context source binding differs')
        rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
        final_question = str(row['question']) + PROMPT_SUFFIX
        text_prompt = prefix + final_question
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text_prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sample = dict(processor(text=[text], images=[image], return_tensors='pt', truncation=False))
        grids = [[int(value) for value in values] for values in sample['image_grid_thw'].tolist()]
        if len(grids) != 1 or grids[0][0] != 1 or grids[0][1] % 2 or grids[0][2] % 2:
            raise ValueError('native image grid must remain one static merged image')
        patches = math.prod(grids[0])
        visual_tokens = patches // 4
        counted = int((sample['input_ids'] == image_token_id).sum())
        if counted != visual_tokens or not 1 <= visual_tokens <= 1024 or patches != sample['pixel_values'].shape[0]:
            raise ValueError('native actual visual-token contract failed')
        ids = sample['input_ids'][0].tolist()
        pixel_sha = _pixel_sha(sample['pixel_values'])
        record = {'question_id': qid, 'image_id': iid, 'condition': condition,
                  'question_sha256': content_sha(str(row['question'])),
                  'question_prompt_sha256': content_sha(final_question),
                  'image_sha256': image_record['sha256'], 'image_slots': 1,
                  'raw_width': image.width, 'raw_height': image.height,
                  'resized_width': grids[0][2] * 14, 'resized_height': grids[0][1] * 14,
                  'image_grid_thw': grids, 'per_image_visual_tokens': [visual_tokens],
                  'visual_tokens': visual_tokens, 'counted_image_tokens': counted,
                  'premerge_patch_tokens': patches, 'input_tokens': len(ids),
                  'input_exceeds_training_768': len(ids) > 768, 'input_truncated': False,
                  'input_ids_sha256': content_sha(ids), 'text_prompt_sha256': content_sha(text_prompt),
                  'chat_template_sha256': content_sha(text), 'rgb_frame_sha256': [rgb_sha],
                  'image_rgb_sha256': rgb_sha, 'pixel_values_sha256': pixel_sha,
                  'cpu_float_pixel_sha256': pixel_sha, 'pixel_values_dtype': 'float32',
                  'pixel_values_shape': list(sample['pixel_values'].shape),
                  'context_sha256': content_sha(context), 'ocr_text_tokens': context_tokens,
                  'ocr_prefix_tokens': prefix_tokens, 'shuffle_source_image_id': donor,
                  'ocr_context_source_image_id': donor if donor is not None else (iid if condition == 'predicted_ocr' else None),
                  'ocr_retained_lines': ocr_record['retained_items_count'] if context else 0,
                  'encoded_input_sha256': content_sha({'ids': ids, 'grid': grids, 'pixel': pixel_sha}),
                  'answers_or_target_used': False, 'question_used_for_ocr_selection': False}
        details.append(record)
        chats.append(messages)
        samples.append(sample)
    batch = collate(samples, processor.tokenizer.pad_token_id)
    for record in details:
        record['padded_input_tokens'] = int(batch['input_ids'].shape[1])
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch, details, chats


def _check_group(records):
    if {record['condition'] for record in records} != set(CONDITIONS):
        raise ValueError('a question lacks all four native-image controls')
    first = records[0]
    fields = ('question_id', 'image_id', 'question_sha256', 'question_prompt_sha256',
              'image_sha256', 'raw_width', 'raw_height', 'resized_width', 'resized_height',
              'image_slots', 'image_grid_thw', 'visual_tokens', 'per_image_visual_tokens',
              'counted_image_tokens', 'premerge_patch_tokens', 'rgb_frame_sha256',
              'pixel_values_sha256', 'pixel_values_shape', 'pixel_values_dtype')
    if any(any(record[key] != first[key] for key in fields) for record in records[1:]):
        raise ValueError(f'four-condition native image/question contract differs: {first["question_id"]}')
    if any(record['input_truncated'] or record['answers_or_target_used'] for record in records):
        raise ValueError('answers entered inference or the input was truncated')
    if any(record['ocr_text_tokens'] > 256 or record['ocr_prefix_tokens'] > 320 for record in records):
        raise ValueError('actual OCR budgets differ from preregistered caps')
    lookup = {record['condition']: record for record in records}
    if lookup['baseline']['ocr_prefix_tokens'] or lookup['baseline']['ocr_text_tokens']:
        raise ValueError('baseline must preserve original prompt with no OCR prefix')
    if lookup['empty_template']['ocr_text_tokens']:
        raise ValueError('empty template control must contain no predicted text')
    if lookup['shuffled_ocr']['shuffle_source_image_id'] == first['image_id']:
        raise ValueError('shuffled context cannot come from the same image')


def _historical_checks(records, config):
    value = config.get('historical_baseline_predictions_path')
    if not value:
        return {'available': False, 'scope': 'no saved native-input metadata reference supplied'}
    path = private_path(value)
    historical = {str(item['question_id']): item for item in map(json.loads, path.read_text().splitlines())}
    baseline = [record for record in records if record['condition'] == 'baseline']
    fields = ('image_id', 'question_sha256', 'question_prompt_sha256', 'image_sha256',
              'image_grid_thw', 'visual_tokens', 'input_tokens',
              'raw_width', 'raw_height', 'resized_width', 'resized_height')
    for record in baseline:
        old = dict(historical[record['question_id']])
        if old['image_grid_thw'] and isinstance(old['image_grid_thw'][0], (int, float)):
            old['image_grid_thw'] = [old['image_grid_thw']]
        if any(record[key] != old[key] for key in fields):
            raise ValueError(f'native baseline differs from saved processor metadata: {record["question_id"]}')
    return {'available': True, 'prediction_file_sha256': sha_file(path),
            'questions_checked': len(baseline), 'fields': list(fields),
            'scope': 'input metadata only; historical cache lacks CPU pixel/input-id hashes; predictions are not reused'}


def prepare_inputs(config):
    """Real first32×4 checks, then complete6468 native-image input checks."""
    validate_config(config)
    started = time.monotonic()
    root = private_path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = private_path(config['manifest_path'])
    manifest = read_manifest(manifest_path)
    rows = manifest['splits']['dev']
    if len(rows) != 1617 or len({row['image_id'] for row in rows}) != 1024:
        raise ValueError('wrong dev coverage')
    processor = processor_for(config['model_root'], config)
    context_lock = prepare_contexts(config, processor, manifest)
    lock_path = _artifact_path(config, 'context_lock_path', 'OCR_CONTEXT_LOCK.json')
    context_lock_sha = sha_file(lock_path)
    sources, versions = source_hashes(), _package_versions()
    checks = {'native_images_pixels_grids_equal_all_conditions': True,
              'original_question_and_suffix_identical': True,
              'original_baseline_prompt_preserved': True,
              'actual_visual_tokens_at_most_1024': True,
              'actual_ocr_text_tokens_at_most_256': True,
              'actual_rendered_prefix_context_tokens_at_most_320': True,
              'no_input_truncation': True, 'no_answers_or_targets_used': True,
              'different_image_shuffled_context': True,
              'same_image_questions_share_predicted_ocr': True,
              'equal_total_input_tokens_claimed': False}

    def check_rows(selected, label):
        records = []
        for begin in range(0, len(selected), 4):
            chunk = [inference_row(row) for row in selected[begin:begin+4]]
            batch_records = []
            for condition in CONDITIONS:
                batch, values, _ = build_ocr_batch(processor, chunk, manifest['images'],
                                                  manifest_path.parent, condition, config, context_lock)
                del batch
                records.extend(values)
                batch_records.extend(values)
            for row in chunk:
                _check_group([value for value in batch_records if value['question_id'] == str(row['question_id'])])
            if begin % 64 == 0 or begin+4 >= len(selected):
                status = {'schema_version': 'rarevlm.ocr_context.cpu_prepare.v1', 'status': 'running',
                          'phase': label, 'questions_completed': min(begin+4, len(selected)),
                          'questions_expected': len(selected), 'records_completed': len(records),
                          'updated_at': now(), 'elapsed_seconds': time.monotonic() - started}
                write_json(root/'CPU_PREPARE_STATUS.json', status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
        return records

    chosen = sorted(rows, key=lambda row: int(row['question_id']))[:32]
    preflight = check_rows(chosen, 'first32_input_only_precheck')
    precheck = {'schema_version': 'rarevlm.ocr_context.precheck.v1', 'status': 'passed', 'complete': True,
                'created_at': now(), 'config_sha256': _config_sha(config),
                'configuration_content_sha256': _config_content_sha(config),
                'context_lock_sha256': context_lock_sha, 'source_hashes': sources,
                'package_versions': versions, 'conditions': list(CONDITIONS),
                'question_ids': [str(row['question_id']) for row in chosen],
                'selection': 'first32 dev question IDs in ascending numeric order; input-only',
                'questions': 32, 'processor_checks': len(preflight), 'records': preflight,
                'checks': checks, 'historical_native_input_check': _historical_checks(preflight, config)}
    write_json(_artifact_path(config, 'input_precheck_path', 'INPUT_PRECHECK.json'), precheck)
    complete = check_rows(rows, 'complete_internal_dev_native_input_contract')
    if len(complete) != 6468:
        raise ValueError('full native input contract is incomplete')
    contract = {'schema_version': 'rarevlm.ocr_context.input_contract.v1', 'status': 'passed', 'complete': True,
                'created_at': now(), 'config_sha256': _config_sha(config),
                'configuration_content_sha256': _config_content_sha(config),
                'context_lock_sha256': context_lock_sha, 'source_hashes': sources,
                'package_versions': versions, 'manifest_sha256': manifest['manifest_sha256'],
                'question_order_sha256': content_sha([str(row['question_id']) for row in rows]),
                'conditions': list(CONDITIONS), 'questions': 1617, 'images': 1024,
                'total_records': len(complete), 'processor_checks': len(complete),
                'records': complete, 'checks': checks,
                'historical_native_input_check': _historical_checks(complete, config),
                'elapsed_seconds': time.monotonic() - started,
                'scope': 'input validation only; no predictions, scoring or cached encoded tensors'}
    write_json(_artifact_path(config, 'input_contract_path', 'INPUT_CONTRACT.json'), contract)
    write_json(root/'CPU_PREPARE_STATUS.json', {'status': 'passed', 'complete': True,
               'completed_at': now(), 'elapsed_seconds': time.monotonic() - started,
               'processor_checks': len(preflight) + len(complete), 'context_lock_sha256': context_lock_sha})
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
                      'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
