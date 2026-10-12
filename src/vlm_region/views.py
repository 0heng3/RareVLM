"""Deterministic input-only region selection, matched controls and RGB canvases.

This module never reads answers, supervision targets, recognized OCR strings or
prediction scores. The transformed PNGs are the actual images fed to the VLM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from PIL import Image

CONDITIONS = ('whole', 'double_whole', 'text_region', 'random_matched')
PRIVATE = Path('/opt/data/private')


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def private_path(path):
    path = Path(path).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError(f'new region artifacts must stay under {PRIVATE}: {path}')
    return path


def rgb_sha(image):
    if image.mode != 'RGB':
        raise ValueError('actual VLM views must be RGB')
    return hashlib.sha256(image.tobytes()).hexdigest()


def input_rules(config):
    """Only frozen geometry/selection fields; ignore controller status/timing."""
    return {'schema_version': 'rarevlm.region_input_rules.v1',
            'full_canvas_wh': list(config.get('full_canvas_wh', [896, 896])),
            'half_canvas_wh': list(config.get('half_canvas_wh', [896, 448])),
            'padding_rgb': list(config.get('padding_rgb', [127, 127, 127])),
            'resize': 'PIL.Image.Resampling.BICUBIC',
            'rounding': 'floor(scale*dimension+0.5), centered integer offsets',
            'threshold': 0.6, 'selector': 'max(score*sqrt(clipped_envelope_area/image_area)); bbox lexicographic ties',
            'padding_fraction_each_side': 0.1, 'round_envelope': 'floor left/top, ceil right/bottom; clamp raw bounds',
            'random_seed': 20261012,
            'random_seed_derivation': 'SHA256 UTF8(str(base_seed)+":"+str(image_id)) as big-endian integer',
            'random_location': 'Python Random(seed).randint independent x/y legal integer translations',
            'fallback': 'full raw image for both detected and random crops',
            'condition_order': list(CONDITIONS), 'patch_size': 14, 'merge_size': 2,
            'visual_tokens_each_question': 1024, 'reference_answers_allowed': False,
            'recognized_ocr_text_allowed': False}


def validate_geometry(config):
    rules = input_rules(config)
    if (rules['full_canvas_wh'] != [896, 896] or rules['half_canvas_wh'] != [896, 448]
            or rules['padding_rgb'] != [127, 127, 127]):
        raise ValueError('region v1 requires frozen 896-square/896x448 letterbox geometry')
    selector = config.get('roi_selector', {})
    if selector.get('score_threshold', 0.6) != 0.6 or selector.get('padding_fraction', 0.1) != 0.1:
        raise ValueError('region v1 selector threshold/padding changed')
    if config.get('random_seed', 20261012) != 20261012:
        raise ValueError('region v1 random location seed changed')
    return rules


def select_region(polygons, scores, width, height):
    """Choose a legal polygon envelope using geometry/confidence only.

    Candidate scores use clipped floating-point envelopes, before padding and
    rounding. Coordinates returned to PIL are integer half-open raw-image boxes.
    """
    width, height = int(width), int(height)
    if width < 1 or height < 1 or len(polygons) != len(scores):
        raise ValueError('invalid image size or unpaired detector polygons/scores')
    eligible = []
    for index, (polygon, score) in enumerate(zip(polygons, scores)):
        try:
            score = float(score)
            points = [(float(point[0]), float(point[1])) for point in polygon]
        except (TypeError, ValueError, IndexError):
            continue
        if (len(points) < 3 or not math.isfinite(score) or not 0.6 <= score <= 1.0
                or any(not math.isfinite(x) or not math.isfinite(y) for x, y in points)):
            continue
        left = max(0.0, min(float(width), min(x for x, _ in points)))
        top = max(0.0, min(float(height), min(y for _, y in points)))
        right = max(0.0, min(float(width), max(x for x, _ in points)))
        bottom = max(0.0, min(float(height), max(y for _, y in points)))
        if right <= left or bottom <= top:
            continue
        box = [left, top, right, bottom]
        area = (right - left) * (bottom - top)
        priority = score * math.sqrt(area / (width * height))
        eligible.append((priority, box, index, score))
    if not eligible:
        return {'bbox_xyxy': [0, 0, width, height], 'fallback': True,
                'fallback_reason': 'no legal score>=0.6 positive clipped envelope',
                'eligible_candidates': 0, 'detector_candidates': len(polygons),
                'selection_index': None, 'score': None, 'priority': None,
                'clipped_unpadded_envelope_xyxy': None}
    priority, box, index, score = min(eligible, key=lambda item: (-item[0], item[1]))
    left, top, right, bottom = box
    dx, dy = 0.1 * (right - left), 0.1 * (bottom - top)
    crop = [max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
            min(width, math.ceil(right + dx)), min(height, math.ceil(bottom + dy))]
    if crop[2] - crop[0] < 1 or crop[3] - crop[1] < 1:
        raise ValueError('positive clipped envelope produced empty crop')
    return {'bbox_xyxy': crop, 'fallback': False, 'fallback_reason': None,
            'eligible_candidates': len(eligible), 'detector_candidates': len(polygons),
            'selection_index': index, 'score': score, 'priority': priority,
            'clipped_unpadded_envelope_xyxy': box}


def random_region(selected, width, height, image_id, base_seed=20261012):
    width, height = int(width), int(height)
    digest = hashlib.sha256(f'{base_seed}:{image_id}'.encode()).hexdigest()
    generator = random.Random(int(digest, 16))
    if selected['fallback']:
        box = [0, 0, width, height]
    else:
        left, top, right, bottom = selected['bbox_xyxy']
        crop_width, crop_height = right - left, bottom - top
        if not (1 <= crop_width <= width and 1 <= crop_height <= height):
            raise ValueError('selected crop outside source dimensions')
        x = generator.randint(0, width - crop_width)
        y = generator.randint(0, height - crop_height)
        box = [x, y, x + crop_width, y + crop_height]
    return {'bbox_xyxy': box, 'fallback': selected['fallback'],
            'seed_sha256': digest, 'base_seed': base_seed,
            'location_scope': 'uniform legal integer translations; informative regions not excluded'}


def letterbox(image, canvas_wh, padding_rgb=(127, 127, 127)):
    width, height = image.size
    canvas_width, canvas_height = map(int, canvas_wh)
    if min(width, height, canvas_width, canvas_height) < 1:
        raise ValueError('nonpositive image/canvas dimension')
    scale = min(canvas_width / width, canvas_height / height)
    resized_width = min(canvas_width, max(1, math.floor(width * scale + 0.5)))
    resized_height = min(canvas_height, max(1, math.floor(height * scale + 0.5)))
    offset = [(canvas_width - resized_width) // 2, (canvas_height - resized_height) // 2]
    resized = image.convert('RGB').resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (canvas_width, canvas_height), tuple(padding_rgb))
    canvas.paste(resized, tuple(offset))
    return canvas, {'canvas_wh': [canvas_width, canvas_height], 'resized_wh': [resized_width, resized_height],
                    'offset_xy': offset, 'scale_before_rounding': scale,
                    'padding_rgb': list(padding_rgb), 'resize': 'PIL.Image.Resampling.BICUBIC'}


def build_image_views(image_id, image_record, image_root, detector_record, config):
    """Prepare all four conditions once per source frame, write actual VLM PNGs."""
    rules = validate_geometry(config)
    image_path = Path(image_root) / image_record['path']
    actual_sha = sha_file(image_path)
    if actual_sha != image_record['sha256'] or actual_sha != detector_record['image_sha256']:
        raise ValueError(f'original image/detector source bytes changed: {image_id}')
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    width, height = image.size
    if ([width, height] != [int(image_record['width']), int(image_record['height'])]
            or [width, height] != [int(detector_record['width']), int(detector_record['height'])]):
        raise ValueError(f'original image/detector dimensions differ: {image_id}')
    selected = select_region(detector_record['polygons'], detector_record['scores'], width, height)
    control = random_region(selected, width, height, image_id)
    detector_record_sha = content_sha(detector_record)
    cache_key = content_sha({'source_image_sha256': actual_sha, 'detector_record_sha256': detector_record_sha,
                             'rules': rules, 'image_id': str(image_id)})
    directory = private_path(config['output_root']) / 'prepared_views' / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    full_box = [0, 0, width, height]
    definitions = {'whole_1024': (full_box, rules['full_canvas_wh']),
                   'whole_512': (full_box, rules['half_canvas_wh']),
                   'text_region_512': (selected['bbox_xyxy'], rules['half_canvas_wh']),
                   'random_region_512': (control['bbox_xyxy'], rules['half_canvas_wh'])}
    slot_records = {}
    crop_seconds = letterbox_seconds = png_seconds = 0.0
    for key, (box, canvas) in definitions.items():
        begun = time.monotonic()
        crop = image.crop(tuple(box))
        elapsed_crop = time.monotonic() - begun
        begun = time.monotonic()
        frame, transformation = letterbox(crop, canvas, rules['padding_rgb'])
        elapsed_letterbox = time.monotonic() - begun
        begun = time.monotonic()
        path = directory / f'{key}.png'
        frame_bytes_sha = rgb_sha(frame)
        if path.exists():
            with Image.open(path) as previous:
                if previous.mode != 'RGB' or previous.size != frame.size or rgb_sha(previous) != frame_bytes_sha:
                    raise ValueError(f'cached actual view bytes changed: {path}')
        else:
            temporary = path.with_suffix('.png.tmp')
            with temporary.open('wb') as stream:
                frame.save(stream, format='PNG', compress_level=1)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        elapsed_png = time.monotonic() - begun
        record = {'key': key, 'path': str(path.resolve()), 'png_sha256': sha_file(path),
                  'rgb_bytes_sha256': frame_bytes_sha, 'mode': 'RGB', 'size_wh': list(frame.size),
                  'bbox_xyxy': list(box), 'raw_crop_wh': [box[2] - box[0], box[3] - box[1]],
                  'transformation': transformation, 'crop_seconds': elapsed_crop,
                  'letterbox_seconds': elapsed_letterbox, 'png_write_verify_seconds': elapsed_png,
                  'actual_vlm_input': True, 'source_image_sha256': actual_sha}
        slot_records[key] = record
        crop_seconds += elapsed_crop
        letterbox_seconds += elapsed_letterbox
        png_seconds += elapsed_png
    selected_size = definitions['text_region_512'][0]
    random_size = definitions['random_region_512'][0]
    if [selected_size[2]-selected_size[0], selected_size[3]-selected_size[1]] != [random_size[2]-random_size[0], random_size[3]-random_size[1]]:
        raise ValueError('random control area/aspect changed')
    keys = {'whole': ['whole_1024'], 'double_whole': ['whole_512', 'whole_512'],
            'text_region': ['whole_512', 'text_region_512'],
            'random_matched': ['whole_512', 'random_region_512']}
    conditions = {}
    for condition, slots in keys.items():
        records = [slot_records[key] for key in slots]
        conditions[condition] = {'condition': condition, 'image_id': str(image_id),
                                'source_image_sha256': actual_sha, 'raw_size_wh': [width, height],
                                'selected_region': selected, 'random_region': control,
                                'views': records, 'image_slots': len(slots),
                                'view_sha256': content_sha([{key: view[key] for key in ('png_sha256', 'rgb_bytes_sha256', 'bbox_xyxy', 'size_wh')} for view in records]),
                                'detector_seconds': float(detector_record['detection_seconds']),
                                'crop_seconds': sum(item['crop_seconds'] for item in records),
                                'letterbox_seconds': sum(item['letterbox_seconds'] for item in records),
                                'png_write_verify_seconds': sum(item['png_write_verify_seconds'] for item in records),
                                'input_rules_sha256': content_sha(rules)}
    return {'source_image_sha256': actual_sha, 'source_rgb_bytes_sha256': rgb_sha(image),
            'raw_size_wh': [width, height], 'image_id': str(image_id),
            'detection_record_sha256': detector_record_sha, 'cache_key': cache_key,
            'unique_frame_preparation_seconds': {'crop': crop_seconds, 'letterbox': letterbox_seconds, 'png': png_seconds},
            'conditions': conditions}


def get_views(row, image_record, image_root, condition, config, view_lock=None):
    """Return exact cached RGB VLM input images and an input-only QA record.

    view_lock is the completed VIEW_LOCK.json object. Without it this function
    prepares the source frame from the pinned detector file, for CPU preflight.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'unknown region condition: {condition}')
    iid = str(row['image_id'])
    if view_lock is None:
        detections = json.loads(Path(config['detections_path']).read_text())
        prepared = build_image_views(iid, image_record, image_root, detections['images'][iid], config)
    else:
        if view_lock['input_rules'] != validate_geometry(config):
            raise ValueError('cached view input rules differ')
        prepared = view_lock['images'][iid]
        if prepared['source_image_sha256'] != image_record['sha256']:
            raise ValueError(f'cached source-image binding differs: {iid}')
    saved = prepared['conditions'][condition]
    images = []
    begun = time.monotonic()
    for record in saved['views']:
        path = private_path(record['path'])
        if sha_file(path) != record['png_sha256']:
            raise ValueError(f'cached input PNG changed: {path}')
        with Image.open(path) as opened:
            if opened.mode != 'RGB' or list(opened.size) != record['size_wh'] or rgb_sha(opened) != record['rgb_bytes_sha256']:
                raise ValueError(f'cached actual RGB frame changed: {path}')
            images.append(opened.copy())
    return images, {**saved, 'question_id': str(row['question_id']),
                    'question_sha256': content_sha(str(row['question'])),
                    'cached_view_load_verify_seconds': time.monotonic() - begun,
                    'answers_or_target_used': False}

