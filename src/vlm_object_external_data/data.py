"""Integrity, exact-image exclusion and fixed external-question contracts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import re

from PIL import Image, ImageFile

from vlm_object_presence.data import PROMPT_SUFFIX, content_sha, image_identity
from vlm_object_presence_v2.data import CATEGORIES

NEGATIVE_SALT = 'object_external_v1_negative'
AUDIT_SALT = 'object_external_v1_audit'
AUDIT_ORDER_SALT = 'object_external_v1_audit_order'
CLASS_BALANCE_SALT = 'object_external_v1_class_balance'
DEDUP_SALT = 'object_external_v1_image_dedup'


def verify_image_bytes(data, metadata, headers):
    """Verify complete original JPEG bytes; never resize or transform storage."""
    headers = {str(k).lower(): str(v) for k, v in headers.items()}
    length = headers.get('content-length', '')
    if not length.isdigit() or int(length) <= 0 or len(data) != int(length):
        raise ValueError('Content-Length missing or different from complete image bytes')
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError('Truncated-image decoding must not be enabled')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG':
            raise ValueError('Official val2017 storage must be JPEG')
        image.verify()
    identity = image_identity(data)  # Reopen, fully load and decode RGB.
    if (identity['width'], identity['height']) != (metadata['width'], metadata['height']):
        raise ValueError('Native JPEG dimensions differ from official image metadata')
    md5 = hashlib.md5(data).hexdigest()
    etag = headers.get('etag')
    token = etag.strip().strip('"') if etag else ''
    is_md5 = bool(re.fullmatch(r'[0-9a-fA-F]{32}', token))
    if is_md5 and md5 != token.lower():
        raise ValueError('Official single-part ETag MD5 differs from complete bytes')
    return {**identity, 'file_sha256': identity['sha256'], 'md5': md5,
        'http_content_length': int(length), 'etag': etag, 'etag_is_md5': is_md5,
        'etag_md5_verified': is_md5, 'etag_not_md5_reason': None if is_md5 else
            ('missing' if not etag else 'non-single-part-MD5 ETag'),
        'official_published_sha256_claimed': False, 'jpeg_fully_decoded': True,
        'stored_native_dimensions': True, 'stored_image_resized_or_cropped': False}


def select_verified_images(inventory_rows, verified_mapping, old_file_shas, old_pixel_shas, seed=42):
    ids = [row['coco_image_id'] for row in inventory_rows]
    if len(set(ids)) != len(ids) or any(iid not in verified_mapping for iid in ids):
        raise ValueError('Every unique metadata candidate needs a verified complete download')
    ordered = sorted(inventory_rows, key=lambda row: content_sha([
        DEDUP_SALT, seed, row['coco_image_id'], verified_mapping[row['coco_image_id']]['pixel_sha256']]))
    retained, excluded, files, pixels = [], [], {}, {}
    for row in ordered:
        iid = row['coco_image_id']
        image = verified_mapping[iid]
        reasons = []
        if image['sha256'] in old_file_shas:
            reasons.append('existing_training_validation_review_benchmark_file_sha')
        if image['pixel_sha256'] in old_pixel_shas:
            reasons.append('existing_training_validation_review_benchmark_pixel_sha')
        duplicate = files.get(image['sha256']) or pixels.get(image['pixel_sha256'])
        if duplicate:
            reasons.append('duplicate_within_new_pool_file_or_pixel_sha')
        if reasons:
            excluded.append({'coco_image_id': iid, 'file_sha256': image['sha256'],
                'pixel_sha256': image['pixel_sha256'], 'reasons': reasons,
                'duplicate_of_coco_image_id': duplicate})
            continue
        files[image['sha256']] = pixels[image['pixel_sha256']] = iid
        retained.append(row)
    return sorted(retained, key=lambda row: row['coco_image_id']), excluded


def build_rows(meta_row, identity):
    if (identity['width'], identity['height']) != (meta_row['width'], meta_row['height']):
        raise ValueError('Question evidence requires the verified official native dimensions')
    category_map = identity['category_map']
    if set(category_map.values()) != set(CATEGORIES) or len(category_map) != 18:
        raise ValueError('External data must retain the fixed 18 official categories')
    positives = {int(item['category_id']): item for item in meta_row['positive_categories']}
    if not set(positives).issubset(set(meta_row['all_annotated_category_ids_including_crowd'])):
        raise ValueError('Every positive category must occur in the official annotation inventory')
    for category, item in positives.items():
        if category not in category_map or not item['qualifying_instances']:
            raise ValueError('Invalid external positive category')
        for box in item['qualifying_instances']:
            if (int(box.get('iscrowd', 0)) != 0 or min(box['bbox_native_metadata_xywh'][2:]) < 24
                or box['segmentation_area_fraction'] < .005):
                raise ValueError('Both visible-size limits must hold on the same noncrowd instance')
    absent = set(category_map) - set(meta_row['all_annotated_category_ids_including_crowd'])
    if not positives or len(absent) < len(positives):
        raise ValueError('External image has no positive or insufficient unique absent categories')
    iid = meta_row['coco_image_id']
    negatives = sorted(absent, key=lambda category: content_sha([
        NEGATIVE_SALT, 42, int(iid), identity['pixel_sha256'], category]))[:len(positives)]
    image_id = f'coco2017:{int(iid):012d}'
    image = {'path': identity['path'], 'sha256': identity['sha256'],
        'file_sha256': identity['sha256'], 'pixel_sha256': identity['pixel_sha256'],
        'width': identity['width'], 'height': identity['height'],
        'coco_width': meta_row['width'], 'coco_height': meta_row['height'],
        'coco_image_id': iid, 'coco_annotation_split': 'val2017',
        'coco_url': meta_row['source_coco_url'], 'official_https_url': meta_row['official_https_url'],
        'source_split': 'external_confirmation', 'source_image_path': 'val2017/' + meta_row['file_name'],
        'source_pair_id': None, 'annotated_noncrowd_category_ids': sorted(positives),
        'annotated_any_including_crowd_category_ids': meta_row['all_annotated_category_ids_including_crowd'],
        'native_original_jpeg': True, 'source_image_resized_or_cropped': False}
    rows = []
    for category in sorted(positives) + negatives:
        present = category in positives
        boxes = [{**box, 'bbox_coco_xywh': box['bbox_native_metadata_xywh'],
                  'bbox_current_image_xywh': box['bbox_native_metadata_xywh']}
                 for box in positives[category]['qualifying_instances']] if present else []
        rows.append({'question_id': f'object_external_v1:{int(iid):012d}:category{category:02d}',
            'question_id_namespace': 'rarevlm.object_external.v1:image_category',
            'image_id': image_id, 'question': f'Is there a {category_map[category]} in the image?',
            'target': 'yes' if present else 'no', 'prompt_suffix': PROMPT_SUFFIX,
            'image_path': identity['path'], 'image_sha256': identity['sha256'],
            'file_sha256': identity['sha256'], 'pixel_sha256': identity['pixel_sha256'],
            'coco_image_id': iid, 'coco_annotation_split': 'val2017',
            'category_id': category, 'category_name': category_map[category], 'presence': present,
            'bbox_evidence': boxes, 'source_split': 'external_confirmation',
            'source_coco_url': meta_row['source_coco_url'], 'official_https_url': meta_row['official_https_url'],
            'label_source': 'official_COCO2017_closed_world_instance_annotations',
            'label_rule': 'same_noncrowd_instance_shortside24px_and_seg_fraction_point005' if present else
                          'absent_from_all_annotations_including_crowd_and_small_instances',
            'human_preference': False, 'model_prediction_used_for_label_or_sampling': False,
            'evaluation_requires_external_admission': True, 'training_allowed': False})
    return rows, image


def freeze_audit(rows, count=64, seed=42):
    if count <= 0 or count % 2:
        raise ValueError('A positive even balanced audit count is required')
    selected, seen = [], set()
    for label in ('yes', 'no'):
        ordered = sorted((row for row in rows if row['target'] == label), key=lambda row:
            content_sha([AUDIT_SALT, seed, row['question_id'], row['pixel_sha256']]))
        chosen = []
        for row in ordered:
            if row['pixel_sha256'] in seen:
                continue
            chosen.append(row)
            seen.add(row['pixel_sha256'])
            if len(chosen) == count//2:
                break
        if len(chosen) != count//2:
            raise ValueError('Insufficient distinct external images for the fixed balanced audit')
        selected += chosen
    return sorted(selected, key=lambda row: content_sha([AUDIT_ORDER_SALT, seed, row['question_id']]))


def class_balanced_subset(rows, seed=42):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['category_id']][row['target']].append(row)
    selected, counts = [], {}
    for category, labels in sorted(groups.items()):
        count = min(len(labels['yes']), len(labels['no']))
        counts[str(category)] = {'original_yes': len(labels['yes']), 'original_no': len(labels['no']),
                                'selected_yes': count, 'selected_no': count}
        for label in ('yes', 'no'):
            selected += sorted(labels[label], key=lambda row: content_sha(
                [CLASS_BALANCE_SALT, seed, row['question_id'], row['pixel_sha256']]))[:count]
    return sorted(selected, key=lambda row: row['question_id']), counts


def corpus_counts(rows, images):
    categories = defaultdict(Counter)
    for row in rows:
        categories[row['category_name']][row['target']] += 1
    return {'questions': len(rows), 'images': len(images),
        'labels': dict(Counter(row['target'] for row in rows)),
        'categories': {key: dict(value) for key, value in sorted(categories.items())}}
