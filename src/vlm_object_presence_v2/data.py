"""Deterministic, prediction-independent transformation of frozen v1 images."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from vlm_object_presence.data import PROMPT_SUFFIX, content_sha

CATEGORIES = ('bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat')
MIN_BOX_SHORT_SIDE = 24.0
MIN_SEGMENTATION_AREA_FRACTION = 0.005
NEGATIVE_SALT = 'object_presence_v2_negative'
AUDIT_SALT = 'object_presence_v2_audit'
AUDIT_ORDER_SALT = 'object_presence_v2_audit_order'
CLASS_BALANCE_SALT = 'object_presence_v2_class_balance'


def qualifies(box):
    """Both limits must be met by the same non-crowd annotated instance."""
    return (int(box.get('iscrowd', 0)) == 0
            and min(box['bbox_current_image_xywh'][2:]) >= MIN_BOX_SHORT_SIDE
            and box['segmentation_area_fraction'] >= MIN_SEGMENTATION_AREA_FRACTION)


def audit_exclusion_sets(rows):
    return {key: {str(row[key]) for row in rows} for key in
            ('image_id', 'pixel_sha256', 'file_sha256', 'coco_image_id')}


def select_split(manifest, old_audit_rows, split, seed=42):
    """Keep every eligible image and all qualified categories, preserving split."""
    excluded = audit_exclusion_sets(old_audit_rows)
    categories = {row['category_id']: row['category_name'] for row in manifest['rows']
                  if row['category_name'] in CATEGORIES}
    if set(categories.values()) != set(CATEGORIES) or len(categories) != 18:
        raise ValueError('Frozen source must identify all 18 official categories')
    category_ids = set(categories)
    positive_by_image = defaultdict(list)
    for row in manifest['rows']:
        if row['target'] == 'yes' and row['category_id'] in category_ids:
            positive_by_image[row['image_id']].append(row)
    result, images, inventory = [], {}, []
    for image_id, image in sorted(manifest['images'].items()):
        identity = {'image_id': image_id, 'pixel_sha256': image['pixel_sha256'],
                    'file_sha256': image['sha256'], 'coco_image_id': image['coco_image_id']}
        record = {**identity, 'source_split': split, 'source_image_path': image['source_image_path']}
        if any(str(identity[key]) in values for key, values in excluded.items()):
            inventory.append({**record, 'admitted': False, 'reasons': ['all_v1_audit_images_excluded']})
            continue
        positives = {row['category_id']: row for row in positive_by_image[image_id]
                     if any(qualifies(box) for box in row['bbox_evidence'])}
        if not positives:
            inventory.append({**record, 'admitted': False, 'reasons': ['no_qualified_18_class_positive']})
            continue
        absent = category_ids - set(image['annotated_any_including_crowd_category_ids'])
        if len(absent) < len(positives):
            inventory.append({**record, 'admitted': False,
                'reasons': ['insufficient_unique_absent_categories'],
                'positive_category_count': len(positives), 'absent_category_count': len(absent)})
            continue
        negatives = sorted(absent, key=lambda category: content_sha(
            [NEGATIVE_SALT, seed, int(image['coco_image_id']), image['pixel_sha256'], category]))[:len(positives)]
        images[image_id] = {**image, 'v2_positive_category_ids': sorted(positives),
                            'v2_category_scope': list(CATEGORIES)}
        for category in sorted(positives) + negatives:
            present = category in positives
            original = positives.get(category)
            name = categories[category]
            evidence = [box for box in original['bbox_evidence'] if qualifies(box)] if present else []
            result.append({'question_id': f'object_presence_v2:{int(image["coco_image_id"]):012d}:category{category:02d}',
                'question_id_namespace': 'rarevlm.object_presence.v2:image_category',
                'image_id': image_id, 'question': f'Is there a {name} in the image?',
                'target': 'yes' if present else 'no', 'prompt_suffix': PROMPT_SUFFIX,
                'image_path': image['path'], 'image_sha256': image['sha256'],
                'file_sha256': image['sha256'], 'pixel_sha256': image['pixel_sha256'],
                'coco_image_id': image['coco_image_id'], 'coco_annotation_split': image['coco_annotation_split'],
                'category_id': category, 'category_name': name, 'presence': present,
                'source_image_path': image['source_image_path'], 'source_pair_id': image['source_pair_id'],
                'source_split': split, 'bbox_evidence': evidence,
                'all_noncrowd_instance_count': len(original['bbox_evidence']) if present else 0,
                'v1_positive_question_id': original['question_id'] if present else None,
                'positive_min_current_bbox_short_side_px': MIN_BOX_SHORT_SIDE,
                'positive_min_segmentation_area_fraction': MIN_SEGMENTATION_AREA_FRACTION,
                'label_source': 'official_coco2017_instance_annotations_closed_world_restricted_18_class',
                'label_rule': 'at_least_one_noncrowd_instance_meeting_both_visible_size_limits' if present else
                              'absent_from_all_annotations_including_crowd',
                'human_preference': False, 'model_prediction_used_for_label_or_sampling': False})
        inventory.append({**record, 'admitted': True, 'reasons': [],
                          'positive_category_count': len(positives),
                          'absent_category_count': len(absent), 'questions': 2*len(positives)})
    return sorted(result, key=lambda row: row['question_id']), images, inventory


def freeze_audit(rows, count=64, seed=42):
    if count % 2:
        raise ValueError('Balanced audit requires an even count')
    selected, seen = [], set()
    for label in ('yes', 'no'):
        candidates = sorted((row for row in rows if row['target'] == label), key=lambda row:
            content_sha([AUDIT_SALT, seed, row['question_id'], row['pixel_sha256']]))
        chosen = []
        for row in candidates:
            if row['pixel_sha256'] in seen:
                continue
            chosen.append(row)
            seen.add(row['pixel_sha256'])
            if len(chosen) == count//2:
                break
        if len(chosen) != count//2:
            raise ValueError('Insufficient distinct images for the fixed balanced audit')
        selected.extend(chosen)
    return sorted(selected, key=lambda row: content_sha(
        [AUDIT_ORDER_SALT, seed, row['question_id']]))


def class_balanced_validation(rows, seed=42):
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


def summary(rows, images):
    classes = defaultdict(Counter)
    for row in rows:
        classes[row['category_name']][row['target']] += 1
    return {'questions': len(rows), 'images': len(images),
            'labels': dict(Counter(row['target'] for row in rows)),
            'categories': {key: dict(value) for key, value in sorted(classes.items())},
            'positive_instance_area_bins': dict(Counter(box['coco_area_bin']
                for row in rows if row['target'] == 'yes' for box in row['bbox_evidence'])),
            'coco_annotation_splits': dict(Counter(image['coco_annotation_split'] for image in images.values()))}


def fresh_cpu_encode(row, processor):
    """Fresh actual encoding, with full token IDs and processed-pixel evidence."""
    from vlm_deep.common import encode
    sample = encode(processor, row, Path(row['image_path']), True)
    ids, labels, pixels = sample['input_ids'], sample['labels'], sample['pixel_values']
    grid = sample['image_grid_thw'].tolist()
    length = int(ids.shape[1])
    visual = sum(int(t*h*w//4) for t, h, w in grid)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    counted = int((ids == image_token_id).sum())
    supervised = int((labels != -100).sum())
    if (len(grid) != 1 or grid[0][0] != 1 or length > 2048 or not 0 < visual <= 1024
            or visual != counted or pixels.shape[0] != visual*4 or supervised <= 0):
        raise ValueError(f'Actual fresh encoding contract failed: {row["question_id"]}')
    return {'question_id': row['question_id'], 'sequence_length': length,
            'visual_tokens': visual, 'counted_image_tokens': counted,
            'assistant_tokens': supervised, 'image_grid_thw': grid,
            'raw_rgb_pixel_sha256': row['pixel_sha256'],
            'processor_pixel_values_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
            'processor_pixel_values_shape': list(pixels.shape),
            'processor_pixel_values_dtype': str(pixels.dtype),
            'input_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            'labels_sha256': hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
            'input_ids': ids[0].tolist(), 'labels': labels[0].tolist(),
            'encoding_origin': 'fresh_v2_cpu', 'v1_encoding_reused': False,
            'truncation': False}
