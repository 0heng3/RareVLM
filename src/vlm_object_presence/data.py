"""CPU-only contracts for a frozen, annotation-grounded yes/no corpus."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import re

from vlm_public_preference.data import content_sha, image_identity, sha_file, write_json

PROMPT_SUFFIX = '\nAnswer yes or no.'
SEED = 42


def definite_coco_source(path):
    """Accept only the explicit official COCO-2017 path namespace in this pool."""
    match = re.fullmatch(r'coco2017/(train2017|val2017)/(\d{12})\.jpg', str(path))
    return (match.group(1), int(match.group(2))) if match else None


def annotation_index(document, split):
    """Retain all categories and crowd records; never derive labels from answers."""
    categories = {int(c['id']): c for c in document['categories']}
    if len(categories) != 80 or len({c['name'] for c in categories.values()}) != 80:
        raise ValueError('Expected the official 80 unique COCO instance categories')
    images = {int(image['id']): image for image in document['images']}
    annotations = defaultdict(list)
    for annotation in document['annotations']:
        if annotation['image_id'] not in images or annotation['category_id'] not in categories:
            raise ValueError('Annotation references an unknown image or category')
        annotations[int(annotation['image_id'])].append(annotation)
    return {'images': images, 'categories': categories, 'annotations': dict(annotations),
            'split': split, 'info': document['info'], 'licenses': document['licenses']}


def positive_box(annotation, official, actual):
    box = [float(value) for value in annotation['bbox']]
    if (len(box) != 4 or not all(math.isfinite(value) for value in box)
            or box[2] <= 0 or box[3] <= 0 or annotation['area'] <= 0):
        raise ValueError('Invalid official non-crowd instance geometry')
    sx, sy = actual['width'] / official['width'], actual['height'] / official['height']
    return {'annotation_id': int(annotation['id']), 'iscrowd': 0,
            'bbox_coco_xywh': box,
            'bbox_current_image_xywh': [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy],
            'segmentation_area_coco_pixels': float(annotation['area']),
            'segmentation_area_fraction': float(annotation['area']) / (official['width'] * official['height']),
            'coco_area_bin': 'small' if annotation['area'] < 32**2 else
                             'medium' if annotation['area'] < 96**2 else 'large'}


def rows_for_image(source, actual, index, source_split, seed=SEED):
    mapping = definite_coco_source(source['source_image_path'])
    if mapping is None or mapping[0] != index['split']:
        raise ValueError('Source path does not match an official COCO annotation split')
    _, coco_id = mapping
    official = index['images'].get(coco_id)
    if official is None or official['file_name'] != f'{coco_id:012d}.jpg':
        raise ValueError('COCO image ID has no exact matching official image metadata')
    # The public pool may resize images. Anisotropic changes beyond integer-rounding
    # tolerance are rejected instead of claiming that official boxes identify pixels.
    ratio_error = abs(actual['width'] * official['height'] - actual['height'] * official['width'])
    if ratio_error > max(official['width'], official['height']) * 2:
        raise ValueError('Public image aspect ratio differs from the official COCO image')
    anns = index['annotations'].get(coco_id, [])
    all_present = {int(a['category_id']) for a in anns}
    positive = defaultdict(list)
    for annotation in anns:
        if int(annotation.get('iscrowd', 0)) == 0:
            positive[int(annotation['category_id'])].append(positive_box(annotation, official, actual))
    absent = set(index['categories']) - all_present
    if len(absent) < len(positive):
        raise ValueError('Cannot balance unique image/category positives and negatives')
    negatives = sorted(absent, key=lambda category: content_sha(
        ['object_presence_negative_v1', seed, coco_id, actual['pixel_sha256'], category]))[:len(positive)]
    image_id = f'coco2017:{coco_id:012d}'
    image = {'path': source['image_path'], 'sha256': actual['sha256'],
             'file_sha256': actual['sha256'], 'pixel_sha256': actual['pixel_sha256'],
             'width': actual['width'], 'height': actual['height'],
             'coco_image_id': f'{coco_id:012d}', 'coco_annotation_split': mapping[0],
             'coco_width': official['width'], 'coco_height': official['height'],
             'coco_url': official.get('coco_url'), 'source_image_path': source['source_image_path'],
             'source_pair_id': source['pair_id'], 'source_split': source_split,
             'annotated_noncrowd_category_ids': sorted(positive),
             'annotated_any_including_crowd_category_ids': sorted(all_present),
             'crowd_annotation_count': sum(int(a.get('iscrowd', 0)) for a in anns),
             'bbox_scaling': 'independent x/y scaling from official metadata to existing image; display evidence only',
             'public_image_is_full_coco_frame': 'metadata/aspect-ratio checked; no byte-identical original-image proof'}
    rows = []
    for category in sorted(positive) + negatives:
        name = index['categories'][category]['name']
        present = category in positive
        qid = f'object_presence_v1:{coco_id:012d}:category{category:02d}'
        rows.append({'question_id': qid, 'question_id_namespace': 'rarevlm.object_presence.v1:image_category',
                     'image_id': image_id, 'question': f'Is there a {name} in the image?',
                     'target': 'yes' if present else 'no', 'prompt_suffix': PROMPT_SUFFIX,
                     'image_path': source['image_path'], 'image_sha256': actual['sha256'],
                     'file_sha256': actual['sha256'], 'pixel_sha256': actual['pixel_sha256'],
                     'coco_image_id': f'{coco_id:012d}', 'coco_annotation_split': mapping[0],
                     'category_id': category, 'category_name': name, 'presence': present,
                     'source_image_path': source['source_image_path'], 'source_pair_id': source['pair_id'],
                     'source_split': source_split, 'bbox_evidence': positive.get(category, []),
                     'label_source': 'official_coco2017_instance_annotations_closed_world',
                     'label_rule': 'at_least_one_noncrowd_instance' if present else
                                   'absent_from_all_annotations_including_crowd',
                     'human_preference': False, 'model_prediction_used_for_label_or_sampling': False})
    return rows, image


def frozen_audit(rows, count=64, seed=SEED):
    if count % 2:
        raise ValueError('Audit count must be even')
    selected, seen_pixels = [], set()
    for label in ('yes', 'no'):
        candidates = [row for row in rows if row['target'] == label]
        candidates.sort(key=lambda row: content_sha(
            ['object_presence_audit_v1', seed, row['question_id'], row['pixel_sha256']]))
        chosen = []
        for row in candidates:
            if row['pixel_sha256'] in seen_pixels:
                continue
            chosen.append(row)
            seen_pixels.add(row['pixel_sha256'])
            if len(chosen) == count//2:
                break
        if len(chosen) != count//2:
            raise ValueError('Insufficient distinct images for a balanced frozen audit')
        selected += chosen
    return sorted(selected, key=lambda row: content_sha(
        ['object_presence_audit_order_v1', seed, row['question_id']]))


def class_balanced_validation(rows, seed=SEED):
    """Grounding diagnostic: exactly balanced labels for each fixed question text."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['category_id']][row['target']].append(row)
    result, counts = [], {}
    for category, labels in sorted(grouped.items()):
        count = min(len(labels['yes']), len(labels['no']))
        counts[str(category)] = {'original_yes': len(labels['yes']),
            'original_no': len(labels['no']), 'selected_yes': count, 'selected_no': count}
        for label in ('yes', 'no'):
            result += sorted(labels[label], key=lambda row: content_sha(
                ['object_presence_class_balance_v1', seed, row['question_id'], row['pixel_sha256']]))[:count]
    return sorted(result, key=lambda row: row['question_id']), counts


def corpus_summary(rows, images):
    categories = defaultdict(Counter)
    areas = Counter()
    for row in rows:
        categories[row['category_name']][row['target']] += 1
        if row['target'] == 'yes':
            areas.update(box['coco_area_bin'] for box in row['bbox_evidence'])
    return {'questions': len(rows), 'images': len(images),
            'labels': dict(Counter(row['target'] for row in rows)),
            'unique_image_category': len({(row['image_id'], row['category_id']) for row in rows}),
            'categories': {name: dict(value) for name, value in sorted(categories.items())},
            'positive_instance_area_bins': dict(areas),
            'coco_annotation_splits': dict(Counter(image['coco_annotation_split'] for image in images.values())),
            'positive_categories_per_image': dict(sorted(Counter(
                len(image['annotated_noncrowd_category_ids']) for image in images.values()).items()))}


def assert_disjoint(train_rows, val_rows):
    for key in ('question_id', 'image_id', 'pixel_sha256', 'file_sha256', 'coco_image_id'):
        if {row[key] for row in train_rows} & {row[key] for row in val_rows}:
            raise ValueError(f'Train/validation overlap: {key}')
    for rows in (train_rows, val_rows):
        if len({row['question_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate image/category question')
        balance = defaultdict(Counter)
        for row in rows:
            balance[row['image_id']][row['target']] += 1
        if any(counts['yes'] != counts['no'] for counts in balance.values()):
            raise ValueError('Per-image yes/no balance failed')


def cpu_encode(row, processor, max_length=2048):
    """Run the original frozen encoder on the actual complete supervised sequence."""
    from vlm_deep.common import encode
    sample = encode(processor, row, Path(row['image_path']), True)
    ids = sample['input_ids']
    length = int(ids.shape[1])
    grid = sample['image_grid_thw'].tolist()
    visual = sum(int(t*h*w//4) for t, h, w in grid)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    counted = int((ids == image_token_id).sum())
    supervised = int((sample['labels'] != -100).sum())
    if length > max_length or not 0 < visual <= 1024 or counted != visual or supervised <= 0:
        raise ValueError(f'Actual encoding contract failed: {row["question_id"]}')
    import hashlib
    return {'question_id': row['question_id'], 'sequence_length': length,
            'visual_tokens': visual, 'counted_image_tokens': counted,
            'assistant_tokens': supervised, 'image_grid_thw': grid,
            'input_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            'labels_sha256': hashlib.sha256(sample['labels'].numpy().tobytes()).hexdigest(),
            'truncation': False}
