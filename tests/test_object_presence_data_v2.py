"""CPU contracts for the fresh restricted-category v2 presence corpus."""
from collections import Counter
from copy import deepcopy

import pytest

from vlm_object_presence.data import content_sha, frozen_audit as v1_frozen_audit
from vlm_object_presence_v2.data import CATEGORIES, freeze_audit, qualifies, select_split


CATEGORY_NAMES = (
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat',
)
CATEGORY_IDS = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 2, 3, 4, 5, 6, 7, 8, 9)
NAMES = dict(zip(CATEGORY_IDS, CATEGORY_NAMES))
CATALOG_IMAGE_ID = 'coco2017:000000090000'


def box(width=24.0, height=24.0, area_fraction=0.005, *, crowd=0):
    return {
        'annotation_id': 1, 'iscrowd': crowd,
        'bbox_coco_xywh': [0.0, 0.0, 200.0, 200.0],
        'bbox_current_image_xywh': [0.0, 0.0, width, height],
        'segmentation_area_fraction': area_fraction,
        'segmentation_area_coco_pixels': 2000.0, 'coco_area_bin': 'medium',
    }


def source_image(number, positives=None, *, annotated=None):
    positives = {16: [box()]} if positives is None else positives
    coco_id = f'{number:012d}'
    image_id = f'coco2017:{coco_id}'
    image = {
        'path': f'/opt/data/private/synthetic/{coco_id}.jpg',
        'sha256': f'file:{number}', 'file_sha256': f'file:{number}',
        'pixel_sha256': f'pixel:{number}', 'width': 320, 'height': 240,
        'coco_image_id': coco_id, 'coco_annotation_split': 'train2017',
        'source_image_path': f'coco2017/train2017/{coco_id}.jpg',
        'source_pair_id': f'source:{number}', 'source_split': 'train',
        'annotated_noncrowd_category_ids': sorted(positives),
        'annotated_any_including_crowd_category_ids': sorted(
            positives if annotated is None else annotated),
    }
    rows = [{
        'question_id': f'object_presence_v1:{coco_id}:category{category:02d}',
        'image_id': image_id, 'category_id': category,
        'category_name': NAMES.get(category, 'person'), 'target': 'yes',
        'bbox_evidence': evidence,
    } for category, evidence in positives.items()]
    return image_id, image, rows


def manifest(*sources):
    # The frozen source must identify all 18 category IDs. This extra image has
    # valid v1 positives that all fail v2 size limits, so it cannot enter v2.
    catalog = source_image(90000, {
        category: [box(12, 12, 0.001)] for category in CATEGORY_IDS})
    all_sources = (*sources, catalog)
    return {
        'rows': [row for _, _, rows in all_sources for row in rows],
        'images': {image_id: image for image_id, image, _ in all_sources},
    }


def old_audit_row(number):
    return {
        'image_id': f'old-image:{number}', 'pixel_sha256': f'old-pixel:{number}',
        'file_sha256': f'old-file:{number}', 'coco_image_id': f'{number+100000:012d}',
    }


def test_category_scope_is_the_frozen_18_names():
    assert isinstance(CATEGORIES, tuple)
    assert CATEGORIES == CATEGORY_NAMES


@pytest.mark.parametrize('width,height,area,crowd,expected', [
    (24.0, 24.0, 0.005, 0, True),
    (200.0, 24.0, 0.005, 0, True),
    (24.0, 200.0, 0.005, 0, True),
    (23.999, 200.0, 0.5, 0, False),
    (200.0, 23.999, 0.5, 0, False),
    (100.0, 100.0, 0.004999, 0, False),
    (100.0, 100.0, 0.5, 1, False),
])
def test_qualifies_uses_inclusive_current_image_geometry_and_segmentation_area(
        width, height, area, crowd, expected):
    assert qualifies(box(width, height, area, crowd=crowd)) is expected


def test_same_instance_must_meet_both_limits_and_small_annotated_objects_are_not_negatives():
    source = manifest(source_image(1, {
        16: [box(23, 100, 0.1), box(100, 100, 0.004)],
        17: [box(), box(100, 100, 0.004)],
        1: [box(100, 100, 0.1)],
    }))
    rows, images, inventory = select_split(source, [], 'train')
    assert set(images) == {'coco2017:000000000001'}
    assert [row['category_id'] for row in rows if row['target'] == 'yes'] == [17]
    assert all(row['category_id'] not in {1, 16} for row in rows)
    assert Counter(row['target'] for row in rows) == {'yes': 1, 'no': 1}
    assert all(qualifies(evidence) for row in rows if row['target'] == 'yes'
               for evidence in row['bbox_evidence'])
    assert any(item['image_id'] == CATALOG_IMAGE_ID and not item['admitted']
               for item in inventory)


def test_all_qualified_positives_and_equal_unique_negatives_are_fixed_without_mutating_v1():
    source = manifest(source_image(7, {16: [box()], 17: [box()], 18: [box()]}))
    before = deepcopy(source)
    rows, images, inventory = select_split(source, [], 'val', seed=42)
    assert source == before
    assert set(images) == {'coco2017:000000000007'}
    assert Counter(row['target'] for row in rows) == {'yes': 3, 'no': 3}
    assert {row['category_id'] for row in rows if row['target'] == 'yes'} == {16, 17, 18}
    assert len({row['category_id'] for row in rows}) == 6
    expected_negative_ids = set(sorted(set(CATEGORY_IDS) - {16, 17, 18}, key=lambda category:
        content_sha(['object_presence_v2_negative', 42, 7, 'pixel:7', category]))[:3])
    assert {row['category_id'] for row in rows if row['target'] == 'no'} == expected_negative_ids
    for row in rows:
        assert row['question_id'] == f'object_presence_v2:000000000007:category{row["category_id"]:02d}'
        assert row['question'] == f'Is there a {row["category_name"]} in the image?'
        assert row['prompt_suffix'] == '\nAnswer yes or no.'
        assert row['source_split'] == 'val'
        assert row['human_preference'] is False
        assert row['model_prediction_used_for_label_or_sampling'] is False
        if row['target'] == 'no':
            assert row['bbox_evidence'] == []
    reversed_source = {'rows': list(reversed(source['rows'])),
                       'images': dict(reversed(list(source['images'].items())))}
    assert (rows, images, inventory) == select_split(reversed_source, [], 'val', seed=42)


def test_crowd_and_unqualified_present_categories_are_protected_from_negative_sampling():
    # Only boat is absent. Every other restricted category is in the complete
    # annotation set, even though only bird has a qualifying non-crowd instance.
    source = manifest(source_image(2, {16: [box()], 17: [box(12, 12, 0.001)]},
                                   annotated=set(CATEGORY_IDS) - {9}))
    rows, images, _ = select_split(source, [], 'train')
    assert len(images) == 1
    assert {(row['category_id'], row['target']) for row in rows} == {(16, 'yes'), (9, 'no')}


def test_insufficient_unique_absent_categories_rejects_the_whole_image():
    source = manifest(source_image(3, {16: [box()], 17: [box()]},
                                   annotated=set(CATEGORY_IDS) - {9}))
    rows, images, inventory = select_split(source, [], 'train')
    assert rows == []
    assert images == {}
    rejected = next(item for item in inventory if item['image_id'] == 'coco2017:000000000003')
    assert rejected['admitted'] is False
    assert 'insufficient_unique_absent_categories' in rejected['reasons']


def test_an_image_without_a_qualified_positive_is_rejected_in_full():
    source = manifest(source_image(4, {16: [box(23, 100, 0.1), box(100, 100, 0.004)]}))
    rows, images, inventory = select_split(source, [], 'train')
    assert rows == []
    assert images == {}
    rejected = next(item for item in inventory if item['image_id'] == 'coco2017:000000000004')
    assert rejected['admitted'] is False
    assert 'no_qualified_18_class_positive' in rejected['reasons']


@pytest.mark.parametrize('identity_key', ['image_id', 'pixel_sha256', 'file_sha256', 'coco_image_id'])
def test_any_identity_in_any_of_the_old_64_rows_excludes_the_entire_image(identity_key):
    first = source_image(5, {16: [box()], 17: [box()]})
    source = manifest(first, source_image(6))
    old = [old_audit_row(number) for number in range(64)]
    image_id, image, _ = first
    old[-1][identity_key] = {
        'image_id': image_id, 'pixel_sha256': image['pixel_sha256'],
        'file_sha256': image['sha256'], 'coco_image_id': image['coco_image_id'],
    }[identity_key]
    rows, images, inventory = select_split(source, old, 'train')
    assert set(images) == {'coco2017:000000000006'}
    assert all(row['image_id'] != image_id for row in rows)
    rejected = next(item for item in inventory if item['image_id'] == image_id)
    assert rejected['admitted'] is False
    assert 'all_v1_audit_images_excluded' in rejected['reasons']


def test_all_64_old_audit_images_are_excluded_rather_than_only_old_questions():
    sources = [source_image(number, {16: [box()], 17: [box()]}) for number in range(1, 66)]
    source = manifest(*sources)
    old = [{
        'image_id': image_id, 'pixel_sha256': image['pixel_sha256'],
        'file_sha256': image['sha256'], 'coco_image_id': image['coco_image_id'],
        'question_id': rows[0]['question_id'], 'target': 'yes',
    } for image_id, image, rows in sources[:64]]
    rows, images, _ = select_split(source, old, 'train')
    assert set(images) == {'coco2017:000000000065'}
    assert len(rows) == 4
    assert not {row['pixel_sha256'] for row in rows} & {row['pixel_sha256'] for row in old}


def audit_candidates(image_count=96):
    return [{
        'question_id': f'object_presence_v2:{number:012d}:category{category:02d}',
        'image_id': f'coco2017:{number:012d}', 'pixel_sha256': f'audit-pixel:{number}',
        'target': label,
    } for number in range(image_count) for category, label in ((16, 'yes'), (17, 'yes'), (18, 'no'), (19, 'no'))]


def test_fresh_audit_is_balanced_64_distinct_pixels_yes_first_then_independent_order():
    candidates = audit_candidates()
    before = deepcopy(candidates)
    selected = freeze_audit(candidates, count=64, seed=42)
    assert candidates == before
    assert len(selected) == 64
    assert Counter(row['target'] for row in selected) == {'yes': 32, 'no': 32}
    assert len({row['pixel_sha256'] for row in selected}) == 64
    assert selected == freeze_audit(list(reversed(candidates)), count=64, seed=42)
    seen, expected = set(), []
    for label in ('yes', 'no'):
        ranked = sorted((row for row in candidates if row['target'] == label), key=lambda row:
            content_sha(['object_presence_v2_audit', 42, row['question_id'], row['pixel_sha256']]))
        label_selection = []
        for row in ranked:
            if row['pixel_sha256'] not in seen:
                label_selection.append(row)
                seen.add(row['pixel_sha256'])
                if len(label_selection) == 32:
                    break
        expected.extend(label_selection)
    expected.sort(key=lambda row: content_sha(['object_presence_v2_audit_order', 42, row['question_id']]))
    assert selected == expected
    assert {row['question_id'] for row in selected} != {
        row['question_id'] for row in v1_frozen_audit(candidates, count=64, seed=42)}


def test_audit_rejects_an_odd_count_and_insufficient_distinct_images_across_labels():
    with pytest.raises(ValueError, match='even'):
        freeze_audit(audit_candidates(), count=63)
    with pytest.raises(ValueError, match='distinct images'):
        freeze_audit(audit_candidates(image_count=63), count=64)
