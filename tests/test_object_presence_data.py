"""Contracts protecting annotation labels, evidence and image-level isolation."""
from copy import deepcopy

import pytest

from vlm_object_presence.data import (PROMPT_SUFFIX, annotation_index, assert_disjoint,
    class_balanced_validation, definite_coco_source, frozen_audit, rows_for_image)


def example():
    source = {'source_image_path': 'coco2017/train2017/000000000001.jpg',
              'image_path': '/opt/data/private/image.jpg', 'pair_id': 'source:1'}
    actual = {'width': 100, 'height': 50, 'sha256': 'file1', 'pixel_sha256': 'pixels1'}
    document = {'info': {}, 'licenses': [],
        'categories': [{'id': n, 'name': f'object{n}'} for n in range(1, 81)],
        'images': [{'id': 1, 'file_name': '000000000001.jpg', 'width': 200, 'height': 100}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [20, 10, 60, 30], 'area': 1800, 'iscrowd': 0},
            {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [80, 10, 20, 20], 'area': 400, 'iscrowd': 0},
            {'id': 3, 'image_id': 1, 'category_id': 2, 'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 1}]}
    return source, actual, annotation_index(document, 'train2017')


def test_positive_all_boxes_crowd_never_negative_and_per_image_balance():
    source, actual, index = example()
    rows, image = rows_for_image(source, actual, index, 'train')
    positive = [row for row in rows if row['target'] == 'yes']
    negative = [row for row in rows if row['target'] == 'no']
    assert len(positive) == len(negative) == 1
    assert len(positive[0]['bbox_evidence']) == 2
    assert positive[0]['bbox_evidence'][0]['bbox_current_image_xywh'] == [10, 5, 30, 15]
    assert negative[0]['category_id'] not in {1, 2}
    assert image['annotated_any_including_crowd_category_ids'] == [1, 2]
    assert all(row['prompt_suffix'] == PROMPT_SUFFIX for row in rows)
    assert rows == rows_for_image(source, actual, index, 'train')[0]


def test_only_explicit_source_ids_and_full_frame_metadata_are_accepted():
    assert definite_coco_source('coco2017/val2017/000000000001.jpg') == ('val2017', 1)
    assert definite_coco_source('random/000000000001.jpg') is None
    assert definite_coco_source('coco2017/train2017/1.jpg') is None
    source, actual, index = example()
    with pytest.raises(ValueError, match='aspect ratio'):
        rows_for_image(source, {**actual, 'height': 100}, index, 'train')
    source['source_image_path'] = 'coco2017/val2017/000000000001.jpg'
    with pytest.raises(ValueError, match='split'):
        rows_for_image(source, actual, index, 'train')


def test_train_validation_identity_overlap_is_rejected():
    source, actual, index = example()
    rows, _ = rows_for_image(source, actual, index, 'train')
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(rows, deepcopy(rows))
    assert_disjoint(rows, [])


def test_frozen_balanced_audit_depends_on_identity_not_answers_or_input_order():
    source, actual, index = example()
    rows, _ = rows_for_image(source, actual, index, 'train')
    many = [{**row, 'question_id': row['question_id']+f':copy{n}', 'pixel_sha256': f'pixel{n}'}
            for n in range(80) for row in rows]
    selected = frozen_audit(many)
    assert len(selected) == 64
    assert sum(row['target'] == 'yes' for row in selected) == 32
    assert len({row['pixel_sha256'] for row in selected}) == 64
    assert selected == frozen_audit(list(reversed(many)))


def test_category_balance_diagnostic_retains_only_matched_labels_deterministically():
    rows = [{'question_id': f'q{n}', 'pixel_sha256': f'p{n}', 'category_id': c, 'target': label}
            for n, (c, label) in enumerate([(1, 'yes'), (1, 'yes'), (1, 'no'), (2, 'yes')])]
    selected, counts = class_balanced_validation(rows)
    assert len(selected) == 2
    assert {row['category_id'] for row in selected} == {1}
    assert sorted(row['target'] for row in selected) == ['no', 'yes']
    assert counts['2']['selected_yes'] == counts['2']['selected_no'] == 0
    assert selected == class_balanced_validation(list(reversed(rows)))[0]
