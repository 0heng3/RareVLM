"""Targeted CPU contracts for known-uncertainty training admission."""
from copy import deepcopy

import pytest

from vlm_semantic_admission_v1 import build_training_view, extract_training_rows


def fixture():
    rows = [{'question_id': 'positive', 'image_id': 'i1', 'complementary_pair_id': 'pair',
             'question': 'Are these buses the same color?', 'target': 'yes',
             'answers': ['yes'] * 10, 'answer_vote_count': 10, 'source_split': 'train',
             'label_source': 'official_vqa_v2_train2014_human_answers'},
            {'question_id': 'negative', 'image_id': 'i2', 'complementary_pair_id': 'pair',
             'question': 'Are these buses the same color?', 'target': 'no',
             'answers': ['no'] * 9 + ['yes'], 'answer_vote_count': 9, 'source_split': 'train',
             'label_source': 'official_vqa_v2_train2014_human_answers'}]
    return {'schema_version': 'rarevlm.vlm_data.v2',
            'splits': {'train_all': [], 'dev': rows, 'final': []},
            'images': {'i1': {'path': '/original/a.jpg', 'sha256': 'a'},
                       'i2': {'path': '/original/b.jpg', 'sha256': 'b'}}}


def review(passed=True, answer='no'):
    return {'status': 'passed', 'rows': [{'review_id': 'T02', 'split': 'train',
            'pair_id': 'pair', 'pair_passed': passed, 'answers': {'A': 'yes', 'B': answer},
            'original_targets': {'A': 'yes', 'B': 'no'}, 'evidence': 'original image review'}]}


@pytest.mark.parametrize('passed,answer', [(False, 'no'), (False, None), (True, None)])
def test_pool_pass_cannot_override_false_or_unknown_and_whole_pair_is_excluded(passed, answer):
    view, decisions, qa, result = build_training_view(fixture(), review(passed, answer), [])
    assert result['excluded_pairs'] == 1
    assert not view['splits']['train_all'] and not view['images']
    assert decisions[0]['admitted'] is False
    assert all(not row['admitted'] for row in qa)
    if answer is None:
        assert 'pretraining_sample_review_unknown' in decisions[0]['exclusion_reasons']


def test_passed_high_consensus_preserves_original_rows_images_and_labels():
    source = fixture()
    before = deepcopy(source)
    view, _, qa, result = build_training_view(source, review(), [])
    assert source == before
    assert view['splits']['train_all'] == source['splits']['dev']
    assert view['images'] == source['images']
    assert all(row['status'] == 'reviewed_original_label_supported' for row in qa)
    assert result['label_edits'] == 0


def test_unreviewed_is_not_claimed_individually_clean_and_low_consensus_is_denied():
    source = fixture()
    _, _, qa, _ = build_training_view(source, {'rows': []}, [])
    assert all(row['status'] == 'official_human_consensus_not_individually_verified' for row in qa)
    source['splits']['dev'][0]['answers'] = ['yes'] * 8 + ['no'] * 2
    source['splits']['dev'][0]['answer_vote_count'] = 8
    view, _, _, result = build_training_view(source, review(), [])
    assert result['excluded_pairs'] == 1 and not view['splits']['train_all']


def test_additional_unknown_vetoes_partner_and_question_rule_applies_without_scores():
    source = fixture()
    extra = [{'question_id': 'negative', 'pair_id': 'pair', 'target': 'no',
              'source_split': 'train', 'classification': 'insufficient_fine_detail'}]
    _, decisions, _, _ = build_training_view(source, review(), extra)
    assert 'additional_training_review_uncertain' in decisions[0]['exclusion_reasons']
    for row in source['splits']['dev']:
        row['question'] = 'Is that the right ball to be using in that situation?'
    view, decisions, _, _ = build_training_view(source, {'rows': []}, [])
    assert not view['splits']['train_all']
    assert 'question_rule:appropriateness_right_ball' in decisions[0]['exclusion_reasons']


@pytest.mark.parametrize('violation', ['final', 'screen', 'source_split', 'extra_image'])
def test_training_only_boundary_rejects_heldout_rows_or_metadata(violation):
    source = fixture()
    if violation in ('final', 'screen'):
        source['splits'][violation] = [deepcopy(source['splits']['dev'][0])]
    elif violation == 'source_split':
        source['splits']['dev'][0]['source_split'] = 'confirm'
    else:
        source['images']['heldout'] = {'path': '/not/read.jpg'}
    with pytest.raises(ValueError):
        extract_training_rows(source)
