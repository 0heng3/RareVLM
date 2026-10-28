import pytest

from vlm_human_vqa.data import human_row


def source(no_votes=9):
    question = {'question_id': 123, 'image_id': 7, 'question': 'Is the car red?'}
    annotation = {'question_id': 123, 'image_id': 7, 'answer_type': 'yes/no',
                  'question_type': 'is the', 'multiple_choice_answer': 'no',
                  'answers': [{'answer_id': i + 1, 'answer': 'no' if i < no_votes else 'yes'} for i in range(10)]}
    return question, annotation, {'coco_image_id': '000000000007'}


def test_human_negative_preserves_question_instead_of_object_absence():
    question, annotation, image = source()
    row = human_row(question, annotation, 'coco2017:000000000007', image, 'train')
    assert row['question'] == 'Is the car red?' and row['target'] == 'no'
    assert row['answer_vote_count'] == 9 and len(row['answers']) == 10
    assert 'presence' not in row and 'category_id' not in row


def test_disagreement_and_identity_mismatches_are_not_silent_labels():
    question, annotation, image = source(no_votes=8)
    assert human_row(question, annotation, 'coco2017:7', image, 'train') is None
    annotation['image_id'] = 8
    with pytest.raises(ValueError, match='identity mismatch'):
        human_row(question, annotation, 'coco2017:7', image, 'train')
