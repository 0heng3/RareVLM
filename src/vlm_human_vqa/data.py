"""Keep actual human questions and their answers; never infer object absence."""
from collections import Counter

from vlm_deep.common import NORMALIZER


def human_row(question, annotation, image_id, image, source_split):
    if question['question_id'] != annotation['question_id'] or question['image_id'] != annotation['image_id']:
        raise ValueError('Human question/annotation identity mismatch')
    if int(image['coco_image_id']) != int(annotation['image_id']):
        raise ValueError('Human annotation/image identity mismatch')
    answers = annotation['answers']
    if len(answers) != 10 or {answer['answer_id'] for answer in answers} != set(range(1, 11)):
        raise ValueError('Expected ten distinct human answer IDs')
    if annotation['answer_type'] != 'yes/no':
        return None
    votes = Counter(NORMALIZER(answer['answer']) for answer in answers)
    target = next((answer for answer in ('yes', 'no') if votes[answer] >= 9), None)
    if target is None:
        return None
    if NORMALIZER(annotation['multiple_choice_answer']) != target:
        raise ValueError('Human majority disagrees with released multiple-choice answer')
    if not question['question'].strip():
        raise ValueError('Empty human question')
    return {'question_id': f'vqa2_train2014:{question["question_id"]}',
            'image_id': image_id, 'question': question['question'],
            'target': target, 'answers': [answer['answer'] for answer in answers],
            'prompt_suffix': '\nAnswer yes or no.',
            'task': 'human_visual_yesno', 'source_split': source_split,
            'source_question_id': question['question_id'], 'question_type': annotation['question_type'],
            'coco_image_id': image['coco_image_id'], 'answer_vote_count': votes[target],
            'label_source': 'official_vqa_v2_train2014_human_answers',
            'negative_rule': 'human_answer_to_original_question; not object_annotation_absence'}


def counts(rows):
    return {'qa': len(rows), 'images': len({row['image_id'] for row in rows}),
            'labels': dict(Counter(row['target'] for row in rows)),
            'votes': dict(Counter(row['answer_vote_count'] for row in rows)),
            'question_types': dict(sorted(Counter(row['question_type'] for row in rows).items())),
            'existence_prefix_qa': sum(row['question'].lower().startswith(('is there ', 'are there ')) for row in rows)}
