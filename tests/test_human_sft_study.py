from collections import Counter

from vlm_human_sft.evaluate import binary_answer, primary_score, strict_binary
from vlm_human_sft.study import slots, update_plan


def test_binary_capability_is_separate_from_complete_answer_format():
    gold = {'target': 'no'}
    assert binary_answer(' NO, there is no tripod.') == 'no'
    assert primary_score(' NO, there is no tripod.', gold, 'human') == 1
    assert not strict_binary(' NO, there is no tripod.')
    assert strict_binary('No!')
    assert binary_answer('nobody') is None and binary_answer('') is None
    assert primary_score('', gold, 'human') == 0


def test_matched_slots_use_all_human_and_full_text_without_fake_new_control_qa():
    text = [{'question_id': f'text:{i}'} for i in range(10)]
    human = [{'question_id': f'human:{i}'} for i in range(3)]
    candidate = slots(text, human, 'human', 42)
    control = slots(text, human, 'control', 42)
    assert len(candidate) == len(control) == 13
    assert candidate[:10] == control[:10]
    assert Counter(source for _, source in candidate) == {'text': 10, 'human': 3}
    assert Counter(source for _, source in control) == {'text': 10, 'repeat_text': 3}
    assert {row['question_id'] for row, _ in control} == {row['question_id'] for row in text}
    plan = update_plan(13, 4, 42)
    assert sorted(index for block in plan for index in block) == list(range(13))
    assert [len(block) for block in plan] == [4, 4, 4, 1]
    assert update_plan(36878, 16, 42)[-1] and len(update_plan(36878, 16, 42)[-1]) == 14
