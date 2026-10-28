from copy import deepcopy

import pytest
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from vlm_attention_scope_repair_v1.admission import apply_visual_reviews
from vlm_attention_scope_repair_v1.generation import raw_greedy_config, evaluate
from vlm_deep.common import content_sha


def source():
    rows = [{'question_id': q, 'image_id': i, 'complementary_pair_id': 'pair',
             'question': 'Is she wearing a skirt?', 'target': t, 'answers': [t] * 10,
             'answer_vote_count': 10, 'source_split': 'train',
             'label_source': 'official_vqa_v2_train2014_human_answers'}
            for q, i, t in [('yes', 'a', 'yes'), ('no', 'b', 'no')]]
    return {'schema_version': 'rarevlm.vlm_data.v2',
            'splits': {'train_all': rows, 'dev': [], 'final': []},
            'images': {i: {'path': '/opt/data/private/' + i + '.jpg', 'sha256': i}
                       for i in ('a', 'b')}}


def review():
    return {'question_id': 'no', 'pair_id': 'pair', 'target': 'no', 'source_split': 'train',
            'question_sha256': content_sha('Is she wearing a skirt?'), 'image_sha256': 'b',
            'classification': 'insufficient_fine_detail',
            'evidence': 'Lower body is occluded; no is not established by the visible frame.'}


def test_uncertain_visible_evidence_vetoes_high_consensus_and_earlier_support():
    m = source(); old = deepcopy(m)
    supported = {**review(), 'classification': 'label_visually_supported'}
    view, _, _, _ = apply_visual_reviews(m, {'rows': []}, [supported], [review()])
    assert not view['training_allowed'] and not view['splits']['train_all']
    assert m == old  # No relabeling, mutation or singleton partner retained.


@pytest.mark.parametrize('field,value', [('source_split', 'dev'), ('image_sha256', 'other'),
                                       ('question_sha256', 'other'), ('target', 'yes')])
def test_reviews_cannot_be_reused_for_other_inputs(field, value):
    r = review(); r[field] = value
    with pytest.raises(ValueError):
        apply_visual_reviews(source(), {'rows': []}, [], [r])


def test_repetition_penalty_can_change_a_first_answer_despite_greedy_decoding():
    logits = torch.tensor([[0., 10., 9.75]])
    prompt = torch.tensor([[0, 1]])
    penalized = RepetitionPenaltyLogitsProcessor(1.05)(prompt, logits.clone())
    assert logits.argmax(-1).item() == 1 and penalized.argmax(-1).item() == 2
    inherited = GenerationConfig(bos_token_id=0, pad_token_id=0, eos_token_id=[3, 0],
                                 repetition_penalty=1.05, bad_words_ids=[[1]],
                                 do_sample=True, temperature=.000001)
    policy = raw_greedy_config(inherited)
    assert policy.repetition_penalty == 1 and policy.bad_words_ids is None
    assert policy.do_sample is False and policy.eos_token_id == [3, 0]
    assert inherited.repetition_penalty == 1.05 and inherited.bad_words_ids == [[1]]


def test_new_policy_cannot_overwrite_a_historical_prediction_cache(tmp_path):
    from types import SimpleNamespace
    (tmp_path / 'INPUT_LOCK.json').write_text('{}')
    model = SimpleNamespace(generation_config=GenerationConfig())
    with pytest.raises(ValueError, match='historical cache'):
        evaluate(model, None, {'max_new_tokens': 32}, 'human', tmp_path)
