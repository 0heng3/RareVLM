from copy import deepcopy

import pytest

from vlm_semantic_admission_v2 import build_training_view


def source():
    rows = [{'question_id': q, 'image_id': i, 'complementary_pair_id': 'p',
             'question': 'Is there a cat?', 'target': t, 'answers': [t] * 10,
             'answer_vote_count': 10, 'source_split': 'train',
             'label_source': 'official_vqa_v2_train2014_human_answers'}
            for q, i, t in [('positive', 'a', 'yes'), ('negative', 'b', 'no')]]
    return {'schema_version': 'rarevlm.vlm_data.v2',
            'splits': {'train_all': rows, 'dev': [], 'final': []},
            'images': {'a': {'path': '/opt/data/private/a.jpg', 'sha256': 'a', 'pixel_sha256': 'pa'},
                       'b': {'path': '/opt/data/private/b.jpg', 'sha256': 'b', 'pixel_sha256': 'pb'}}}


def test_all_rejected_pool_is_explicitly_not_training_allowed():
    review = {'rows': [{'split': 'train', 'pair_id': 'p', 'pair_passed': False,
                        'answers': {'A': None, 'B': None},
                        'original_targets': {'A': 'yes', 'B': 'no'}}]}
    view, _, _, summary = build_training_view(source(), review, [])
    assert view['training_allowed'] is False and summary['training_allowed'] is False
    assert not view['splits']['train_all'] and not view['images']


@pytest.mark.parametrize('identity', ['image_id', 'path', 'sha256', 'pixel_sha256'])
def test_opposite_labels_cannot_share_the_same_visual_input(identity):
    m = source()
    if identity == 'image_id':
        m['splits']['train_all'][1]['image_id'] = 'a'
        del m['images']['b']
    else:
        m['images']['b'][identity] = m['images']['a'][identity]
    with pytest.raises(ValueError, match='Opposite targets share the same visual input'):
        build_training_view(m, {'rows': []}, [])


def test_distinct_pair_preserves_original_input_and_does_not_unescape_prompts():
    m = source()
    for q in m['splits']['train_all']:
        q['prompt_suffix'] = '\nAnswer yes or no.'
    before = deepcopy(m)
    view, _, _, summary = build_training_view(m, {'rows': []}, [])
    assert m == before
    assert view['splits']['train_all'] == before['splits']['train_all']
    assert view['images'] == before['images']
    assert view['training_allowed'] is True and summary['training_allowed'] is True
    assert all(ord(q['prompt_suffix'][0]) == 10 for q in view['splits']['train_all'])
