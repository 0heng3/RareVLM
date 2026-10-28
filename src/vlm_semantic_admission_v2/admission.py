"""Disallow an empty training pool and contradictory same-image yes/no pairs.

Historical v1 sources/artifacts remain immutable. No prompt or label is changed.
"""

from copy import deepcopy
from pathlib import Path

from vlm_semantic_admission_v1 import POLICY as BASE_POLICY
from vlm_semantic_admission_v1 import build_training_view as build_v1


POLICY = {**deepcopy(BASE_POLICY), 'boundary_revision': 2,
          'empty_pool_training_allowed': False,
          'opposite_targets_require_distinct_visual_inputs': True}


def build_training_view(manifest, prior_review, additional_review, policy=POLICY):
    view, decisions, qa_admissions, summary = build_v1(
        manifest, prior_review, additional_review, policy)
    rows = view['splits']['train_all']
    lookup = {row['question_id']: row for row in rows}
    for pair in view['complementary_pairs']['train']:
        yes, no = [lookup[pair[key]] for key in ('positive_question_id', 'negative_question_id')]
        a, b = [view['images'][row['image_id']] for row in (yes, no)]
        equal_fields = [key for key in ('sha256', 'pixel_sha256')
                        if a.get(key) and b.get(key) and a[key] == b[key]]
        same_path = Path(a['path']).resolve() == Path(b['path']).resolve()
        if yes['image_id'] == no['image_id'] or same_path or equal_fields:
            raise ValueError(f'Opposite targets share the same visual input: {pair["pair_id"]}')
    allowed = bool(rows)
    view['training_allowed'] = allowed
    view['semantic_admission']['boundary_revision'] = 2
    summary.update(training_allowed=allowed, boundary_revision=2,
                   admitted_pairs_visual_identity_checked=len(view['complementary_pairs']['train']))
    return view, decisions, qa_admissions, summary
