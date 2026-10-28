"""Regressions for the identified loss-precision and label-construction faults."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from vlm_fp32_adapter.loss import answer_loss
from vlm_object_presence_v3.data import repair_rows, summary


class HeadOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 11, bias=False, dtype=torch.bfloat16)

    def get_output_embeddings(self):
        return self.head

    def forward(self, hidden, use_cache=False):
        return SimpleNamespace(logits=self.head(hidden))


def test_no_amp_ce_and_backward_equal_explicit_fp32_reference():
    torch.manual_seed(42)
    model = HeadOnly()
    hidden = torch.randn(2, 7, 3, dtype=torch.bfloat16)
    labels = torch.tensor([[-100]*5 + [2, 7], [-100]*4 + [1, 2, 8]])
    batch = {'hidden': hidden, 'labels': labels}
    actual, _, tokens = answer_loss(model, batch)
    full = model(hidden).logits[:, :-1].float()
    reference = F.cross_entropy(full.reshape(-1, 11), labels[:, 1:].reshape(-1),
                                ignore_index=-100, reduction='none').reshape(2, 6).sum(1) / torch.tensor([2, 3])
    assert actual.dtype == torch.float32 and tokens == 5
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.mean().backward()
    actual_gradient = model.head.weight.grad.clone()
    model.zero_grad(); reference.mean().backward()
    torch.testing.assert_close(actual_gradient, model.head.weight.grad, rtol=0, atol=0)
    # Show that the regression case distinguishes the old BF16 CE calculation.
    old = F.cross_entropy(full.to(torch.bfloat16).reshape(-1, 11), labels[:, 1:].reshape(-1),
                          ignore_index=-100, reduction='none').reshape(2, 6).sum(1) / torch.tensor([2, 3])
    assert not torch.equal(actual, old)


def test_invalid_answer_layout_is_rejected_before_head_slicing():
    model = HeadOnly()
    with pytest.raises(ValueError, match='contiguous'):
        answer_loss(model, {'hidden': torch.ones(1, 4, 3, dtype=torch.bfloat16),
                            'labels': torch.tensor([[-100, 1, -100, 2]])})
    assert not model.head._forward_pre_hooks


def test_unknown_is_not_no_and_small_outside_18_class_positive_is_retained():
    base = {'image_id': 'i', 'pixel_sha256': 'p', 'file_sha256': 'f', 'coco_image_id': '1',
            'question_id': 'object_presence_v1:1:category90', 'category_id': 90, 'category_name': 'toothbrush'}
    rows = [{**base, 'target': 'yes', 'bbox_evidence': [{'coco_area_bin': 'small'}]},
            {**base, 'question_id': 'object_presence_v1:1:category17', 'category_id': 17,
             'category_name': 'cat', 'target': 'no', 'bbox_evidence': []}]
    manifest = {'rows': rows, 'images': {'i': {'path': 'image'}}}
    positive, unknown, images = repair_rows(manifest, [])
    assert positive[0]['small_only'] and positive[0]['category_name'] == 'toothbrush'
    assert unknown[0]['target'] is None and unknown[0]['presence'] is None
    assert not any(row['supervision_eligible'] for row in positive + unknown)
    assert not summary(positive, unknown, images)['training_allowed']
    assert manifest['rows'][1]['target'] == 'no'  # historical input unchanged
    assert repair_rows(manifest, [{**base}]) == ([], [], {})
