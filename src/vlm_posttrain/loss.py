"""Frozen assistant-mask/sum-logp implementation and the two paired objectives."""
from __future__ import annotations

import torch

from vlm_alignment_signal.loss import completion_logps, dpo_loss, sequence_logps


def pair_objective(condition, logps, counts, reference=None, beta=.1):
    """SFT averages chosen answer tokens; DPO uses unnormalised sequence sums."""
    if logps.ndim != 1 or counts.shape != logps.shape or not bool((counts > 0).all()):
        raise ValueError('invalid assistant log probabilities/counts')
    if condition == 'sft':
        if logps.numel() != 1:
            raise ValueError('SFT requires exactly the chosen completion')
        return -logps[0] / counts[0]
    if condition != 'dpo' or logps.numel() != 2 or reference is None:
        raise ValueError('DPO requires chosen/rejected and a static reference')
    if reference.shape != logps.shape or reference.requires_grad:
        raise ValueError('reference must be a frozen chosen/rejected pair')
    return dpo_loss(logps[:1], logps[1:], reference[:1], reference[1:], beta).sum()


def normalize_pair_loss(loss, actual_pairs):
    """Accumulation uses real pairs, including the incomplete final update."""
    if type(actual_pairs) is not int or actual_pairs <= 0:
        raise ValueError('normalizer must count real pairs in the current update')
    return loss / actual_pairs


class StaticReference:
    """A value-only cache. Lookups return copies, never aliases to actor tensors."""
    def __init__(self, records):
        self._records = {}
        for record in records:
            key = str(record['question_id'])
            values = torch.tensor(record['logps'], dtype=torch.float32)
            counts = torch.tensor(record['counts'], dtype=torch.long)
            if key in self._records or values.shape != (2,) or counts.shape != (2,):
                raise ValueError('invalid/duplicate static reference record')
            if not bool(torch.isfinite(values).all()) or not bool((counts > 0).all()):
                raise ValueError('nonfinite/empty static reference')
            self._records[key] = (record['input_sha256'], values, counts)

    def lookup(self, question_id, input_sha256, device='cpu'):
        key = str(question_id)
        if key not in self._records:
            raise ValueError('missing initial-SFT reference')
        expected, values, counts = self._records[key]
        if expected != input_sha256:
            raise ValueError('static reference input changed')
        return values.clone().to(device), counts.clone().to(device)
