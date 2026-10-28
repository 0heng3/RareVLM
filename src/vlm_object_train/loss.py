"""Per-question assistant CE, with independent object and replay means."""
from __future__ import annotations

import math

import torch

# Provenance: this frozen helper preserves the original assistant/end-token
# mask and tail-head calculation. No DPO objective or reference is used here.
from vlm_alignment_signal.loss import completion_logps, sequence_logps


def per_question_ce(logps, counts):
    if logps.ndim != 1 or logps.numel() == 0 or counts.shape != logps.shape or not bool((counts > 0).all()):
        raise ValueError('expected nonempty per-question assistant logps/counts')
    return -logps / counts


def normalized_qa_loss(ce, source, actual_object_count, actual_replay_count, replay_lambda=1.):
    """One real microbatch's contribution to independent object/replay means.

    The last real object block can contain fewer than sixteen questions. Its
    object denominator is independent of the fixed eight-question replay block.
    """
    if not isinstance(ce, torch.Tensor) or ce.ndim > 1 or ce.numel() == 0:
        raise ValueError('microbatch must contain a nonempty per-question CE vector')
    if type(actual_object_count) is not int or actual_object_count <= 0:
        raise ValueError('object denominator must count real questions')
    if type(actual_replay_count) is not int or actual_replay_count < 0:
        raise ValueError('invalid replay denominator')
    if not math.isfinite(replay_lambda) or replay_lambda <= 0:
        raise ValueError('invalid replay coefficient')
    if source == 'object':
        return ce.sum() / actual_object_count
    if source == 'replay' and actual_replay_count > 0:
        return ce.sum() * replay_lambda / actual_replay_count
    raise ValueError('unknown source or replay absent from this update')
