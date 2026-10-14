"""Label-free schedules; no answers, predictions or evaluation scores accepted."""
from __future__ import annotations
from collections import Counter


def make_batches(question_ids, lengths, policy):
    ids = [str(q) for q in question_ids]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('question IDs must be unique and nonempty')
    size = policy['batch_size']
    if not isinstance(size, int) or size <= 0:
        raise ValueError('positive integer batch size required')
    if any(not isinstance(lengths[q], int) or lengths[q] <= 0 for q in ids):
        raise ValueError('positive encoded input lengths required')
    if policy['order'] == 'input_tokens_ascending_numeric_qid_tiebreak':
        ids.sort(key=lambda q: (lengths[q], int(q)))
    elif policy['order'] != 'manifest':
        raise ValueError('unknown order rule')
    batches = [ids[i:i+size] for i in range(0, len(ids), size)]
    if Counter(q for b in batches for q in b) != Counter(question_ids):
        raise ValueError('schedule drops or duplicates questions')
    return batches


def padding_summary(batches, lengths):
    useful = sum(lengths[q] for b in batches for q in b)
    slots = sum(len(b) * max(lengths[q] for q in b) for b in batches)
    return {'batches': len(batches), 'questions': sum(map(len, batches)),
            'useful_input_tokens': useful, 'padded_input_slots': slots,
            'padding_slots': slots-useful, 'padding_fraction': (slots-useful)/slots,
            'scope': 'input padding only; not FLOPs, KV padding or attention map'}
