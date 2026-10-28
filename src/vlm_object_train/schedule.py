"""Input-only matched object order and complete-pool deterministic replay."""
from __future__ import annotations

import hashlib

import torch


def replay_stream(rows, count, seed):
    """Hash-shuffle the complete pool; finish each permutation before cycling.

    Sampling reads question IDs only. Targets, scores, dev errors and answers
    cannot change replay selection. A prefix usually consumes only part of the
    original train_all pool, which must be reported as partial coverage.
    """
    if not rows or type(count) is not int or count < 0:
        raise ValueError('invalid replay pool/request')
    keys = [str(row['question_id']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate replay question IDs')
    result, cycle = [], 0
    while len(result) < count:
        def key(index):
            payload = f'object-replay-v1|{seed}|{cycle}|{keys[index]}'.encode()
            return hashlib.sha256(payload).hexdigest(), keys[index]
        order = sorted(range(len(rows)), key=key)
        result.extend(order[:count - len(result)])
        cycle += 1
    return result


def build_schedule(object_rows, replay_rows, seed=42, epochs=1, accumulation=16, replay_per_update=8):
    """Condition-independent object blocks; replay is consumed by the B arm only."""
    if not object_rows or min(epochs, accumulation, replay_per_update) <= 0:
        raise ValueError('empty object schedule')
    blocks = []
    for epoch in range(epochs):
        order = torch.randperm(len(object_rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        blocks.extend({'epoch': epoch, 'object_indices': order[start:start + accumulation]}
                      for start in range(0, len(order), accumulation))
    replay = replay_stream(replay_rows, len(blocks) * replay_per_update, seed)
    for step, block in enumerate(blocks):
        block['replay_indices'] = replay[step * replay_per_update:(step + 1) * replay_per_update]
    return blocks


def actual_indices(block, condition):
    if condition not in ('object_only', 'object_replay'):
        raise ValueError('unknown supervised object training condition')
    return block['object_indices'], block['replay_indices'] if condition == 'object_replay' else []


def coverage(rows, consumed_indices):
    unique = set(consumed_indices)
    return {'pool_questions': len(rows), 'actual_exposures': len(consumed_indices),
            'unique_questions_consumed': len(unique),
            'unique_images_consumed': len({str(rows[index]['image_id']) for index in unique}),
            'question_coverage_fraction': len(unique) / len(rows),
            'complete_pool_consumed': len(unique) == len(rows),
            'full_cycles_completed': len(consumed_indices) // len(rows)}
