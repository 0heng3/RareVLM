"""Train-only batch-max mining, with an explicit per-image/epoch hard quota."""
import numpy as np
from clip_retrieval.train import choose_negatives


def sample_pool(condition, eligible, scores, seed, step, epoch_hard_counts,
                total=64, hard=32, cap=8):
    eligible = np.asarray(eligible, dtype=np.int64)
    if total <= 0 or not 0 <= hard <= total or cap <= 0:
        raise ValueError('Invalid sampling budget/cap')
    if len(eligible) < total or len(np.unique(eligible)) != len(eligible):
        raise ValueError('Need a unique eligible pool covering the full budget')
    if np.any(eligible < 0) or np.any(eligible >= len(epoch_hard_counts)):
        raise ValueError('Eligible index outside training pool')
    if not np.isfinite(scores[eligible]).all() or np.any(epoch_hard_counts < 0):
        raise ValueError('Nonfinite mining scores or invalid counts')
    if condition == 'mixed_distill':
        pool = choose_negatives('mixed_hard_negatives', eligible, scores,
                                seed, step, total, hard)
        return pool, pool[:hard], pool[hard:]
    if condition != 'capped_distill':
        raise ValueError('Unregistered negative policy')
    candidates = eligible[epoch_hard_counts[eligible] < cap]
    order = np.lexsort((candidates, -scores[candidates]))
    mined = candidates[order[:hard]]
    remaining = eligible[~np.isin(eligible, mined)]
    rng = np.random.default_rng(seed * 1000003 + step + 7919)
    random = rng.choice(remaining, total - len(mined), replace=False)
    return np.concatenate([mined, random]), mined, random


def anchor_stream_hash(seed, images=28000, batch=32, epochs=10):
    import hashlib
    digest = bytes(32)
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch * 1000003).permutation(images)
        for start in range(0, images, batch):
            digest = hashlib.sha256(digest + order[start:start+batch].astype('<i8').tobytes()).digest()
    return digest.hex()
