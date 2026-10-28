"""Selection sees only actual features, SHA and slots; never noise labels."""
import hashlib
import numpy as np


def select_slots(scores, image_sha, method, seed, keep=4):
    scores=np.asarray(scores)
    if scores.shape != (len(image_sha),5) or not np.isfinite(scores).all():
        raise ValueError('Require finite five-caption train-only scores')
    if method=='score':
        order=np.argsort(-scores,axis=1,kind='stable')
    elif method=='random':
        order=np.array([sorted(range(5),key=lambda j:hashlib.sha256(
            f'rarevlm-quality-v1:{seed}:{sha}:{j}'.encode()).digest()) for sha in image_sha])
    else:raise ValueError('Unknown registered selector')
    if keep not in [4,5]:raise ValueError('Unsupported caption budget')
    # Restore annotation slot order: loss is permutation invariant, tensor order is reproducible.
    return np.sort(order[:,:keep],axis=1)


def reassign_slots(image_sha, seed):
    """One slot/image is permuted among different image SHA, retaining vocabulary."""
    n=len(image_sha);rng=np.random.default_rng(seed)
    slots=rng.integers(0,5,size=n);order=rng.permutation(n);sha=np.asarray(image_sha)
    for shift in range(1,n):
        donors=np.roll(order,shift)
        if np.all(sha[order]!=sha[donors]):break
    else:raise ValueError('No different-SHA cyclic reassignment')
    indices=np.arange(n*5,dtype=np.int64).reshape(n,5)
    indices[order,slots[order]]=donors*5+slots[donors]
    if len(np.unique(indices))!=n*5:raise ValueError('Caption vocabulary not conserved')
    return indices
