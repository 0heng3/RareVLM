"""Routing has no targets: confidence margin and the frozen threshold only."""
import hashlib
import numpy as np
from clip_retrieval_rerank.common import pair_indices,reorder,retrieve_pair_scores


def image_partition(rows,calibration_images=500,salt='rarevlm-adaptive-v1:'):
    sha=[r['image_sha256'] for r in rows]
    if len(rows)!=1000 or len(set(sha))!=1000:raise ValueError('Need1000 unique source SHA groups')
    order=sorted(range(len(rows)),key=lambda i:hashlib.sha256((salt+sha[i]).encode()).hexdigest())
    cal=np.zeros(len(rows),dtype=bool);cal[order[:calibration_images]]=True
    return cal,~cal


def route_mask(margin,threshold):
    margin=np.asarray(margin)
    if not np.isfinite(margin).all() or not np.isfinite(threshold):raise ValueError('Nonfinite routing input')
    return margin<=threshold


def selected_keys(pools,masks):
    parts=[]
    for direction,pool in pools.items():
        mask=masks[direction]
        if mask.shape!=(len(pool),) or mask.dtype!=bool:raise ValueError('Invalid query route mask')
        parts.append(pair_indices(pool,direction).reshape(pool.shape)[mask].reshape(-1))
    return np.unique(np.concatenate(parts))


def routed_ranking(pool,direction,mask,keys,scores):
    result=pool.copy()
    if mask.any():
        requests=pair_indices(pool,direction).reshape(pool.shape)[mask]
        values=retrieve_pair_scores(requests,keys,scores)
        result[mask]=reorder(pool[mask],values)
    return result
