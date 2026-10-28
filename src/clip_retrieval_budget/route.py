"""Budget decisions never receive target labels or old ITM scores."""
import numpy as np
from clip_retrieval_rerank.common import pair_indices,reorder,retrieve_pair_scores


def budgets_from_margin(margin,fast_threshold,full_cutoff,small=20,full=50):
    margin=np.asarray(margin)
    if not np.isfinite(margin).all() or not np.isfinite([fast_threshold,full_cutoff]).all():raise ValueError('Nonfinite budget input')
    if not 0<small<=full:raise ValueError('Invalid candidate budgets')
    return np.where(margin>fast_threshold,0,np.where(margin<full_cutoff,full,small)).astype(np.int16)


def budget_keys(pools,budgets,small=20):
    parts=[]
    for direction,pool in pools.items():
        budget=budgets[direction]
        if budget.shape!=(len(pool),) or not np.isin(budget,[0,small,pool.shape[1]]).all():raise ValueError('Unregistered row budget')
        matrix=pair_indices(pool,direction).reshape(pool.shape)
        for k in [small,pool.shape[1]]:
            if (budget==k).any():parts.append(matrix[budget==k,:k].reshape(-1))
    return np.unique(np.concatenate(parts)) if parts else np.empty(0,dtype=np.int64)


def budget_ranking(pool,direction,budget,keys,scores,small=20):
    if budget.shape!=(len(pool),) or not np.isin(budget,[0,small,pool.shape[1]]).all():raise ValueError('Invalid row budget')
    result=pool.copy();requests=pair_indices(pool,direction).reshape(pool.shape)
    for k in [small,pool.shape[1]]:
        mask=budget==k
        if mask.any():
            value=retrieve_pair_scores(requests[mask,:k],keys,scores)
            result[mask,:k]=reorder(pool[mask,:k],value)
    # Top20 reorders only that prefix; remaining30 retain CLIP order without ITM.
    return result
