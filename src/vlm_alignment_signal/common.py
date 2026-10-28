import numpy as np
from vlm_probe.evaluate import write_json as original_write_json

def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    return x

def write_json(path,value):original_write_json(path,native(value))

def group_advantages(rewards):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or len(r)!=4 or not np.isfinite(r).all():raise ValueError('Require four finite actual rewards')
    std=float(r.std())
    return np.zeros(4) if std==0 else (r-r.mean())/(std+1e-8)

def preference(responses,scores,normalizer,chosen_min=.9,rejected_max=0.):
    good=[i for i,(a,s) in enumerate(zip(responses,scores)) if a.strip() and s>=chosen_min]
    bad=[i for i,(a,s) in enumerate(zip(responses,scores)) if a.strip() and s<=rejected_max]
    good.sort(key=lambda i:(-scores[i],i));bad.sort(key=lambda i:(scores[i],i))
    for c in good:
        for r in bad:
            if normalizer(responses[c])!=normalizer(responses[r]):return {'chosen_index':c,'rejected_index':r}
    return None
