"""Matched multi-positive contrastive residual adapters on complete frozen features."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import unicodedata
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from methods import ResidualAdapter
from vlm_probe.evaluate import now,write_json,sha_file,content_sha,lock_file,package_versions
from vlm_joint.evaluate import visible_gpu_binding
from .data import DEFAULT_CONFIG


class DualAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.image=ResidualAdapter(512,64);self.text=ResidualAdapter(512,64)
    def image_features(self,x):return F.normalize(self.image(x),dim=-1)
    def text_features(self,x):return F.normalize(self.text(x),dim=-1)


def multi_positive_ce(logits,positive):
    counts=positive.sum(dim=1)
    if not bool((counts>0).all()):raise ValueError('query without positives')
    targets=positive.to(logits.dtype)/counts[:,None]
    return -(targets*F.log_softmax(logits,dim=1)).sum(dim=1)


def contrastive_loss(model,images,texts,anchors,pool,groups,temperature):
    n=len(anchors);image_ids=np.concatenate([anchors,pool])
    image_indices=torch.as_tensor(image_ids,device=images.device)
    caption_indices=(image_indices[:,None]*5+torch.arange(5,device=images.device)[None,:]).reshape(-1)
    zi=model.image_features(images[image_indices]);zt=model.text_features(texts[caption_indices])
    group=groups[image_ids];caption_group=np.repeat(group,5)
    pos_image=torch.as_tensor(group[:n,None]==caption_group[None,:],device=images.device)
    pos_text=torch.as_tensor(caption_group[:n*5,None]==group[None,:],device=images.device)
    image_loss=multi_positive_ce(zi[:n]@zt.T/temperature,pos_image).mean()
    text_loss=multi_positive_ce(zt[:n*5]@zi.T/temperature,pos_text).reshape(n,5).mean(dim=1).mean()
    return (image_loss+text_loss)/2


def caption_key(value):return ' '.join(unicodedata.normalize('NFC',value).split()).casefold()


def eligible_negatives(anchors,groups,caption_sets,inverted):
    excluded=set(map(int,anchors));anchor_groups=set(groups[anchors])
    excluded.update(i for i,g in enumerate(groups) if g in anchor_groups)
    for a in anchors:
        for key in caption_sets[int(a)]:excluded.update(inverted[key])
    return np.array([i for i in range(len(groups)) if i not in excluded],dtype=np.int64)


def choose_negatives(condition,eligible,hard_scores,seed,step,total=64,hard=32):
    if len(eligible)<total:raise ValueError('insufficient eligible negatives,no budget fallback')
    rng=np.random.default_rng(seed*1000003+step+7919)
    if condition=='uniform_negatives':return rng.choice(eligible,total,replace=False)
    if condition!='mixed_hard_negatives':raise ValueError('unknown sampling strategy')
    # Stable tie break on the fixed image index; no evaluation data can enter.
    order=np.lexsort((eligible,-hard_scores[eligible]));mined=eligible[order[:hard]]
    remaining=eligible[~np.isin(eligible,mined)]
    return np.concatenate([mined,rng.choice(remaining,total-hard,replace=False)])


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(config_path,condition,seed):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);path=root/'features/train.pt'
    marker=json.loads(path.with_suffix('.json').read_text())
    if marker['features_sha256']!=sha_file(path):raise ValueError('training features changed')
    payload=torch.load(path,map_location='cpu',weights_only=False);n=len(payload['image_ids'])
    if n<10000 or len(payload['captions'])!=n*5:raise ValueError('complete training features required')
    torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=DualAdapter().to(device)
    if sum(p.numel() for p in model.parameters())!=cfg['trainable_parameters']:raise ValueError('adapter parameter count mismatch')
    initial_hash=tensor_hash(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    directory=root/'runs'/condition/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    from rarevlm_paths import SOURCE_ROOT
    lock={'config_sha256':sha_file(config_path),'feature_sha256':marker['features_sha256'],
        'source_sha256':{name:sha_file(SOURCE_ROOT/name) for name in ['clip_retrieval/train.py','methods.py']},
        'condition':condition,'seed':seed,'initial_state_sha256':initial_hash,
        'hardware':visible_gpu_binding(),'packages':package_versions(),'train_images':n,
        'positive_caption_pairs_per_epoch':n*5,'trainable_parameters':132480,'checkpoint':'Final@10'}
    lock_file(directory/'INPUT_LOCK.json',lock)
    if (directory/'TRAIN_RESULT.json').exists():
        previous=json.loads((directory/'TRAIN_RESULT.json').read_text())
        if previous['checkpoint_sha256']!=sha_file(directory/'final.pt'):raise ValueError('completed adapter changed')
        return previous
    images=payload['image_features'].to(device);texts=payload['text_features'].to(device)
    query=F.normalize(texts.reshape(n,5,512).mean(dim=1),dim=-1)
    _,groups=np.unique(payload['image_sha256'],return_inverse=True)
    caption_sets=[{caption_key(c) for c in payload['captions'][i*5:(i+1)*5]} for i in range(n)]
    inverted=defaultdict(set)
    for i,keys in enumerate(caption_sets):
        for key in keys:inverted[key].add(i)
    batch=cfg['anchor_images_per_update'];updates=(n+batch-1)//batch;planned=updates*cfg['epochs']
    start=0;active_previous=0.;resume=directory/'checkpoint.pt';negative_occurrences=np.zeros(n,dtype=np.int64)
    if resume.exists():
        checkpoint=torch.load(resume,map_location=device,weights_only=False)
        if checkpoint['input_lock_sha256']!=content_sha(lock):raise ValueError('resume lock differs')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);start=checkpoint['step'];active_previous=checkpoint['active_seconds']
        negative_occurrences=checkpoint['negative_occurrences']
    started=time.monotonic();positive_exposure=0
    for step in range(start,planned):
        epoch,within=divmod(step,updates)
        permutation=np.random.default_rng(seed+epoch*1000003).permutation(n)
        anchors=permutation[within*batch:(within+1)*batch]
        eligible=eligible_negatives(anchors,groups,caption_sets,inverted)
        with torch.no_grad():scores=(query[torch.as_tensor(anchors,device=device)]@images.T).max(dim=0).values.cpu().numpy()
        negatives=choose_negatives(condition,eligible,scores,seed,step,cfg['extra_negative_images'],cfg['hard_negative_images'])
        negative_occurrences[negatives]+=1
        optimizer.zero_grad(set_to_none=True)
        loss=contrastive_loss(model,images,texts,anchors,negatives,groups,cfg['temperature'])
        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite contrastive loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise ValueError('nonfinite gradients')
        optimizer.step();positive_exposure+=len(anchors)*5
        if (step+1)%50==0 or step+1==planned:
            elapsed=active_previous+time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'completed_updates':step+1,'planned_updates':planned,
                'epoch':epoch+1,'loss':float(loss),'active_seconds':elapsed,'eta_seconds':(planned-step-1)*elapsed/(step+1),
                'positive_caption_exposure_completed_epochs':epoch*n*5,'last_anchor_images':len(anchors),'negative_images':len(negatives)})
            print('TRAINING',condition,seed,step+1,'/',planned,float(loss),flush=True)
        if (step+1)%250==0 or step+1==planned:
            value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'input_lock_sha256':content_sha(lock),
                'active_seconds':active_previous+time.monotonic()-started,'negative_occurrences':negative_occurrences}
            tmp=resume.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(resume)
    final=directory/'final.pt';torch.save(model.state_dict(),final);np.save(directory/'negative_occurrences.npy',negative_occurrences)
    result={'status':'completed','completed_at':now(),'condition':condition,'seed':seed,'checkpoint_sha256':sha_file(final),
        'initial_state_sha256':initial_hash,'input_lock_sha256':content_sha(lock),'completed_updates':planned,
        'positive_caption_pair_exposures':n*5*cfg['epochs'],'train_images':n,'epochs':cfg['epochs'],'trainable_parameters':132480,
        'active_seconds':active_previous+time.monotonic()-started,'negative_image_slots':64,
        'negative_image_exposures':int(negative_occurrences.sum()),'unique_negative_images':int((negative_occurrences>0).sum()),
        'max_negative_image_exposure':int(negative_occurrences.max()),
        'scope':'only two residual adapters trained;encoder frozen via cached features;negative identities intentionally differ'}
    write_json(directory/'TRAIN_RESULT.json',result);write_json(directory/'PROGRESS.json',{'status':'completed','completed_updates':planned,'planned_updates':planned})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--condition',choices=['uniform_negatives','mixed_hard_negatives'],required=True)
    parser.add_argument('--seed',type=int,choices=[0,42,200],required=True);args=parser.parse_args()
    train(args.config,args.condition,args.seed)
