"""Reuse immutable complete caches and record real-query loss diagnostics."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from clip_retrieval.train import caption_key, choose_negatives, eligible_negatives, multi_positive_ce
from vlm_probe.evaluate import lock_file, now, sha_file, write_json
from .loss import positive_set_ce


def prepare(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    root.mkdir(parents=True,exist_ok=True)
    if json.loads((old/'COMPLETION.json').read_text())['status']!='completed':raise ValueError('Baseline incomplete')
    provenance={'baseline_config_sha256':sha_file(Path(cfg['baseline_config'])),
        'baseline_study_lock_sha256':sha_file(old/'STUDY_INPUT_LOCK.json'),
        'feature_input_lock_sha256':sha_file(old/'FEATURE_INPUT_LOCK.json'),
        'manifest_sha256':sha_file(old/'DATA_MANIFEST.json'),'index_sha256':sha_file(old/'IMAGE_INDEX.json'),
        'features':{},'scope':'read-only reuse; no encoder,split,image,prompt or test-label selection changes'}
    for split,images in [('train',28000),('dev',1000),('test',1000)]:
        path=old/'features'/f'{split}.pt';marker=json.loads(path.with_suffix('.json').read_text())
        if marker['images']!=images or marker['captions']!=images*5 or marker['features_sha256']!=sha_file(path):
            raise ValueError('Complete frozen cache contract changed')
        provenance['features'][split]=marker
    for name in ['features','DATA_MANIFEST.json','IMAGE_INDEX.json']:
        link=root/name;target=(old/name).resolve()
        if link.exists() or link.is_symlink():
            if link.resolve()!=target:raise ValueError('Unexpected reuse path')
        else:link.symlink_to(target,target_is_directory=target.is_dir())
    lock_file(root/'REUSED_DATA_LOCK.json',provenance)
    if (root/'CPU_LOSS_PRECHECK.json').exists():return provenance
    torch.set_num_threads(2)
    data=torch.load(root/'features/train.pt',map_location='cpu',weights_only=False)
    n=len(data['image_ids']);_,groups=np.unique(data['image_sha256'],return_inverse=True)
    caption_sets=[{caption_key(c) for c in data['captions'][i*5:(i+1)*5]} for i in range(n)];inverted=defaultdict(set)
    for i,keys in enumerate(caption_sets):
        for key in keys:inverted[key].add(i)
    anchors=np.random.default_rng(42).permutation(n)[:32]
    eligible=eligible_negatives(anchors,groups,caption_sets,inverted)
    queries=F.normalize(data['text_features'].reshape(n,5,512).mean(1),dim=-1)
    scores=(queries[anchors]@data['image_features'].T).max(0).values.numpy()
    pool=choose_negatives('uniform_negatives',eligible,scores,42,0,64,32)
    image_ids=np.concatenate([anchors,pool]);caption_ids=(image_ids[:,None]*5+np.arange(5)[None,:]).reshape(-1)
    logits=data['image_features'][anchors]@data['text_features'][caption_ids].T/cfg['temperature']
    positive=torch.as_tensor(groups[anchors,None]==np.repeat(groups[image_ids],5)[None,:])
    one=logits[:1].double();mask=positive[:1]
    x=one.clone().requires_grad_(True);mean=multi_positive_ce(x,mask)
    mean_grad=torch.autograd.grad(mean.sum(),x)[0]
    x=one.clone().requires_grad_(True);event=positive_set_ce(x,mask)
    event_grad=torch.autograd.grad(event.sum(),x)[0]
    p=one.softmax(1);within=one.masked_fill(~mask,-torch.inf).softmax(1)
    if not torch.allclose(event_grad,p-within,atol=1e-12):raise ValueError('Real-query set gradient failed')
    if not torch.isfinite(mean_grad).all() or not torch.isfinite(event_grad).all():raise ValueError('Nonfinite real-query gradient')
    value={'status':'passed','created_at':now(),'source_feature_sha256':provenance['features']['train']['features_sha256'],
        'sample_rule':'seed42 epoch0 first full 32-anchor batch,step0 unchanged uniform64 sampler',
        'image_ids':image_ids.tolist(),'caption_ids':caption_ids.tolist(),'anchor_ids':anchors.tolist(),
        'image_logits_shape':list(logits.shape),'text_logits_shape':[160,96],
        'first_anchor_id':int(anchors[0]),'first_anchor_image_id':data['image_ids'][anchors[0]],
        'first_anchor_captions':data['captions'][anchors[0]*5:(anchors[0]+1)*5],
        'first_anchor_logits':one[0].tolist(),'first_anchor_probability':p[0].tolist(),
        'first_anchor_positive':mask[0].tolist(),'mean_positive_targets':(mask.double()/mask.sum()).flatten().tolist(),
        'set_positive_effective_targets':within[0].tolist(),'mean_gradient':mean_grad[0].tolist(),'set_gradient':event_grad[0].tolist(),
        'mean_positive_loss':float(mean),'positive_set_loss':float(event),'positive_probability_mass':float(p[mask].sum()),
        'scope':'real cached frozen-CLIP logits/analytical engineering precheck,not a new retrieval benchmark result'}
    write_json(root/'CPU_LOSS_PRECHECK.json',value)
    exposure=[];full=[]
    for pth in sorted((old/'runs').rglob('negative_occurrences.npy')):
        x=np.load(pth);total=int(x.sum());ordered=np.sort(x.astype(np.float64));count=len(x)
        if count!=n or total!=560000:raise ValueError('Historical negative exposure incomplete')
        row={'condition':pth.parent.parent.name,'seed':int(pth.parent.name[4:]),'source_sha256':sha_file(pth),
            'negative_slots_total':total,'images':count,'mean_exposure':float(x.mean()),'max_exposure':int(x.max()),
            'zero_exposure_images':int((x==0).sum()),'top_one_percent_exposure_share':float(ordered[-280:].sum()/total),
            'gini':float(2*np.dot(np.arange(1,count+1),ordered)/(count*total)-(count+1)/count)}
        exposure.append(row);full.append((row['condition'],row['seed'],x))
    write_json(root/'NEGATIVE_EXPOSURE_DIAGNOSTIC.json',{'status':'completed','rows':exposure,
        'scope':'saved v1 sampling counts,not evidence of semantic false negatives or causal degradation'})
    folder=root/'learning';folder.mkdir(exist_ok=True)
    with (folder/'negative_exposure.csv').open('w') as f:
        f.write('condition,seed,train_image_index,negative_occurrences\n')
        for condition,seed,x in full:
            for i,count in enumerate(x):f.write(f'{condition},{seed},{i},{int(count)}\n')
    return provenance
