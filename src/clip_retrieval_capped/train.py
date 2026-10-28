"""Full Final@10 trainings; serialize quota state and both exposure types."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import DualAdapter,caption_key,eligible_negatives,tensor_hash
from clip_retrieval_distill.loss import contrastive_distill_loss
from rarevlm_paths import SOURCE_ROOT
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import content_sha,lock_file,now,package_versions,sha_file,write_json
from .sampler import sample_pool


def train(config_path,condition,seed):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root'])
    if condition not in cfg['new_training_conditions'] or seed not in cfg['training_seeds']:raise ValueError('Unregistered training')
    source=root/'features/train.pt';marker=json.loads(source.with_suffix('.json').read_text())
    if sha_file(source)!=marker['features_sha256']:raise ValueError('Frozen training cache changed')
    payload=torch.load(source,map_location='cpu',weights_only=False);n=len(payload['image_ids'])
    if n!=28000 or len(payload['captions'])!=140000:raise ValueError('Complete training pool required')
    torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=DualAdapter().to(device)
    initial=tensor_hash(model.state_dict());parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    bound=json.loads((root/'REUSED_BASELINES.json').read_text())['checkpoints'][str(seed)]
    if parameters!=cfg['trainable_parameters'] or initial!=bound['initial_state_sha256']:raise ValueError('Matched initialization/parameter count failed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    directory=root/'runs'/condition/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    names=['clip_retrieval_capped/train.py','clip_retrieval_capped/sampler.py','clip_retrieval_distill/loss.py','clip_retrieval/train.py','methods.py']
    lock={'config_sha256':sha_file(config_path),'feature_sha256':marker['features_sha256'],
          'source_sha256':{name:sha_file(SOURCE_ROOT/name) for name in names},'condition':condition,'seed':seed,
          'initial_state_sha256':initial,'hardware':visible_gpu_binding(),'packages':package_versions(),
          'train_images':n,'trainable_parameters':parameters,'checkpoint':'Final@10','tf32':False,
          'distill_weight':cfg['distill_weight'],'hard_cap':cfg['hard_exposure_cap_per_image_per_epoch'] if condition=='capped_distill' else None,
          'baseline_binding_sha256':sha_file(root/'REUSED_BASELINES.json')}
    lock_file(directory/'INPUT_LOCK.json',lock)
    if (directory/'TRAIN_RESULT.json').exists():
        previous=json.loads((directory/'TRAIN_RESULT.json').read_text())
        if sha_file(directory/'final.pt')!=previous['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return previous
    images=payload['image_features'].to(device);texts=payload['text_features'].to(device)
    query=F.normalize(texts.reshape(n,5,512).mean(1),dim=-1)
    _,groups=np.unique(payload['image_sha256'],return_inverse=True)
    sets=[{caption_key(c) for c in payload['captions'][i*5:i*5+5]} for i in range(n)]
    inverted=defaultdict(set)
    for i,keys in enumerate(sets):
        for key in keys:inverted[key].add(i)
    batch=cfg['anchor_images_per_update'];updates=(n+batch-1)//batch;planned=updates*cfg['epochs']
    hard_counts=np.zeros((cfg['epochs'],n),dtype=np.int64);random_counts=np.zeros_like(hard_counts)
    stream=bytes(32);anchor_stream=bytes(32);start=0;previous_seconds=0.;shortages=0
    resume=directory/'checkpoint.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['input_lock_sha256']!=content_sha(lock):raise ValueError('Resume provenance/hardware differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step']
        hard_counts=state['hard_counts'];random_counts=state['random_counts'];previous_seconds=state['active_seconds']
        stream=bytes.fromhex(state['sampling_stream_sha256']);anchor_stream=bytes.fromhex(state['anchor_stream_sha256'])
        shortages=state['hard_shortage_updates']
        if int(hard_counts.sum()+random_counts.sum())!=start*64:raise ValueError('Resume exposure state differs')
        if condition=='capped_distill' and hard_counts.max()>8:raise ValueError('Invalid resumed quota state')
    started=time.monotonic();last_epoch=-1
    for step in range(start,planned):
        epoch,within=divmod(step,updates)
        if epoch!=last_epoch:
            permutation=np.random.default_rng(seed+epoch*1000003).permutation(n);last_epoch=epoch
        anchors=permutation[within*batch:within*batch+batch]
        eligible=eligible_negatives(anchors,groups,sets,inverted)
        with torch.no_grad():scores=(query[torch.as_tensor(anchors,device=device)]@images.T).max(0).values.cpu().numpy()
        negatives,mined,random=sample_pool(condition,eligible,scores,seed,step,hard_counts[epoch],
            cfg['extra_negative_images'],cfg['hard_negative_images'],cfg['hard_exposure_cap_per_image_per_epoch'])
        if len(np.unique(negatives))!=64 or not np.isin(negatives,eligible).all():raise ValueError('Invalid negative identities/budget')
        if condition=='capped_distill' and (hard_counts[epoch,mined]>=8).any():raise ValueError('Quota violated before entry')
        if len(mined)<32:shortages+=1
        if step==0 or (condition=='capped_distill' and not (directory/'FIRST_CAP_EVENT.json').exists()):
            raw=eligible[np.lexsort((eligible,-scores[eligible]))[:32]]
            blocked=raw[hard_counts[epoch,raw]>=8]
            name='FIRST_GPU_POOL.json' if step==0 else 'FIRST_CAP_EVENT.json'
            if step==0 or len(blocked):
                write_json(directory/name,{'status':'passed','scope':'actual GPU training pool before optimizer update',
                    'step':step,'epoch':epoch+1,'anchor_ids':anchors.tolist(),'hard_ids':mined.tolist(),'random_ids':random.tolist(),
                    'raw_top32_ids':raw.tolist(),'raw_top32_scores':scores[raw].tolist(),
                    'raw_top32_epoch_hard_counts_before':hard_counts[epoch,raw].tolist(),'blocked_raw_top_ids':blocked.tolist(),
                    'selected_hard_scores':scores[mined].tolist(),'selected_hard_counts_before':hard_counts[epoch,mined].tolist(),
                    'feature_sha256':marker['features_sha256'],'tf32':False})
        hard_counts[epoch,mined]+=1;random_counts[epoch,random]+=1
        stream=hashlib.sha256(stream+anchors.astype('<i8').tobytes()+negatives.astype('<i8').tobytes()).digest()
        anchor_stream=hashlib.sha256(anchor_stream+anchors.astype('<i8').tobytes()).digest()
        optimizer.zero_grad(set_to_none=True)
        loss=contrastive_distill_loss(model,images,texts,anchors,negatives,groups,cfg['temperature'],cfg['distill_weight'])
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite gradient')
        optimizer.step()
        if (step+1)%100==0 or step+1==planned:
            elapsed=previous_seconds+time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'completed_updates':step+1,'planned_updates':planned,
                'epoch':epoch+1,'loss':float(loss),'gradient_norm':float(norm),'active_seconds':elapsed,
                'eta_seconds':(planned-step-1)*elapsed/(step+1),'max_current_epoch_hard':int(hard_counts[epoch].max()),
                'hard_shortage_updates':shortages})
            print('TRAIN',condition,seed,step+1,'/',planned,float(loss),flush=True)
        if (step+1)%250==0 or step+1==planned:
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                'input_lock_sha256':content_sha(lock),'active_seconds':previous_seconds+time.monotonic()-started,
                'hard_counts':hard_counts,'random_counts':random_counts,'hard_shortage_updates':shortages,
                'sampling_stream_sha256':stream.hex(),'anchor_stream_sha256':anchor_stream.hex()}
            tmp=resume.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(resume)
    if anchor_stream.hex()!=bound['expected_anchor_stream_sha256']:raise ValueError('Full anchor order differs')
    if int(hard_counts.sum()+random_counts.sum())!=planned*64:raise ValueError('Incomplete total exposure')
    if condition=='capped_distill' and hard_counts.max()>8:raise ValueError('Final quota violated')
    counts=(hard_counts+random_counts).sum(0)
    torch.save(model.state_dict(),directory/'final.pt')
    for name,array in [('negative_occurrences',counts),('epoch_hard_counts',hard_counts),('epoch_random_counts',random_counts)]:np.save(directory/(name+'.npy'),array)
    top=max(1,int(np.ceil(n*.01)));share=float(np.sort(counts)[-top:].sum()/counts.sum())
    result={'status':'completed','completed_at':now(),'condition':condition,'seed':seed,
        'checkpoint_sha256':sha_file(directory/'final.pt'),'initial_state_sha256':initial,'input_lock_sha256':content_sha(lock),
        'completed_updates':planned,'positive_caption_pair_exposures':n*5*cfg['epochs'],'train_images':n,'epochs':cfg['epochs'],
        'trainable_parameters':parameters,'active_seconds':previous_seconds+time.monotonic()-started,
        'negative_image_slots':64,'negative_image_exposures':int(counts.sum()),'hard_image_exposures':int(hard_counts.sum()),
        'random_image_exposures':int(random_counts.sum()),'hard_shortage_updates':shortages,
        'negative_occurrences_sha256':sha_file(directory/'negative_occurrences.npy'),
        'epoch_hard_counts_sha256':sha_file(directory/'epoch_hard_counts.npy'),
        'epoch_random_counts_sha256':sha_file(directory/'epoch_random_counts.npy'),
        'sampling_stream_sha256':stream.hex(),'anchor_stream_sha256':anchor_stream.hex(),
        'unique_negative_images':int((counts>0).sum()),'max_negative_image_exposure':int(counts.max()),
        'max_hard_image_exposure_in_one_epoch':int(hard_counts.max()),'per_epoch_max_hard':hard_counts.max(1).tolist(),
        'top1pct_total_negative_exposure_share':share,'distill_weight':cfg['distill_weight'],
        'scope':'unchanged meanCE+KL;full train-only mining;hard quota excludes random exposure;not semantic false-negative labels'}
    write_json(directory/'TRAIN_RESULT.json',result)
    write_json(directory/'PROGRESS.json',{'status':'completed','completed_updates':planned,'planned_updates':planned})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();train(args.config,args.condition,args.seed)
