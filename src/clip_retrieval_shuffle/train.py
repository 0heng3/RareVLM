"""Three entropy-preserving negative-relation control trainings."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from clip_retrieval.train import (DualAdapter, caption_key, choose_negatives,
    contrastive_loss, eligible_negatives, tensor_hash)
from rarevlm_paths import SOURCE_ROOT
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, sha_file, write_json
from .loss import contrastive_shuffle_loss


def train(config_path, condition, seed):
    cfg = json.loads(config_path.read_text())
    if condition not in cfg['new_training_conditions'] or seed not in cfg['training_seeds']:
        raise ValueError('Unregistered condition or seed')
    root = Path(cfg['output_root']); source = root / 'features/train.pt'
    marker = json.loads(source.with_suffix('.json').read_text())
    if sha_file(source) != marker['features_sha256']:
        raise ValueError('Frozen feature cache changed')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    n = len(payload['image_ids'])
    if n != 28000 or len(payload['captions']) != n * 5:
        raise ValueError('Require the complete frozen train pool')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', 0)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = DualAdapter().to(device)
    initial = tensor_hash(model.state_dict())
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != cfg['trainable_parameters']:
        raise ValueError('Parameter count mismatch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    directory = root / 'runs' / condition / f'seed{seed}'; directory.mkdir(parents=True, exist_ok=True)
    sources = ['clip_retrieval_shuffle/train.py','clip_retrieval_shuffle/loss.py','clip_retrieval/train.py','methods.py']
    lock = {'config_sha256':sha_file(config_path), 'feature_sha256':marker['features_sha256'],
        'source_sha256':{name:sha_file(SOURCE_ROOT/name) for name in sources},
        'condition':condition, 'seed':seed, 'initial_state_sha256':initial,
        'hardware':visible_gpu_binding(), 'packages':package_versions(),
        'train_images':n, 'positive_caption_pairs_per_epoch':n*5,
        'trainable_parameters':parameters, 'checkpoint':'Final@10',
        'tf32':False, 'negative_sampler':'unchanged v1 uniform_negatives', 'distill_weight':cfg['distill_weight'], 'teacher':'normalized frozen CLIP features, no gradient', 'permutation':cfg['permutation']}
    lock_file(directory/'INPUT_LOCK.json',lock)
    if (directory/'TRAIN_RESULT.json').exists():
        result=json.loads((directory/'TRAIN_RESULT.json').read_text())
        if result['checkpoint_sha256'] != sha_file(directory/'final.pt'):
            raise ValueError('Completed checkpoint changed')
        return result
    images = payload['image_features'].to(device); texts = payload['text_features'].to(device)
    query = F.normalize(texts.reshape(n,5,512).mean(dim=1),dim=-1)
    _,groups = np.unique(payload['image_sha256'],return_inverse=True)
    caption_sets = [{caption_key(c) for c in payload['captions'][i*5:(i+1)*5]} for i in range(n)]
    inverted=defaultdict(set)
    for i,keys in enumerate(caption_sets):
        for key in keys: inverted[key].add(i)
    batch=cfg['anchor_images_per_update']; updates=(n+batch-1)//batch; planned=updates*cfg['epochs']
    start=0; elapsed_previous=0.; negative_occurrences=np.zeros(n,dtype=np.int64); sampling_digest=bytes(32)
    resume=directory/'checkpoint.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['input_lock_sha256']!=content_sha(lock):raise ValueError('Resume lock mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];elapsed_previous=state['active_seconds'];negative_occurrences=state['negative_occurrences']
        sampling_digest=bytes.fromhex(state['sampling_stream_sha256'])
    started=time.monotonic()
    objective=contrastive_shuffle_loss
    for step in range(start,planned):
        epoch,within=divmod(step,updates)
        permutation=np.random.default_rng(seed+epoch*1000003).permutation(n)
        anchors=permutation[within*batch:(within+1)*batch]
        eligible=eligible_negatives(anchors,groups,caption_sets,inverted)
        # Retain the v1 computation and RNG contract; uniform sampling ignores these scores.
        with torch.no_grad(): scores=(query[torch.as_tensor(anchors,device=device)]@images.T).max(dim=0).values.cpu().numpy()
        negatives=choose_negatives('uniform_negatives',eligible,scores,seed,step,cfg['extra_negative_images'],32)
        sampling_digest=hashlib.sha256(sampling_digest+anchors.astype('<i8').tobytes()+negatives.astype('<i8').tobytes()).digest()
        negative_occurrences[negatives]+=1
        optimizer.zero_grad(set_to_none=True)
        out=objective(model,images,texts,anchors,negatives,groups,cfg['temperature'],cfg['distill_weight'],permutation_seed=seed,step=step,return_details=(step==0))
        if step==0:
            loss,details=out
            save_first_update(directory,payload,details,seed)
        else:loss=out
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite gradient')
        optimizer.step()
        if (step+1)%100==0 or step+1==planned:
            elapsed=elapsed_previous+time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),
                'completed_updates':step+1,'planned_updates':planned,'epoch':epoch+1,
                'loss':float(loss),'active_seconds':elapsed,'eta_seconds':(planned-step-1)*elapsed/(step+1)})
            print('TRAIN',condition,seed,step+1,'/',planned,float(loss),flush=True)
        if (step+1)%250==0 or step+1==planned:
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                'input_lock_sha256':content_sha(lock),'active_seconds':elapsed_previous+time.monotonic()-started,
                'negative_occurrences':negative_occurrences,'sampling_stream_sha256':sampling_digest.hex()}
            tmp=resume.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(resume)
    torch.save(model.state_dict(),directory/'final.pt');np.save(directory/'negative_occurrences.npy',negative_occurrences)
    result={'status':'completed','completed_at':now(),'condition':condition,'seed':seed,
        'checkpoint_sha256':sha_file(directory/'final.pt'),'initial_state_sha256':initial,
        'input_lock_sha256':content_sha(lock),'completed_updates':planned,
        'positive_caption_pair_exposures':n*5*cfg['epochs'],'train_images':n,'epochs':cfg['epochs'],
        'trainable_parameters':parameters,'active_seconds':elapsed_previous+time.monotonic()-started,
        'negative_image_slots':64,'negative_image_exposures':int(negative_occurrences.sum()),
        'negative_occurrences_sha256':sha_file(directory/'negative_occurrences.npy'),
        'sampling_stream_sha256':sampling_digest.hex(),
        'unique_negative_images':int((negative_occurrences>0).sum()),'max_negative_image_exposure':int(negative_occurrences.max()),
        'distill_weight':cfg['distill_weight'],'permutation':cfg['permutation'],'scope':'negative-only teacher probability bijection; positive targets/entropy fixed; full Final@10'}
    write_json(directory/'TRAIN_RESULT.json',result)
    write_json(directory/'PROGRESS.json',{'status':'completed','completed_updates':planned,'planned_updates':planned})
    return result


def save_first_update(directory,payload,details,seed):
    checks={}
    for direction in ['i','t']:
        before=details['teacher_logits_'+direction].log_softmax(1)
        after=details['target_'+direction];mask=details['positive_'+direction]
        mapping=details['mapping_'+direction]
        columns=torch.arange(mask.shape[1],device=mask.device).expand(mask.shape[0],-1)
        if not torch.equal(mapping.sort(1).values,columns) or not torch.equal(mapping[mask],columns[mask]):
            raise ValueError('CUDA permutation not a positive-fixed bijection')
        if not torch.equal(before[mask],after[mask]) or not torch.equal(before.sort(1).values,after.sort(1).values):
            raise ValueError('CUDA teacher target values changed')
        a=before.double().exp();b=after.double().exp();gt=mask.double()/mask.sum(1)[:,None]
        def entropy(p):return -(p*p.clamp_min(1e-300).log()).sum(1)
        checks[direction]={'positive_fixed':True,'negative_multiset_fixed':True,
            'probability_mass_max_error':float((a.sum(1)-b.sum(1)).abs().max()),
            'teacher_entropy_max_error':float((entropy(a)-entropy(b)).abs().max()),
            'mixed_entropy_max_error':float((entropy((gt+a)/2)-entropy((gt+b)/2)).abs().max())}
        if max(v for k,v in checks[direction].items() if k.endswith('error'))>1e-10:
            raise ValueError('CUDA entropy/mass contract failed')
    original=details['teacher_logits_i'].log_softmax(1)[0].detach().cpu()
    changed=details['target_i'][0].detach().cpu()
    student=details['student_logits_i'][0].log_softmax(0).cpu()
    positive=details['positive_i'][0].cpu()
    anchor=int(details['image_ids'][0]);caps=details['caption_ids'].cpu().tolist()
    write_json(directory/'FIRST_UPDATE_DIAGNOSTIC.json',{'status':'passed','created_at':now(),
        'seed':seed,'step':0,'hardware':visible_gpu_binding(),
        'first_anchor_id':anchor,'first_anchor_image_id':payload['image_ids'][anchor],
        'first_anchor_captions':payload['captions'][anchor*5:(anchor+1)*5],
        'image_ids':details['image_ids'].tolist(),'caption_ids':caps,
        'positive':positive.tolist(),'teacher_original_logp':original.tolist(),
        'teacher_shuffled_logp':changed.tolist(),'student_logp':student.tolist(),
        'negative_permutation':details['mapping_i'][0].cpu().tolist(),
        'actual_cuda_contracts':checks,
        'I2T_shape':list(details['target_i'].shape),'T2I_shape':list(details['target_t'].shape),
        'precision':'actual first CUDA update FP32 logits/targets; before optimizer step',
        'scope':'real training control target,not attention or final trained prediction'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();train(args.config,args.condition,args.seed)
