"""Bind immutable inputs; construct train-only policies before dev evaluation."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import lock_file,now,sha_file,write_json
from .selection import select_slots,reassign_slots


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError('Unexpected linked input')
    else:path.symlink_to(target.resolve(),target_is_directory=target.is_dir())


def verify_guard(root):
    guard=json.loads((root/'FROZEN_GUARD.json').read_text())
    for path,digest in guard['protected_sha256'].items():
        if sha_file(Path(path))!=digest:raise ValueError('Historical file changed:'+path)
    return len(guard['protected_sha256'])


def prepare(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    source=Path(cfg['feature_root']);root.mkdir(parents=True,exist_ok=True)
    oldlock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    for name,digest in oldlock['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Baseline source changed:'+name)
    if json.loads((old/'COMPLETION.json').read_text())['status']!='completed':raise ValueError('Baseline incomplete')
    records={'baseline_lock_sha256':sha_file(old/'STUDY_INPUT_LOCK.json'),'feature_markers':{},'reference_cells':{}}
    for split,n in [('train',28000),('dev',1000),('test',1000)]:
        p=source/'features'/f'{split}.pt';m=json.loads(p.with_suffix('.json').read_text())
        if m['images']!=n or m['features_sha256']!=sha_file(p):raise ValueError('Feature contract differs')
        records['feature_markers'][split]=m
    for name in ['features','DATA_MANIFEST.json','IMAGE_INDEX.json']:link(root/name,source/name)
    for seed in cfg['training_seeds']:
        run=old/'runs/distill'/f'seed{seed}';r=json.loads((run/'TRAIN_RESULT.json').read_text())
        if r['checkpoint_sha256']!=sha_file(run/'final.pt'):raise ValueError('Reference weight changed')
        records.setdefault('checkpoints',{})[str(seed)]={'path':str(run),'result_sha256':sha_file(run/'TRAIN_RESULT.json')}
        for split in ['dev','test']:
            p=old/'evaluation'/f'distill_seed{seed}'/split;m=json.loads((p/'EVALUATION.json').read_text())
            if m['per_query_sha256']!=sha_file(p/'per_query.npz'):raise ValueError('Reference predictions changed')
            records['reference_cells'][f'{seed}/{split}']={'source':str(p),'result_sha256':sha_file(p/'EVALUATION.json'),
                'per_query_sha256':m['per_query_sha256']}
    lock_file(root/'REUSED_BASELINES.json',records)
    guard=root/'FROZEN_GUARD.json'
    if not guard.exists():
        paths=[p for folder in ['src','configs','results'] for p in (PROJECT_ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ['.py','.json'] and 'clip_retrieval_quality' not in str(p)]
        paths += [p for p in (PROJECT_ROOT/'docs').glob('*PROTOCOL*.md') if 'QUALITY' not in p.name]
        paths += [p for p in [PROJECT_ROOT/'docs/RESUME.md',
            Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf')] if p.exists()]
        write_json(guard,{'created_at':now(),'protected_sha256':{str(p):sha_file(p) for p in sorted(set(paths))}})
    verify_guard(root)
    policy=root/'selection/POLICY_LOCK.json'
    if policy.exists():
        identity=json.loads(policy.read_text())
        if identity['config_sha256']!=sha_file(config_path) or identity['selection_npz_sha256']!=sha_file(root/'selection/indices.npz'):
            raise ValueError('Selection policy changed')
        return records
    start=time.monotonic();torch.set_num_threads(2)
    payload=torch.load(root/'features/train.pt',map_location='cpu',weights_only=False)
    n=len(payload['image_ids']);clean=np.arange(n*5).reshape(n,5)
    noisy=reassign_slots(payload['image_sha256'],cfg['noise_seed']);image=payload['image_features'];text=payload['text_features']
    if n!=28000 or len(text)!=140000:raise ValueError('Full train-only cache required')
    arrays={'clean_actual':clean,'noisy_actual':noisy};diagnostics={}
    for branch,actual in [('clean',clean),('noisy',noisy)]:
        score=(image[:,None,:]*text[torch.as_tensor(actual)]).sum(-1).numpy()
        arrays[branch+'_scores']=score
        for method in ['random','score']:
            condition=branch+'_'+method+'4'
            slots=select_slots(score,payload['image_sha256'],method,cfg['selection_seed'])
            chosen=np.take_along_axis(actual,slots,axis=1)
            arrays[condition]=chosen;arrays[condition+'_slots']=slots
            # Known reassignment flag is derived AFTER selection, never supplied to selector/loss.
            altered=actual!=clean;selected_flag=np.take_along_axis(altered,slots,axis=1)
            kept=np.take_along_axis(score,slots,axis=1)
            mask=np.ones((n,5),dtype=bool);np.put_along_axis(mask,slots,False,axis=1)
            diagnostics[condition]={'images':n,'retained_captions':chosen.size,
                'retained_reassigned_pairs':int(selected_flag.sum()),
                'retained_reassigned_pair_fraction':float(selected_flag.mean()),
                'removed_reassigned_pairs':int(altered.sum()-selected_flag.sum()),
                'mean_kept_cosine':float(kept.mean()),'mean_removed_cosine':float(score[mask].mean()),
                'scope':'reassignment provenance diagnostic,not verified semantic incorrectness'}
    directory=root/'selection';directory.mkdir(exist_ok=True)
    np.savez(directory/'indices.npz',**arrays)
    write_json(directory/'DIAGNOSTICS.json',{'created_at':now(),'original_pairs':n*5,'reassigned_pairs':n,
        'conditions':diagnostics,'train_only_selection_seconds':time.monotonic()-start,
        'no_eval_features_read_for_selection':True,'noise_flags_not_selector_input':True})
    lock_file(policy,{'config_sha256':sha_file(config_path),'feature_sha256':records['feature_markers']['train']['features_sha256'],
        'selection_source_sha256':sha_file(SOURCE_ROOT/'clip_retrieval_quality/selection.py'),
        'selection_npz_sha256':sha_file(directory/'indices.npz'),'selection_seed':cfg['selection_seed'],'noise_seed':cfg['noise_seed'],
        'selector_input':'train-only actual frozen image/text features,source image SHA,slots;no corruption flags/groundtruth targets',
        'fitted_dev_parameters':0})
    return records
