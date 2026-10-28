"""Bind historical assets read-only and declare the SHA-only dev partition."""
import json
from pathlib import Path
import numpy as np
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import lock_file,sha_file,now
from clip_retrieval_rerank.common import write_json
from .route import image_partition


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError('Unexpected inherited path')
    else:path.symlink_to(target.resolve(),target_is_directory=target.is_dir())


def bind(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    root.mkdir(parents=True,exist_ok=True);oldcfg=json.loads(Path(cfg['rerank_config']).read_text())
    oldlock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    done=json.loads((old/'COMPLETION.json').read_text())
    if done['status']!='completed' or done['analysis_sha256']!=sha_file(old/'analysis/ANALYSIS.json'):raise ValueError('Old full rerank incomplete/changed')
    if oldlock['config_sha256']!=sha_file(Path(cfg['rerank_config'])):raise ValueError('Old configuration changed')
    for name,digest in oldlock['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Historical source changed:'+name)
    for name in ['training_seeds','candidate_k','text_max_tokens','vision_batch_size','pair_batch_size','pair_chunk_size',
                 'pair_scoring_roles','dtype','matching_prefix_token_id','positive_itm_class','model_weight_sha256','model_revision']:
        if cfg[name]!=oldcfg[name]:raise ValueError('Fixed matching contract differs:'+name)
    ready=json.loads((old/'MODEL_READY.json').read_text())
    if ready['revision']!=cfg['model_revision'] or sha_file(Path(cfg['model_root'])/'pytorch_model.bin')!=cfg['model_weight_sha256']:raise ValueError('Pinned model changed')
    for name,meta in ready['files'].items():
        if sha_file(Path(cfg['model_root'])/name)!=meta['sha256']:raise ValueError('Pinned model configuration/tokenizer changed')
    records={'old_study_lock_sha256':sha_file(old/'STUDY_INPUT_LOCK.json'),'old_analysis_sha256':done['analysis_sha256'],
        'old_config_sha256':sha_file(Path(cfg['rerank_config'])),'model_ready_sha256':sha_file(old/'MODEL_READY.json'),
        'features':{},'inputs':{},'candidates':{},'checkpoints':{},'old_full_cells':{},
        'scope':'bound historical3CLIP+1BLIP and complete old rankings;not new model training'}
    for name in ['MODEL_READY.json']:link(root/name,old/name)
    for name in ['DATA_MANIFEST.json','IMAGE_INDEX.json']:link(root/name,Path(cfg['feature_root'])/name)
    for split in ['dev','test']:
        feature=Path(cfg['feature_root'])/'features'/f'{split}.pt';marker=json.loads(feature.with_suffix('.json').read_text())
        if marker['features_sha256']!=sha_file(feature) or marker['images']!=1000:raise ValueError('Feature cache changed')
        records['features'][split]=marker
        prepared=old/'prepared'/split;meta=json.loads((prepared/'INPUTS.json').read_text())
        if meta['pixels_sha256']!=sha_file(prepared/'pixels.pt') or meta['tokens_sha256']!=sha_file(prepared/'tokens.pt'):raise ValueError('Prepared inputs changed')
        records['inputs'][split]=meta;link(root/'prepared'/split,prepared)
        candidate=old/'candidates'/split;meta=json.loads((candidate/'CANDIDATES.json').read_text())
        if any(sha_file(candidate/name)!=digest for name,digest in meta['files'].items()):raise ValueError('Old candidate file changed')
        records['candidates'][split]=meta
        for seed in cfg['training_seeds']:
            cell=old/'evaluation'/f'rerank_seed{seed}'/split;record=json.loads((cell/'EVALUATION.json').read_text())
            if record['per_query_sha256']!=sha_file(cell/'per_query.npz'):raise ValueError('Old full ranking changed')
            records['old_full_cells'][split+'/'+str(seed)]={'per_query_sha256':record['per_query_sha256'],'source':str(cell)}
    for seed in cfg['training_seeds']:
        d=Path(cfg['baseline_root'])/'runs/distill'/f'seed{seed}';result=json.loads((d/'TRAIN_RESULT.json').read_text())
        if sha_file(d/'final.pt')!=result['checkpoint_sha256']:raise ValueError('Adapter changed')
        records['checkpoints'][str(seed)]={'source':str(d),'checkpoint_sha256':result['checkpoint_sha256']}
    lock_file(root/'REUSED_ASSETS.json',records)
    rows=json.loads((root/'DATA_MANIFEST.json').read_text())['splits']['dev']
    cal,confirm=image_partition(rows,cfg['calibration_images'],cfg['partition_salt'])
    partition={'created_by':'SHA-only fixed rule,without answers/scores','salt':cfg['partition_salt'],
        'dev_manifest_sha256':sha_file(root/'DATA_MANIFEST.json'),'cal_image_indices':np.flatnonzero(cal).tolist(),
        'confirm_image_indices':np.flatnonzero(confirm).tolist(),'cal_image_sha256':[rows[i]['image_sha256'] for i in np.flatnonzero(cal)],
        'confirm_image_sha256':[rows[i]['image_sha256'] for i in np.flatnonzero(confirm)],'gallery_images':1000,'gallery_captions':5000}
    lock_file(root/'PARTITION.json',partition)
    guard=root/'FROZEN_GUARD.json'
    if not guard.exists():
        protected=[]
        for folder in ['src','configs','results']:
            protected.extend(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and 'clip_retrieval_adaptive' not in str(p))
        protected.extend(p for p in (PROJECT_ROOT/'docs').glob('*PROTOCOL*.md') if 'ADAPTIVE' not in p.name)
        protected.extend([PROJECT_ROOT/'docs/RESUME.md',Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf'),
            old/'MODEL_READY.json',old/'analysis/ANALYSIS.json',old/'COMPLETION.json'])
        protected.extend(Path(cfg['model_root'])/name for name in ready['files'])
        for split in ['dev','test']:
            protected.extend(Path(cfg['feature_root'])/'features'/f'{split}.{suffix}' for suffix in ['pt','json'])
            protected.extend(old/'prepared'/split/name for name in ['pixels.pt','tokens.pt','INPUTS.json','INPUT_LOCK.json'])
            protected.extend(old/'candidates'/split/name for name in ['CANDIDATES.json','INPUT_LOCK.json',*records['candidates'][split]['files']])
            for seed in cfg['training_seeds']:
                protected.extend(old/'evaluation'/f'rerank_seed{seed}'/split/name for name in ['per_query.npz','EVALUATION.json'])
        for seed in cfg['training_seeds']:
            protected.extend(Path(records['checkpoints'][str(seed)]['source'])/name for name in ['final.pt','TRAIN_RESULT.json','INPUT_LOCK.json'])
        lock_file(guard,{'files':{str(p):sha_file(p) for p in sorted(set(protected)) if p.exists()}})
    verify_guard(root)
    return records


def verify_guard(root):
    hashes=json.loads((root/'FROZEN_GUARD.json').read_text())['files']
    if any(sha_file(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Protected historical source/result/resume changed')
    return {'status':'passed','protected_files':len(hashes)}
