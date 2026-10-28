"""Read-only lineage bindings, legacy guard and real frozen-feature preflight."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from clip_retrieval.train import DualAdapter,caption_key,eligible_negatives,tensor_hash,choose_negatives
from clip_retrieval_distill.loss import distill_terms
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import lock_file,now,sha_file,write_json
from .sampler import sample_pool,anchor_stream_hash


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError('Unexpected inherited path:'+str(path))
    else:path.symlink_to(target.resolve(),target_is_directory=target.is_dir())


def prepare(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    feature_root=Path(cfg['feature_root']);root.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path(cfg['baseline_config']).read_text())
    oldlock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(Path(cfg['baseline_config']))!=oldlock['config_sha256']:raise ValueError('Baseline config changed')
    if json.loads((old/'COMPLETION.json').read_text())['status']!='completed':raise ValueError('Baseline not completed')
    for name,digest in oldlock['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Frozen baseline source changed:'+name)
    for name in ['training_seeds','epochs','checkpoint_selection','anchor_images_per_update','extra_negative_images',
                 'trainable_parameters','temperature','learning_rate','weight_decay','gradient_clip_norm','distill_weight']:
        if cfg[name]!=previous[name]:raise ValueError('Baseline training contract differs:'+name)
    records={'baseline_config_sha256':sha_file(Path(cfg['baseline_config'])),
             'baseline_study_lock_sha256':sha_file(old/'STUDY_INPUT_LOCK.json'),
             'baseline_analysis_sha256':sha_file(old/'analysis/ANALYSIS.json'),
             'upstream_source_sha256':sha_file(old/'upstream/SOURCES.json'),
             'feature_input_lock_sha256':sha_file(feature_root/'FEATURE_INPUT_LOCK.json'),
             'feature_markers':{},'checkpoints':{},'reused_cells':{},
             'scope':'three bound historical distill checkpoints,not fresh independent baselines'}
    for split,n in [('train',28000),('dev',1000),('test',1000)]:
        p=feature_root/'features'/f'{split}.pt';marker=json.loads(p.with_suffix('.json').read_text())
        if marker['images']!=n or marker['features_sha256']!=sha_file(p):raise ValueError('Frozen feature cache changed')
        records['feature_markers'][split]=marker
    for name in ['features','DATA_MANIFEST.json','IMAGE_INDEX.json']:link(root/name,feature_root/name)
    for seed in cfg['training_seeds']:
        directory=old/'runs/distill'/f'seed{seed}';result=json.loads((directory/'TRAIN_RESULT.json').read_text())
        if result['checkpoint_sha256']!=sha_file(directory/'final.pt'):raise ValueError('Reused checkpoint changed')
        if result['negative_occurrences_sha256']!=sha_file(directory/'negative_occurrences.npy'):raise ValueError('Baseline exposure changed')
        if result['completed_updates']!=8750 or result['positive_caption_pair_exposures']!=1400000:raise ValueError('Partial baseline')
        records['checkpoints'][str(seed)]={'source':str(directory),'train_result_sha256':sha_file(directory/'TRAIN_RESULT.json'),
            **{k:result[k] for k in ['checkpoint_sha256','initial_state_sha256','sampling_stream_sha256','negative_occurrences_sha256']},
            'expected_anchor_stream_sha256':anchor_stream_hash(seed)}
        link(root/'runs/distill'/f'seed{seed}',directory)
        for split in ['dev','test']:
            key=f'distill_seed{seed}';source=old/'evaluation'/key/split
            cell=json.loads((source/'EVALUATION.json').read_text());input_lock=json.loads((source/'INPUT_LOCK.json').read_text())
            if cell['per_query_sha256']!=sha_file(source/'per_query.npz'):raise ValueError('Bound rankings changed')
            if input_lock['feature_sha256']!=records['feature_markers'][split]['features_sha256']:raise ValueError('Bound gallery differs')
            if input_lock['checkpoint_sha256']!=result['checkpoint_sha256']:raise ValueError('Bound prediction model differs')
            records['reused_cells'][key+'/'+split]={'source':str(source),'evaluation_sha256':sha_file(source/'EVALUATION.json'),
                'per_query_sha256':cell['per_query_sha256'],'input_lock_sha256':sha_file(source/'INPUT_LOCK.json')}
            link(root/'evaluation'/key/split,source)
    lock_file(root/'REUSED_BASELINES.json',records)
    guard=root/'FROZEN_GUARD.json'
    if guard.exists():
        hashes=json.loads(guard.read_text())['protected_sha256']
        if any(sha_file(Path(p))!=v for p,v in hashes.items()):raise ValueError('Protected historical file changed')
    else:
        paths=[]
        for folder in ['src','configs','results']:
            for p in (PROJECT_ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix in ['.py','.json'] and 'clip_retrieval_capped' not in str(p):paths.append(p)
        paths+=list((PROJECT_ROOT/'docs').glob('*PROTOCOL*.md'))
        paths=[p for p in paths if 'CAPPED' not in p.name]
        for p in [PROJECT_ROOT/'docs/RESUME.md',Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf')]:
            if p.exists():paths.append(p)
        write_json(guard,{'created_at':now(),'protected_sha256':{str(p):sha_file(p) for p in sorted(set(paths))},
            'scope':'historical scientific sources/configs/results/protocols and formal resume;mutable project entrances excluded'})
    if (root/'REAL_CPU_PRECHECK.json').exists():return records
    torch.set_num_threads(2);payload=torch.load(root/'features/train.pt',map_location='cpu',weights_only=False)
    n=len(payload['image_ids'])
    if n!=28000 or len(payload['captions'])!=140000:raise ValueError('Require full training pool')
    anchors=np.random.default_rng(42).permutation(n)[:32]
    _,groups=np.unique(payload['image_sha256'],return_inverse=True)
    sets=[{caption_key(c) for c in payload['captions'][i*5:i*5+5]} for i in range(n)]
    inverted=defaultdict(set)
    for i,keys in enumerate(sets):
        for key in keys:inverted[key].add(i)
    eligible=eligible_negatives(anchors,groups,sets,inverted)
    query=F.normalize(payload['text_features'].reshape(n,5,512).mean(1),dim=-1)
    scores=(query[anchors]@payload['image_features'].T).max(0).values.numpy()
    counts=np.zeros(n,dtype=np.int64)
    b,bh,br=sample_pool('mixed_distill',eligible,scores,42,0,counts)
    c,ch,cr=sample_pool('capped_distill',eligible,scores,42,0,counts)
    original=choose_negatives('mixed_hard_negatives',eligible,scores,42,0)
    if not np.array_equal(b,original) or not np.array_equal(b,c):raise ValueError('Original/inactive-cap sampler mismatch')
    forced=counts.copy();forced[bh[:16]]=cfg['hard_exposure_cap_per_image_per_epoch']
    altered,ah,ar=sample_pool('capped_distill',eligible,scores,42,0,forced)
    if len(np.unique(altered))!=64 or any(forced[ah]>=8):raise ValueError('Real-pool cap/budget failed')
    torch.manual_seed(42);model=DualAdapter()
    if tensor_hash(model.state_dict())!=records['checkpoints']['42']['initial_state_sha256']:raise ValueError('Baseline initialization differs')
    supervised,relation=distill_terms(model,payload['image_features'],payload['text_features'],anchors,c,groups,cfg['temperature'])
    loss=supervised+cfg['distill_weight']*relation
    gradients=torch.autograd.grad(loss,tuple(model.parameters()))
    if abs(float(relation))>1e-6 or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Real CPU loss/gradient failed')
    write_json(root/'REAL_CPU_PRECHECK.json',{'status':'passed','created_at':now(),
        'feature_sha256':records['feature_markers']['train']['features_sha256'],
        'scope':'real full-train-cache seed42 step0 CPU diagnostic;forced quota state is a capacity check,not trained exposure or benchmark',
        'anchor_ids':anchors.tolist(),'first_anchor_image_id':payload['image_ids'][int(anchors[0])],
        'captions':payload['captions'][int(anchors[0])*5:int(anchors[0])*5+5],
        'i2t_shape':[32,480],'t2i_shape':[160,96],'eligible_images':len(eligible),
        'original_pool':original.tolist(),'uncapped_hard':bh.tolist(),'capped_inactive_exact':True,
        'forced_at_cap':bh[:16].tolist(),'forced_capped_hard':ah.tolist(),
        'original_hard_scores':scores[bh].tolist(),'forced_capped_scores':scores[ah].tolist(),
        'supervised_loss':float(supervised.detach()),'identity_teacher_kl':float(relation.detach()),
        'finite_all_parameter_gradients':True,'all64_unique_and_eligible':True})
    return records
