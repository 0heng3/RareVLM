"""Read-only bindings to the completed teacher study and complete feature cache."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import DualAdapter, tensor_hash
from clip_retrieval_distill.loss import contrastive_distill_loss
from rarevlm_paths import SOURCE_ROOT, PROJECT_ROOT
from vlm_probe.evaluate import lock_file, now, sha_file, write_json
from .loss import contrastive_shuffle_loss, shuffle_terms


def prepare(config_path):
    cfg = json.loads(config_path.read_text()); root = Path(cfg['output_root'])
    old = Path(cfg['baseline_root']); features = Path(cfg['feature_root'])
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads(Path(cfg['baseline_config']).read_text())
    oldlock = json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(Path(cfg['baseline_config'])) != oldlock['config_sha256']:
        raise ValueError('Historical configuration changed')
    for rel, digest in oldlock['sources'].items():
        if sha_file(SOURCE_ROOT/rel) != digest:
            raise ValueError('Historical dependency changed: '+rel)
    if json.loads((old/'COMPLETION.json').read_text())['status'] != 'completed':
        raise ValueError('Historical teacher study incomplete')
    for key in ['training_seeds','epochs','anchor_images_per_update','extra_negative_images',
                'temperature','trainable_parameters','learning_rate','weight_decay',
                'gradient_clip_norm','checkpoint_selection','distill_weight']:
        if cfg[key] != previous[key]:
            raise ValueError('Training protocol mismatch: '+key)
    protected = {str(SOURCE_ROOT/rel): digest for rel, digest in oldlock['sources'].items()}
    records = {'baseline_config_sha256': sha_file(Path(cfg['baseline_config'])),
               'baseline_source_lock_sha256': sha_file(old/'STUDY_INPUT_LOCK.json'),
               'feature_input_lock_sha256': sha_file(features/'FEATURE_INPUT_LOCK.json'),
               'checkpoints': {}, 'reused_cells': {}, 'feature_markers': {},
               'scope': 'Bound historical controls; not fresh independent repetitions'}
    def protect(path):
        protected[str(path)] = sha_file(path)
    def link(destination, source):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError('Unexpected inherited destination')
        else:
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    for path in [Path(cfg['baseline_config']), old/'STUDY_INPUT_LOCK.json',
                 old/'REUSED_BASELINES.json', old/'COMPLETION.json', old/'analysis/ANALYSIS.json',
                 PROJECT_ROOT/'docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md',
                 PROJECT_ROOT/'results/clip_retrieval_distill_v1_metrics.json',
                 PROJECT_ROOT/'docs/RESUME.md',
                 Path('/opt/data/private/longtail_vlm/龚智恒-中国科学技术大学-日常实习-CV.pdf')]:
        protect(path)
    if protected[str(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md')] != oldlock['protocol_sha256']:
        raise ValueError('Historical protocol changed')
    for split, n in [('train',28000),('dev',1000),('test',1000)]:
        path = features/'features'/f'{split}.pt'
        marker = json.loads(path.with_suffix('.json').read_text())
        if marker['images'] != n or sha_file(path) != marker['features_sha256']:
            raise ValueError('Incomplete or modified cache: '+split)
        records['feature_markers'][split] = marker
        protect(path); protect(path.with_suffix('.json'))
    for name in ['features','DATA_MANIFEST.json','IMAGE_INDEX.json']:
        link(root/name, features/name)
    for condition in ['uniform_mean','distill']:
        for seed in cfg['training_seeds']:
            source = old/'runs'/condition/f'seed{seed}'
            result = json.loads((source/'TRAIN_RESULT.json').read_text())
            for filename, key in [('final.pt','checkpoint_sha256'),('negative_occurrences.npy','negative_occurrences_sha256')]:
                if sha_file(source/filename) != result[key]:
                    raise ValueError('Historical training artifact changed')
                protect(source/filename)
            protect(source/'TRAIN_RESULT.json'); protect(source/'INPUT_LOCK.json')
            records['checkpoints'][f'{condition}_seed{seed}'] = {'source':str(source),
                'train_result_sha256':sha_file(source/'TRAIN_RESULT.json'),
                'checkpoint_sha256':result['checkpoint_sha256'],
                'initial_state_sha256':result['initial_state_sha256'],
                'sampling_stream_sha256':result['sampling_stream_sha256'],
                'negative_occurrences_sha256':result['negative_occurrences_sha256']}
            link(root/'runs'/condition/f'seed{seed}', source)
    for split in ['dev','test']:
        for key in ['frozen',*[f'{c}_seed{s}' for c in ['uniform_mean','distill'] for s in cfg['training_seeds']]]:
            source = old/'evaluation'/key/split
            cell = json.loads((source/'EVALUATION.json').read_text())
            inp = json.loads((source/'INPUT_LOCK.json').read_text())
            if cell['per_query_sha256'] != sha_file(source/'per_query.npz'):
                raise ValueError('Historical rankings changed')
            if inp['feature_sha256'] != records['feature_markers'][split]['features_sha256']:
                raise ValueError('Historical ranking cache differs')
            if key != 'frozen' and inp['checkpoint_sha256'] != records['checkpoints'][key]['checkpoint_sha256']:
                raise ValueError('Historical ranking checkpoint differs')
            for name in ['EVALUATION.json','INPUT_LOCK.json','per_query.npz']:
                protect(source/name)
            records['reused_cells'][f'{key}/{split}'] = {'source':str(source),
                'evaluation_sha256':sha_file(source/'EVALUATION.json'),
                'per_query_sha256':cell['per_query_sha256'], 'input_lock_sha256':sha_file(source/'INPUT_LOCK.json')}
            link(root/'evaluation'/key/split, source)
    lock_file(root/'REUSED_BASELINES.json', records)
    lock_file(root/'FROZEN_GUARD.json', {'files':protected})
    if (root/'REAL_MECHANISM_PRECHECK.json').exists():
        return records
    torch.set_num_threads(2)
    payload = torch.load(root/'features/train.pt', map_location='cpu', weights_only=False)
    sample = json.loads((Path(previous['baseline_root'])/'CPU_LOSS_PRECHECK.json').read_text())
    anchors = np.asarray(sample['anchor_ids']); image_ids = np.asarray(sample['image_ids'])
    pool = image_ids[len(anchors):]; _, groups = np.unique(payload['image_sha256'], return_inverse=True)
    torch.manual_seed(42); model = DualAdapter()
    if tensor_hash(model.state_dict()) != records['checkpoints']['distill_seed42']['initial_state_sha256']:
        raise ValueError('Initialization differs')
    with torch.no_grad():
        oldloss = contrastive_distill_loss(model,payload['image_features'],payload['text_features'],anchors,pool,groups,cfg['temperature'],1.)
        newloss = contrastive_shuffle_loss(model,payload['image_features'],payload['text_features'],anchors,pool,groups,cfg['temperature'],1.,42,0,shuffle=False)
        _, _, details = shuffle_terms(model,payload['image_features'],payload['text_features'],anchors,pool,groups,cfg['temperature'],42,0)
    if not torch.allclose(oldloss,newloss,atol=1e-6):
        raise ValueError('Actual original objective not reproduced')
    errors = {}
    for direction in ['i','t']:
        original = details['teacher_logits_'+direction].log_softmax(1)
        changed = details['target_'+direction]; mask = details['positive_'+direction]
        columns = torch.arange(mask.shape[1]).expand(mask.shape[0],-1)
        mapping = details['mapping_'+direction]
        if not torch.equal(mapping.sort(1).values,columns) or not torch.equal(mapping[mask],columns[mask]):
            raise ValueError('Negative mapping contract failed')
        if not torch.equal(original[mask],changed[mask]) or not torch.equal(original.sort(1).values,changed.sort(1).values):
            raise ValueError('Teacher values changed instead of permuting')
        def entropy(p):return -(p*p.clamp_min(1e-300).log()).sum(1)
        a = original.double().exp(); b = changed.double().exp()
        gt = mask.double()/mask.sum(1)[:,None]
        errors[direction] = {'probability_sum_max_error':float((a.sum(1)-b.sum(1)).abs().max()),
             'teacher_entropy_max_error':float((entropy(a)-entropy(b)).abs().max()),
             'mixed_target_entropy_max_error':float((entropy((gt+a)/2)-entropy((gt+b)/2)).abs().max()),
             'positive_probabilities_bitwise_fixed':True, 'negative_probability_multiset_bitwise_fixed':True}
        if max(v for k,v in errors[direction].items() if k.endswith('error')) > 1e-10:
            raise ValueError('Actual target entropy/mass differs')
    write_json(root/'REAL_MECHANISM_PRECHECK.json',{'status':'passed','created_at':now(),
        'train_feature_sha256':records['feature_markers']['train']['features_sha256'],
        'precision':'FP32 real train cache/CPU objective; float64 entropy diagnostic',
        'i2t_logits_shape':list(details['target_i'].shape),'t2i_logits_shape':list(details['target_t'].shape),
        'old_objective':float(oldloss),'new_unshuffled_objective':float(newloss), 'contracts':errors,
        'scope':'Real CPU contract; CUDA first-update actual permutation archived separately'})
    return records


def verify_guard(root):
    guard = json.loads((root/'FROZEN_GUARD.json').read_text())
    changed = [name for name,digest in guard['files'].items() if sha_file(Path(name)) != digest]
    if changed:
        raise ValueError('Protected historical file changed: '+str(changed))
    return {'passed':True,'protected_files':len(guard['files'])}
