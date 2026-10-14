"""Bind reused checkpoints/rankings and capture a real teacher/student query."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
import numpy as np

from clip_retrieval.train import DualAdapter,tensor_hash,contrastive_loss
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import lock_file,now,sha_file,write_json
from .loss import distill_terms,teacher_kl


def prepare(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root']);features=Path(cfg['feature_root'])
    root.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path(cfg['baseline_config']).read_text());oldlock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(Path(cfg['baseline_config']))!=oldlock['config_sha256']:raise ValueError('Baseline config changed')
    for rel,digest in oldlock['sources'].items():
        if sha_file(SOURCE_ROOT/rel)!=digest:raise ValueError('Frozen baseline source changed')
    if json.loads((old/'COMPLETION.json').read_text())['status']!='completed':raise ValueError('Baseline incomplete')
    for key in ['training_seeds','epochs','anchor_images_per_update','extra_negative_images','temperature',
                'trainable_parameters','learning_rate','weight_decay','gradient_clip_norm','checkpoint_selection']:
        if cfg[key]!=previous[key]:raise ValueError('Baseline training protocol differs:'+key)
    records={'baseline_config_sha256':sha_file(Path(cfg['baseline_config'])),'baseline_study_lock_sha256':sha_file(old/'STUDY_INPUT_LOCK.json'),
        'baseline_analysis_sha256':sha_file(old/'analysis/dev/ANALYSIS.json'),'feature_input_lock_sha256':sha_file(features/'FEATURE_INPUT_LOCK.json'),
        'feature_markers':{},'checkpoints':{},'reused_dev_cells':{},'scope':'historical bound controls,not fresh independent repetitions'}
    for split,n in [('train',28000),('dev',1000),('test',1000)]:
        marker=json.loads((features/'features'/f'{split}.json').read_text())
        if marker['images']!=n or marker['features_sha256']!=sha_file(features/'features'/f'{split}.pt'):raise ValueError('Frozen cache changed')
        records['feature_markers'][split]=marker
    def link(path,target):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() or path.is_symlink():
            if path.resolve()!=target.resolve():raise ValueError('Unexpected inherited path')
        else:path.symlink_to(target.resolve(),target_is_directory=target.is_dir())
    for name in ['features','DATA_MANIFEST.json','IMAGE_INDEX.json']:link(root/name,features/name)
    for seed in cfg['training_seeds']:
        source=old/'runs/uniform_mean'/f'seed{seed}';train=json.loads((source/'TRAIN_RESULT.json').read_text())
        if train['checkpoint_sha256']!=sha_file(source/'final.pt') or train['negative_occurrences_sha256']!=sha_file(source/'negative_occurrences.npy'):
            raise ValueError('Reused checkpoint/exposure changed')
        records['checkpoints'][str(seed)]={'source':str(source),'train_result_sha256':sha_file(source/'TRAIN_RESULT.json'),
            'checkpoint_sha256':train['checkpoint_sha256'],'initial_state_sha256':train['initial_state_sha256'],
            'sampling_stream_sha256':train['sampling_stream_sha256'],'negative_occurrences_sha256':train['negative_occurrences_sha256']}
        link(root/'runs/uniform_mean'/f'seed{seed}',source)
    for key in ['frozen',*[f'uniform_mean_seed{s}' for s in cfg['training_seeds']]]:
        source=old/'evaluation'/key/'dev';cell=json.loads((source/'EVALUATION.json').read_text())
        if cell['per_query_sha256']!=sha_file(source/'per_query.npz'):raise ValueError('Historical dev rankings changed')
        input_lock=json.loads((source/'INPUT_LOCK.json').read_text())
        if input_lock['feature_sha256']!=records['feature_markers']['dev']['features_sha256']:raise ValueError('Dev features differ')
        records['reused_dev_cells'][key]={'source':str(source),'evaluation_sha256':sha_file(source/'EVALUATION.json'),
            'per_query_sha256':cell['per_query_sha256'],'input_lock_sha256':sha_file(source/'INPUT_LOCK.json')}
        link(root/'evaluation'/key/'dev',source)
    lock_file(root/'REUSED_BASELINES.json',records)
    upstream=root/'upstream';upstream.mkdir(exist_ok=True)
    archive=old/'upstream/mlfoundations__open_clip'
    for rel,name in [('src/open_clip/loss.py','open_clip_loss.py'),('LICENSE','LICENSE')]:
        destination=upstream/name
        if destination.exists() and sha_file(destination)!=sha_file(archive/rel):raise ValueError('Upstream archive differs')
        if not destination.exists():shutil.copyfile(archive/rel,destination)
    lock_file(upstream/'SOURCES.json',{'repository':'mlfoundations/open_clip','commit':'7067b350050dc409bfdfdbb3f0cc6c9a1651c000',
        'files':{name:sha_file(upstream/name) for name in ['open_clip_loss.py','LICENSE']},
        'url':'https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/loss.py'})
    if (root/'REAL_DISTILL_PRECHECK.json').exists():return records
    torch.set_num_threads(2)
    payload=torch.load(root/'features/train.pt',map_location='cpu',weights_only=False)
    sample=json.loads((old/'CPU_LOSS_PRECHECK.json').read_text())
    anchors=np.asarray(sample['anchor_ids']);image_ids=np.asarray(sample['image_ids']);pool=image_ids[len(anchors):]
    _,groups=np.unique(payload['image_sha256'],return_inverse=True)
    torch.manual_seed(42);identity=DualAdapter()
    if tensor_hash(identity.state_dict())!=records['checkpoints']['42']['initial_state_sha256']:raise ValueError('Initialization changed')
    with torch.no_grad():
        supervised,relation=distill_terms(identity,payload['image_features'],payload['text_features'],anchors,pool,groups,cfg['temperature'])
        reference=contrastive_loss(identity,payload['image_features'],payload['text_features'],anchors,pool,groups,cfg['temperature'])
    if abs(float(relation))>1e-6 or not torch.allclose(supervised,reference,atol=1e-6):raise ValueError('Real identity precheck failed')
    trained=DualAdapter();trained.load_state_dict(torch.load(old/'runs/uniform_mean/seed42/final.pt',map_location='cpu',weights_only=True));trained.eval()
    caption_ids=torch.as_tensor(sample['caption_ids']);image_indices=torch.as_tensor(image_ids)
    with torch.no_grad():
        teacher=F.normalize(payload['image_features'][anchors],dim=-1)@F.normalize(payload['text_features'][caption_ids],dim=-1).T/cfg['temperature']
        student=trained.image_features(payload['image_features'][anchors])@trained.text_features(payload['text_features'][caption_ids]).T/cfg['temperature']
    teacher=teacher[:1].double();student=student[:1].double().requires_grad_(True)
    kl=teacher_kl(student,teacher).sum();gradient=torch.autograd.grad(kl,student)[0]
    p_teacher=teacher.softmax(1);p_student=student.detach().softmax(1)
    if not torch.allclose(gradient,p_student-p_teacher,atol=1e-12):raise ValueError('Real KL gradient failed')
    cls=next(n for n in ast.parse((upstream/'open_clip_loss.py').read_text()).body if getattr(n,'name',None)=='DistillClipLoss')
    method=next(n for n in cls.body if getattr(n,'name',None)=='dist_loss');namespace={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned_open_clip_dist_loss','exec'),namespace)
    upstream_ce=namespace['dist_loss'](None,teacher,student.detach())
    entropy=-(p_teacher*teacher.log_softmax(1)).sum(1).mean()
    if not torch.allclose(upstream_ce-entropy,kl.detach(),atol=1e-12):raise ValueError('Upstream entropy/gradient relation failed')
    value={'status':'passed','created_at':now(),'precision':'FP32 cached features/checkpoint,CPU logits;float64 probability/gradient diagnostic',
        'scope':'real training query and fixed historical baseline,not new benchmark or new trained candidate prediction',
        'train_feature_sha256':records['feature_markers']['train']['features_sha256'],
        'baseline_checkpoint_sha256':records['checkpoints']['42']['checkpoint_sha256'],
        'identity_supervised_loss':float(supervised),'identity_relation_loss':float(relation),
        'first_anchor_id':sample['first_anchor_id'],'first_anchor_image_id':sample['first_anchor_image_id'],
        'captions':sample['first_anchor_captions'],'positive':sample['first_anchor_positive'],
        'teacher_logits':teacher[0].tolist(),'baseline_student_logits':student.detach()[0].tolist(),
        'teacher_probability':p_teacher[0].tolist(),'baseline_probability':p_student[0].tolist(),
        'kl_gradient':gradient[0].tolist(),'baseline_query_kl':float(kl),
        'upstream_soft_target_ce':float(upstream_ce),'teacher_entropy':float(entropy),
        'upstream_minus_entropy_equals_kl':True,'i2t_logits_shape':[32,480],'t2i_logits_shape':[160,96]}
    write_json(root/'REAL_DISTILL_PRECHECK.json',value);return records
