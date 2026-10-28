"""Bind old adapters/features and construct exact declared BLIP inputs."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from rarevlm_paths import SOURCE_ROOT,PROJECT_ROOT
from clip_retrieval.features import read_image
from vlm_probe.evaluate import lock_file,sha_file,now
from .common import write_json


def bind(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    feature_root=Path(cfg['feature_root']);root.mkdir(parents=True,exist_ok=True)
    oldlock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text());oldcfg=PROJECT_ROOT/'configs/clip_retrieval_distill_v1.json'
    if oldlock['config_sha256']!=sha_file(oldcfg):raise ValueError('Old config changed')
    if json.loads((old/'COMPLETION.json').read_text())['status']!='completed':raise ValueError('Old study incomplete')
    protected={str(SOURCE_ROOT/name):digest for name,digest in oldlock['sources'].items()}
    if any(sha_file(Path(name))!=digest for name,digest in protected.items()):raise ValueError('Historical scientific source changed')
    protected.update({str(p):sha_file(p) for p in [oldcfg,old/'STUDY_INPUT_LOCK.json',old/'analysis/ANALYSIS.json',
        PROJECT_ROOT/'docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md',PROJECT_ROOT/'results/clip_retrieval_distill_v1_metrics.json',
        PROJECT_ROOT/'docs/RESUME.md',Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf')]})
    record={'checkpoints':{},'feature_markers':{},'reused_evaluations':{},'scope':'historical three adapted checkpoints,not new training seeds'}
    for split in ['dev','test']:
        marker=json.loads((feature_root/f'features/{split}.json').read_text())
        if marker['images']!=1000 or sha_file(feature_root/f'features/{split}.pt')!=marker['features_sha256']:
            raise ValueError('Frozen gallery cache changed')
        record['feature_markers'][split]=marker
        protected[str(feature_root/f'features/{split}.pt')]=marker['features_sha256']
        for seed in cfg['training_seeds']:
            path=old/f'evaluation/distill_seed{seed}/{split}';cell=json.loads((path/'EVALUATION.json').read_text())
            if sha_file(path/'per_query.npz')!=cell['per_query_sha256']:raise ValueError('Historical ranks changed')
            record['reused_evaluations'][f'{split}/seed{seed}']={'source':str(path),'per_query_sha256':cell['per_query_sha256']}
            for name in ['EVALUATION.json','INPUT_LOCK.json','per_query.npz']:protected[str(path/name)]=sha_file(path/name)
    for seed in cfg['training_seeds']:
        path=old/f'runs/distill/seed{seed}';result=json.loads((path/'TRAIN_RESULT.json').read_text())
        if sha_file(path/'final.pt')!=result['checkpoint_sha256']:raise ValueError('Historical adapter changed')
        record['checkpoints'][str(seed)]={'source':str(path),'checkpoint_sha256':result['checkpoint_sha256']}
        for name in ['TRAIN_RESULT.json','INPUT_LOCK.json','final.pt']:protected[str(path/name)]=sha_file(path/name)
    for name in ['DATA_MANIFEST.json','IMAGE_INDEX.json','FEATURE_INPUT_LOCK.json']:
        protected[str(feature_root/name)]=sha_file(feature_root/name)
    lock_file(root/'REUSED_BASELINES.json',record);lock_file(root/'FROZEN_GUARD.json',{'files':protected})
    return record


def inputs(config_path,split):
    started=time.monotonic()
    from transformers import BlipProcessor
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);directory=root/'prepared'/split
    directory.mkdir(parents=True,exist_ok=True);processor=BlipProcessor.from_pretrained(cfg['model_root'],local_files_only=True)
    # Official BLIP init_tokenizer; checkpoint already contains these two rows.
    processor.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    processor.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    vocab=json.loads((Path(cfg['model_root'])/'config.json').read_text())['text_config']['vocab_size']
    if len(processor.tokenizer)!=vocab:raise ValueError('Tokenizer/checkpoint vocabulary differs; no embedding resize')
    manifest_path=Path(cfg['feature_root'])/'DATA_MANIFEST.json';manifest=json.loads(manifest_path.read_text())
    rows=manifest['splits'][split];index=json.loads((Path(cfg['feature_root'])/'IMAGE_INDEX.json').read_text())['offsets']
    identity={'config_sha256':sha_file(config_path),'manifest_sha256':sha_file(manifest_path),'source_sha256':sha_file(Path(__file__)),
        'processor':processor.image_processor.to_dict(),'text_max_tokens':cfg['text_max_tokens'],'split':split}
    lock_file(directory/'INPUT_LOCK.json',identity)
    if (directory/'INPUTS.json').exists():
        previous=json.loads((directory/'INPUTS.json').read_text())
        if sha_file(directory/'pixels.pt')!=previous['pixels_sha256'] or sha_file(directory/'tokens.pt')!=previous['tokens_sha256']:
            raise ValueError('Prepared inputs changed')
        return previous
    if len(rows)!=1000 or any(len(r['captions'])!=5 for r in rows):raise ValueError('Full gallery required')
    captions=[c for row in rows for c in row['captions']]
    tokens=processor.tokenizer(captions,padding='max_length',truncation=True,max_length=cfg['text_max_tokens'],return_tensors='pt')
    lengths=[len(processor.tokenizer.encode(c)) for c in captions]
    enc=processor.tokenizer.convert_tokens_to_ids('[ENC]')
    if enc==processor.tokenizer.unk_token_id or enc!=30523:raise ValueError('Pinned matching encoder token missing')
    tokens['input_ids'][:,0]=enc
    torch.save({'input_ids':tokens['input_ids'],'attention_mask':tokens['attention_mask']},directory/'tokens.pt')
    handles={};parts=[]
    try:
        for start in range(0,1000,32):
            images=[read_image(row,index,handles) for row in rows[start:start+32]]
            parts.append(processor.image_processor(images=images,return_tensors='pt')['pixel_values'])
            write_json(directory/'PROGRESS.json',{'status':'preparing','images':min(start+32,1000)})
    finally:
        for stream in handles.values():stream.close()
    pixels=torch.cat(parts);assert pixels.shape==(1000,3,384,384)
    torch.save(pixels,directory/'pixels.pt')
    result={'status':'completed','completed_at':now(),'split':split,'images':1000,'captions':5000,
        'pixel_shape':list(pixels.shape),'token_shape':list(tokens['input_ids'].shape),'dtype':'float32CPU pixels,forwardBF16',
        'matching_prefix':'[ENC] token30523,official BLIP ITM path; differs default HF processor CLS101',
        'text_truncated_captions':sum(n>cfg['text_max_tokens'] for n in lengths),'text_max_actual_tokens':int(tokens['attention_mask'].sum(1).max()),
        'cpu_input_preparation_seconds':time.monotonic()-started,
        'pixels_sha256':sha_file(directory/'pixels.pt'),'tokens_sha256':sha_file(directory/'tokens.pt')}
    write_json(directory/'INPUTS.json',result);return result


def verify_guard(root):
    guard=json.loads((root/'FROZEN_GUARD.json').read_text())['files']
    changed=[name for name,digest in guard.items() if sha_file(Path(name))!=digest]
    if changed:raise ValueError('Protected inputs changed: '+str(changed))
    return {'passed':True,'protected_files':len(guard)}
