"""Frozen complete CLIP features; efficient tar offsets and resumable chunks."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import tarfile
import time
import torch
import torch.nn.functional as F
from PIL import Image
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,content_sha,lock_file,package_versions
from vlm_joint.evaluate import visible_gpu_binding
from .data import DEFAULT_CONFIG


def prepare_index(root):
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());indices={}
    for split,rows in manifest['splits'].items():
        for row in rows:indices.setdefault(row['archive_path'],{})[row['member']]=row['image_id']
    offsets={}
    for path,needed in indices.items():
        with tarfile.open(path,'r') as tar:
            for member in tar:
                if member.name in needed:offsets[needed[member.name]]={'archive_path':path,'offset':member.offset_data,'size':member.size}
    if len(offsets)!=sum(len(rows) for rows in manifest['splits'].values()):raise ValueError('offset index incomplete')
    value={'manifest_sha256':manifest['manifest_sha256'],'source_sha256':sha_file(Path(__file__)),'offsets':offsets}
    lock_file(root/'IMAGE_INDEX.json',value);return value


def read_image(row,index,handles):
    item=index[row['image_id']];path=item['archive_path']
    if path not in handles:handles[path]=Path(path).open('rb')
    stream=handles[path];stream.seek(item['offset']);raw=stream.read(item['size'])
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('indexed image bytes changed')
    with Image.open(io.BytesIO(raw)) as image:return image.convert('RGB')


def extract(config_path=DEFAULT_CONFIG):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root'])
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());idx=json.loads((root/'IMAGE_INDEX.json').read_text())
    if idx['manifest_sha256']!=manifest['manifest_sha256']:raise ValueError('image index mismatch')
    model_root=Path(cfg['model_root'])
    if sha_file(model_root/'pytorch_model.bin')!=cfg['model_weight_sha256']:raise ValueError('CLIP weights mismatch')
    torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    from transformers import CLIPModel,CLIPTokenizerFast
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    pre=json.loads((model_root/'preprocessor_config.json').read_text())
    transform=transforms.Compose([transforms.Resize(pre['size'],interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(pre['crop_size']),transforms.ToTensor(),transforms.Normalize(pre['image_mean'],pre['image_std'])])
    model=CLIPModel.from_pretrained(str(model_root),local_files_only=True,use_safetensors=False,
        torch_dtype=torch.float32,attn_implementation=cfg['clip_attention_implementation']).eval().to(device)
    model.requires_grad_(False);tokenizer=CLIPTokenizerFast.from_pretrained(str(model_root),local_files_only=True)
    if model.config.projection_dim!=512 or model.config.text_config.max_position_embeddings!=77:raise ValueError('pinned CLIP shape mismatch')
    source_names=['clip_retrieval/features.py','clip_retrieval/data.py','methods.py','rarevlm_paths.py']
    hardware=visible_gpu_binding()
    lock={'config_sha256':sha_file(config_path),'manifest_sha256':manifest['manifest_sha256'],
        'index_sha256':sha_file(root/'IMAGE_INDEX.json'),'weight_sha256':cfg['model_weight_sha256'],
        'source_hashes':{name:sha_file(SOURCE_ROOT/name) for name in source_names},
        'packages':package_versions(),'hardware':hardware,'precision':'float32','tf32':False,
        'image_transform':pre,'text_max_tokens':77,'frozen_parameters':True}
    lock_file(root/'FEATURE_INPUT_LOCK.json',lock);handles={};started=time.monotonic()
    try:
        for split,rows in manifest['splits'].items():
            directory=root/'feature_chunks'/split;directory.mkdir(parents=True,exist_ok=True)
            n=len(rows);size=cfg['extract_image_batch_size'];parts=[];truncated=0
            for begin in range(0,n,size):
                path=directory/f'{begin:06d}.pt';meta=path.with_suffix('.json');chunk=rows[begin:begin+size]
                if path.exists() and meta.exists():
                    marker=json.loads(meta.read_text())
                    if marker['input_lock_sha256']!=content_sha(lock) or marker['features_sha256']!=sha_file(path):raise ValueError('feature cache changed')
                else:
                    images=torch.stack([transform(read_image(r,idx['offsets'],handles)) for r in chunk]).to(device)
                    captions=[c for row in chunk for c in row['captions']]
                    lengths=[len(tokenizer.encode(c,add_special_tokens=True)) for c in captions]
                    with torch.inference_mode():
                        image_features=F.normalize(model.get_image_features(pixel_values=images).float(),dim=-1).cpu()
                        text_parts=[]
                        for text_begin in range(0,len(captions),cfg['extract_text_batch_size']):
                            texts=captions[text_begin:text_begin+cfg['extract_text_batch_size']]
                            tokens=tokenizer(texts,padding=True,truncation=True,max_length=77,return_tensors='pt')
                            text_parts.append(F.normalize(model.get_text_features(**{k:v.to(device) for k,v in tokens.items()}).float(),dim=-1).cpu())
                        text_features=torch.cat(text_parts)
                    payload={'image_features':image_features,'text_features':text_features,
                        'image_ids':[r['image_id'] for r in chunk],'image_sha256':[r['image_sha256'] for r in chunk],
                        'captions':captions,'caption_to_image':torch.arange(len(chunk)).repeat_interleave(5),
                        'caption_token_lengths':lengths}
                    tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path)
                    marker={'features_sha256':sha_file(path),'input_lock_sha256':content_sha(lock),
                        'start':begin,'images':len(chunk),'captions':len(captions),'truncated_captions':sum(l>77 for l in lengths)}
                    write_json(meta,marker)
                truncated+=marker['truncated_captions'];parts.append(path)
                write_json(root/'FEATURE_STATUS.json',{'status':'extracting','split':split,'completed_images':min(begin+size,n),
                    'total_images':n,'elapsed_seconds':time.monotonic()-started,'updated_at':now()})
                if begin==0 or (begin//size+1)%20==0:print('EXTRACTED',split,min(begin+size,n),'/',n,flush=True)
            chunks=[torch.load(p,map_location='cpu',weights_only=False) for p in parts]
            payload={'image_features':torch.cat([p['image_features'] for p in chunks]),
                'text_features':torch.cat([p['text_features'] for p in chunks]),
                'image_ids':[i for p in chunks for i in p['image_ids']],
                'image_sha256':[i for p in chunks for i in p['image_sha256']],
                'captions':[c for p in chunks for c in p['captions']],
                'caption_to_image':torch.arange(n).repeat_interleave(5),'manifest_sha256':manifest['manifest_sha256'],
                'feature_input_lock_sha256':content_sha(lock),'truncated_captions':truncated}
            if payload['image_features'].shape!=(n,512) or payload['text_features'].shape!=(n*5,512):raise ValueError('complete feature shape mismatch')
            path=root/'features'/f'{split}.pt';path.parent.mkdir(exist_ok=True)
            tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path)
            write_json(path.with_suffix('.json'),{'status':'completed','images':n,'captions':n*5,
                'features_sha256':sha_file(path),'feature_input_lock_sha256':content_sha(lock),'truncated_captions':truncated})
        write_json(root/'FEATURE_STATUS.json',{'status':'completed','completed_at':now(),'elapsed_seconds':time.monotonic()-started,'hardware':hardware})
    finally:
        for handle in handles.values():handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    extract(args.config)
