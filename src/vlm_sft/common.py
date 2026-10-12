"""Shared, auditable data/masking/metric contracts for generative VQA."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
_cache=Path(os.environ.get('XDG_CACHE_HOME',PROJECT_ROOT/'.cache'))
os.environ.setdefault('CUDA_CACHE_PATH',str(_cache/'cuda'))
os.environ.setdefault('TRITON_CACHE_DIR',str(_cache/'triton'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(_cache/'torchinductor'))

PROMPT_SUFFIX = '\nAnswer with a short phrase only.'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def normalize_answer(value):
    """Declared compact normalization; not a full benchmark evaluator clone."""
    value=str(value).lower().strip().replace('\n',' ').replace('\t',' ')
    value=re.sub(r'(?<=\d),(?=\d)','',value)
    value=re.sub(r'(?<!\d)\.|\.(?!\d)','',value)
    value=re.sub(r'[^\w\s\."]', ' ',value)
    value=value.replace('"',' ')
    numbers={'none':'0','zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    return ' '.join(numbers.get(w,w) for w in value.split() if w not in {'a','an','the'})


def consensus_score(prediction,answers):
    """Exact leave-one-annotator-out VQA consensus formula, declared normalization."""
    if len(answers)!=10:raise ValueError('requires exactly ten human answers')
    p=normalize_answer(prediction);normalized=[normalize_answer(a) for a in answers]
    return sum(min(sum(p==a for j,a in enumerate(normalized) if j!=i)/3.,1.) for i in range(10))/10


def supervised_answer(answers):
    counts=Counter(normalize_answer(a) for a in answers)
    if '' in counts:del counts['']
    if not counts:raise ValueError('no usable answer')
    return min(counts,key=lambda a:(-counts[a],a))


def validate_records(splits):
    seen_ids=set();seen_pixels=set()
    for split,records in splits.items():
        if not records:raise ValueError(f'empty split: {split}')
        for record in records:
            if len(record['answers'])!=10 or not record['question'].strip():raise ValueError('invalid VQA annotation')
            if record['image_id'] in seen_ids or record['pixel_sha256'] in seen_pixels:raise ValueError('same image/content appears across selected records')
            seen_ids.add(record['image_id']);seen_pixels.add(record['pixel_sha256'])
            if not supervised_answer(record['answers']):raise ValueError('invalid supervision')


def read_manifest(path,verify_images=False):
    path=Path(path);m=json.loads(path.read_text())
    if m.get('schema_version')!='rarevlm.generative_vqa.v1':raise ValueError('manifest schema mismatch')
    if m['manifest_sha256']!=content_sha({k:v for k,v in m.items() if k!='manifest_sha256'}):raise ValueError('manifest content changed')
    validate_records(m['splits'])
    if verify_images:
        for records in m['splits'].values():
            for r in records:
                p=path.parent/r['image']
                if sha_file(p)!=r['image_sha256']:raise ValueError('image artifact changed')
    return m


def user_messages(question,image=None):
    # No answer or evaluation target enters generation messages.
    image_item={'type':'image'}
    if image is not None:image_item['image']=image
    return [{'role':'user','content':[image_item,{'type':'text','text':question+PROMPT_SUFFIX}]}]


def answer_labels(full_ids,prefix_ids):
    import torch
    if full_ids.ndim!=2 or prefix_ids.ndim!=2 or full_ids.shape[0]!=1 or prefix_ids.shape[0]!=1:raise ValueError('microbatch must contain one image/question')
    n=prefix_ids.shape[1]
    if n>=full_ids.shape[1] or not torch.equal(full_ids[:,:n],prefix_ids):raise ValueError('chat template prefix mismatch or missing answer tokens')
    labels=full_ids.clone();labels[:,:n]=-100
    return labels,n


def encode_record(processor,record,image_root,supervised):
    from PIL import Image
    image=Image.open(Path(image_root)/record['image']).convert('RGB')
    messages=user_messages(record['question'])
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    prefix_inputs=processor(text=[prefix],images=[image],return_tensors='pt')
    if not supervised:return dict(prefix_inputs)
    messages=messages+[{'role':'assistant','content':[{'type':'text','text':record['answer']}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    full=processor(text=[text],images=[image],return_tensors='pt')
    labels,prefix_length=answer_labels(full['input_ids'],prefix_inputs['input_ids'])
    full=dict(full);full['labels']=labels
    return full


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def frozen_parameter_hash(model):
    """Bind every frozen parameter's name and values before/after adaptation."""
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
            # Packed 4-bit bytes alone do not bind the dequantized weights.
            state=getattr(p,'quant_state',None)
            if state is not None:
                for key,value in sorted(state.as_dict(packed=False).items()):
                    h.update(key.encode())
                    if hasattr(value,'detach'):h.update(value.detach().cpu().contiguous().numpy().tobytes())
                    else:h.update(str(value).encode())
    return h.hexdigest()
