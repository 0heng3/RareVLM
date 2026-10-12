from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vlm_sft.common import content_sha,sha_file,write_json,frozen_parameter_hash,to_device
from .third_party.vqa_eval_metric import EvalAIAnswerProcessor

PROMPT_SUFFIX='\nAnswer the question using a single word or phrase.'
NORMALIZER=EvalAIAnswerProcessor()


def consensus(prediction,answers):
    if len(answers)!=10:raise ValueError('TextVQA requires ten human answers')
    p=NORMALIZER(prediction);refs=[NORMALIZER(a) for a in answers]
    return sum(min(sum(p==a for j,a in enumerate(refs) if j!=i)/3,1) for i in range(10))/10


def training_target(answers):
    values=[str(a).strip() for a in answers if str(a).strip()]
    if not values:return None
    counts=Counter(values)
    return min(counts,key=lambda x:(-counts[x],x))


def read_manifest(path,verify_images=False):
    path=Path(path);m=json.loads(path.read_text())
    if m['schema_version']!='rarevlm.vlm_data.v2':raise ValueError('wrong manifest version')
    if content_sha({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:raise ValueError('manifest changed')
    previous=set()
    for name in ('train_all','dev','final'):
        rows=m['splits'][name];ids={r['image_id'] for r in rows}
        if previous&ids:raise ValueError('image leakage across train/dev/final')
        previous|=ids
        if len({r['question_id'] for r in rows})!=len(rows):raise ValueError('duplicate question IDs')
    if {r['question_id'] for r in m['splits']['train_10k']}-{r['question_id'] for r in m['splits']['train_all']}:raise ValueError('stage1 outside training pool')
    if verify_images:
        for item in m['images'].values():
            if sha_file(path.parent/item['path'])!=item['sha256']:raise ValueError('image changed')
    return m


def messages(question,suffix=PROMPT_SUFFIX):
    return [{'role':'user','content':[{'type':'image'},{'type':'text','text':question+suffix}]}]


def encode(processor,row,image_path,supervised):
    from PIL import Image
    from vlm_sft.common import answer_labels
    image=Image.open(image_path).convert('RGB')
    msg=messages(row['question'],row.get('prompt_suffix',PROMPT_SUFFIX))
    prefix=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    p=processor(text=[prefix],images=[image],return_tensors='pt')
    if not supervised:return dict(p)
    if not row.get('target'):raise ValueError('missing training target')
    full=msg+[{'role':'assistant','content':[{'type':'text','text':row['target']}]}]
    text=processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
    batch=dict(processor(text=[text],images=[image],return_tensors='pt'))
    batch['labels'],_=answer_labels(batch['input_ids'],p['input_ids'])
    return batch


def collate(samples,pad_token_id):
    """Left padding puts all supervised answer spans in the common sequence tail."""
    import torch
    length=max(x['input_ids'].shape[1] for x in samples)
    n=len(samples);result={}
    for key,fill in [('input_ids',pad_token_id),('attention_mask',0),('labels',-100)]:
        if key not in samples[0]:continue
        out=torch.full((n,length),fill,dtype=samples[0][key].dtype)
        for i,x in enumerate(samples):out[i,-x[key].shape[1]:]=x[key][0]
        result[key]=out
    for key in ('pixel_values','image_grid_thw'):
        result[key]=torch.cat([x[key] for x in samples],dim=0)
    return result


def answer_loss(model,batch,tail=True):
    """Exact per-question assistant CE; optional head slicing avoids ignored logits."""
    import torch.nn.functional as F
    labels=batch['labels']
    counts=(labels!=-100).sum(dim=1)
    if not bool((counts>0).all()):raise ValueError('no supervised answer token')
    keep=int(counts.max())+1
    handle=None
    if tail:
        underlying=model.module if hasattr(model,'module') else model
        handle=underlying.get_output_embeddings().register_forward_pre_hook(
            lambda module,args:(args[0][:,-keep:,:],))
    try:
        output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
    finally:
        if handle is not None:handle.remove()
    logits=output.logits
    targets=labels[:,-logits.shape[1]:][:,1:]
    shifted=logits[:,:-1,:].contiguous()
    values=F.cross_entropy(shifted.view(-1,shifted.shape[-1]),targets.reshape(-1),
                           ignore_index=-100,reduction='none').view(targets.shape)
    valid=targets!=-100
    if not bool((valid.sum(dim=1)==counts).all()):raise ValueError('head slicing lost answer targets')
    return values.sum(dim=1)/counts,values.sum(),int(counts.sum())
