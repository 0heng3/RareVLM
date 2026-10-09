"""Pinned-version viewer transport fallback for TextVQA engineering data."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .assets import DATASET_ID,DATASET_REVISION
from .common import content_sha,sha_file,supervised_answer,validate_records,write_json


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as r:return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(1)


def get_page(split,offset,length,raw):
    path=raw/f'{split}_{offset}_{length}.json'
    if path.exists():data=path.read_bytes()
    else:
        params=urllib.parse.urlencode({'dataset':DATASET_ID,'config':'default','split':split,'offset':offset,'length':length})
        data=get('https://datasets-server.huggingface.co/rows?'+params)
        path.write_bytes(data)
    page=json.loads(data)
    if page.get('partial'):raise ValueError('partial dataset viewer response')
    if len(page['rows'])!=length:raise ValueError('viewer row count mismatch')
    records=[];excluded=[]
    for item in page['rows']:
        r=item['row']
        if item.get('truncated_cells'):raise ValueError('truncated viewer annotation')
        if f'/--/{DATASET_REVISION}/--/' not in r['image']['src']:raise ValueError('dataset viewer revision differs from pinned release')
        if len(r['answers'])!=10 or not r['question'].strip():raise ValueError('invalid VQA annotations')
        try:answer=supervised_answer(r['answers'])
        except ValueError:
            excluded.append({'row':item['row_idx'],'question_id':str(r['question_id']),'reason':'all normalized human answers empty'})
            continue
        records.append({'image_id':str(r['image_id']),'question_id':str(r['question_id']),
            'question':r['question'],'answers':r['answers'],'answer':answer,
            'image_url':r['image']['src'],'source_split':split,'source_row':item['row_idx']})
    return records,{'file':path.name,'sha256':hashlib.sha256(data).hexdigest(),'source_split':split,'offset':offset,'length':length,'excluded_unusable_answers':excluded}


def select(rows,seed):
    by_image={}
    for r in rows:
        old=by_image.get(r['image_id'])
        if old is None or content_sha([seed,'question',r['question_id']])<content_sha([seed,'question',old['question_id']]):by_image[r['image_id']]=r
    return sorted(by_image.values(),key=lambda r:content_sha([seed,'image',r['image_id']]))


def get_image(row,destination):
    from PIL import Image
    cache=destination.parent/'viewer_images'/f"{row['image_id']}.jpg";cache.parent.mkdir(parents=True,exist_ok=True)
    sidecar=cache.with_suffix('.source.json')
    if cache.exists():
        data=cache.read_bytes();transport=json.loads(sidecar.read_text())['transport'] if sidecar.exists() else 'pinned HF viewer image CDN'
    else:
        try:
            url=f"https://open-images-dataset.s3.amazonaws.com/train/{row['image_id']}.jpg"
            data=get(url);transport='public Open Images S3 original by pinned annotation image ID'
        except Exception:
            data=get(row['image_url']);transport='pinned HF viewer image CDN'
        image=Image.open(io.BytesIO(data)).convert('RGB');image.load()
        temporary=cache.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(cache)
    image=Image.open(io.BytesIO(data)).convert('RGB');image.load()
    write_json(sidecar,{'transport':transport,'source_bytes_sha256':hashlib.sha256(data).hexdigest()})
    pixel=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    filename=f'images/{pixel}.png';p=destination/filename;p.parent.mkdir(parents=True,exist_ok=True);image.save(p,format='PNG')
    result={k:v for k,v in row.items() if k!='image_url'}
    result.update({'image':filename,'image_sha256':sha_file(p),'pixel_sha256':pixel,
                   'source_image_sha256':hashlib.sha256(data).hexdigest(),'image_transport':transport})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text());raw=args.asset_root/'viewer_metadata';raw.mkdir(parents=True,exist_ok=True)
    destination=args.asset_root/'prepared';path=destination/'manifest.json'
    if path.exists():raise FileExistsError('frozen prepared manifest already exists')
    requests=[(split,offset,min(100,n-offset)) for split,n in [('train',4000),('validation',1000)] for offset in range(0,n,100)]
    pools={'train':[],'validation':[]};source_pages=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(get_page,split,offset,length,raw):split for split,offset,length in requests}
        for future in concurrent.futures.as_completed(futures):
            rows,meta=future.result();pools[futures[future]]+=rows;source_pages.append(meta)
            print('metadata pages',len(source_pages),'/',len(requests),flush=True)
    final_pool=select(pools['validation'],config['data_seed']);final_ids={r['image_id'] for r in final_pool}
    train_pool=select([r for r in pools['train'] if r['image_id'] not in final_ids],config['data_seed'])
    n=config['internal_validation_images'];t=config['training_images'];f=config['final_evaluation_images']
    if len(train_pool)<n+t or len(final_pool)<f:raise ValueError('declared viewer source pool too small')
    selected={'validation':train_pool[:n],'train':train_pool[n:n+t],'final':final_pool[:f]};splits={}
    for split,rows in selected.items():
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures=[executor.submit(get_image,row,destination) for row in rows]
            records=[]
            for future in futures:
                records.append(future.result())
                if len(records)%64==0:print('images',split,len(records),'/',len(rows),flush=True)
        splits[split]=records
    validate_records(splits)
    manifest={'schema_version':'rarevlm.generative_vqa.v1','dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,
        'transport':'public HF Dataset Viewer; revision validated in every source image URL',
        'source_metadata':sorted(source_pages,key=lambda r:r['file']),
        'source_pool_rows':{'train':4000,'official_validation':1000},'data_seed':config['data_seed'],
        'selection':'deterministic question/image hash; one QA/image; selected image-ID and decoded-pixel checks',
        'pool_unique_train_images':len(train_pool),'pool_unique_final_images':len(final_pool),'splits':splits,
        'scope':'image-disjoint fixed small engineering subset; no full benchmark or pretraining-disjointness claim'}
    manifest['manifest_sha256']=content_sha(manifest);write_json(path,manifest)
    print('prepared manifest',manifest['manifest_sha256'],{k:len(v) for k,v in splits.items()},flush=True)


if __name__=='__main__':main()
