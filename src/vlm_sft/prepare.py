"""Prepare image-disjoint VQA instructions from pinned, embedded-image shards."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT

from .assets import DATASET_ID,DATASET_REVISION,PARQUETS
from .common import content_sha,sha_file,supervised_answer,validate_records,write_json


def read_rows(path):
    import pyarrow.parquet as pq
    from PIL import Image
    table=pq.read_table(path)
    rows=[]
    for i,r in enumerate(table.to_pylist()):
        image_record=r.get('image') or r.get('images')
        if isinstance(image_record,list):image_record=image_record[0]
        if not isinstance(image_record,dict) or not image_record.get('bytes'):raise ValueError('expected an embedded image in each source row')
        try:
            image=Image.open(io.BytesIO(image_record['bytes'])).convert('RGB');image.load()
        except Exception as e:raise ValueError(f'undecodable pinned source image row {i}') from e
        answers=r.get('answers')
        if isinstance(answers,dict):answers=answers.get('answer')
        if not isinstance(answers,list) or len(answers)!=10:raise ValueError('source must have ten human answers')
        try:answer=supervised_answer(answers)
        except ValueError:continue
        question=r.get('question')
        if not isinstance(question,str) or not question.strip():raise ValueError('empty source question')
        pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        rows.append({'image_id':str(r.get('image_id') or image_record.get('path') or pixels),
                     'question_id':str(r.get('question_id',i)),'question':question,'answers':answers,
                     'answer':answer,'pixel_sha256':pixels,'pil_image':image,
                     'source_shard':path.name,'source_row':i})
    return rows


def one_per_image(rows,seed):
    grouped=defaultdict(list)
    for row in rows:grouped[row['image_id']].append(row)
    chosen=[min(values,key=lambda r:content_sha([seed,'question',r['question_id'],r['question']])) for values in grouped.values()]
    chosen.sort(key=lambda r:content_sha([seed,'image',r['image_id']]))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text())
    destination=args.asset_root/'prepared';manifest_path=destination/'manifest.json'
    if manifest_path.exists():raise FileExistsError('prepared manifest already exists; preserve its frozen selection')
    for relative,digest in PARQUETS.items():
        if sha_file(args.asset_root/'raw_data'/Path(relative).name)!=digest:raise ValueError('source shard changed')
    train_rows=[]
    for relative in PARQUETS:
        if '/train-' in relative:train_rows+=read_rows(args.asset_root/'raw_data'/Path(relative).name)
    final_rows=read_rows(args.asset_root/'raw_data'/'validation-00000-of-00003.parquet')
    final_ids={r['image_id'] for r in final_rows};final_pixels={r['pixel_sha256'] for r in final_rows}
    train_rows=[r for r in train_rows if r['image_id'] not in final_ids and r['pixel_sha256'] not in final_pixels]
    pool=one_per_image(train_rows,config['data_seed']);final=one_per_image(final_rows,config['data_seed'])
    # Also prevent distinct image IDs with identical decoded pixel content.
    seen=set();unique=[]
    for r in pool:
        if r['pixel_sha256'] not in seen:unique.append(r);seen.add(r['pixel_sha256'])
    pool=unique
    seen=set();unique=[]
    for r in final:
        if r['pixel_sha256'] not in seen:unique.append(r);seen.add(r['pixel_sha256'])
    final=unique
    n=config['internal_validation_images'];t=config['training_images'];f=config['final_evaluation_images']
    if len(pool)<n+t or len(final)<f:raise ValueError(f'fixed source pool too small: train={len(pool)}, final={len(final)}')
    selected={'validation':pool[:n],'train':pool[n:n+t],'final':final[:f]}
    splits={}
    for split,rows in selected.items():
        records=[]
        for r in rows:
            r=dict(r);image=r.pop('pil_image');filename=f"images/{r['pixel_sha256']}.png"
            path=destination/filename;path.parent.mkdir(parents=True,exist_ok=True)
            image.save(path,format='PNG')
            r['image']=filename;r['image_sha256']=sha_file(path);records.append(r)
        splits[split]=records
    validate_records(splits)
    manifest={'schema_version':'rarevlm.generative_vqa.v1','dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,
        'source_shards_sha256':{Path(k).name:v for k,v in PARQUETS.items()},'data_seed':config['data_seed'],
        'selection':'one question per image; deterministic SHA image/question order; decoded pixel deduplication; final source is official validation',
        'pool_unique_train_images':len(pool),'pool_unique_final_images':len(final),'splits':splits,
        'scope':'fixed small engineering subset; no full benchmark/pretraining-disjointness claim'}
    manifest['manifest_sha256']=content_sha(manifest)
    write_json(manifest_path,manifest)
    print('prepared',manifest['manifest_sha256'],{k:len(v) for k,v in splits.items()},flush=True)


if __name__=='__main__':main()
