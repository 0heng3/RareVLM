"""Full fixed TextVQA annotations, official images and image-disjoint splits."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter,defaultdict
from pathlib import Path
from .common import content_sha,sha_file,training_target,write_json
from .download import get_bytes,official_images
from vlm_sft.assets import DATASET_ID,DATASET_REVISION


def annotations(root,pilot):
    raw=root/'annotations';raw.mkdir(parents=True,exist_ok=True)
    jobs=[(split,offset,min(100,n-offset)) for split,n in [('train',34602),('validation',5000)] for offset in range(0,n,100)]
    def page(job):
        split,offset,length=job;path=raw/f'{split}_{offset}_{length}.json'
        old=pilot/'viewer_metadata'/path.name
        if not path.exists():
            if old.exists():shutil.copyfile(old,path)
            else:
                from urllib.parse import urlencode
                params=urlencode({'dataset':DATASET_ID,'config':'default','split':split,'offset':offset,'length':length})
                data=get_bytes('https://datasets-server.huggingface.co/rows?'+params)
                tmp=path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(path)
        d=json.loads(path.read_text())
        if d.get('partial') or len(d['rows'])!=length:raise ValueError('partial/incorrect annotation page')
        records=[]
        for item in d['rows']:
            r=item['row']
            critical={'question','answers','image','image_id','question_id','image_width','image_height'}
            if critical.intersection(item.get('truncated_cells',[])):raise ValueError('truncated model input/evaluation annotation')
            if f'/--/{DATASET_REVISION}/--/' not in r['image']['src']:raise ValueError('viewer revision changed')
            if len(r['answers'])!=10 or not r['question'].strip():raise ValueError('invalid annotation')
            records.append({'image_id':str(r['image_id']),'question_id':str(r['question_id']),
                'question':r['question'],'answers':r['answers'],'target':training_target(r['answers']),
                'annotation_width':r['image_width'],'annotation_height':r['image_height'],
                'source_split':split,'source_row':item['row_idx']})
        return split,records,{'path':path.name,'sha256':sha_file(path),'split':split,'offset':offset,'length':length}
    pool=defaultdict(list);provenance=[];done=0
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for future in concurrent.futures.as_completed([executor.submit(page,j) for j in jobs]):
            split,rows,meta=future.result();pool[split]+=rows;provenance.append(meta);done+=1
            if done%20==0 or done==len(jobs):print('full annotation pages',done,'/',len(jobs),flush=True)
    for rows in pool.values():rows.sort(key=lambda r:r['source_row'])
    write_json(root/'ANNOTATIONS.json',{'splits':dict(pool),'source_pages':sorted(provenance,key=lambda x:x['path']),
        'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'transport':'complete pinned HF mirror annotations; official image archive'})
    return pool,provenance


def build(root,pilot,config,pool,source_pages,archive,archive_meta):
    from PIL import Image
    prepared=root/'prepared';prepared.mkdir(exist_ok=True);manifest=prepared/'manifest.json'
    if manifest.exists():raise FileExistsError('prepared manifest already frozen')
    train=pool['train'];final=pool['validation']
    if len(train)!=34602 or len(final)!=5000 or len({r['image_id'] for r in final})!=3166:raise ValueError('not the full official benchmark')
    image_ids={r['image_id'] for r in train+final}
    references={r['image_id']:r for r in train+final};images={}
    with zipfile.ZipFile(archive) as z:
        lookup={}
        for info in z.infolist():
            if info.is_dir():continue
            name=Path(info.filename).name
            if Path(name).suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
            image_id=Path(name).stem
            if image_id in lookup:raise ValueError('duplicate archive image ID')
            lookup[image_id]=info.filename
        missing=image_ids-set(lookup)
        if missing:raise ValueError(f'official archive missing {len(missing)} annotated images')
        for i,image_id in enumerate(sorted(image_ids)):
            path=prepared/'images'/f'{image_id}.jpg';path.parent.mkdir(exist_ok=True)
            if not path.exists():
                tmp=path.with_suffix('.tmp');tmp.write_bytes(z.read(lookup[image_id]));tmp.replace(path)
            image=Image.open(path).convert('RGB');image.load()
            ref=references[image_id]
            if image.size!=(int(ref['annotation_width']),int(ref['annotation_height'])):
                raise ValueError(f'official image/annotation dimensions disagree: {image_id}: {image.size} vs {(ref["annotation_width"],ref["annotation_height"])}')
            pixel=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            images[image_id]={'path':str(path.relative_to(prepared)),'sha256':sha_file(path),
                'pixel_sha256':pixel,'width':image.width,'height':image.height,'archive_member':lookup[image_id]}
            if (i+1)%1000==0:print('official decoded image audit',i+1,'/',len(image_ids),flush=True)
    final_pixels={images[r['image_id']]['pixel_sha256'] for r in final}
    final_ids={r['image_id'] for r in final};excluded=[];usable=[]
    for r in train:
        if not r['target']:excluded.append({'question_id':r['question_id'],'reason':'all raw human answers empty'})
        elif r['image_id'] in final_ids or images[r['image_id']]['pixel_sha256'] in final_pixels:
            excluded.append({'question_id':r['question_id'],'reason':'train image overlaps final'})
        else:usable.append(r)
    old=json.loads((pilot/'prepared/manifest.json').read_text())
    exposed={r['image_id'] for name in ('train','validation') for r in old['splits'][name]}
    ids=sorted({r['image_id'] for r in usable},key=lambda x:content_sha([config['data_seed'],'deep_v2_image',x]))
    candidates=[x for x in ids if x not in exposed]
    dev_ids=set(candidates[:config['dev_images']]);dev_pixels={images[x]['pixel_sha256'] for x in dev_ids}
    dev=[r for r in usable if r['image_id'] in dev_ids]
    train_all=[]
    for r in usable:
        if r['image_id'] in dev_ids:continue
        if images[r['image_id']]['pixel_sha256'] in dev_pixels:
            excluded.append({'question_id':r['question_id'],'reason':'train pixel overlaps internal dev'})
        else:train_all.append(r)
    train_all.sort(key=lambda r:(content_sha([config['data_seed'],'deep_v2_image',r['image_id']]),content_sha(['question',r['question_id']])))
    stage1=train_all[:config['stage1_training_questions']]
    if len(stage1)!=10000:raise ValueError('not enough stage1 questions')
    counts=Counter(r['target'] for r in train_all)
    m={'schema_version':'rarevlm.vlm_data.v2','dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,
       'source_pages':sorted(source_pages,key=lambda x:x['path']),'official_archive':archive_meta,
       'data_seed':config['data_seed'],'config_sha256':content_sha(config),'images':images,
       'splits':{'train_10k':stage1,'train_all':train_all,'dev':dev,'final':final},
       'source_question_counts':{'train':len(train),'final':len(final)},
       'source_image_counts':{'train':len({r['image_id'] for r in train}),'final':3166},
       'excluded_training_annotations':excluded,'historical_pilot_images_excluded_from_new_dev':len(exposed),
       'train_raw_answer_frequency':dict(counts),
       'selection':'fixed SHA image order, image-disjoint dev, all questions retained; stage1 first 10000 QA in SHA order',
       'evaluation_exposure':'official val partially evaluated in historical pilot; public validation benchmark, not unseen test',
       'pretraining_overlap':'unknown'}
    m['manifest_sha256']=content_sha(m);write_json(manifest,m)
    print('FULL DATA READY',json.dumps({k:len(v) for k,v in m['splits'].items()}),m['manifest_sha256'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_deep_v2.json')
    args=p.parse_args();args.asset_root.mkdir(parents=True,exist_ok=True);c=json.loads(args.config.read_text())
    if (args.asset_root/'prepared/manifest.json').exists():
        from .common import read_manifest
        m=read_manifest(args.asset_root/'prepared/manifest.json')
        print('reusing fixed full manifest',m['manifest_sha256']);return
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        image_job=executor.submit(official_images,args.asset_root)
        if (args.asset_root/'ANNOTATIONS.json').exists():
            a=json.loads((args.asset_root/'ANNOTATIONS.json').read_text());pool=a['splits'];meta=a['source_pages']
        else:pool,meta=annotations(args.asset_root,args.pilot_root)
        archive,archive_meta=image_job.result()
    build(args.asset_root,args.pilot_root,c,pool,meta,archive,archive_meta)


if __name__=='__main__':main()
