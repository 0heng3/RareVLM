"""Resumable pinned WebDataset archives and image-level train/dev/test manifests."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request
from PIL import Image
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,private_path,lock_file,content_sha

DEFAULT_CONFIG=PROJECT_ROOT/'configs/clip_retrieval_v1.json'


def download_one(root,source,item):
    relative=item['path']
    if not re.fullmatch(r'(train|test)/[0-9]+\.tar',relative):raise ValueError('unexpected archive path')
    destination=root/'archives'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix('.tar.part');state_path=root/'download_progress'/relative.replace('/','_')
    state_path=state_path.with_suffix('.json')
    if destination.exists():
        if destination.stat().st_size==item['size'] and sha_file(destination)==item['sha256']:
            write_json(state_path,{'status':'completed','bytes':item['size'],'total':item['size'],'cache_verified':True});return
        raise ValueError('complete archive changed;do not silently replace')
    for attempt in range(5):
        start=partial.stat().st_size if partial.exists() else 0;begun=time.monotonic();received=start
        try:
            url=f'https://huggingface.co/datasets/{source["repository"]}/resolve/{source["revision"]}/{relative}?download=true&rarevlm_nonce={time.time_ns()}'
            headers={'User-Agent':'RareVLM-public-benchmark','Range':f'bytes={start}-'} if start else {'User-Agent':'RareVLM-public-benchmark'}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                if start and response.status!=206:start=received=0
                if start:
                    match=re.match(r'bytes (\d+)-\d+/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=start or int(match[2])!=item['size']:raise ValueError('unsafe resume range')
                with partial.open('ab' if start else 'wb') as stream:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        stream.write(chunk);received+=len(chunk)
                        elapsed=time.monotonic()-begun
                        write_json(state_path,{'status':'downloading','path':relative,'bytes':received,'total':item['size'],
                            'attempt':attempt,'updated_at':now(),'session_bytes_per_second':(received-start)/max(elapsed,1e-8)})
            if partial.stat().st_size!=item['size'] or sha_file(partial)!=item['sha256']:raise ValueError('archive size/SHA differs')
            partial.replace(destination)
            write_json(state_path,{'status':'completed','path':relative,'bytes':item['size'],'total':item['size'],
                'sha256':item['sha256'],'completed_at':now()})
            print('VERIFIED_ARCHIVE',relative,item['size'],flush=True);return
        except Exception as error:
            write_json(state_path,{'status':'retrying' if attempt<4 else 'failed','path':relative,
                'bytes':partial.stat().st_size if partial.exists() else 0,'total':item['size'],
                'attempt':attempt,'error_type':type(error).__name__,'updated_at':now()})
            if attempt==4:raise RuntimeError(f'archive download failed:{relative}:{type(error).__name__}') from None
            time.sleep(min(5*(attempt+1),20))


def download(root):
    root=private_path(root);source=json.loads((root/'DATA_SOURCE_LOCK.json').read_text())
    owner=(root/'DOWNLOAD.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_file(root/'DOWNLOAD_INPUT_LOCK.json',{'source_sha256':sha_file(Path(__file__)),
        'metadata_sha256':sha_file(root/'DATA_SOURCE_LOCK.json'),'root':str(root)})
    (root/'download_progress').mkdir(exist_ok=True)
    write_json(root/'DATA_STATUS.json',{'status':'downloading','started_at':now(),'files':len(source['files']),
        'total_bytes':source['total_archive_bytes'],'source_revision':source['revision']})
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(download_one,root,source,item) for item in sorted(source['files'],key=lambda x:(x['size'],x['path']))]
            for future in futures:future.result()
        write_json(root/'ARCHIVES_READY.json',{'status':'passed','completed_at':now(),
            'metadata_sha256':sha_file(root/'DATA_SOURCE_LOCK.json'),'files':source['files']})
        write_json(root/'DATA_STATUS.json',{'status':'archives_completed','updated_at':now(),'total_bytes':source['total_archive_bytes']})
    except Exception as error:
        write_json(root/'DATA_STATUS.json',{'status':'failed','updated_at':now(),'error':str(error)});raise


def manifest(config_path=DEFAULT_CONFIG):
    cfg=json.loads(config_path.read_text());root=private_path(cfg['output_root'])
    if not (root/'ARCHIVES_READY.json').exists():raise ValueError('full archives not ready')
    source=json.loads((root/'DATA_SOURCE_LOCK.json').read_text());records={'train':[],'test':[]}
    for item in source['files']:
        path=root/'archives'/item['path']
        if path.stat().st_size!=item['size'] or sha_file(path)!=item['sha256']:raise ValueError('archive changed')
        split,filename=item['path'].split('/');entries={}
        with tarfile.open(path,'r') as tar:
            for member in tar:
                if not member.isfile():continue
                if not re.fullmatch(r'[A-Za-z0-9_-]+\.(jpg|jpeg|png|txt)',member.name):raise ValueError('unexpected webdataset member')
                key,extension=member.name.rsplit('.',1);entries.setdefault(key,{})[extension]=member
            for key,fields in sorted(entries.items()):
                image_key=next((e for e in ('jpg','jpeg','png') if e in fields),None)
                if not image_key or 'txt' not in fields:raise ValueError('incomplete image/caption pair')
                raw=tar.extractfile(fields[image_key]).read();captions=tar.extractfile(fields['txt']).read().decode('utf-8').splitlines()
                if len(captions)!=5 or any(not c.strip() for c in captions):raise ValueError('five complete human captions required')
                with Image.open(io.BytesIO(raw)) as image:width,height=image.size;image.verify()
                records[split].append({'image_id':split+'/'+filename+'/'+key,'archive_path':str(path),
                    'member':fields[image_key].name,'image_sha256':hashlib.sha256(raw).hexdigest(),
                    'width':width,'height':height,'captions':captions,'source_split':split})
    train_sha={r['image_sha256'] for r in records['train']};test_sha={r['image_sha256'] for r in records['test']}
    if train_sha&test_sha:raise ValueError('identical image bytes cross supplied train/test')
    groups=sorted(train_sha,key=lambda s:hashlib.sha256(f'{cfg["split_seed"]}|{s}'.encode()).hexdigest())
    if len(groups)<cfg['dev_images']+10000:raise ValueError('full training pool required;not sample fallback')
    dev_groups=set(groups[:cfg['dev_images']]);dev=[r for r in records['train'] if r['image_sha256'] in dev_groups]
    train=[r for r in records['train'] if r['image_sha256'] not in dev_groups]
    if len(records['test'])!=1000:raise ValueError('expected supplied full1000-image test')
    output={'schema_version':'rarevlm.retrieval.data.v1','dataset_repository':source['repository'],
        'dataset_revision':source['revision'],'source_lock_sha256':sha_file(root/'DATA_SOURCE_LOCK.json'),
        'data_source_sha256':sha_file(Path(__file__)),'splits':{'train':train,'dev':dev,'test':records['test']},
        'counts':{'source_train':len(records['train']),'train':len(train),'dev':len(dev),'test':len(records['test'])},
        'caption_counts':{k:5*len(v) for k,v in {'train':train,'dev':dev,'test':records['test']}.items()},
        'pretraining_overlap':'unknown','dev_rule':cfg['dev_selection'],'upstream_image_representation':'supplied webdataset JPEG/PNG,not claiming original full-resolution Flickr files'}
    output['manifest_sha256']=content_sha(output);lock_file(root/'DATA_MANIFEST.json',output)
    write_json(root/'DATA_STATUS.json',{'status':'manifest_completed','completed_at':now(),'counts':output['counts'],
        'manifest_sha256':output['manifest_sha256']});print('FULL_MANIFEST',output['counts'],flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','manifest'])
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    download(Path(cfg['output_root'])) if args.action=='download' else manifest(args.config)
