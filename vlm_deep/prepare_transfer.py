"""Full pinned DocVQA validation/MMStar; verified Parquet assets and pixel audits."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import time
from pathlib import Path
from .common import content_sha,sha_file,write_json

DATASETS={
 'docvqa':{'repository':'lmms-lab-encoder/DocVQA','revision':'539088ef8a8ada01ac8e2e6d4e372586748a265e','questions':5349},
 'mmstar':{'repository':'Lin-Chen/MMStar','revision':'bc98d668301da7b14f648724866e57302778ab27','questions':1500}}


def download(spec,item,path):
    import requests
    expected=item['lfs']['oid'];size=item['size']
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=size or sha_file(path)!=expected:raise ValueError('cached transfer asset corrupted')
        return {'file':item['path'],'sha256':expected,'bytes':size,'transport':'cached pinned asset'}
    tmp=path.with_suffix('.download');last=None
    for attempt in range(6):
        domain='huggingface.co' if attempt<4 else 'hf-mirror.com'
        offset=tmp.stat().st_size if tmp.exists() else 0
        if offset>=size:offset=0
        url=f'https://{domain}/datasets/{spec["repository"]}/resolve/{spec["revision"]}/{item["path"]}?download=true&rarevlm_v2={time.time_ns()}'
        try:
            with requests.get(url,headers={'Range':f'bytes={offset}-{size-1}'},stream=True,timeout=(15,60)) as r:
                r.raise_for_status()
                if r.status_code==206:
                    if r.headers.get('Content-Range')!=f'bytes {offset}-{size-1}/{size}':raise ValueError('wrong transfer response range')
                elif r.status_code==200:offset=0
                else:raise ValueError('unexpected transfer HTTP status')
                with tmp.open('ab' if offset else 'wb') as f:
                    for block in r.iter_content(1024*1024):
                        if block:f.write(block)
            if tmp.stat().st_size!=size:raise ValueError('transfer download incomplete')
            if sha_file(tmp)!=expected:tmp.unlink();raise ValueError('transfer asset hash mismatch')
            tmp.replace(path);print('TRANSFER FILE READY',path.name,size,flush=True)
            return {'file':item['path'],'sha256':expected,'bytes':size,'transport':domain}
        except Exception as e:
            last=e;print('transfer retry',path.name,attempt+1,str(e),flush=True);time.sleep(2)
    raise RuntimeError(f'transfer asset failed: {path.name}: {last}')


def prepare(root,task,files):
    from PIL import Image
    import pyarrow.parquet as pq
    spec=DATASETS[task];destination=root/'transfer'/task;destination.mkdir(parents=True,exist_ok=True)
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if content_sha({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:raise ValueError('changed transfer manifest')
        print('fixed transfer manifest reused',task,len(m['records']),flush=True);return
    selected=[x for x in files if (x['path'].startswith('DocVQA/validation-') if task=='docvqa' else x['path']=='mmstar.parquet')]
    if len(selected)!=(6 if task=='docvqa' else 1):raise ValueError('incomplete transfer shard listing')
    selected.sort(key=lambda x:x['path'])
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        results=list(executor.map(lambda item:download(spec,item,destination/'raw'/Path(item['path']).name),selected))
    records=[];images={};source_to_image={}
    for item in selected:
        file=destination/'raw'/Path(item['path']).name
        for batch in pq.ParquetFile(file).iter_batches(batch_size=16,use_threads=False):
            for row in batch.to_pylist():
                if not row['question'].strip():raise ValueError('empty transfer question')
                value=row.get('image',row.get('images'))
                if isinstance(value,list):value=value[0]
                if isinstance(value,dict):data=value['bytes']
                elif isinstance(value,str):data=base64.b64decode(value.split(',',1)[1] if value.startswith('data:') else value)
                else:data=value
                source_hash=hashlib.sha256(data).hexdigest()
                if source_hash in source_to_image:image_id=source_to_image[source_hash]
                else:
                    image=Image.open(io.BytesIO(data)).convert('RGB');image.load()
                    pixel=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    image_id=task+':'+pixel;source_to_image[source_hash]=image_id
                    if image_id not in images:
                        p=destination/'images'/f'{source_hash}.image';p.parent.mkdir(exist_ok=True)
                        if not p.exists():p.write_bytes(data)
                        images[image_id]={'path':str(p.relative_to(destination)),'sha256':source_hash,
                            'pixel_sha256':pixel,'width':image.width,'height':image.height}
                if task=='docvqa':
                    qid=str(row.get('questionId',row.get('question_id')))
                    answers=row['answers']
                    if isinstance(answers,dict):answers=answers['text']
                    if not answers or any(not isinstance(a,str) for a in answers):raise ValueError('invalid DocVQA references')
                    record={'question_id':qid,'image_id':image_id,'question':row['question'],'answers':answers,'target':None}
                else:
                    qid=str(row['index']);target=str(row['answer']).strip().upper()
                    if target not in 'ABCD' or len(target)!=1:raise ValueError('invalid choice target')
                    sub=row.get('l2_category',row.get('l2-category'))
                    if not sub:raise ValueError('missing MMStar subcategory')
                    record={'question_id':qid,'image_id':image_id,'question':row['question'].strip(),
                        'target':target,'subcategory':sub,'category':row['category'],
                        'prompt_suffix':"\nAnswer with the option's letter from the given choices directly"}
                record.update({'source_shard':item['path'],'source_row':len(records)});records.append(record)
                if len(records)%500==0:print('TRANSFER RECORDS',task,len(records),flush=True)
    if len(records)!=spec['questions'] or len({r['question_id'] for r in records})!=len(records):raise ValueError('not the full transfer benchmark')
    m={'schema_version':'rarevlm.transfer.v2','task':task,**spec,'files':results,
        'records':records,'images':images,'metric':'pinned lmms-eval ANLS' if task=='docvqa' else 'pinned MCQ extraction and macro L2-category accuracy',
        'pretraining_overlap':'unknown','selection':'all validation/benchmark questions; no score-dependent filtering'}
    m['manifest_sha256']=content_sha(m);write_json(manifest_path,m)
    print('TRANSFER MANIFEST READY',task,len(records),len(images),m['manifest_sha256'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    args=p.parse_args();jobs=[]
    for task in DATASETS:
        files=json.loads((args.asset_root/(task+'_files.json')).read_text());jobs.append((task,files))
    write_json(args.asset_root/'TRANSFER_SOURCE_LOCK.json',DATASETS)
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        futures=[executor.submit(prepare,args.asset_root,task,files) for task,files in jobs]
        for f in futures:f.result()


if __name__=='__main__':main()
