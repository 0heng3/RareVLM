"""Fetch only the pinned public BLIP artifacts; no credentials or model upgrade."""
import hashlib
import json
import time
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download


def bounded_weight(study,root,pin,item):
    name=item['rfilename'];destination=root/name
    if destination.exists():return destination
    size=item['size'];width=4*1024*1024;parts=root/'.range_weight';parts.mkdir(exist_ok=True)
    def fetch(index):
        begin=index*width;end=min(begin+width,size)-1;path=parts/f'{index:04d}.part'
        if path.exists() and path.stat().st_size==end-begin+1:return index
        for attempt in range(12):
            try:
                url=f'https://huggingface.co/{pin["repository"]}/resolve/{pin["revision"]}/{name}?download=true&rv_range={time.time_ns()}'
                request=urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}','User-Agent':'RareVLM-pinned-model'})
                with urllib.request.urlopen(request,timeout=40) as response:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or tuple(map(int,match.groups()))!=(begin,end,size):raise ValueError('Bounded range header differs')
                    data=response.read(end-begin+1)
                    if len(data)!=end-begin+1:raise ValueError('Incomplete bounded response')
                temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path);return index
            except Exception:
                if attempt==11:raise
                time.sleep(min(2*(attempt+1),10))
    total=(size+width-1)//width;done=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,i) for i in range(total)]):
            future.result();done+=1
            (study/'DOWNLOAD_STATUS.json').write_text(json.dumps({'status':'downloading','file':name,
                'transport':'4MiB bounded ranges,4threads','completed_chunks':done,'total_chunks':total})+'\n')
            if done%16==0:print('WEIGHT_RANGES',done,'/',total,flush=True)
    temp=destination.with_suffix('.bin.part')
    with temp.open('wb') as out:
        for i in range(total):out.write((parts/f'{i:04d}.part').read_bytes())
    h=hashlib.sha256()
    with temp.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    if temp.stat().st_size!=size or h.hexdigest()!=item['lfs']['sha256']:raise ValueError('Assembled model SHA failed')
    temp.replace(destination);return destination


def main():
    study=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/clip_retrieval_rerank_v1')
    pin=json.loads((study/'MODEL_PIN.json').read_text());root=Path(pin['model_root']);root.mkdir(parents=True,exist_ok=True)
    records={}
    for item in sorted(pin['files'],key=lambda v:v['rfilename']=='pytorch_model.bin'):
        name=item['rfilename']
        (study/'DOWNLOAD_STATUS.json').write_text(json.dumps({'status':'downloading','file':name})+'\n')
        for attempt in range(1,5):
            try:
                path=bounded_weight(study,root,pin,item) if name=='pytorch_model.bin' else Path(hf_hub_download(pin['repository'],name,revision=pin['revision'],local_dir=root,token=False))
                break
            except Exception as error:
                (study/'DOWNLOAD_STATUS.json').write_text(json.dumps({'status':'retrying','file':name,'attempt':attempt,'error':type(error).__name__})+'\n')
                print('RETRY',name,attempt,type(error).__name__,flush=True)
                if attempt==4:raise
                time.sleep(3)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        digest=h.hexdigest()
        if path.stat().st_size!=item['size']:raise ValueError('Model artifact size differs: '+name)
        if item.get('lfs') and digest!=item['lfs']['sha256']:raise ValueError('Model LFS SHA differs')
        records[name]={'sha256':digest,'size':path.stat().st_size}
        print('VERIFIED',name,path.stat().st_size,flush=True)
    record={'status':'completed','repository':pin['repository'],'revision':pin['revision'],'model_root':str(root),'files':records}
    (study/'MODEL_READY.json').write_text(json.dumps(record,indent=2)+'\n')
    (study/'DOWNLOAD_STATUS.json').write_text(json.dumps({'status':'completed','files':len(records)})+'\n')
    print('MODEL_READY',flush=True)


if __name__=='__main__':main()
