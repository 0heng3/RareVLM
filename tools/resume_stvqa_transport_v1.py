"""Curl range recovery for fixed public assets; science inputs stay unchanged."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import requests
from fetch_stvqa_source_v1 import ROOT,FILES,REPO,REV
from vlm_deep.common import sha_file,write_json


def main():
    write_json(ROOT/'TRANSPORT_ATTEMPT1.json',{'status':'single_connection_time_budget_exceeded',
        'reason':'Reached300seconds on first shard; no model or optimizer started',
        'source_plan_preserved':True,'model_calls':0,'optimizer_updates':0})
    assets=[(ROOT/'source'/p,ROOT/'source'/p.replace('.parquet','.parquet.partial'),
        f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{p}?download=true',n,h) for p,n,h in FILES]
    assets.append((ROOT/'source/mmf_annotations.tar.gz',None,
        'https://dl.fbaipublicfiles.com/mmf/data/datasets/stvqa/defaults/annotations/annotations.tar.gz',
        46466749,'c2b1168542babbccb966bd2f33c7e10d2238aa27cfef17d8f447df96056afbe7'))
    write_json(ROOT/'TRANSPORT_PLAN.json',{'status':'frozen','connections_per_file':8,'max_seconds_per_range':180,
        'source_files_sha256':{str(Path(__file__)):sha_file(Path(__file__))},
        'assets':[{'path':str(p),'bytes':n,'sha256':h} for p,_,_,n,h in assets],
        'scope':'Transport recovery only; fixed data revisions, all science gates unchanged','model_calls':0})
    records=[]
    for index,(dest,prefix,url,expected,digest) in enumerate(assets):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size==expected and sha_file(dest)==digest:
            records.append({'path':str(dest),'bytes':expected,'sha256':digest,'reused':True});continue
        initial=prefix.stat().st_size if prefix and prefix.exists() else 0
        if initial>=expected:raise ValueError('Unexpected partial size')
        width=(expected-initial+7)//8;parts=[];started=time.monotonic()
        for a in range(initial,expected,width):
            b=min(expected-1,a+width-1);part=ROOT/'transport_parts'/f'{index}_{a}_{b}.bin';part.parent.mkdir(exist_ok=True)
            parts.append((a,b,part))
        def retrieve(item):
            a,b,path=item;separator='&' if '?' in url else '?';uri=url+separator+f'range_part={a}-{b}'
            log=path.with_suffix('.stderr');header=path.with_suffix('.headers')
            with log.open('w') as error:
                result=subprocess.run(['curl','--fail','--location','--silent','--show-error',
                    '--connect-timeout','20','--max-time','180','--max-filesize',str(b-a+1),
                    '--range',f'{a}-{b}','--dump-header',str(header),'--output',str(path),uri],stderr=error)
            if result.returncode or not path.exists() or path.stat().st_size!=b-a+1:
                raise RuntimeError(f'Range {a}:{b} failed; see {log.name}')
            headers=header.read_text().lower()
            if f'content-range: bytes {a}-{b}/{expected}' not in headers:raise ValueError('Unexpected range response')
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(retrieve,q) for q in parts]
            while not all(q.done() for q in futures):
                count=initial+sum(q.stat().st_size for _,_,q in parts if q.exists())
                write_json(ROOT/'TRANSPORT_PROGRESS.json',{'status':'range_download','asset':index,'completed_files':index,
                    'downloaded_bytes':count,'expected_bytes':expected,'active_seconds':time.monotonic()-started,
                    'connections':8,'model_calls':0,'optimizer_updates':0})
                time.sleep(2)
            for future in futures:future.result()
        assembled=dest.with_suffix(dest.suffix+'.reassembled')
        with assembled.open('wb') as output:
            if initial:
                with prefix.open('rb') as source:shutil.copyfileobj(source,output)
            for _,_,path in parts:
                with path.open('rb') as source:shutil.copyfileobj(source,output)
        if assembled.stat().st_size!=expected or sha_file(assembled)!=digest:raise ValueError('Full recovered SHA differs')
        assembled.replace(dest);records.append({'path':str(dest),'bytes':expected,'sha256':digest,'active_seconds':time.monotonic()-started})
        for _,_,path in parts:path.unlink()
    response=requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REV}/README.md',timeout=30);response.raise_for_status()
    (ROOT/'source/README.md').write_text(response.text)
    write_json(ROOT/'DOWNLOAD_COMPLETE.json',{'status':'completed','repo':REPO,'revision':REV,'split':'train',
        'files':records,'readme_sha256':sha_file(ROOT/'source/README.md'),'transport_recovery':True,
        'model_calls':0,'optimizer_updates':0})
    print({'status':'source_download_complete','files':len(records),'bytes':sum(q['bytes'] for q in records)},flush=True)


if __name__=='__main__':main()
