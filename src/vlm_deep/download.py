"""Bounded, resumable range download for the public official image archive."""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
import threading
import urllib.request
import zipfile
from pathlib import Path
from vlm_sft.common import sha_file,write_json

ZIP_URL='https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
_sessions=threading.local()


def get_bytes(url,attempts=4,timeout=35):
    last=None
    for attempt in range(attempts):
        try:
            import requests
            if not hasattr(_sessions,'http'):_sessions.http=requests.Session()
            with _sessions.http.get(url,timeout=(12,timeout)) as r:
                if r.status_code==429:
                    time.sleep(min(float(r.headers.get('Retry-After','15')),60))
                r.raise_for_status();return r.content
        except Exception as e:
            last=e
            if attempt+1<attempts:time.sleep(1+attempt)
    raise RuntimeError(f'download failed: {url}: {last}')


def official_images(root,workers=8):
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    archive=raw/'train_val_images.zip';record=raw/'OFFICIAL_IMAGES.json'
    if archive.exists() and record.exists():
        meta=json.loads(record.read_text())
        if sha_file(archive)!=meta['sha256']:raise ValueError('official archive changed')
        return archive,meta
    req=urllib.request.Request(ZIP_URL+'?rarevlm_v2=size',headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(req,timeout=35) as r:
        header=r.headers.get('Content-Range','')
        match=re.fullmatch(r'bytes \d+-\d+/(\d+)',header)
        if not match:raise ValueError('official server must provide range size')
        size=int(match.group(1))
    chunk=32*1024**2;parts=raw/'zip_parts';parts.mkdir(exist_ok=True)
    def fetch(start):
        end=min(start+chunk,size)-1;path=parts/f'{start:012d}.part';lock=path.with_suffix('.json')
        if path.exists() and lock.exists() and path.stat().st_size==end-start+1:
            if sha_file(path)==json.loads(lock.read_text())['sha256']:return path
        error=None
        for attempt in range(5):
            tmp=path.with_suffix('.tmp')
            try:
                url=ZIP_URL+f'?rarevlm_v2=part_{start}_{end}_{attempt}'
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r,tmp.open('wb') as f:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('server returned the wrong range')
                    while True:
                        b=r.read(1024*1024)
                        if not b:break
                        f.write(b)
                if tmp.stat().st_size!=end-start+1:raise ValueError('truncated archive part')
                tmp.replace(path);write_json(lock,{'start':start,'end':end,'sha256':sha_file(path)})
                return path
            except Exception as e:error=e;time.sleep(1+attempt)
        raise RuntimeError(f'archive range {start}-{end} failed: {error}')
    starts=list(range(0,size,chunk));done=0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch,s) for s in starts]):
            future.result();done+=1
            if done%8==0 or done==len(starts):print('official image archive parts',done,'/',len(starts),flush=True)
    tmp=archive.with_suffix('.tmp')
    with tmp.open('wb') as f:
        for s in starts:
            with (parts/f'{s:012d}.part').open('rb') as p:
                while True:
                    b=p.read(8*1024**2)
                    if not b:break
                    f.write(b)
    if tmp.stat().st_size!=size:raise ValueError('wrong assembled archive size')
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'archive CRC failure: {bad}')
    tmp.replace(archive);meta={'url':ZIP_URL,'bytes':size,'sha256':sha_file(archive),'all_member_crc_verified':True}
    write_json(record,meta);return archive,meta
