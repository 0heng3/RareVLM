"""Explicit transport recovery:bounded HTTP ranges,original hashes/partial prefixes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from vlm_probe.evaluate import now,write_json,sha_file,lock_file,private_path


def one(root,source,item):
    relative=item['path'];destination=root/'archives'/relative;partial=destination.with_suffix('.tar.part')
    progress=root/'download_progress'/relative.replace('/','_');progress=progress.with_suffix('.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.stat().st_size!=item['size'] or sha_file(destination)!=item['sha256']:raise ValueError('existingverifiedarchive changed')
        return
    begun=time.monotonic();initial=partial.stat().st_size if partial.exists() else 0
    if initial>item['size']:raise ValueError('partial exceeds pinned size')
    while (partial.stat().st_size if partial.exists() else 0)<item['size']:
        start=partial.stat().st_size if partial.exists() else 0;end=min(start+4*1024*1024,item['size'])-1
        for retry in range(20):
            try:
                url=f'https://huggingface.co/datasets/{source["repository"]}/resolve/{source["revision"]}/{relative}?download=true&rv_range={time.time_ns()}'
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'RareVLM-public-benchmark'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or tuple(map(int,match.groups()))!=(start,end,item['size']):raise ValueError('bounded content-range differs')
                    data=response.read(end-start+1)
                    if len(data)!=end-start+1:raise ValueError('incompleteboundedbody')
                with partial.open('ab') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
                elapsed=time.monotonic()-begun;received=end+1
                write_json(progress,{'status':'downloading','path':relative,'bytes':received,'total':item['size'],
                    'transport':'bounded4MiBranges','updated_at':now(),'session_bytes_per_second':(received-initial)/max(elapsed,1e-8),
                    'retry':retry})
                if (received-initial)//(4*1024*1024)%16==0:print('RANGE_PROGRESS',relative,received,'/',item['size'],flush=True)
                break
            except Exception as error:
                write_json(progress,{'status':'retrying','path':relative,'bytes':start,'total':item['size'],
                    'transport':'bounded4MiBranges','retry':retry,'error_type':type(error).__name__,'updated_at':now()})
                if retry==19:raise RuntimeError(f'boundedrangefailed:{relative}:{type(error).__name__}') from None
                time.sleep(min(2*(retry+1),20))
    if sha_file(partial)!=item['sha256']:raise ValueError(f'fullarchiveSHAfailed:{relative};prefixpreservedforaudit')
    partial.replace(destination);write_json(progress,{'status':'completed','path':relative,'bytes':item['size'],'total':item['size'],
        'sha256':item['sha256'],'transport':'bounded4MiBranges','completed_at':now()});print('VERIFIED_RECOVERED',relative,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=private_path(args.root)
    owner=(root/'DOWNLOAD.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads((root/'DATA_SOURCE_LOCK.json').read_text())
    lock_file(root/'DOWNLOAD_V2_INPUT_LOCK.json',{'source_sha256':sha_file(Path(__file__)),
        'data_source_lock_sha256':sha_file(root/'DATA_SOURCE_LOCK.json'),'transport':'4MiBbounded ranges;20 retries perrange',
        'scientific_protocol_changed':False,'full_archive_SHA_required':True})
    write_json(root/'DATA_STATUS.json',{'status':'downloading','updated_at':now(),'transport':'bounded_ranges_v2','total_bytes':source['total_archive_bytes']})
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one,root,source,item) for item in source['files']]
            for future in futures:future.result()
        write_json(root/'ARCHIVES_READY.json',{'status':'passed','completed_at':now(),'metadata_sha256':sha_file(root/'DATA_SOURCE_LOCK.json'),
            'files':source['files'],'transport_source_sha256':sha_file(Path(__file__))})
        write_json(root/'DATA_STATUS.json',{'status':'archives_completed','updated_at':now(),'transport':'bounded_ranges_v2'})
    except Exception as error:
        write_json(root/'DATA_STATUS.json',{'status':'failed','updated_at':now(),'error':str(error),'transport':'bounded_ranges_v2'});raise


if __name__=='__main__':main()
