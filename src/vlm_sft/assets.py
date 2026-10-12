"""Pinned public assets with SHA checks; images/weights remain local artifacts."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .common import sha_file,write_json

MODEL_ID='Qwen/Qwen2.5-VL-3B-Instruct'
HF_MODEL_REVISION='66285546d2b821cf421d4f5eb2576359d3770cd3'
MS_MODEL_REVISION='1b5a7674d40a0ac9acef541a46fed12d710bd36b'
DATASET_ID='lmms-lab-encoder/textvqa'
DATASET_REVISION='9c0699cd19768ac5ab97568f6b3cbac4c0062884'
WEIGHTS={'model-00001-of-00002.safetensors':'41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8',
         'model-00002-of-00002.safetensors':'365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4'}
PARQUETS={'data/train-00000-of-00020.parquet':'1e0af509ea969d78ad8f243e0205800d1a82b5094e1c8063848ada49b6679d89',
          'data/train-00001-of-00020.parquet':'7f7f03af1de34ceb5dbb0f2e668334d80394f18e69b640a6aabcacc1e8efb04c',
          'data/validation-00000-of-00003.parquet':'491649b5b8add2c5bde13a1e36707002fcd11898699255d65e2e568ac4fbddff'}


def download(url,path,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha_file(path)!=expected:raise ValueError(f'cached asset hash mismatch: {path.name}')
        return {'file':path.name,'sha256':expected,'bytes':path.stat().st_size,'reused':True}
    temporary=path.with_suffix(path.suffix+'.download')
    last=None
    for attempt in range(3):
        try:
            # A nonce avoids reusing a stale signed redirect or cached partial response.
            request_url=url+('&' if '?' in url else '?')+f'rarevlm_download={time.time_ns()}'
            with urllib.request.urlopen(request_url,timeout=45) as response,temporary.open('wb') as stream:
                while True:
                    block=response.read(8*1024*1024)
                    if not block:break
                    stream.write(block)
            actual=sha_file(temporary)
            if actual!=expected:raise ValueError(f'download SHA mismatch: {path.name}: {actual}')
            temporary.replace(path)
            print('ready',path.name,path.stat().st_size,flush=True)
            return {'file':path.name,'sha256':actual,'bytes':path.stat().st_size,'reused':False}
        except Exception as error:
            last=error;print('retry',path.name,attempt+1,str(error),flush=True)
            time.sleep(2)
    raise RuntimeError(f'asset download failed: {path.name}: {last}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--model',action='store_true');p.add_argument('--data',action='store_true')
    args=p.parse_args()
    if not (args.model or args.data):p.error('choose model and/or data')
    jobs=[];model_metadata={}
    if args.model:
        url=f'https://modelscope.cn/api/v1/models/{MODEL_ID}/repo/files?Revision={MS_MODEL_REVISION}&Recursive=true'
        with urllib.request.urlopen(url,timeout=30) as r:metadata=json.load(r)
        selected=[f for f in metadata['Data']['Files'] if f['Name'] in set(WEIGHTS)|{'LICENSE','config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.json','generation_config.json','model.safetensors.index.json'}]
        if not all(any(f['Name']==name and f['Sha256']==digest for f in selected) for name,digest in WEIGHTS.items()):raise ValueError('model mirror weights differ from official pinned Hugging Face weights')
        for f in selected:
            params=urllib.parse.urlencode({'Revision':MS_MODEL_REVISION,'FilePath':f['Path']})
            jobs.append((f'https://modelscope.cn/api/v1/models/{MODEL_ID}/repo?{params}',args.output_root/'model'/f['Name'],f['Sha256']))
        model_metadata={'model_id':MODEL_ID,'official_hf_revision':HF_MODEL_REVISION,'download_transport':'Qwen official ModelScope repository','modelscope_revision':MS_MODEL_REVISION,'files':{f['Name']:f['Sha256'] for f in selected}}
    if args.data:
        for name,digest in PARQUETS.items():
            url=f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}?download=true'
            jobs.append((url,args.output_root/'raw_data'/Path(name).name,digest))
    args.output_root.mkdir(parents=True,exist_ok=True)
    if model_metadata:write_json(args.output_root/'model_provenance.json',model_metadata)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,*job) for job in jobs]
        results=[future.result() for future in futures]
    write_json(args.output_root/('model_downloads.json' if args.model and not args.data else 'asset_downloads.json'),results)
    if args.data:write_json(args.output_root/'data_provenance.json',{'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'parquets':PARQUETS,'pool':'first two training shards; first official-validation shard; fixed engineering subset'})
    print('all requested public assets verified',flush=True)


if __name__=='__main__':main()
