"""Pinned public train-only data retrieval; no model or training."""
import hashlib
from pathlib import Path
import time
import requests
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json,sha_file

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_stvqa_source_v1'
REPO='HuggingFaceM4/the_cauldron'
REV='847a98a779b1652d65111daf20c972dfcd333605'
FILES=[('st_vqa/train-00000-of-00002-8f04db983951cc0f.parquet',359226537,'a7a8dbbe2d4238ec3cb2419189b2ad49dd954d1c725e8acc6d30a05bd681957c'),
       ('st_vqa/train-00001-of-00002-9f9c1472593ff618.parquet',361236353,'f85d7305b013e2ed0e407abba7f07eafcd2b3de0d8626fce2573ab32292b4c7f')]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'DOWNLOAD_COMPLETE.json').exists():raise FileExistsError('Source download already complete')
    protocol=PROJECT_ROOT/'docs/VLM_STVQA_SOURCE_V1_PROTOCOL.md'
    write_json(ROOT/'SOURCE_PLAN.json',{'repo':REPO,'revision':REV,'split':'train','subset':'st_vqa',
        'files':[{'path':p,'bytes':n,'sha256':h} for p,n,h in FILES],
        'source_files':{str(p):sha_file(p) for p in (Path(__file__),protocol)},'model_calls':0,'optimizer_updates':0})
    session=requests.Session();records=[]
    for index,(relative,expected,digest) in enumerate(FILES):
        dest=ROOT/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size==expected and sha_file(dest)==digest:
            records.append({'path':str(dest),'bytes':expected,'sha256':digest,'reused':True});continue
        partial=dest.with_suffix('.parquet.partial');start=time.monotonic();last=0
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{relative}?download=true'
        with session.get(url,stream=True,timeout=(25,45)) as response:
            response.raise_for_status();h=hashlib.sha256();count=0
            with partial.open('wb') as handle:
                for chunk in response.iter_content(2**20):
                    if not chunk:continue
                    count+=len(chunk)
                    if count>expected or time.monotonic()-start>300:raise RuntimeError('Download byte/time budget exceeded')
                    handle.write(chunk);h.update(chunk)
                    if count-last>=32*2**20:
                        write_json(ROOT/'DOWNLOAD_PROGRESS.json',{'status':'downloading_train_source','completed_files':index,
                            'file':relative,'downloaded_bytes':count,'expected_bytes':expected,
                            'active_seconds':time.monotonic()-start,'model_calls':0,'optimizer_updates':0});last=count
            if count!=expected or h.hexdigest()!=digest:raise ValueError('Source bytes or SHA mismatch')
        partial.replace(dest);records.append({'path':str(dest),'bytes':count,'sha256':digest,'active_seconds':time.monotonic()-start})
    readme=session.get(f'https://huggingface.co/datasets/{REPO}/raw/{REV}/README.md',timeout=30);readme.raise_for_status()
    (ROOT/'source/README.md').write_text(readme.text)
    write_json(ROOT/'DOWNLOAD_COMPLETE.json',{'status':'completed','repo':REPO,'revision':REV,'split':'train',
        'files':records,'readme_sha256':sha_file(ROOT/'source/README.md'),'model_calls':0,'optimizer_updates':0})
    print({'status':'download_complete','files':len(records),'bytes':sum(x['bytes'] for x in records)},flush=True)


if __name__=='__main__':main()
