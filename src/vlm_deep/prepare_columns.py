"""Read full annotation columns by verified HTTP ranges, avoiding image payloads."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from .common import training_target,write_json
from .prepare import build
from .download import official_images
from vlm_sft.assets import DATASET_ID,DATASET_REVISION


class RangeReader(io.RawIOBase):
    def __init__(self,url,size):
        import requests
        self.url=url;self.size=size;self.position=0;self.http=requests.Session();self.ranges=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        if whence==0:self.position=offset
        elif whence==1:self.position+=offset
        elif whence==2:self.position=self.size+offset
        else:raise ValueError('bad seek')
        return self.position
    def read(self,n=-1):
        if n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if n<=0:return b''
        start=self.position;end=start+n-1;last=None
        for attempt in range(4):
            try:
                url=self.url+f'?download=true&rv_column={start}_{end}_{time.time_ns()}'
                with self.http.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,45)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                        raise ValueError('annotation range response incorrect')
                    data=r.content
                if len(data)!=n:raise ValueError('annotation range truncated')
                self.position+=n
                self.ranges.append({'start':start,'end':end,'sha256':hashlib.sha256(data).hexdigest()})
                return data
            except Exception as e:last=e;time.sleep(1+attempt)
        raise RuntimeError(f'column range failed {start}-{end}: {last}')
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def close(self):
        if not self.closed:self.http.close()
        super().close()


def extract(root,item):
    import pyarrow.parquet as pq
    name=Path(item['path']).name;cache=root/'column_annotations'/name.replace('.parquet','.json')
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():return json.loads(cache.read_text())
    url=f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{item["path"]}'
    with RangeReader(url,item['size']) as reader:
        file=pq.ParquetFile(reader)
        columns=['image_id','question_id','question','answers','image_width','image_height']
        table=file.read(columns=columns,use_threads=False);rows=table.to_pylist()
        if table.num_rows!=file.metadata.num_rows:raise ValueError('incomplete column rows')
        result={'source_file':item['path'],'revision':DATASET_REVISION,'lfs_sha256_advertised':item['lfs']['oid'],
            'entire_parquet_downloaded_and_sha_verified':False,
            'transport':'TLS HTTP ranges with exact Content-Range checks; annotation columns only',
            'ranges':reader.ranges,'rows':rows}
    write_json(cache,result);print('ANNOTATION COLUMN SHARD',name,len(rows),'bytes',sum(r['end']-r['start']+1 for r in result['ranges']),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    args=p.parse_args();root=args.asset_root;c=json.loads(args.config.read_text())
    items=json.loads((root/'hf_parquet_metadata.download').read_text())
    selected=[x for x in items if '/train-' in x['path'] or '/validation-' in x['path']]
    selected.sort(key=lambda x:x['path'])
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results=list(executor.map(lambda item:extract(root,item),selected))
    pools={'train':[],'validation':[]};provenance=[]
    for item in results:
        split='train' if '/train-' in item['source_file'] else 'validation'
        offset=len(pools[split])
        for i,r in enumerate(item['rows']):
            if len(r['answers'])!=10 or not r['question'].strip():raise ValueError('invalid full annotation')
            pools[split].append({'image_id':str(r['image_id']),'question_id':str(r['question_id']),
                'question':r['question'],'answers':r['answers'],'target':training_target(r['answers']),
                'annotation_width':r['image_width'],'annotation_height':r['image_height'],
                'source_split':split,'source_row':offset+i})
        provenance.append({'path':item['source_file'],**{k:v for k,v in item.items() if k!='rows'}})
    # Independent previously fetched viewer pages must agree on all used fields.
    lookup={split:{r['source_row']:r for r in rows} for split,rows in pools.items()};checked=0
    for page in (root/'annotations').glob('*.json'):
        d=json.loads(page.read_text());split=page.name.split('_')[0]
        for item in d['rows']:
            r=item['row'];canonical=lookup[split][item['row_idx']]
            for key in ['image_id','question_id','question','answers','annotation_width','annotation_height']:
                original=r[{'annotation_width':'image_width','annotation_height':'image_height'}.get(key,key)]
                if key in ['image_id','question_id']:original=str(original)
                if original!=canonical[key]:raise ValueError(f'viewer/Parquet annotation disagreement: {page.name}: {key}')
            checked+=1
    write_json(root/'ANNOTATIONS.json',{'splits':pools,'source_pages':provenance,
        'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'transport':'complete pinned Parquet annotation columns by HTTP range',
        'independent_cached_viewer_rows_verified':checked})
    archive,meta=official_images(root)
    build(root,args.pilot_root,c,pools,provenance,archive,meta)


if __name__=='__main__':main()
