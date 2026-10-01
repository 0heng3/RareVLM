"""Download pinned public CLIP assets and optional CIFAR/category metadata."""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from project_paths import (B32_REVISION, B16_REVISION, CLIP_B32, CLIP_B16,
                           CIFAR100_ROOT, INAT2018_ROOT)

WEIGHTS = {'b32': 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f',
           'b16': 'ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0'}


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['b32','b16'])
    p.add_argument('--cifar100',action='store_true')
    p.add_argument('--inat-categories',action='store_true')
    args=p.parse_args()
    if not (args.model or args.cifar100 or args.inat_categories): p.error('choose at least one asset')
    if args.model:
        from huggingface_hub import snapshot_download
        patch='32' if args.model=='b32' else '16'
        directory=CLIP_B32 if args.model=='b32' else CLIP_B16
        revision=B32_REVISION if args.model=='b32' else B16_REVISION
        snapshot_download(repo_id=f'openai/clip-vit-base-patch{patch}',revision=revision,local_dir=str(directory),
            allow_patterns=['config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json',
                            'vocab.json','merges.txt','special_tokens_map.json','pytorch_model.bin'])
        actual=file_sha(directory/'pytorch_model.bin')
        if actual!=WEIGHTS[args.model]: raise ValueError('downloaded model weight hash mismatch')
        print('model ready:',directory,'SHA256:',actual)
    if args.cifar100:
        from torchvision.datasets import CIFAR100
        CIFAR100(str(CIFAR100_ROOT),train=True,download=True)
        CIFAR100(str(CIFAR100_ROOT),train=False,download=True)
        print('CIFAR-100 ready:',CIFAR100_ROOT)
    if args.inat_categories:
        url='https://ml-inat-competition-datasets.s3.amazonaws.com/2018/categories.json.tar.gz'
        with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
        if hashlib.sha256(data).hexdigest()!='2287211e8da582419dc2ca66736005fdb079e0a89eaf38936a441d8277a8162e':
            raise ValueError('official category archive hash mismatch')
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            categories=json.load(archive.extractfile('categories.json'))
        if sorted(x['id'] for x in categories)!=list(range(8142)): raise ValueError('invalid category IDs')
        if any(not x['name'] or x['name'].isnumeric() for x in categories): raise ValueError('obfuscated names')
        INAT2018_ROOT.mkdir(parents=True,exist_ok=True)
        (INAT2018_ROOT/'categories.json').write_text(json.dumps(categories,ensure_ascii=False)+'\n')
        print('official species names ready:',INAT2018_ROOT/'categories.json')


if __name__=='__main__': main()
