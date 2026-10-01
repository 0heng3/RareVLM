"""Offline checks for delivered source tables, figures and file bindings."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    provenance=json.loads((ROOT/'results/provenance.json').read_text())
    for name,digest in provenance['public_csv_sha256'].items():assert sha(ROOT/'results'/name)==digest,name
    with (ROOT/'results/main_results.csv').open() as f:main=list(csv.DictReader(f))
    with (ROOT/'results/seed_results.csv').open() as f:seeds=list(csv.DictReader(f))
    assert len(main)==28 and len(seeds)==76
    readme=(ROOT/'README.md').read_text()
    labels={'imagenet_lt_b32':'ImageNet-LT · B/32 · fixed view','imagenet_lt_b16':'ImageNet-LT · B/16 · fixed view',
            'imagenet_lt_b32_online':'ImageNet-LT · B/32 · online views','inat2018_b32':'iNaturalist 2018 · B/32 · fixed view'}
    for setting,label in labels.items():
        values={r['method']:r for r in main if r['setting']==setting}
        pairs=[f"{float(values[m]['oa']):.2f} / {float(values[m]['few']):.2f}" for m in ('zero_shot','ce','la','ce_val_alpha')]
        expected='| '+label+' | '+' | '.join(pairs)+' |'
        assert expected in readme,f'homepage table differs from CSV: {setting}'
    for row in main:
        members=[r for r in seeds if (r['setting'],r['method'])==(row['setting'],row['method'])]
        assert len(members)==int(row['n_runs'])
        if row['method']!='zero_shot':assert {int(r['seed']) for r in members}=={0,42,200}
        for metric in ('oa','many','medium','few','macro_f1','nll','ece15'):
            values=[float(r[metric]) for r in members]
            assert all(math.isfinite(v) and (0<=v<=100 if metric!='nll' else v>=0) for v in values)
            assert math.isclose(statistics.mean(values),float(row[metric]),abs_tol=1e-8)
            if len(values)>1:assert math.isclose(statistics.stdev(values),float(row[metric+'_sd']),abs_tol=1e-8)
    figure=json.loads((ROOT/'assets/figure_provenance.json').read_text())
    assert figure['source_csv_sha256']==sha(ROOT/'results/seed_results.csv')
    for name,digest in figure['files'].items():assert sha(ROOT/'assets'/name)==digest,name
    for stem in ('overview','main_results'):
        with Image.open(ROOT/'assets'/f'{stem}.png') as im:
            assert im.width>=2000 and im.height>=1000
            if 'A' in im.getbands():assert im.getchannel('A').getextrema()==(255,255)
    lock=json.loads((ROOT/'results/research_code_lock.json').read_text())
    assert lock['new_research_experiments']==0
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest,f'frozen research module changed: {name}'
    for name,item in json.loads((ROOT/'SOURCE_ORIGINS.json').read_text())['files'].items():
        assert sha(ROOT/name)==item['portable_sha256'],name
    for setting,s in provenance['settings'].items():
        assert s['train_seeds']==[0,42,200] and s['prompt']=='a photo of a {}.'
        assert len(s['model_weight_sha256'])==64 and len(s['manifest_sha256'])==64
        assert set(s['ordered_split_records_sha256'])=={'train','val','test'}
    citation=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    assert citation['cff-version']=='1.2.0' and citation['type']=='software' and citation['authors']
    assert citation['repository-code']=='https://github.com/0heng3/RareVLM'
    for path in ROOT.rglob('*.md'):
        if any(p in ('.git','.tmp','.cache','artifacts','data','.venv') for p in path.relative_to(ROOT).parts):continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (path.parent/target.split('#')[0]).resolve().exists(),f'broken link: {path.name}: {target}'
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,f'package hash mismatch: {name}'
    print(f'PASS: {len(main)} aggregate rows, {len(seeds)} seed records, frozen modules, images, links and {len(manifest["files"])} package hashes')


if __name__=='__main__':check()
