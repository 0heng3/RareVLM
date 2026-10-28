"""Full clean gallery; failed branch cannot access new test rankings."""
import argparse
import json
from pathlib import Path
from clip_retrieval_set.evaluate import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--condition',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','test'],required=True);a=p.parse_args()
    if a.split=='test':
        cfg=json.loads(a.config.read_text());dev=json.loads((Path(cfg['output_root'])/'analysis/dev/ANALYSIS.json').read_text())
        if a.condition.split('_')[0] not in dev['gate']['passed_branches']:raise ValueError('Branch dev gate failed; no test')
    evaluate(a.config,a.condition,a.seed,a.split)
