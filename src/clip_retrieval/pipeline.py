"""Full-data download -> frozen features ->6 matched trainings ->complete rankings."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,lock_file
from research_runtime.queue import run_queue
from .data import DEFAULT_CONFIG,manifest
from .features import prepare_index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root'])
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={str(p.relative_to(SOURCE_ROOT)):sha_file(p) for p in sorted((SOURCE_ROOT/'clip_retrieval').glob('*.py'))}
    sources['methods.py']=sha_file(SOURCE_ROOT/'methods.py');sources['research_runtime/queue.py']=sha_file(SOURCE_ROOT/'research_runtime/queue.py')
    identity={'config_sha256':sha_file(args.config),'source_hashes':sources,'data_source_lock_sha256':sha_file(root/'DATA_SOURCE_LOCK.json')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,**kw})
    status('waiting_full_dataset')
    while not (root/'ARCHIVES_READY.json').exists():
        data=json.loads((root/'DATA_STATUS.json').read_text())
        if data['status']=='failed':status('failed',reason='dataset_download_failed');raise RuntimeError('full dataset download failed,no subsetfallback')
        status('waiting_full_dataset');time.sleep(30)
    status('preparing_full_manifest');m=manifest(args.config);prepare_index(root)
    status('extracting_features',counts=m['counts'])
    jobs=[{'key':'clip_features','command':[sys.executable,'-m','clip_retrieval.features','--config',str(args.config)],'completion_marker':str(root/'FEATURE_STATUS.json')}]
    run_queue(root/'feature_queue',jobs,identity)
    status('training',counts=m['counts']);jobs=[]
    for seed in cfg['training_seeds']:
        for condition in cfg['conditions']:
            key=f'{condition}_seed{seed}'
            jobs.append({'key':key,'command':[sys.executable,'-m','clip_retrieval.train','--config',str(args.config),'--condition',condition,'--seed',str(seed)],
                'completion_marker':str(root/'runs'/condition/f'seed{seed}'/'TRAIN_RESULT.json')})
    run_queue(root/'training_queue',jobs,identity)
    status('evaluating_full_dev_and_test');jobs=[]
    for condition,seeds in [('frozen',[0]),*[(c,cfg['training_seeds']) for c in cfg['conditions']]]:
        for seed in seeds:
            key='frozen' if condition=='frozen' else f'{condition}_seed{seed}'
            jobs.append({'key':key,'command':[sys.executable,'-m','clip_retrieval.evaluate','--config',str(args.config),'--condition',condition,'--seed',str(seed)],
                'completion_marker':str(root/'evaluation'/key/'test/EVALUATION.json')})
    run_queue(root/'evaluation_queue',jobs,identity)
    status('analyzing');from .analysis import analyze
    result=analyze(args.config)
    from .learning import build
    build(root,result)
    status('completed',gates=result['gates']);print('RETRIEVAL_STUDY_COMPLETE',result['gates'],flush=True)


if __name__=='__main__':main()
