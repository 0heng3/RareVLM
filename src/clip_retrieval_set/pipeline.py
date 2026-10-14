"""Full six trainings -> seven dev rankings -> fixed gate -> optional test."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file, now, sha_file, write_json
from .analysis import analyze
from .prepare import prepare


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(args.config)
    source_names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'clip_retrieval_set').glob('*.py'))]
    source_names += ['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/analysis.py',
        'clip_retrieval/features.py','methods.py','research_runtime/queue.py']
    identity={'config_sha256':sha_file(args.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in source_names},
        'reused_data_lock_sha256':sha_file(root/'REUSED_DATA_LOCK.json'),
        'upstream_archive_sha256':sha_file(root/'upstream/SOURCES.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_SET_V1_PROTOCOL.md')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**extra):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,**extra})
    if (root/'COMPLETION.json').exists():
        status('completed',previous_completion=True);return
    try:
        status('training');jobs=[]
        for seed in cfg['training_seeds']:
            for condition in cfg['conditions']:
                jobs.append({'key':f'{condition}_seed{seed}','command':[sys.executable,'-m','clip_retrieval_set.train',
                    '--config',str(args.config),'--condition',condition,'--seed',str(seed)],
                    'completion_marker':str(root/'runs'/condition/f'seed{seed}'/'TRAIN_RESULT.json')})
        run_queue(root/'training_queue',jobs,identity)
        def evaluate_split(split):
            status('evaluating_'+split);jobs=[]
            for condition,seeds in [('frozen',[0]),*[(c,cfg['training_seeds']) for c in cfg['conditions']]]:
                for seed in seeds:
                    key='frozen' if condition=='frozen' else f'{condition}_seed{seed}'
                    jobs.append({'key':key,'command':[sys.executable,'-m','clip_retrieval_set.evaluate','--config',str(args.config),
                        '--condition',condition,'--seed',str(seed),'--split',split],
                        'completion_marker':str(root/'evaluation'/key/split/'EVALUATION.json')})
            run_queue(root/(split+'_queue'),jobs,identity);return analyze(args.config,split)
        dev=evaluate_split('dev');test=evaluate_split('test') if dev['gate']['passed'] else None
        status('building_learning',dev_gate=dev['gate']['passed'])
        from .learning import build
        build(args.config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,
            'test_state':'completed' if test is not None else 'not_run_dev_gate_failed','new_training_runs':6,
            'config_sha256':sha_file(args.config),'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json')}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',result['test_state'],dev['gate'],flush=True)
    except BaseException as error:
        status('failed',reason=repr(error));raise
    finally:owner.close()


if __name__=='__main__':main()
