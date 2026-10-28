"""Twelve cached-feature trainings; branch-specific dev-gated clean test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,now,sha_file,write_json
from .prepare import prepare,verify_guard
from .analysis import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);a=parser.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();prepare(a.config)
    names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'clip_retrieval_quality').glob('*.py'))]
    names += ['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/analysis.py',
        'clip_retrieval/features.py','clip_retrieval_set/evaluate.py','clip_retrieval_distill/loss.py',
        'methods.py','research_runtime/queue.py']
    identity={'config_sha256':sha_file(a.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
        'selection_policy_sha256':sha_file(root/'selection/POLICY_LOCK.json'),
        'baseline_bindings_sha256':sha_file(root/'REUSED_BASELINES.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_QUALITY_V1_PROTOCOL.md'),
        'cpu_contract_sha256':sha_file(root/'CPU_CONTRACTS.json'),
        'upstream_source_sha256':sha_file(root/'upstream/SOURCES.json')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,**kw})
    if (root/'COMPLETION.json').exists():status('completed',previous_completion=True);return
    try:
        status('training');jobs=[]
        for condition in cfg['conditions']:
            for seed in cfg['training_seeds']:
                jobs.append({'key':f'{condition}_seed{seed}',
                    'command':[sys.executable,'-m','clip_retrieval_quality.train','--config',str(a.config),
                        '--condition',condition,'--seed',str(seed)],
                    'completion_marker':str(root/'runs'/condition/f'seed{seed}'/'TRAIN_RESULT.json')})
        run_queue(root/'training_queue',jobs,identity)
        def score(split,branches):
            status('evaluating_'+split);jobs=[]
            for branch in branches:
                for method in ['random','score']:
                    condition=branch+'_'+method+'4'
                    for seed in cfg['training_seeds']:
                        key=f'{condition}_seed{seed}'
                        jobs.append({'key':key,'command':[sys.executable,'-m','clip_retrieval_quality.evaluate',
                            '--config',str(a.config),'--condition',condition,'--seed',str(seed),'--split',split],
                            'completion_marker':str(root/'evaluation'/key/split/'EVALUATION.json')})
            run_queue(root/(split+'_queue'),jobs,identity)
            return analyze(a.config,split,branches)
        dev=score('dev',['clean','noisy']);passed=dev['gate']['passed_branches']
        test=score('test',passed) if passed else None
        status('building_learning',passed_dev_branches=passed)
        from .learning import build
        build(a.config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,'new_training_runs':12,
            'new_dev_cells':12,'new_test_cells':6*len(passed),'test_state':'completed' if test else 'not_run_dev_gate_failed',
            'test_branches':passed,'controller_seconds':time.monotonic()-started,
            'protected_files_unchanged':verify_guard(root),'config_sha256':sha_file(a.config),
            'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json')}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'test_state':result['test_state']})
        status('completed',test_branches=passed,test_state=result['test_state'])
        print('COMPLETED',passed,result['test_state'],flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()


if __name__=='__main__':main()
