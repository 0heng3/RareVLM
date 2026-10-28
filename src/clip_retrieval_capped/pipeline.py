"""Six queued trainings, six dev cells, joint-gated historical test and review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,now,sha_file,write_json
from .prepare import prepare
from .analysis import analyze
from .learning import build


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=None
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'existing_tasks_untouched':True,**kw})
    try:
        status('preparing');prepare(args.config)
        names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'clip_retrieval_capped').glob('*.py'))]
        names+=['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/analysis.py','clip_retrieval/features.py',
                'clip_retrieval_set/evaluate.py','clip_retrieval_distill/loss.py','methods.py','research_runtime/queue.py']
        identity={'config_sha256':sha_file(args.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
            'baseline_bindings_sha256':sha_file(root/'REUSED_BASELINES.json'),
            'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_CAPPED_V1_PROTOCOL.md')}
        lock_file(root/'STUDY_INPUT_LOCK.json',identity)
        if (root/'COMPLETION.json').exists():status('completed',previous_completion=True);return
        build(args.config);status('training');jobs=[]
        for seed in cfg['training_seeds']:
            for condition in cfg['new_training_conditions']:
                jobs.append({'key':f'{condition}_seed{seed}',
                    'command':[sys.executable,'-m','clip_retrieval_capped.train','--config',str(args.config),'--condition',condition,'--seed',str(seed)],
                    'completion_marker':str(root/'runs'/condition/f'seed{seed}'/'TRAIN_RESULT.json')})
        run_queue(root/'training_queue',jobs,identity)
        def score(split):
            status('evaluating_'+split);jobs=[]
            for seed in cfg['training_seeds']:
                for condition in cfg['new_training_conditions']:
                    jobs.append({'key':f'{condition}_seed{seed}',
                        'command':[sys.executable,'-m','clip_retrieval_capped.evaluate','--config',str(args.config),'--condition',condition,'--seed',str(seed),'--split',split],
                        'completion_marker':str(root/'evaluation'/f'{condition}_seed{seed}'/split/'EVALUATION.json')})
            run_queue(root/(split+'_queue'),jobs,identity)
            return analyze(args.config,split)
        dev=score('dev');test=score('test') if dev['gate']['passed'] else None
        status('building_learning',dev_gate=dev['gate']['passed']);build(args.config,dev,test)
        for name,digest in identity['sources'].items():
            if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Frozen source changed:'+name)
        if sha_file(args.config)!=identity['config_sha256'] or sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_CAPPED_V1_PROTOCOL.md')!=identity['protocol_sha256']:raise ValueError('Frozen protocol changed')
        guard=json.loads((root/'FROZEN_GUARD.json').read_text())
        if any(sha_file(Path(p))!=digest for p,digest in guard['protected_sha256'].items()):raise ValueError('Historical source/result/resume changed')
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,'new_training_runs':6,
            'reused_baseline_checkpoints':3,'new_dev_cells':6,'reused_dev_cells':3,
            'new_test_cells':6 if test else 0,'reused_test_cells':3 if test else 0,
            'test_state':'completed' if test else 'not_run_dev_gate_failed',
            'config_sha256':sha_file(args.config),'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
            'verification':{'historical_scientific_and_resume_guard':'unchanged','protected_files':len(guard['protected_sha256']),
                'all_scored_top10_independently_rescored':True,'hard_quota_and_complete_exposure':'passed'}}
        write_json(root/'analysis/ANALYSIS.json',result)
        portable={**result,'source_analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'source_study':'clip_retrieval_capped_v1'}
        write_json(PROJECT_ROOT/'results/clip_retrieval_capped_v1_metrics.json',portable)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',dev['gate'],result['test_state'],flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()


if __name__=='__main__':main()
