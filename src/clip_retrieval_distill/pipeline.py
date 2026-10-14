"""Three new trainings, reused dev controls, gated full historical test."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(args.config)
    names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'clip_retrieval_distill').glob('*.py'))]
    names+=['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/analysis.py','clip_retrieval/features.py',
        'clip_retrieval_set/evaluate.py','methods.py','research_runtime/queue.py']
    identity={'config_sha256':sha_file(args.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
        'baseline_bindings_sha256':sha_file(root/'REUSED_BASELINES.json'),'upstream_sha256':sha_file(root/'upstream/SOURCES.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,**kw})
    if (root/'COMPLETION.json').exists():status('completed',previous_completion=True);return
    try:
        status('training');jobs=[{'key':f'distill_seed{s}','command':[sys.executable,'-m','clip_retrieval_distill.train','--config',str(args.config),
            '--condition','distill','--seed',str(s)],'completion_marker':str(root/'runs/distill'/f'seed{s}'/'TRAIN_RESULT.json')} for s in cfg['training_seeds']]
        run_queue(root/'training_queue',jobs,identity)
        def score(split):
            status('evaluating_'+split);jobs=[]
            conditions=[('distill',cfg['training_seeds'])] if split=='dev' else [('frozen',[0]),*[(c,cfg['training_seeds']) for c in cfg['conditions']]]
            for condition,seeds in conditions:
                for seed in seeds:
                    key='frozen' if condition=='frozen' else f'{condition}_seed{seed}'
                    jobs.append({'key':key,'command':[sys.executable,'-m','clip_retrieval_distill.evaluate','--config',str(args.config),
                        '--condition',condition,'--seed',str(seed),'--split',split],
                        'completion_marker':str(root/'evaluation'/key/split/'EVALUATION.json')})
            run_queue(root/(split+'_queue'),jobs,identity);return analyze(args.config,split)
        dev=score('dev');test=score('test') if dev['gate']['passed'] else None
        status('building_learning',dev_gate=dev['gate']['passed'])
        from .learning import build
        build(args.config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,'new_training_runs':3,'reused_baseline_checkpoints':3,
            'new_dev_cells':3,'reused_dev_cells':4,'new_test_cells':7 if test else 0,
            'test_state':'completed' if test else 'not_run_dev_gate_failed','config_sha256':sha_file(args.config),
            'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json')}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',dev['gate'],result['test_state'],flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()


if __name__=='__main__':main()
