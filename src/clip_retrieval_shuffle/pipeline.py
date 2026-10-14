"""Three new controls, bound history, dev gate, optional test confirmation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file, now, sha_file, write_json
from .prepare import prepare, verify_guard
from .analysis import analyze


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text())
    root=Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+'); fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**extra):
        write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
                                      'existing_tasks_untouched':True,**extra})
    try:
        status('preparing'); prepare(args.config)
        names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'clip_retrieval_shuffle').glob('*.py'))]
        names+=['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/analysis.py',
                'clip_retrieval/features.py','clip_retrieval_set/evaluate.py','methods.py',
                'research_runtime/queue.py','vlm_joint/pipeline.py','vlm_precision/pipeline.py']
        identity={'config_sha256':sha_file(args.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
            'baseline_bindings_sha256':sha_file(root/'REUSED_BASELINES.json'),
            'data_manifest_sha256':sha_file(root/'DATA_MANIFEST.json'),
            'image_index_sha256':sha_file(root/'IMAGE_INDEX.json'),
            'contract_test_sha256':sha_file(PROJECT_ROOT/'tests/test_retrieval_shuffle_contract.py'),
            'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_SHUFFLE_V1_PROTOCOL.md')}
        lock_file(root/'STUDY_INPUT_LOCK.json',identity)
        if (root/'COMPLETION.json').exists():
            verify_guard(root); status('completed',previous_completion=True); return
        status('training')
        jobs=[{'key':f'negative_shuffle_seed{s}',
               'command':[sys.executable,'-m','clip_retrieval_shuffle.train','--config',str(args.config),'--condition','negative_shuffle','--seed',str(s)],
               'completion_marker':str(root/'runs/negative_shuffle'/f'seed{s}'/'TRAIN_RESULT.json')} for s in cfg['training_seeds']]
        run_queue(root/'training_queue',jobs,identity)
        def score(split):
            status('evaluating_'+split)
            jobs=[{'key':f'negative_shuffle_seed{s}',
                   'command':[sys.executable,'-m','clip_retrieval_shuffle.evaluate','--config',str(args.config),
                              '--condition','negative_shuffle','--seed',str(s),'--split',split],
                   'completion_marker':str(root/'evaluation'/f'negative_shuffle_seed{s}'/split/'EVALUATION.json')} for s in cfg['training_seeds']]
            run_queue(root/(split+'_queue'),jobs,identity)
            return analyze(args.config,split)
        dev=score('dev'); test=score('test') if dev['gate']['passed'] else None
        status('building_learning',dev_gate=dev['gate']['passed'])
        from .learning import build
        build(args.config,dev,test)
        if sha_file(root/'DATA_MANIFEST.json')!=identity['data_manifest_sha256'] or sha_file(root/'IMAGE_INDEX.json')!=identity['image_index_sha256']:
            raise ValueError('Data manifest/image index changed during the study')
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,
            'new_training_runs':3,'reused_baseline_checkpoints':6,'new_dev_cells':3,'reused_dev_cells':7,
            'new_test_cells':3 if test else 0,'reused_test_cells':7 if test else 0,
            'test_state':'completed' if test else 'not_run_dev_gate_failed',
            'config_sha256':sha_file(args.config),'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json')}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'analysis/VERIFICATION.json',{'status':'passed','completed_at':now(),
            'frozen_guard':verify_guard(root),'training_contracts':dev['training_contracts'],
            'independent_top10_rescore_dev':dev['verification'],
            'independent_top10_rescore_test':test['verification'] if test else None})
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',dev['gate'],result['test_state'],flush=True)
    except BaseException as error:
        status('failed',reason=repr(error)); raise
    finally:
        owner.close()


if __name__=='__main__':main()
