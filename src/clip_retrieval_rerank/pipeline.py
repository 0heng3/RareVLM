"""Bound baseline -> exact candidates -> native preflight -> complete gated rerank."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from rarevlm_paths import SOURCE_ROOT,PROJECT_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,sha_file,now
from .common import write_json
from .prepare import bind,inputs,verify_guard
from .analysis import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**extra):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),
        'controller_pid':os.getpid(),'existing_tasks_untouched':True,**extra})
    try:
        if (root/'COMPLETION.json').exists():verify_guard(root);status('completed');return
        status('binding_baselines');bind(args.config)
        source_names=[str(x.relative_to(SOURCE_ROOT)) for x in sorted((SOURCE_ROOT/'clip_retrieval_rerank').glob('*.py'))]
        source_names+=['clip_retrieval/train.py','clip_retrieval/evaluate.py','clip_retrieval/features.py',
                      'clip_retrieval/analysis.py','methods.py','research_runtime/queue.py']
        import transformers.models.blip.modeling_blip as blip
        import transformers.models.blip.modeling_blip_text as text
        import transformers.models.blip.image_processing_blip as image
        import transformers.models.blip.processing_blip as processor
        upstream={str(Path(inspect.getfile(m))):sha_file(Path(inspect.getfile(m))) for m in [blip,text,image,processor]}
        phase_a={'config_sha256':sha_file(args.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in source_names},
                 'baseline_bindings_sha256':sha_file(root/'REUSED_BASELINES.json'),'installed_transformers_sources':upstream}
        lock_file(root/'PREPARATION_INPUT_LOCK.json',phase_a)
        while not (root/'MODEL_READY.json').exists():
            status('waiting_pinned_model');time.sleep(15)
        ready=json.loads((root/'MODEL_READY.json').read_text())
        if ready['revision']!=cfg['model_revision'] or ready['files']['pytorch_model.bin']['sha256']!=cfg['model_weight_sha256']:
            raise ValueError('Model pin mismatch')
        for name,meta in ready['files'].items():
            if sha_file(Path(cfg['model_root'])/name)!=meta['sha256']:raise ValueError('Model artifact changed')
        def execute(split):
            split_start=time.monotonic();status('preparing_'+split);inputs(args.config,split)
            identity=dict(phase_a,model_ready_sha256=sha_file(root/'MODEL_READY.json'),split=split)
            jobs=[{'key':'candidates','command':[sys.executable,'-m','clip_retrieval_rerank.candidates',
                '--config',str(args.config),'--split',split],
                'completion_marker':str(root/'candidates'/split/'CANDIDATES.json')}]
            status('building_candidates_'+split);run_queue(root/(split+'_candidate_queue'),jobs,identity)
            preflight_jobs=[{'key':f'preflight_role{role}',
                'command':[sys.executable,'-m','clip_retrieval_rerank.matcher','--config',str(args.config),
                           '--split',split,'--role',str(role),'--mode','preflight'],
                'completion_marker':str(root/'matcher'/split/f'role{role}'/'PREFLIGHT.json')} for role in range(cfg['pair_scoring_roles'])]
            status('native_preflight_'+split);run_queue(root/(split+'_preflight_queue'),preflight_jobs,identity)
            reports=[json.loads((root/'matcher'/split/f'role{role}'/'PREFLIGHT.json').read_text()) for role in range(cfg['pair_scoring_roles'])]
            a=np.asarray(reports[0]['cached_logits']);b=np.asarray(reports[1]['cached_logits'])
            if reports[0]['probe_pair_keys']!=reports[1]['probe_pair_keys'] or np.abs(a-b).max()>cfg['native_cached_max_logit_abs']:
                raise ValueError('Cross-role matching preflight differs')
            write_json(root/f'{split.upper()}_PREFLIGHT_COMPLETE.json',{'status':'passed','reports':reports,
                'cross_role_max_logit_difference':float(np.abs(a-b).max()),'quality_reference_used':False})
            if split=='dev':
                main_lock=dict(phase_a,model_ready_sha256=sha_file(root/'MODEL_READY.json'),
                    protocol_sha256=sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_RERANK_V1_PROTOCOL.md'),
                    dev_preflight_sha256=sha_file(root/'DEV_PREFLIGHT_COMPLETE.json'))
                lock_file(root/'STUDY_INPUT_LOCK.json',main_lock)
            jobs=[{'key':f'score_role{role}',
                'command':[sys.executable,'-m','clip_retrieval_rerank.matcher','--config',str(args.config),
                    '--split',split,'--role',str(role),'--mode','score'],
                'completion_marker':str(root/'matcher'/split/f'role{role}'/'MATCH_COMPLETE.json')} for role in range(cfg['pair_scoring_roles'])]
            queue=root/(split+'_score_queue');queue.mkdir(exist_ok=True)
            state=queue/'STATUS.json'
            if not state.exists():
                write_json(state,{'jobs':{f'score_role{role}':{'state':'waiting_gpu','gpu_uuid':report['gpu_uuid'],
                    'contention_observations':0} for role,report in enumerate(reports)}})
            status('matching_'+split,logical_pairs=900000,unique_pairs=json.loads((root/'candidates'/split/'CANDIDATES.json').read_text())['unique_pairs'])
            run_queue(queue,jobs,identity)
            status('analyzing_'+split);result=analyze(args.config,split)
            write_json(root/(split+'_PHASE_TIMES.json'),{'split_pipeline_wall_seconds':time.monotonic()-split_start,
                'includes':'CPU prepare,baseline regenerate,resourcewait,native64pair preflight,cacheload,matching,sort,analysis',
                'scope':'offline fixed gallery pipeline; not online request latency'})
            return result
        dev=execute('dev');test=execute('test') if dev['gate']['passed'] else None
        status('building_learning')
        from .learning import build
        build(args.config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,
            'new_training_runs':0,'bound_clip_checkpoints':3,'fixed_blip_checkpoints':1,
            'new_dev_reranked_cells':3,'reused_dev_baseline_cells':3,
            'new_test_reranked_cells':3 if test else 0,'test_state':'completed' if test else 'not_run_dev_gate_failed',
            'config_sha256':sha_file(args.config),'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json')}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'analysis/VERIFICATION.json',{'status':'passed','frozen_guard':verify_guard(root),
            'dev_baseline_and_pool_checks':dev['baseline_checks'],'test_baseline_and_pool_checks':test['baseline_checks'] if test else None,
            'scientific_sources_unchanged':all(sha_file(SOURCE_ROOT/n)==digest for n,digest in phase_a['sources'].items())})
        public=dict(result,source_study='clip_retrieval_rerank_v1',source_analysis_sha256=sha_file(root/'analysis/ANALYSIS.json'))
        write_json(PROJECT_ROOT/'results/clip_retrieval_rerank_v1_metrics.json',public)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',dev['gate'],result['test_state'],flush=True)
    except BaseException as error:status('failed',reason=repr(error));raise
    finally:owner.close()


if __name__=='__main__':main()
