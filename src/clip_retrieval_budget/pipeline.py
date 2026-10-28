"""Bound SHA folds/margins -> cal-only0/20/50 -> fresh matches -> fixed gate."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,sha_file,now
from clip_retrieval_rerank.common import write_json
from .prepare import bind,verify_guard
from .calibrate import calibrate,plan
from .analysis import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),'existing_tasks_untouched':True,**kw})
    try:
        status('binding_assets');bind(a.config)
        names=[str(x.relative_to(SOURCE_ROOT)) for x in sorted((SOURCE_ROOT/'clip_retrieval_budget').glob('*.py'))]
        names.extend(str(x.relative_to(SOURCE_ROOT)) for x in sorted((SOURCE_ROOT/'clip_retrieval_adaptive').glob('*.py')))
        names+=['clip_retrieval_rerank/common.py','clip_retrieval/train.py','clip_retrieval/analysis.py','clip_retrieval/features.py','methods.py','research_runtime/queue.py']
        import transformers.models.blip.modeling_blip as blip
        import transformers.models.blip.modeling_blip_text as text
        upstream={str(Path(inspect.getfile(m))):sha_file(Path(inspect.getfile(m))) for m in [blip,text]}
        identity={'config_sha256':sha_file(a.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
            'assets_sha256':sha_file(root/'REUSED_ASSETS.json'),'partition_sha256':sha_file(root/'PARTITION.json'),
            'protocol_sha256':sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md'),'installed_transformers_sources':upstream}
        lock_file(root/'STUDY_INPUT_LOCK.json',identity)
        if (root/'COMPLETION.json').exists():verify_guard(root);status('completed',previous_completion=True);return
        def execute(split):
            started=time.monotonic();status('confidence_'+split)
            jobs=[{'key':'confidence','command':[sys.executable,'-m','clip_retrieval_budget.confidence','--config',str(a.config),'--split',split],
                'completion_marker':str(root/'confidence'/split/'CONFIDENCE.json')}]
            if split=='test':run_queue(root/(split+'_confidence_queue'),jobs,identity)
            if split=='dev':status('calibrating');calibrate(a.config)
            status('planning_'+split);route=plan(a.config,split)
            run_identity={**identity,'policy_sha256':sha_file(root/'POLICY.json'),'route_plan_sha256':sha_file(root/'routes'/split/'PLAN.json'),'split':split}
            if split=='dev' and route['unique_pair_reduction']<cfg['gate']['unique_pair_reduction_min']:
                rejected={'status':'calibration_rejected_before_gpu','split':'dev','gate':{'passed':False,'checks':{'planned_pair_reduction':False}},'cost':route,'new_prediction_cells':0,'reason':'minimal calibration-feasible policy cannot meet fixed30%pair target'}
                write_json(root/'analysis/dev/ANALYSIS.json',rejected);return rejected
            status('matching_'+split,full_unique_pairs=route['full_unique_pairs'],adaptive_unique_pairs=route['adaptive_unique_pairs'])
            jobs=[{'key':f'match_role{role}','command':[sys.executable,'-m','clip_retrieval_budget.matcher','--config',str(a.config),'--split',split,'--role',str(role)],
                'completion_marker':str(root/'matcher'/split/f'role{role}'/'MATCH_COMPLETE.json')} for role in range(cfg['pair_scoring_roles'])]
            run_queue(root/(split+'_match_queue'),jobs,run_identity)
            status('analyzing_'+split);result=analyze(a.config,split)
            write_json(root/(split+'_PHASE_TIMES.json'),{'split_wall_seconds':time.monotonic()-started,
                'scope':'confidence+queue+cal(dev)+route+model/vision/nativepreflight+fresh full/adaptive+3timing repeats+analysis;not per-request latency'})
            return result
        dev=execute('dev');test=execute('test') if dev['gate']['passed'] else None
        status('building_learning')
        from .learning import build
        build(a.config,dev,test)
        if any(sha_file(SOURCE_ROOT/n)!=digest for n,digest in identity['sources'].items()):raise ValueError('Frozen study source changed')
        if any(sha_file(Path(p))!=digest for p,digest in upstream.items()):raise ValueError('Installed matching source changed')
        if sha_file(a.config)!=identity['config_sha256'] or sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md')!=identity['protocol_sha256']:raise ValueError('Protocol changed')
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,'new_training_runs':0,
            'bound_clip_checkpoints':3,'fixed_blip_checkpoints':1,'new_dev_cells':dev['new_prediction_cells'],'new_test_cells':6 if test else 0,
            'test_state':'completed' if test else ('not_run_calibration_rejected' if dev['status']=='calibration_rejected_before_gpu' else 'not_run_dev_gate_failed'),'config_sha256':sha_file(a.config),
            'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'policy_sha256':sha_file(root/'POLICY.json'),'frozen_guard':verify_guard(root)}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(PROJECT_ROOT/'results/clip_retrieval_budget_v1_metrics.json',{**result,
            'source_analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'source_study':'clip_retrieval_budget_v1'})
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'])
        print('COMPLETED',dev['gate'],result['test_state'],flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()


if __name__=='__main__':main()
