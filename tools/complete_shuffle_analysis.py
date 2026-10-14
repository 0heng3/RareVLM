"""Recover reporting only: dtype-aware independent nDCG check, locked science."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import numpy as np
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import now,sha_file,write_json
from research_runtime.queue import run_queue
from clip_retrieval_shuffle.prepare import verify_guard
from clip_retrieval_shuffle.learning import build
import clip_retrieval_shuffle.learning as learning_module
import clip_retrieval_shuffle.analysis as analysis_module


NUMERIC_CHECKS=[]


def native_json(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:native_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native_json(v) for v in value]
    return value


def learning_json_writer(path,value):
    # Presentation bookkeeping only; keep source files and all scientific values.
    return write_json(path,native_json(value))


def independent_numeric_rescore(path):
    errors={}
    with np.load(path/'per_query.npz') as z:
        for prefix,count in [('I2T',1000),('T2I',5000)]:
            ranking=z[prefix.lower()+'_top10']
            relevant=ranking//5==np.arange(count)[:,None] if prefix=='I2T' else ranking==(np.arange(count)//5)[:,None]
            for k in [1,5,10]:
                if not np.array_equal(relevant[:,:k].any(1).astype(np.float32),z[f'{prefix}_R@{k}']):
                    raise ValueError('Exact independent Recall check failed')
            discount=1/np.log2(np.arange(10,dtype=np.float64)+2)
            ideal=discount[:5].sum() if prefix=='I2T' else discount[0]
            expected=(relevant*discount).sum(1)/ideal
            recorded=z[prefix+'_nDCG@10']
            if recorded.dtype!=np.float32:raise ValueError('Unexpected stored metric dtype')
            # Float32 log/reciprocal/ten-term reductions/cumulative IDCG/division:
            # an explicit conservative 8*eps consistency bound, not a score gate.
            tolerance=8*np.finfo(np.float32).eps
            error=float(np.max(np.abs(expected-recorded)))
            if error>tolerance:raise ValueError('Independent nDCG exceeds FP32 arithmetic bound')
            errors[prefix]={'max_float64_vs_saved_float32_error':error,'absolute_dtype_consistency_bound':float(tolerance)}
    NUMERIC_CHECKS.append({'directory':str(path),'recall_exact':True,'ndcg':errors})
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root'])
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original_status=json.loads((root/'STATUS.json').read_text())
    identity=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    if identity['config_sha256']!=sha_file(args.config):raise ValueError('Frozen config changed')
    if any(sha_file(SOURCE_ROOT/name)!=digest for name,digest in identity['sources'].items()):
        raise ValueError('Frozen scientific source changed')
    if sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_SHUFFLE_V1_PROTOCOL.md')!=identity['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for name,key in [('DATA_MANIFEST.json','data_manifest_sha256'),('IMAGE_INDEX.json','image_index_sha256'),('REUSED_BASELINES.json','baseline_bindings_sha256')]:
        if sha_file(root/name)!=identity[key]:raise ValueError('Study input changed')
    def status(phase,**extra):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),
        'report_recovery_pid':os.getpid(),'existing_tasks_untouched':True,**extra})
    previous_audit_path=root/'analysis/REPORT_EXPORT_RECOVERY.json'
    previous_audit=json.loads(previous_audit_path.read_text()) if previous_audit_path.exists() else {}
    failures=list(previous_audit.get('failure_status_history',[]))
    if not failures and previous_audit.get('original_status'):failures.append(previous_audit['original_status'])
    if original_status not in failures:failures.append(original_status)
    audit={'started_at':now(),'original_status':previous_audit.get('original_status',original_status),
        'failure_status_history':failures,
        'original_controller_log_sha256':sha_file(root/'controller.log'),
        'recovery_source_sha256':sha_file(Path(__file__)),'study_source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'reason':'original independent verifier used1e-7 absolute bound; observed1.18834e-7 in float64-vs-CUDAFP32 nDCG',
        'change':'Only independent numerical consistency verifier overridden and NumPy scalar presentation JSON conversion; exact Recall,8float32eps nDCG bound; original analyze/statistics/gates/source untouched',
        'new_training':0,'new_dev_predictions':0,'quality_thresholds_changed':False}
    write_json(root/'analysis/REPORT_EXPORT_RECOVERY.json',audit)
    try:
        status('recovering_report')
        analysis_module.rescore_top10=independent_numeric_rescore
        learning_module.write_json=learning_json_writer
        dev=analysis_module.analyze(args.config,'dev');test=None
        print('RECOVERED_DEV',dev['comparisons'],dev['gate'],flush=True)
        if dev['gate']['passed']:
            status('evaluating_test')
            jobs=[{'key':f'negative_shuffle_seed{s}',
                'command':[sys.executable,'-m','clip_retrieval_shuffle.evaluate','--config',str(args.config),
                           '--condition','negative_shuffle','--seed',str(s),'--split','test'],
                'completion_marker':str(root/'evaluation'/f'negative_shuffle_seed{s}'/'test/EVALUATION.json')} for s in cfg['training_seeds']]
            run_queue(root/'test_queue',jobs,identity)
            test=analysis_module.analyze(args.config,'test')
        status('building_learning');build(args.config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,
            'new_training_runs':3,'reused_baseline_checkpoints':6,'new_dev_cells':3,'reused_dev_cells':7,
            'new_test_cells':3 if test else 0,'reused_test_cells':7 if test else 0,
            'test_state':'completed' if test else 'not_run_dev_gate_failed',
            'config_sha256':sha_file(args.config),'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
            'report_recovery_source_sha256':sha_file(Path(__file__))}
        guard=verify_guard(root)
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'analysis/VERIFICATION.json',{'status':'passed','completed_at':now(),
            'frozen_guard':guard,'training_contracts':dev['training_contracts'],
            'independent_top10_numeric_checks':NUMERIC_CHECKS,'scientific_sources_and_protocol':'unchanged',
            'recovery_source_sha256':sha_file(Path(__file__))})
        audit.update(status='completed',completed_at=now(),new_test_predictions=3 if test else 0,
                     numeric_checks=NUMERIC_CHECKS,frozen_guard=guard)
        write_json(root/'analysis/REPORT_EXPORT_RECOVERY.json',audit)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'dev_gate':dev['gate']['passed'],
            'test_state':result['test_state'],'report_recovery':True})
        public=dict(result);public.update(source_analysis_sha256=sha_file(root/'analysis/ANALYSIS.json'),source_study='clip_retrieval_shuffle_v1')
        write_json(PROJECT_ROOT/'results/clip_retrieval_shuffle_v1_metrics.json',public)
        status('completed',dev_gate=dev['gate']['passed'],test_state=result['test_state'],report_recovery=True)
        print('COMPLETED_REPORT',dev['gate'],result['test_state'],flush=True)
    except BaseException as error:
        status('failed',reason=repr(error),report_recovery=True);raise
    finally:owner.close()


if __name__=='__main__':main()
