"""Finish presentation from already-native archived JSON; no new predictions."""
import fcntl
import json
import os
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import sha_file,now
from clip_retrieval_rerank.common import write_json
from clip_retrieval_rerank.prepare import verify_guard
from clip_retrieval_rerank.learning import build


def main():
    config=PROJECT_ROOT/'configs/clip_retrieval_rerank_v1.json';cfg=json.loads(config.read_text());root=Path(cfg['output_root'])
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        original=json.loads((root/'STATUS.json').read_text());lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
        if sha_file(config)!=lock['config_sha256'] or any(sha_file(SOURCE_ROOT/n)!=h for n,h in lock['sources'].items()):
            raise ValueError('Frozen source/config changed')
        if sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_RERANK_V1_PROTOCOL.md')!=lock['protocol_sha256']:
            raise ValueError('Frozen protocol changed')
        dev=json.loads((root/'analysis/dev/ANALYSIS.json').read_text())
        test=json.loads((root/'analysis/test/ANALYSIS.json').read_text()) if dev['gate']['passed'] else None
        write_json(root/'analysis/REPORT_EXPORT_RECOVERY.json',{'original_status':original,
            'recovery_source_sha256':sha_file(Path(__file__)),'created_at':now(),
            'reason':'in-memory numpy bool in learning HTML json.dumps; read archived native JSON instead',
            'new_model_predictions':0,'science_or_gates_changed':False})
        build(config,dev,test)
        result={'status':'completed','completed_at':now(),'dev':dev,'test':test,
            'new_training_runs':0,'bound_clip_checkpoints':3,'fixed_blip_checkpoints':1,
            'new_dev_reranked_cells':3,'reused_dev_baseline_cells':3,'new_test_reranked_cells':3 if test else 0,
            'test_state':'completed' if test else 'not_run_dev_gate_failed','config_sha256':sha_file(config),
            'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'report_recovery_source_sha256':sha_file(Path(__file__))}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'analysis/VERIFICATION.json',{'status':'passed','frozen_guard':verify_guard(root),
            'dev_baseline_and_pool_checks':dev['baseline_checks'],'test_baseline_and_pool_checks':test['baseline_checks'] if test else None,
            'scientific_sources_and_protocol':'unchanged','new_prediction_from_report_recovery':0})
        write_json(PROJECT_ROOT/'results/clip_retrieval_rerank_v1_metrics.json',dict(result,
            source_study='clip_retrieval_rerank_v1',source_analysis_sha256=sha_file(root/'analysis/ANALYSIS.json')))
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
            'dev_gate':dev['gate']['passed'],'test_state':result['test_state'],'report_recovery':True})
        write_json(root/'STATUS.json',{'status':'completed','updated_at':now(),'report_recovery_pid':os.getpid(),
            'existing_tasks_untouched':True,'dev_gate':dev['gate']['passed'],'test_state':result['test_state']})
        print('REPORT_COMPLETED','dev_gate',dev['gate']['passed'],'test_gate',test['gate']['passed'] if test else None,flush=True)
    finally:owner.close()


if __name__=='__main__':main()
