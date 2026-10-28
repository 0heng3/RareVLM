import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,now,sha_file
from .common import write_json
from .prepare import prepare
from .analysis import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();prepare(a.config)
    names=[str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'vlm_alignment_signal').glob('*.py'))]
    names+=['vlm_deep/common.py','vlm_deep/model.py','vlm_deep/third_party/vqa_eval_metric.py',
        'vlm_sft/common.py','vlm_probe/evaluate.py','vlm_precision/evaluate.py','vlm_joint/evaluate.py','research_runtime/queue.py']
    identity={'config_sha256':sha_file(a.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
        'data_bindings_sha256':sha_file(root/'DATA_BINDINGS.json'),
        'cpu_contract_sha256':sha_file(root/'CPU_CONTRACTS.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/VLM_ALIGNMENT_SIGNAL_V1_PROTOCOL.md'),
        'upstream_sha256':sha_file(root/'upstream/SOURCES.json')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,**kw})
    if (root/'COMPLETION.json').exists():status('completed',previous_completion=True);return
    try:
        status('sampling');jobs=[{'key':f'role{r}','command':[sys.executable,'-m','vlm_alignment_signal.generate',
            '--config',str(a.config),'--role',str(r)],'completion_marker':str(root/'roles'/f'role{r}'/'COMPLETE.json')} for r in cfg['roles']]
        run_queue(root/'sampling_queue',jobs,identity)
        status('analyzing');result=analyze(a.config)
        status('building_learning')
        from .learning import build
        build(a.config,result)
        result['controller_seconds']=time.monotonic()-start;write_json(root/'analysis/ANALYSIS.json',result)
        write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'optimizer_updates':0,
            'dpo_supply_passed':result['gates']['dpo_supply_passed'],'grpo_supply_passed':result['gates']['grpo_supply_passed']})
        status('completed',gates=result['gates']);print('SIGNAL_AUDIT_COMPLETED',result['summary'],flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()

if __name__=='__main__':main()
