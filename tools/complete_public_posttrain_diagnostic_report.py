"""CPU report recovery: adapt the caller's --seed 42 to frozen --seed42 CLI.

All science/config/loss/gates/checkpoints and predictions remain byte-identical.
GPU queues are replaced only by strict checks that every marker already exists.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import content_sha,lock_file,now,sha_file
from vlm_alignment_signal.common import write_json
import vlm_posttrain_study.pipeline as pipeline
from vlm_posttrain_study.learning import build

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg_path=args.config.resolve();cfg=json.loads(cfg_path.read_text());root=Path(cfg['output_root'])
    if (root/'COMPLETION.json').exists():
        print('Already completed; no recovery work requested.');return
    started=time.monotonic();old_status=json.loads((root/'STATUS.json').read_text())
    lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(cfg_path)!=lock['config_sha256']:raise ValueError('Original config changed')
    protected={str(SOURCE_ROOT/name):digest for name,digest in lock['sources'].items()}
    protected.update(lock['data_sha256'])
    protected.update(json.loads((root/'INITIALIZATION_CV_GUARD.json').read_text()))
    for condition in ['initial','sft','dpo']:
        for task in ['dev','pope','heldout']:
            directory=root/'runs'/condition/'seed42'/task
            for name in ['predictions.jsonl','EVALUATION.json','INPUT_LOCK.json']:
                file=directory/name;protected[str(file)]=sha_file(file)
        if condition!='initial':
            directory=root/'runs'/condition/'seed42/train'
            for file in [directory/'TRAIN_RESULT.json',*(directory/'adapter').glob('*')]:
                if file.is_file():protected[str(file)]=sha_file(file)
    for file,digest in protected.items():
        if sha_file(Path(file))!=digest:raise ValueError('Pre-recovery original changed:'+file)
    binding={'helper_sha256':sha_file(Path(__file__)),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
             'protected_sha256':protected,'original_failed_status':old_status,
             'recovery':'CLI-only --seed 42 to --seed42;completed queues must never launch workers'}
    lock_file(root/'REPORT_RECOVERY_INPUT_LOCK.json',binding)
    original_run=subprocess.run
    def cpu_only_run(command,*positional,**options):
        command=list(command)
        if '-m' not in command:raise ValueError('Recovery permits CPU analysis modules only')
        module=command[command.index('-m')+1]
        if module not in ['vlm_posttrain_eval.analysis','vlm_posttrain_pope.analysis']:
            raise ValueError('Refusing non-analysis worker during recovery:'+module)
        if module=='vlm_posttrain_pope.analysis':
            index=command.index('--seed')
            if command[index+1]!='42':raise ValueError('Frozen analysis is seed42 only')
            command[index:index+2]=['--seed42']
        return original_run(command,*positional,**options)
    def completed_queue_only(directory,jobs,identity):
        state=json.loads((directory/'STATUS.json').read_text())
        saved=json.loads((directory/'QUEUE_INPUT_LOCK.json').read_text())
        if state['status']!='completed' or saved['identity']!=identity or saved['jobs']!=jobs:
            raise ValueError('Recovery cannot schedule or change queue jobs')
        for job in jobs:
            record=state['jobs'][job['key']]
            marker=Path(job['completion_marker'])
            if record['state']!='completed' or record.get('exit_code')!=0 or not marker.exists():
                raise ValueError('GPU job not actually completed:'+job['key'])
            if json.loads(marker.read_text()).get('status') not in ['completed','passed']:
                raise ValueError('GPU marker not valid:'+str(marker))
    # Frozen pipeline supplies all original analysis and final aggregation logic.
    # Only CLI spelling and fully completed queue reuse are adapted here.
    subprocess.run=cpu_only_run;pipeline.run_queue=completed_queue_only
    sys.argv=[str(Path(pipeline.__file__)),'--config',str(cfg_path)]
    try:pipeline.main()
    finally:subprocess.run=original_run
    for file,digest in protected.items():
        if sha_file(Path(file))!=digest:raise ValueError('Recovery modified protected science/predictions:'+file)
    result_path=root/'analysis/ANALYSIS.json';result=json.loads(result_path.read_text())
    if result['formal_optimizer_updates']!=446 or result['discarded_engineering_optimizer_updates']!=4:
        raise ValueError('Recovery unexpectedly changed optimizer update accounting')
    launch=json.loads((root/'LAUNCH.json').read_text())
    last_compute=max(json.loads((root/'branches'/f'{c}_seed42/COMPLETE.json').read_text())['completed_at'] for c in ['initial','sft','dpo'])
    recovery={'status':'completed','recovered_at':now(),'helper_sha256':sha_file(Path(__file__)),
              'recovery_cpu_session_seconds':time.monotonic()-started,'science_and_prediction_files_unchanged':len(protected),
              'extra_training_updates':0,'extra_gpu_predictions':0,'source_config_loss_gates_unchanged':True,
              'original_error':'caller used --seed 42 but frozen POPE CLI declares --seed42',
              'recovery_input_lock_sha256':sha_file(root/'REPORT_RECOVERY_INPUT_LOCK.json')}
    result.update(first_launch_at=launch['launched_at'],last_gpu_work_completed_at=last_compute,
                  gpu_workflow_wall_seconds=(datetime.fromisoformat(last_compute)-datetime.fromisoformat(launch['launched_at'])).total_seconds(),
                  report_recovery=recovery)
    write_json(result_path,result);write_json(PROJECT_ROOT/'results/vlm_public_posttrain_diagnostic_v1_metrics.json',result)
    build(cfg,result)
    completion=json.loads((root/'COMPLETION.json').read_text());completion.update(analysis_sha256=sha_file(result_path),report_recovery=recovery)
    write_json(root/'COMPLETION.json',completion);write_json(root/'REPORT_RECOVERY.json',recovery)
    print(json.dumps({'status':'completed','recovery':recovery,'gpu_workflow_wall_seconds':result['gpu_workflow_wall_seconds']},ensure_ascii=False))

if __name__=='__main__':main()
