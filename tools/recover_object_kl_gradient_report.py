#!/usr/bin/env python
"""CPU-only exact reproduction of the original two-thread gradient summary.

No scientific source, numeric tolerance, raw point, tensor, model, or queue is
changed. Original failure evidence and the separate thread study are retained.
"""
from __future__ import annotations

import os

# Fix the native thread pools before importing torch or any scientific module.
# Do not resize a live PyTorch/OpenMP pool: one earlier readonly attempt crashed.
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import time


def load_launcher():
    file = Path(__file__).resolve().parent/'run_object_kl_gradient_study.py'
    spec = importlib.util.spec_from_file_location('frozen_gradient_launcher_for_cpu_recovery', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preserve_bytes(source, destination, sha_file):
    value = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != value:
            raise ValueError(f'original recovery evidence already differs: {destination}')
    else:
        destination.write_bytes(value)
    if sha_file(source) != sha_file(destination):
        raise ValueError('original failure snapshot must retain exact bytes')


def recover(config_file):
    started = time.monotonic()
    import torch
    from vlm_probe.evaluate import now, read_json, sha_file, write_json

    if torch.get_num_threads() != 2 or torch.cuda.is_initialized():
        raise ValueError('recovery needs fresh two-thread CPU process and no CUDA context')
    launcher = load_launcher()
    config_file = launcher.absolute(config_file)
    config, root, lock = launcher.verify_lock(config_file)
    recovery_file = root/'RECOVERY.json'
    completion_file = root/'COMPLETION.json'
    if recovery_file.exists() or completion_file.exists():
        if not recovery_file.exists() or not completion_file.exists():
            raise ValueError('partial prior recovery needs independent inspection; no overwrite')
        previous, complete = read_json(recovery_file), read_json(completion_file)
        launcher.verify_files(previous['protected_files_sha256'])
        launcher.verify_files(complete['artifact_files_sha256'])
        if (previous['recovery_source_sha256'] != sha_file(Path(__file__).resolve())
            or previous['study_input_lock_sha256'] != sha_file(root/'STUDY_INPUT_LOCK.json')
            or previous['exact_frozen_aggregate_checks_passed'] is not True
            or complete['status'] != 'completed_diagnostic' or complete['goal_complete'] is not False):
            raise ValueError('completed CPU recovery identity changed')
        return complete

    original_status_file = root/'STATUS.json'
    original_status = read_json(original_status_file)
    if (original_status.get('status') != 'execution_or_engineering_failure'
        or original_status.get('error') != 'CPU independent norm/cosine/shadow recomputation differs'
        or original_status.get('scientific_failure') is not False
        or original_status.get('goal_complete') is not False):
        raise ValueError('this recovery is restricted to the recorded CPU exact-comparison failure')
    if (root/'ANALYSIS.json').exists():
        raise ValueError('refusing to overwrite an existing scientific analysis')
    launch_file = root/'LAUNCH.json'
    launch = read_json(launch_file)
    proc = Path('/proc')/str(launch['controller_pid'])/'stat'
    if proc.exists():
        raw = proc.read_text()
        if raw[raw.rfind(')')+2:].split()[19] == str(launch['process_starttime_ticks']):
            raise ValueError('original controller is still alive; no concurrent recovery')

    float_review_file = root/'independent_review/FLOAT_REDUCTION_REVIEW.json'
    float_review = read_json(float_review_file)
    if (float_review.get('schema_version') != 'rarevlm.object_kl.gradient.float_reduction_review.v1'
        or float_review.get('all_input_layout_identity_tensor_and_file_SHA_checks_passed') is not True
        or float_review.get('all_recomputed_leaves_exact_at_2_threads') is not True
        or float_review.get('all_thread_variants_within_rel1e_12_abs1e_15_without_structural_changes') is not True):
        raise ValueError('independent exact-two-thread diagnosis is required first')
    launcher.verify_files(float_review['frozen_artifact_files_SHA256_unchanged'])

    status_snapshot = root/'RECOVERY_ORIGINAL_STATUS.json'
    log_snapshot = root/'RECOVERY_ORIGINAL_CONTROLLER.log'
    preserve_bytes(original_status_file, status_snapshot, sha_file)
    preserve_bytes(root/'controller.log', log_snapshot, sha_file)
    protected = {**lock['source_sha256'], **lock['files_sha256']}
    group_receipts = []
    for group in launcher.GROUPS:
        file = root/'groups'/group/'COMPLETE.json'
        receipt = launcher.read_completion(file, config_file, root/'STUDY_INPUT_LOCK.json', config)
        protected.update(receipt['artifact_files_sha256'])
        protected[str(file)] = sha_file(file)
        group_receipts.append(file)
    for file in (root/'STUDY_INPUT_LOCK.json', root/'STUDY.json', launch_file,
                 root/'controller.log', status_snapshot, log_snapshot, float_review_file,
                 root/'queue/STATUS.json', root/'queue/QUEUE_INPUT_LOCK.json',
                 root/'queue/RESOURCE_SNAPSHOTS.jsonl', *(root/'queue/logs').glob('*.log')):
        protected[str(file)] = sha_file(file)
    queue = read_json(root/'queue/STATUS.json')
    if queue.get('status') != 'completed' or any(
        queue['jobs'][group].get('state') != 'completed' or queue['jobs'][group].get('exit_code') != 0
        for group in launcher.GROUPS):
        raise ValueError('both original GPU groups must have exited successfully')
    launcher.verify_files(protected)

    # This is the original CPU aggregator, including its exact POINT/RESULT
    # comparisons and all hashes/layout/parameter/teacher/input/UUID invariants.
    # Never call the original run(), run_queue(), or the GPU probe run().
    analysis = launcher.aggregate(config_file, root, config)
    if (torch.get_num_threads() != 2 or torch.cuda.is_initialized()
        or analysis['optimizer_updates'] != 0 or analysis['generation_questions'] != 0
        or analysis['performance_evaluation'] is not False or analysis['goal_complete'] is not False
        or [point['checkpoint_step'] for point in analysis['points']] != [25,100,175]):
        raise ValueError('exact CPU aggregate produced an unexpected diagnostic contract')
    launcher.verify_lock(config_file)
    launcher.verify_files(protected)

    analysis_file = root/'ANALYSIS.json'
    write_json(analysis_file, analysis)
    original_started = datetime.fromisoformat(launch['launched_at'])
    original_failed = datetime.fromisoformat(original_status['updated_at'])
    source_file = Path(__file__).resolve()
    recovery = {'schema_version':'rarevlm.object_kl.gradient.cpu_recovery.v1','status':'completed',
        'completed_at':now(),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'original_status_snapshot':str(status_snapshot),'original_status_sha256':sha_file(status_snapshot),
        'original_controller_log':str(root/'controller.log'),'original_controller_log_sha256':sha_file(root/'controller.log'),
        'original_controller_log_snapshot':str(log_snapshot),'original_controller_log_snapshot_sha256':sha_file(log_snapshot),
        'original_workflow_started_at':launch['launched_at'],'original_failure_at':original_status['updated_at'],
        'original_workflow_wall_seconds':(original_failed-original_started).total_seconds(),
        'original_workflow_wall_scope':'launch to original controller failure; includes queue/loading/GPU/CPU, not pure GPU time',
        'cpu_recovery_seconds':time.monotonic()-started,'cpu_recovery_scope':'successful independent CPU verification/aggregation only',
        'recovery_source_file':str(source_file),'recovery_source_sha256':sha_file(source_file),
        'native_thread_environment':{name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'actual_torch_num_threads':torch.get_num_threads(),
        'independent_float_review_file':str(float_review_file),'independent_float_review_sha256':sha_file(float_review_file),
        'exact_frozen_aggregate_checks_passed':True,'numeric_tolerance_added':False,
        'original_derived_POINT_and_RESULT_values_preserved':True,'frozen_globals_patched':False,
        'analysis_file':str(analysis_file),'analysis_sha256':sha_file(analysis_file),
        'protected_file_count':len(protected),'protected_files_sha256':dict(sorted(protected.items())),
        'protected_bytes_unchanged_before_after':True,
        'original_diagnostic_new_training_qa_forward_backward':96,
        'recovery_new_model_forwards':0,'recovery_new_training_qa_forward_backward':0,
        'recovery_gpu_operations':0,'recovery_new_predictions':0,'recovery_optimizer_updates':0,
        'prior_readonly_inspection_attempt':{'pid':420400,'exit_code':139,
            'error':'UCX signal11 while varying live CPU thread pools; no recovery file or model operation was produced'},
        'performance_evaluation':False,'goal_complete':False}
    write_json(recovery_file,recovery)
    artifacts = {str(file):sha_file(file) for file in (analysis_file,recovery_file,status_snapshot,
        log_snapshot,float_review_file,*group_receipts)}
    complete = {'schema_version':'rarevlm.object_kl.gradient.trial.v1','status':'completed_diagnostic',
        'completed_at':now(),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'artifact_files_sha256':artifacts,'cpu_recovery_receipt_file':str(recovery_file),
        'cpu_recovery_receipt_sha256':sha_file(recovery_file),'original_failure_preserved':True,
        'original_workflow_wall_seconds':recovery['original_workflow_wall_seconds'],
        'cpu_recovery_seconds':recovery['cpu_recovery_seconds'],'optimizer_updates':0,'generation_questions':0,
        'performance_evaluation':False,'scientific_performance_success':False,'goal_complete':False,
        'automatic_runtime_fallback':False,'existing_tasks_untouched':True}
    launcher.verify_files(protected)
    write_json(completion_file,complete)
    write_json(original_status_file,complete)
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    value=recover(args.config)
    print(value['status'],value['cpu_recovery_receipt_file'],flush=True)


if __name__=='__main__':
    main()
