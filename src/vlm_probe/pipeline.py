"""Durable two-GPU fixed-checkpoint visual-budget diagnostic controller."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = PROJECT_ROOT


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=REPO/'configs/vlm_budget_probe_v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop_requested)
    budgets = config['visual_token_budgets']
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    owner = (root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locked = {
        'schema_version': 'rarevlm.visual_budget_pipeline.v1',
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'asset_root': str(args.asset_root.resolve()),
        'model_root': str(args.model_root.resolve()),
        'roles': config['roles'], 'budgets': budgets,
        'batch_size': config['batch_size'], 'gpu_assignment': {'base': 0, 'adapted': 1},
        'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {'rarevlm_paths.py': hashlib.sha256((SOURCE_ROOT/'rarevlm_paths.py').read_bytes()).hexdigest()},
    }
    lock_path = root/'PIPELINE_INPUT_LOCK.json'
    prior = read_json(lock_path)
    if prior is not None and prior != locked:
        raise ValueError('pipeline configuration/source changed; use a new output root')
    write_json(lock_path, locked)
    started = now()
    write_json(root/'PID.json', {'pid': os.getpid(), 'started_at': started, 'python': sys.executable})

    def status(phase, processes=None, error=None):
        cells = []
        for role in config['roles']:
            for budget in budgets:
                directory = root/'runs'/role/f'budget_{budget}'
                evaluation = read_json(directory/'EVALUATION.json')
                progress = read_json(directory/'PROGRESS.json')
                cells.append({'role': role, 'visual_token_budget': budget,
                              'complete': evaluation is not None,
                              'score_percent': evaluation.get('score_percent') if evaluation else None,
                              'progress': progress})
        data = {'schema_version': 'rarevlm.visual_budget_status.v1',
                'phase': phase, 'updated_at': now(), 'started_at': started,
                'pid': os.getpid(), 'completed_cells': sum(c['complete'] for c in cells),
                'total_cells': len(cells), 'cells': cells,
                'role_status': {role: read_json(root/'runs'/role/'STATUS.json') for role in config['roles']},
                'scope': config['scope'], 'config_sha256': locked['config_sha256']}
        if processes:
            data['workers'] = {role: {'pid': proc.pid, 'returncode': proc.poll()}
                               for role, proc in processes.items()}
        if error:
            data['error'] = error
        write_json(root/'STATUS.json', data)

    def launch(phase, preflight=False):
        processes = {}
        logs = []
        for gpu, role in enumerate(config['roles']):
            cmd = [sys.executable, '-m', 'vlm_probe.evaluate', '--asset-root', str(args.asset_root),
                   '--model-root', str(args.model_root), '--output-root', str(root),
                   '--config', str(args.config), '--role', role, '--batch-size', str(config['batch_size']),
                   '--budgets', *map(str, budgets)]
            if role == 'adapted':
                cmd += ['--adapter-dir', config['fixed_adapter_dir']]
            if preflight:
                cmd += ['--preflight-only']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
            env.pop('RANK', None)
            env.pop('WORLD_SIZE', None)
            env.pop('LOCAL_RANK', None)
            log = (root/'logs'/f'{role}_{phase}.log').open('a')
            logs.append(log)
            processes[role] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                status(phase, processes)
                failed = [role for role, proc in processes.items() if proc.poll() not in (None, 0)]
                if failed:
                    for proc in processes.values():
                        if proc.poll() is None:
                            proc.terminate()
                    for proc in processes.values():
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    message = f'{phase} worker failed: {failed}; inspect logs; numerical settings were not changed'
                    status('failed', processes, error=message)
                    raise RuntimeError(message)
                if all(proc.poll() == 0 for proc in processes.values()):
                    status(phase + '_complete', processes)
                    break
                time.sleep(15)
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes.values():
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            for log in logs:
                log.close()

    try:
        status('preflight')
        launch('preflight', preflight=True)
        launch('evaluation')
        status('analysis')
        analysis_script = REPO/'tools/analyze_vlm_budget_probe.py'
        if not analysis_script.exists():
            raise FileNotFoundError('analysis script unavailable; six cells retained for recovery')
        with (root/'logs/analysis.log').open('a') as log:
            subprocess.run([sys.executable, str(analysis_script), '--asset-root', str(args.asset_root),
                            '--probe-root', str(root)], cwd=REPO, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = read_json(root/'analysis/ANALYSIS.json')
        if result is None or not result.get('complete', False):
            raise ValueError('analysis did not certify all six complete cells')
        status('complete')
        print('BUDGET PROBE COMPLETE', now(), flush=True)
    except KeyboardInterrupt as exc:
        status('interrupted', error=f'{type(exc).__name__}: {exc}; completed prediction batches can be resumed')
        raise
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
