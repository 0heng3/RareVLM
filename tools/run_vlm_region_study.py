"""Persistent, gated launch: real CPU contract -> GPU preflight -> full dev."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path('/opt/data/private')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read(config_path)
    root = Path(config['output_root']).resolve()
    if PRIVATE not in root.parents or PRIVATE not in config_path.parents:
        raise ValueError('Persistent workspace required')
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / 'LAUNCHER.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'schema_version':'rarevlm.region_launcher.v1','pid':os.getpid(),
             'started_at':now(),'config_sha256':sha(config_path),'stage':'waiting_cpu_inputs',
             'new_training':0,'full_predictions':6468,'existing_jobs_untouched':True,'events':[]}
    status_path = root / 'LAUNCHER_STATUS.json'

    def status(stage, **fields):
        state.update(stage=stage, updated_at=now(), **fields)
        write(status_path, state)
        print(stage, now(), flush=True)

    def unchanged():
        if sha(config_path) != state['config_sha256']:
            raise ValueError('Frozen configuration changed during launch')

    def run(command, stage, expected_phase=None):
        unchanged()
        status(stage)
        started = time.monotonic()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run(command, cwd=REPO, env=env, check=True)
        event = {'stage':stage,'completed_at':now(),'wall_seconds':time.monotonic()-started,
                 'returncode':result.returncode,'command':command}
        state['events'].append(event)
        if expected_phase and (read(root / 'STATUS.json') or {}).get('phase') != expected_phase:
            raise RuntimeError(f'{stage} returned without expected phase {expected_phase}')
        write(status_path, state)

    try:
        status('waiting_cpu_inputs')
        while True:
            unchanged()
            contract = read(root / 'INPUT_CONTRACT.json')
            precheck = read(root / 'INPUT_PRECHECK.json')
            if contract and precheck:
                if (contract.get('status') != 'passed' or contract.get('processor_checks') != 6468
                        or contract.get('actual_visual_tokens_min') != 1024
                        or contract.get('actual_visual_tokens_max') != 1024
                        or contract.get('input_truncated_count') != 0
                        or precheck.get('status') != 'passed' or precheck.get('processor_checks') != 128):
                    raise ValueError('Real input contract failed')
                break
            time.sleep(30)
        pipeline = [sys.executable, '-m', 'vlm_region.pipeline',
                    '--asset-root', str(Path(config['manifest_path']).parent.parent),
                    '--model-root', config['model_root'], '--output-root', str(root),
                    '--config', str(config_path)]
        if not (root / 'EXECUTION_READY.json').exists():
            run(pipeline + ['--prepare-only'], 'freezing_execution')
        run([sys.executable, str(REPO / 'tools/build_vlm_region_reader.py'),
             '--config', str(config_path)], 'building_real_input_reader')
        run(pipeline + ['--preflight-only'], 'gpu_preflight', 'preflight_completed')
        ready = read(root / 'PRECHECK_READY.json')
        if not ready or ready.get('status') != 'passed':
            raise RuntimeError('No passed GPU precheck artifact')
        run(pipeline, 'full_dev_evaluation', 'complete')
        run([sys.executable, str(REPO / 'tools/build_vlm_region_reader.py'),
             '--config', str(config_path)], 'refreshing_reader_with_results')
        status('complete')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error),
               active_gpu_workers_were_not_terminated=True)
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
