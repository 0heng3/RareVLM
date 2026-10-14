"""Persistent OCR study: CPU contracts -> GPU preflight -> full dev -> reader."""
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
    if not path.exists() or not path.stat().st_size:
        return None
    return json.loads(path.read_text())


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read(config_path)
    root = Path(config['output_root']).resolve()
    if PRIVATE not in root.parents or PRIVATE not in config_path.parents:
        raise ValueError('persistent workspace required')
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / 'LAUNCHER.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'schema_version': 'rarevlm.ocr_context.launcher.v1', 'pid': os.getpid(),
             'started_at': now(), 'config_sha256': sha(config_path), 'stage': 'waiting_cpu_inputs',
             'new_training': 0, 'full_predictions': 6468,
             'existing_jobs_untouched': True, 'events': []}
    status_path = root / 'LAUNCHER_STATUS.json'

    def status(stage, **fields):
        state.update(stage=stage, updated_at=now(), **fields)
        write(status_path, state)
        print(stage, now(), flush=True)

    def unchanged():
        if sha(config_path) != state['config_sha256']:
            raise ValueError('frozen configuration changed during launch')
        preparation = read(root / 'PREPARATION.json')
        for name, digest in preparation['frozen_files_sha256'].items():
            if sha(REPO / name) != digest:
                raise ValueError(f'historical source/configuration changed: {name}')
        for name, digest in preparation['resume_files_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('resume changed during study')

    def run(command, stage, expected_phase=None):
        unchanged()
        status(stage)
        started = time.monotonic()
        env = dict(os.environ)
        env['PYTHONPATH'] = str(REPO / 'src')
        result = subprocess.run(command, cwd=REPO, env=env, check=True)
        event = {'stage': stage, 'completed_at': now(),
                 'wall_seconds': time.monotonic() - started,
                 'returncode': result.returncode, 'command': command}
        state['events'].append(event)
        if expected_phase and (read(root / 'STATUS.json') or {}).get('phase') != expected_phase:
            raise RuntimeError(f'{stage} returned without expected phase {expected_phase}')
        write(status_path, state)

    try:
        status('waiting_cpu_inputs')
        while True:
            unchanged()
            contract = read(Path(config['input_contract_path']))
            precheck = read(Path(config['input_precheck_path']))
            ready = read(Path(config['ocr_ready_path']))
            if contract and precheck and ready:
                records = contract.get('records', [])
                if (contract.get('status') != 'passed' or not contract.get('complete')
                        or contract.get('processor_checks') != 6468 or len(records) != 6468
                        or contract.get('config_sha256') != state['config_sha256']
                        or any(not 1 <= item['visual_tokens'] <= 1024 or item['input_truncated']
                               for item in records)
                        or precheck.get('status') != 'passed' or precheck.get('processor_checks') != 128
                        or precheck.get('config_sha256') != state['config_sha256']
                        or not ready.get('complete') or ready.get('images') != 1024):
                    raise ValueError('real OCR/native input contract failed')
                break
            cpu = read(root / 'CPU_PREPARE_STATUS.json') or {}
            ocr = read(root / 'recognizer/OCR_STATUS.json') or {}
            if cpu.get('status') == 'failed' or ocr.get('status') == 'failed':
                raise RuntimeError('CPU prerequisite explicitly failed; inspect preparation logs')
            time.sleep(30)

        pipeline = [sys.executable, '-m', 'vlm_ocr.pipeline',
                    '--asset-root', config['asset_root'], '--model-root', config['model_root'],
                    '--output-root', str(root), '--config', str(config_path)]
        run([sys.executable, '-m', 'vlm_ocr.analysis', '--asset-root', config['asset_root'],
             '--output-root', str(root), '--config', str(config_path), '--prepare-only'],
            'preparing_analysis_contract')
        run(pipeline + ['--prepare-only'], 'freezing_execution')
        run([sys.executable, str(REPO / 'tools/build_vlm_ocr_reader.py'),
             '--config', str(config_path)], 'building_real_input_reader')
        run(pipeline + ['--preflight-only'], 'gpu_preflight', 'preflight_completed')
        if (read(root / 'PRECHECK_READY.json') or {}).get('status') != 'passed':
            raise RuntimeError('no passed GPU precheck artifact')
        run(pipeline, 'full_dev_evaluation', 'complete')
        run([sys.executable, str(REPO / 'tools/build_vlm_ocr_reader.py'),
             '--config', str(config_path)], 'refreshing_reader_with_results')
        unchanged()
        result = read(root / 'analysis/ANALYSIS.json')
        if not result or not result.get('complete') or result.get('validated_predictions') != 6468:
            raise ValueError('complete scored result absent')
        status('complete', quality_gate=result.get('gate'))
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error),
               active_gpu_workers_were_not_terminated=True)
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
