"""Persist one compact OCR followup; its 97.5% analysis owns the final gate."""
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
EXPECTED_PREDICTIONS = 6468
COMPACT_CONFIDENCE = 0.975


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    if not path.exists() or not path.stat().st_size:
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def private_path(path):
    path = path.resolve()
    if PRIVATE not in path.parents:
        raise ValueError('persistent workspace required')
    return path


def process_matches(pid, command):
    """Wait only for the recorded component, never an unrelated reused PID."""
    try:
        raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return raw.rstrip(b'\0').decode().split('\0') == command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = private_path(args.config)
    config = read(config_path)
    if not config or config.get('schema_version') != 'rarevlm.ocr_compact.v1':
        raise ValueError('requires the frozen compact followup configuration')
    if config['compact_statistics']['two_sided_confidence'] != COMPACT_CONFIDENCE:
        raise ValueError('authoritative compact confidence must remain 97.5%')
    if config['scope']['new_training_runs'] != 0 or config['expected_predictions'] != EXPECTED_PREDICTIONS:
        raise ValueError('only this fixed zero-training four-cell followup is authorized')
    root = private_path(Path(config['output_root']))
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / 'LAUNCHER.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'LAUNCHER_STATUS.json'
    previous = read(status_path) or {}
    config_sha = sha(config_path)
    launcher_sha = sha(Path(__file__))
    if previous and (previous.get('config_sha256') != config_sha
                     or previous.get('launcher_source_sha256') != launcher_sha):
        raise ValueError('launcher/config changed; refuse to resume a different study')
    state = {**previous, 'schema_version': 'rarevlm.ocr_compact.launcher.v1', 'pid': os.getpid(),
             'started_at': previous.get('started_at', now()), 'resumed_at': now(),
             'config_sha256': config_sha, 'launcher_source_sha256': launcher_sha,
             'new_training': 0, 'new_recognition_invocations': 0,
             'full_predictions': EXPECTED_PREDICTIONS, 'existing_jobs_untouched': True,
             'authoritative_analysis': 'compact_analysis/ANALYSIS.json',
             'authoritative_confidence_level': COMPACT_CONFIDENCE,
             'automatic_official_evaluation': False, 'automatic_replication': False,
             'events': previous.get('events', [])}
    source_freeze_path = root / 'COMPACT_SOURCE_FREEZE.json'

    def status(stage, **fields):
        state.update(stage=stage, updated_at=now(), **fields)
        write(status_path, state)
        print(stage, now(), flush=True)

    def unchanged():
        if sha(config_path) != config_sha or sha(Path(__file__)) != launcher_sha:
            raise ValueError('frozen configuration/launcher changed during launch')
        preparation = read(root / 'PREPARATION.json')
        if not preparation:
            raise ValueError('root PREPARATION.json is required before launch')
        if preparation.get('config_sha256') != config_sha:
            raise ValueError('preparation/current compact configuration differ')
        for name, digest in preparation['frozen_files_sha256'].items():
            if sha(REPO / name) != digest:
                raise ValueError(f'historical source/configuration changed: {name}')
        for name, digest in preparation['resume_files_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('resume changed during study')
        freeze = read(source_freeze_path)
        if freeze:
            if freeze['config_sha256'] != config_sha:
                raise ValueError('compact source freeze/configuration differ')
            current_files = sorted(str(p.relative_to(REPO))
                                   for p in (REPO / 'src/vlm_ocr_compact').rglob('*.py'))
            if current_files != freeze['compact_module_files']:
                raise ValueError('compact module source file set changed')
            for path, digest in freeze['files_sha256'].items():
                if sha(Path(path)) != digest:
                    raise ValueError(f'compact source/readiness changed: {path}')

    def run(command, stage, expected_phase=None):
        unchanged()
        status(stage)
        env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONDONTWRITEBYTECODE='1')
        env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
        started = time.monotonic()
        # Popen.wait deliberately avoids subprocess.run's automatic kill on
        # launcher interruption. The component queue keeps its own inherited
        # UUID admission locks and can finish/recover without this launcher.
        child = subprocess.Popen(command, cwd=REPO, env=env)
        state['active_child'] = {'pid': child.pid, 'command': command, 'stage': stage, 'started_at': now()}
        write(status_path, state)
        returncode = child.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        state.pop('active_child', None)
        state['events'].append({'stage': stage, 'completed_at': now(),
                                'wall_seconds': time.monotonic() - started,
                                'returncode': returncode, 'command': command})
        unchanged()
        if expected_phase and (read(root / 'STATUS.json') or {}).get('phase') != expected_phase:
            raise RuntimeError(f'{stage} returned without expected phase {expected_phase}')
        write(status_path, state)

    def freeze_sources():
        module_files = sorted((REPO / 'src/vlm_ocr_compact').rglob('*.py'))
        if not module_files or not (REPO / 'src/vlm_ocr_compact/analysis.py').is_file():
            raise ValueError('compact module/analysis source not ready')
        paths = [*module_files, Path(__file__).resolve(), REPO / 'src/rarevlm_paths.py',
                 REPO / 'tools/build_vlm_ocr_compact_reader.py', root / 'ANALYSIS_READY.json',
                 root / 'COMPACT_ANALYSIS_READY.json', root / 'EXECUTION_READY.json',
                 *[Path(config[key]) for key in ('context_lock_path', 'input_precheck_path',
                                                'input_contract_path', 'ocr_ready_path')]]
        compact_ready = read(root / 'COMPACT_ANALYSIS_READY.json')
        if not compact_ready or compact_ready.get('config_sha256') != config_sha:
            raise ValueError('compact analysis readiness is missing or belongs to another configuration')
        payload = {'schema_version': 'rarevlm.ocr_compact.source_freeze.v1',
                   'config_sha256': config_sha, 'launcher_source_sha256': launcher_sha,
                   'compact_module_files': [str(p.relative_to(REPO)) for p in module_files],
                   'files_sha256': {str(p.resolve()): sha(p) for p in paths},
                   'authoritative_analysis': 'compact_analysis/ANALYSIS.json',
                   'authoritative_confidence_level': COMPACT_CONFIDENCE,
                   'scope': 'own compact sources,reader,launcher,path helper and input/execution/analysis readiness; historical sources guarded by PREPARATION'}
        if source_freeze_path.exists() and read(source_freeze_path) != payload:
            raise ValueError('existing compact source freeze differs')
        write(source_freeze_path, payload)
        state['compact_source_freeze_sha256'] = sha(source_freeze_path)
        unchanged()

    try:
        unchanged()
        prior_child = previous.get('active_child')
        if prior_child:
            while process_matches(prior_child['pid'], prior_child['command']):
                status('waiting_previous_component', active_child=prior_child)
                unchanged()
                time.sleep(30)
            state.pop('active_child', None)
        status('waiting_cpu_inputs')
        while True:
            unchanged()
            context = read(Path(config['context_lock_path']))
            contract = read(Path(config['input_contract_path']))
            precheck = read(Path(config['input_precheck_path']))
            ready = read(Path(config['ocr_ready_path']))
            if context and contract and precheck and ready:
                records = contract.get('records', [])
                if (context.get('status') != 'passed' or not context.get('complete')
                        or context.get('config_sha256') != config_sha or len(context.get('images', {})) != 1024
                        or contract.get('status') != 'passed' or not contract.get('complete')
                        or contract.get('processor_checks') != EXPECTED_PREDICTIONS
                        or len(records) != EXPECTED_PREDICTIONS or contract.get('config_sha256') != config_sha
                        or any(not 1 <= item['visual_tokens'] <= 1024 or item['input_truncated'] for item in records)
                        or precheck.get('status') != 'passed' or not precheck.get('complete')
                        or precheck.get('processor_checks') != 128 or precheck.get('config_sha256') != config_sha
                        or not ready.get('complete') or ready.get('images') != 1024):
                    raise ValueError('complete compact/native input contract failed')
                break
            cpu = read(root / 'CPU_PREPARE_STATUS.json') or {}
            if cpu.get('status') == 'failed':
                raise RuntimeError('CPU prerequisite explicitly failed; inspect preparation logs')
            status('waiting_cpu_inputs')
            time.sleep(30)

        pipeline = [sys.executable, '-m', 'vlm_ocr.pipeline', '--asset-root', config['asset_root'],
                    '--model-root', config['model_root'], '--output-root', str(root), '--config', str(config_path)]
        analysis_args = ['--asset-root', config['asset_root'], '--output-root', str(root), '--config', str(config_path)]
        compact_analysis = [sys.executable, '-m', 'vlm_ocr_compact.analysis', *analysis_args]
        reader = [sys.executable, str(REPO / 'tools/build_vlm_ocr_compact_reader.py'), '--config', str(config_path)]
        if not source_freeze_path.exists():
            run([sys.executable, '-m', 'vlm_ocr.analysis', *analysis_args, '--prepare-only'],
                'preparing_component_analysis_contract')
            run(compact_analysis + ['--prepare-only'], 'preparing_authoritative_compact_analysis_contract')
            run(pipeline + ['--prepare-only'], 'freezing_component_execution')
            freeze_sources()
        else:
            unchanged()
        run(reader, 'building_real_compact_input_reader')
        run(pipeline + ['--preflight-only'], 'gpu_preflight', 'preflight_completed')
        if (read(root / 'PRECHECK_READY.json') or {}).get('status') != 'passed':
            raise RuntimeError('no passed GPU precheck artifact')
        run(pipeline, 'full_dev_evaluation_component_analysis', 'complete')
        component_result = read(root / 'analysis/ANALYSIS.json')
        if not component_result or not component_result.get('complete') or component_result.get('validated_predictions') != EXPECTED_PREDICTIONS:
            raise ValueError('complete validated component report absent')
        state['component_95percent_gate'] = component_result.get('gate')
        state['component_analysis_authoritative'] = False
        run(compact_analysis, 'authoritative_compact_97_5percent_analysis')
        result = read(root / 'compact_analysis/ANALYSIS.json')
        if not result or not result.get('complete') or result.get('validated_predictions') != EXPECTED_PREDICTIONS or 'gate' not in result:
            raise ValueError('complete authoritative compact report absent')
        run(reader, 'refreshing_compact_reader_with_results')
        unchanged()
        status('complete', quality_gate=result['gate'],
               quality_gate_source='compact_analysis/ANALYSIS.json',
               compact_analysis_sha256=sha(root / 'compact_analysis/ANALYSIS.json'),
               validated_predictions=result['validated_predictions'],
               automatic_expansion=False)
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error),
               active_gpu_workers_were_not_terminated=True)
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
