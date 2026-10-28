#!/usr/bin/env python
"""Run only the new behavior-replay CPU contracts and bind their exact source files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / 'src'))
    config_file = args.config.resolve()
    config = json.loads(config_file.read_text())
    root = Path(config['output_root']).resolve()
    root.relative_to('/opt/data/private')
    if (root / 'STUDY_INPUT_LOCK.json').exists():
        raise RuntimeError('refusing to replace CPU evidence for an already frozen study')
    if any((root / name).exists() for name in ('CPU_CONTRACTS.json', 'cpu_contracts.log', 'cpu_contracts.xml')):
        raise RuntimeError('preserve original CPU receipt/log/XML before an explicit preparation revision')
    root.mkdir(parents=True, exist_ok=True)
    tests = ['tests/test_object_behavior_replay_data.py',
             'tests/test_object_behavior_replay_train.py',
             'tests/test_object_behavior_replay_study.py']
    fixed_files = [config_file, Path(config['study_protocol']), Path(__file__).resolve(),
                   project / 'pyproject.toml', project / 'tests/test_object_presence_training.py',
                   *[project / name for name in tests]]
    for file in fixed_files:
        if not file.is_file():
            raise RuntimeError(f'CPU prerequisite missing: {file}')
    from vlm_object_behavior_replay.study import study_sources
    from vlm_probe.evaluate import package_versions
    scientific_before = study_sources()
    before = {**scientific_before, **{str(file): digest(file) for file in fixed_files}}
    command = [sys.executable, '-m', 'pytest', '-q', *tests,
               '--junitxml=' + str(root / 'cpu_contracts.xml')]
    environment = dict(os.environ, PYTHONPATH=str(project / 'src'))
    started = time.monotonic()
    with (root / 'cpu_contracts.log').open('w') as log:
        result = subprocess.run(command, cwd=project, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    scientific_after = study_sources()
    after = {**scientific_after, **{str(file): digest(file) for file in fixed_files}}
    suites = ET.parse(root / 'cpu_contracts.xml').getroot().iter('testsuite')
    counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, 0))
    passed = (result.returncode == 0 and counts['tests'] > 0
              and counts['failures'] == counts['errors'] == counts['skipped'] == 0
              and before == after)
    files = [*fixed_files, root / 'cpu_contracts.log', root / 'cpu_contracts.xml']
    receipt = {'schema_version': 'rarevlm.object_behavior_replay.cpu_contracts.v1',
        'status': 'passed' if passed else 'failed',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'command': command,
        'return_code': result.returncode, 'counts': counts, 'elapsed_seconds': elapsed,
        'source_unchanged_during_tests': before == after,
        'files_sha256': {str(file): digest(file) for file in files},
        'scientific_source_sha256': scientific_after,
        'scientific_source_sha256_before': scientific_before, 'scientific_source_sha256_after': scientific_after,
        'source_and_fixed_files_sha256_before': before, 'source_and_fixed_files_sha256_after': after,
        'packages': package_versions(),
        'new_GPU_operations': 0, 'new_pretrained_model_forwards': 0,
        'CPU_fixture_forwards_are_tests': True, 'study_frozen': False}
    (root / 'CPU_CONTRACTS.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'counts', 'elapsed_seconds',
                                                  'source_unchanged_during_tests')}, ensure_ascii=False))
    if not passed:
        raise SystemExit('new behavior-replay CPU contracts failed; inspect cpu_contracts.log')


if __name__ == '__main__':
    main()
