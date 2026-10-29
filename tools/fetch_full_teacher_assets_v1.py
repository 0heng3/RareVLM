"""Acquire public teacher bytes from ModelScope, verified against pinned HF hashes."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_full_teacher_assets_v1')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def digest(path, lfs):
    algorithm = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        algorithm.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            algorithm.update(chunk)
    return algorithm.hexdigest()


def main():
    if (ROOT / 'TRANSPORT_COMPLETE.json').exists():
        raise FileExistsError('Verified assets already complete')
    source = json.loads((ROOT / 'MODEL_SOURCE.json').read_text())
    model = ROOT / 'model'
    wanted = [f for f in source['files'] if f['name'] != '.gitattributes']
    deadline = time.monotonic() + 1800

    def fetch(item):
        name = item['name']; path = model / name
        expected = item['lfs_sha256'] or item['blob_id']
        if path.exists() and path.stat().st_size == item['size'] and digest(path, bool(item['lfs_sha256'])) == expected:
            return name
        partial = path.with_name(path.name + '.partial')
        last_error = None
        for attempt in range(3):
            if time.monotonic() > deadline:
                raise TimeoutError('Teacher asset acquisition exceeded1800 seconds; partials retained')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item['size']:
                break
            try:
                url = (f"https://modelscope.cn/models/{source['repository']}/resolve/master/{name}" if item['lfs_sha256']
                       else f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{name}")
                with requests.get(url, headers={'Range': f'bytes={offset}-'}, stream=True, timeout=(10, 25)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Resume response does not start at saved byte offset')
                    with partial.open('ab') as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if time.monotonic() > deadline:
                                raise TimeoutError('Acquisition time budget exceeded')
                            stream.write(chunk)
                            if stream.tell() > item['size']:
                                raise ValueError('Received more bytes than the fixed source declares')
                if partial.stat().st_size == item['size']:
                    break
            except (requests.RequestException, ValueError) as error:
                last_error = type(error).__name__
                write(ROOT / 'file_status' / (name + '.json'), {'status': 'transport_retry', 'attempt': attempt + 1,
                    'error_type': last_error, 'bytes': partial.stat().st_size if partial.exists() else 0})
        if not partial.exists() or partial.stat().st_size != item['size']:
            raise RuntimeError(f'Incomplete asset {name}; last transport error={last_error}')
        if digest(partial, bool(item['lfs_sha256'])) != expected:
            raise ValueError('Mirror differs from pinned official hash: ' + name)
        os.replace(partial, path)
        write(ROOT / 'file_status' / (name + '.json'), {'status': 'verified', 'bytes': item['size'], 'hash': expected})
        return name

    total = sum(f['size'] for f in wanted)
    started = time.monotonic(); completed = []; failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, item): item['name'] for item in wanted}
        while pending:
            ready, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for future in ready:
                name = pending.pop(future)
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append({'file': name, 'error_type': type(error).__name__, 'message': str(error)})
            received = sum(max((model / f['name']).stat().st_size if (model / f['name']).exists() else 0,
                               (model / (f['name'] + '.partial')).stat().st_size if (model / (f['name'] + '.partial')).exists() else 0) for f in wanted)
            write(ROOT / 'TRANSPORT_PROGRESS.json', {'status': 'downloading' if pending else 'failed' if failures else 'verifying_complete',
                'bytes': received, 'expected_bytes': total, 'verified_files': len(completed), 'expected_files': len(wanted),
                'elapsed_seconds': time.monotonic() - started, 'optimizer_updates': 0, 'gpu_calls': 0, 'failures': failures})
    if failures:
        raise RuntimeError('Teacher assets incomplete; inspect TRANSPORT_PROGRESS.json. No GPU work admitted.')
    write(ROOT / 'TRANSPORT_COMPLETE.json', {'status': 'completed', 'repository': source['repository'], 'revision': source['revision'],
        'verified_files': len(completed), 'bytes': total, 'all_pinned_official_hashes_matched': True,
        'transport': 'ModelScope public mirror, fixed HF Git/LFS hashes', 'elapsed_seconds': time.monotonic() - started,
        'gpu_calls': 0, 'optimizer_updates': 0})
    print('TEACHER_ASSETS_HASH_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
