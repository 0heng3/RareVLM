"""Download only an explicitly named official source, pinning revision and LFS SHA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_public_preference.data import sha_file, write_json


def ranged_download(url, path, total_bytes, connections):
    chunk_size = 4 * 1024 * 1024
    part_root = path.parent / (path.name + '.parts')
    part_root.mkdir(parents=True, exist_ok=True)
    bounds = [(start, min(start + chunk_size, total_bytes) - 1)
              for start in range(0, total_bytes, chunk_size)]
    def fetch(bound):
        start, end = bound
        part = part_root / f'{start:012d}.part'
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        for attempt in range(4):
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected_range = f'bytes {start}-{end}/{total_bytes}'
                    if response.status_code != 206 or response.headers.get('content-range') != expected_range:
                        raise ValueError('Official CDN did not honor exact byte range')
                    with part.with_suffix('.downloading').open('wb') as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                temporary = part.with_suffix('.downloading')
                if temporary.stat().st_size != end-start+1:
                    raise ValueError('Incomplete official CDN range')
                temporary.replace(part)
                return part
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(min(attempt+1, 3))
    with ThreadPoolExecutor(max_workers=connections) as pool:
        for index, _ in enumerate(pool.map(fetch, bounds), 1):
            if index % 16 == 0:
                print('verified_size_parts', index, '/', len(bounds), flush=True)
    with path.with_suffix(path.suffix + '.part').open('wb') as target:
        for start, _ in bounds:
            with (part_root / f'{start:012d}.part').open('rb') as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    target.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', choices=['openbmb/RLHF-V-Dataset', 'openbmb/RLAIF-V-Dataset'], required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--connections', type=int, default=16)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('Public data must remain under persistent workspace')
    root.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/datasets/{args.repository}/revision/{args.revision}?blobs=true'
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != args.revision:
        raise ValueError('Official revision mismatch')
    lock = {'repository': args.repository, 'revision': args.revision,
            'metadata_url': url, 'license': metadata.get('cardData', {}).get('license'), 'files': {}}
    write_json(root / 'OFFICIAL_METADATA.json', metadata)
    for item in metadata['siblings']:
        name = item['rfilename']
        if not (name == 'README.md' or name.endswith('.parquet')):
            continue
        path = root / name
        expected = item.get('lfs', {}).get('sha256')
        if path.exists() and path.stat().st_size == item['size'] and (not expected or sha_file(path) == expected):
            pass
        else:
            download_url = f'https://huggingface.co/datasets/{args.repository}/resolve/{args.revision}/{name}'
            if name.endswith('.parquet') and args.connections > 1:
                ranged_download(download_url, path, item['size'], args.connections)
            else:
                with requests.get(download_url, stream=True, timeout=120) as download:
                    download.raise_for_status()
                    with path.with_suffix(path.suffix + '.part').open('wb') as handle:
                        for chunk in download.iter_content(4 * 1024 * 1024):
                            handle.write(chunk)
            temporary = path.with_suffix(path.suffix + '.part')
            if temporary.stat().st_size != item['size'] or (expected and sha_file(temporary) != expected):
                raise ValueError(f'Download integrity mismatch: {name}')
            temporary.replace(path)
            part_root = path.parent / (path.name + '.parts')
            if part_root.exists():
                for part in part_root.iterdir():
                    part.unlink()
                part_root.rmdir()
        lock['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha_file(path),
                                'official_lfs_sha256': expected, 'path': str(path)}
        print(name, lock['files'][name]['bytes'], lock['files'][name]['sha256'], flush=True)
    write_json(root / 'DOWNLOAD_LOCK.json', lock)


if __name__ == '__main__':
    main()
