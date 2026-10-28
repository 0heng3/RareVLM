"""Persistent per-image official-HTTPS downloads, never a mirror or replacement."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import requests

from vlm_object_presence.data import sha_file, write_json
from .data import verify_image_bytes

_local = threading.local()


def utc():
    return datetime.now(timezone.utc).isoformat()


def download_one(metadata, root, max_attempts=3):
    """Resume only a complete, byte-and-pixel-verified existing image receipt."""
    root = Path(root)
    iid = metadata['coco_image_id']
    url = f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{iid}.jpg'
    if metadata['official_https_url'] != url or metadata['file_name'] != iid + '.jpg':
        raise ValueError('Only the exact predeclared official JPEG source is allowed')
    destination, receipt_path = root / 'images' / metadata['file_name'], root / 'downloads' / f'{iid}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if previous and previous.get('status') == 'verified':
        if previous['url'] != url or not destination.is_file():
            raise ValueError('Verified download receipt/path/source differs')
        verified = verify_image_bytes(destination.read_bytes(), metadata, previous['headers'])
        if any(verified[key] != previous['identity'][key] for key in ('sha256', 'pixel_sha256', 'width', 'height')):
            raise ValueError('Previously verified image bytes or pixels changed')
        return previous, True
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
    temporary = root / 'download_tmp' / f'{iid}.part'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    attempts = list(previous.get('attempts', [])) if previous else []
    for number in range(max_attempts):
        begun = time.monotonic()
        attempt = {'attempt_number': len(attempts)+1, 'started_at_utc': utc(), 'url': url,
                   'method': 'GET', 'partial_bytes': 0}
        try:
            with _local.session.get(url, stream=True, allow_redirects=False, timeout=(10, 30)) as response:
                headers = dict(response.headers)
                attempt.update(status_code=response.status_code, response_url=response.url,
                    content_length=headers.get('Content-Length'), etag=headers.get('ETag'))
                if response.status_code != 200 or response.url != url:
                    raise ValueError('Official JPEG request did not return exact-source HTTP200')
                with temporary.open('wb') as handle:
                    for block in response.iter_content(128*1024):
                        if block:
                            handle.write(block)
                            attempt['partial_bytes'] += len(block)
                identity = verify_image_bytes(temporary.read_bytes(), metadata, headers)
            temporary.replace(destination)
            attempt.update(outcome='verified', elapsed_seconds=time.monotonic()-begun)
            attempts.append(attempt)
            receipt = {'schema_version': 'rarevlm.object_external.download.v1', 'status': 'verified',
                'coco_image_id': iid, 'source_coco_url': metadata['source_coco_url'], 'url': url,
                'path': str(destination.resolve()), 'verified_at_utc': utc(), 'headers': headers,
                'identity': identity, 'attempts': attempts,
                'official_published_sha256_claimed': False, 'resume_policy': 'reuse only complete verified JPEG receipts; partial files never admitted'}
            write_json(receipt_path, receipt)
            return receipt, False
        except (requests.RequestException, OSError, ValueError) as error:
            attempt.update(outcome='failed', error_type=type(error).__name__, error=str(error),
                           elapsed_seconds=time.monotonic()-begun)
            if temporary.exists():
                attempt['partial_file_sha256'] = sha_file(temporary)
                attempt['partial_bytes_on_disk'] = temporary.stat().st_size
            attempts.append(attempt)
            write_json(receipt_path, {'schema_version': 'rarevlm.object_external.download.v1',
                'status': 'failed', 'coco_image_id': iid, 'url': url,
                'source_coco_url': metadata['source_coco_url'], 'attempts': attempts,
                'partial_file_not_verified': str(temporary) if temporary.exists() else None})
            if number+1 < max_attempts:
                time.sleep(number+1)
    return json.loads(receipt_path.read_text()), False
