"""Strict supervised rows and persistent image/source bindings; no evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import PROMPT_SUFFIX, read_manifest, sha_file

PRIVATE = Path('/opt/data/private')
OBJECT_SUFFIX = '\nAnswer yes or no.'
EXPECTED_REPLAY_MANIFEST_SHA256 = '01c95ce0b346070ecd0e98fe52f9c67e2e4ddf47de6951fd8b8cf5ece376455a'


def private_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError(f'artifact outside persistent workspace: {path}')
    return path


def check_rows(rows, prompt_suffix, object_questions=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected nonempty supervised rows')
    ids = set()
    for row in rows:
        for key in ('question_id', 'image_id', 'question', 'target', 'image_path', 'pixel_sha256'):
            if key not in row or not str(row[key]).strip():
                raise ValueError(f'missing supervised field: {key}')
        if str(row['question_id']) in ids:
            raise ValueError('duplicate supervised question ID')
        ids.add(str(row['question_id']))
        if object_questions and row['target'] not in ('yes', 'no'):
            raise ValueError('object target must be annotation-supervised lowercase yes/no')
        if row.get('prompt_suffix') != prompt_suffix:
            raise ValueError('row prompt suffix differs from its frozen task prompt')
        if not Path(row['image_path']).is_absolute():
            raise ValueError('manifest image_path must be absolute')
        private_path(row['image_path'])
        if row.get('split', 'train') not in ('train', 'train_all'):
            raise ValueError('training row is from a heldout/evaluation split')
    return rows


def read_training_rows(config):
    """Read all train_all targets, without making any generation/evaluation call.

    The frozen TextVQA manifest validates its own digest and image-disjoint
    splits. Replay exposes only train_all to the sampler and trainer.
    """
    object_path = private_path(config['train_manifest'])
    document = json.loads(object_path.read_text())
    if document.get('split', 'train') not in ('train', 'train_all'):
        raise ValueError('object manifest must be a training split')
    objects = check_rows(document.get('rows'), OBJECT_SUFFIX, object_questions=True)
    object_sha = sha_file(object_path)
    admission_binding = read_admission(config, object_sha)
    replay_path = private_path(config['replay_manifest'])
    if sha_file(replay_path) != EXPECTED_REPLAY_MANIFEST_SHA256:
        raise ValueError('original train_all manifest SHA differs from the frozen replay source')
    replay_manifest = read_manifest(replay_path, verify_images=False)
    if len(replay_manifest['splits']['train_all']) != 32985:
        raise ValueError('replay must be the complete original 32985-QA train_all pool')
    replays = []
    for row in replay_manifest['splits']['train_all']:
        image = replay_manifest['images'][row['image_id']]
        replays.append({**row, 'image_path': str(private_path(replay_path.parent / image['path'])),
                        'pixel_sha256': image['pixel_sha256'], 'file_sha256': image['sha256'],
                        'prompt_suffix': PROMPT_SUFFIX, 'split': 'train_all'})
    check_rows(replays, PROMPT_SUFFIX)
    # Exact file/pixel overlap with TextVQA dev/final is rejected independently
    # of COCO/TextVQA identifier namespaces. Other heldouts are data-agent locks.
    heldout_images = {str(row['image_id']) for split in ('dev', 'final')
                      for row in replay_manifest['splits'][split]}
    heldout_pixels = {replay_manifest['images'][iid]['pixel_sha256'] for iid in heldout_images}
    heldout_files = {replay_manifest['images'][iid]['sha256'] for iid in heldout_images}
    if any(row['pixel_sha256'] in heldout_pixels or row.get('file_sha256') in heldout_files for row in objects):
        raise ValueError('object image overlaps TextVQA dev/final pixels or files')
    binding = {'train_manifest_sha256': object_sha,
               'replay_manifest_sha256': sha_file(replay_path),
               'replay_manifest_internal_sha256': replay_manifest['manifest_sha256'],
               'replay_source_split': 'train_all', 'replay_complete_pool_questions': len(replays),
               **admission_binding}
    return objects, replays, binding


def read_admission(config, train_manifest_sha256):
    """Mandatory evidence/protocol gate, shared by preflight and formal entry."""
    path = private_path(config['data_admission'])
    admission = json.loads(path.read_text())
    protocol_path = private_path(config['dataset_protocol'])
    protocol_sha = sha_file(protocol_path)
    if (admission.get('status') != 'admitted'
            or admission.get('train_manifest_sha256') != train_manifest_sha256
            or admission.get('dataset_protocol_sha') != protocol_sha):
        raise ValueError('object data admission status/manifest/dataset protocol binding missing or changed')
    return {'data_admission_sha256': sha_file(path), 'dataset_protocol_sha': protocol_sha}


def verify_image_bindings(rows):
    """Check actual RGB/file bytes for every image that this condition consumes."""
    from PIL import Image
    checked = {}
    for row in rows:
        path = private_path(row['image_path'])
        key = str(path)
        if key not in checked:
            with Image.open(path) as opened:
                rgb = opened.convert('RGB')
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            checked[key] = {'pixel_sha256': pixels, 'file_sha256': sha_file(path)}
        if checked[key]['pixel_sha256'] != row['pixel_sha256']:
            raise ValueError('supervised image RGB pixels changed')
        if row.get('file_sha256') and row['file_sha256'] != checked[key]['file_sha256']:
            raise ValueError('supervised image file changed')
    return checked
