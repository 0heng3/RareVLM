"""Freeze a train-only selection split for the iNaturalist 2018 replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from extract_imagenet_lt_features import atomic_json_save, content_sha256
from extract_features import sha256_file

NUM_CLASSES = 8142
from project_paths import INAT2018_ROOT

DEFAULT_ROOT = INAT2018_ROOT
DEFAULT_CATEGORIES = DEFAULT_ROOT / "categories.json"
OFFICIAL_CATEGORIES_CONTENT_SHA256 = 'bdc1771c7a44d5533f2aff03f7b9c6c3879f87a3452387a694a01894e9354ee7'
OFFICIAL_CATEGORIES_ARCHIVE_SHA256 = '2287211e8da582419dc2ca66736005fdb079e0a89eaf38936a441d8277a8162e'


def records_from_json(data: dict) -> list[dict]:
    images = {item['id']: item for item in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('duplicate image IDs')
    annotations = {item['image_id']: item['category_id'] for item in data['annotations']}
    if len(annotations) != len(data['annotations']) or set(annotations) != set(images):
        raise ValueError('annotations are not one-to-one with images')
    records = []
    for image in data['images']:
        label, path = annotations[image['id']], image['file_name']
        parts = PurePosixPath(path).parts
        if (len(parts) != 4 or parts[0] != 'train_val2018' or '..' in parts
                or PurePosixPath(path).is_absolute() or '\\' in path
                or parts[2] != str(label) or not 0 <= label < NUM_CLASSES):
            raise ValueError(f'invalid image record: {image}')
        records.append({'path': path, 'label': label})
    return records


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 'rarevlm.inat2018.v1':
        raise ValueError('wrong manifest schema')
    if content_sha256({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest.get('manifest_sha256'):
        raise ValueError('manifest hash mismatch')
    if set(manifest['splits']) != {'train', 'val', 'test'}:
        raise ValueError('wrong split names')
    names = manifest['class_names']
    if len(names) != NUM_CLASSES or len(set(names)) != NUM_CLASSES or any(not n or n.isnumeric() for n in names):
        raise ValueError('invalid scientific class names')
    seen = set()
    for split, rows in manifest['splits'].items():
        for row in rows:
            if set(row) != {'path', 'label'} or row['path'] in seen or not 0 <= row['label'] < NUM_CLASSES:
                raise ValueError('duplicate or malformed split record')
            seen.add(row['path'])
        counts = Counter(row['label'] for row in rows)
        if len(counts) != NUM_CLASSES or (split == 'val' and set(counts.values()) != {1}) or (split == 'test' and set(counts.values()) != {3}):
            raise ValueError('split omits a class or has wrong balanced count')
    counts = Counter(row['label'] for row in manifest['splits']['train'])
    if manifest['class_counts'] != [counts[c] for c in range(NUM_CLASSES)]:
        raise ValueError('training counts mismatch')
    expected_groups = {'many': [c for c in range(NUM_CLASSES) if counts[c] > 100],
                       'medium': [c for c in range(NUM_CLASSES) if 20 <= counts[c] <= 100],
                       'few': [c for c in range(NUM_CLASSES) if counts[c] < 20]}
    if manifest['groups'] != expected_groups:
        raise ValueError('frequency groups mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--categories', type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        existing = load_manifest(args.output)
        print('reusing frozen manifest', existing['manifest_sha256'])
        return
    train_path, test_path = args.image_root/'train2018.json', args.image_root/'val2018.json'
    source_train = json.loads(train_path.read_text())
    source_test = json.loads(test_path.read_text())
    categories = sorted(json.loads(args.categories.read_text()), key=lambda item: item['id'])
    if content_sha256(categories) != OFFICIAL_CATEGORIES_CONTENT_SHA256:
        raise ValueError('category contents differ from the official un-obfuscated species table')
    ids = [item['id'] for item in categories]
    if ids != list(range(NUM_CLASSES)) or any(set(ids) != {x['id'] for x in source['categories']} for source in (source_train, source_test)):
        raise ValueError('category IDs mismatch')
    original_train, test = records_from_json(source_train), records_from_json(source_test)
    if len(original_train) != 437513 or len(test) != 24426:
        raise ValueError('unexpected official split sizes')
    by_class = defaultdict(list)
    for row in original_train:
        by_class[row['label']].append(row)
    # Select the smallest hash in each class; invariant to JSON/worker order.
    chosen = {min(rows, key=lambda row: hashlib.sha256(f"2026|{row['label']}|{row['path']}".encode()).digest())['path']
              for rows in by_class.values()}
    train = [row for row in original_train if row['path'] not in chosen]
    val = [row for row in original_train if row['path'] in chosen]
    counts = Counter(row['label'] for row in train)
    manifest = {
        'schema_version': 'rarevlm.inat2018.v1', 'num_classes': NUM_CLASSES,
        'dataset': 'iNaturalist 2018', 'image_root': str(args.image_root.resolve()),
        'splits': {'train': train, 'val': val, 'test': test},
        'split_semantics': {'train': 'official train minus selection validation', 'val': 'one image per class from official train; selection only', 'test': 'official val; locked final top-1 research evaluation, not competition test'},
        'split_seed': 2026, 'selection_rule': 'minimum SHA256(2026|class_id|relative_path) per class',
        'class_names': [item['name'] for item in categories],
        'categories': categories, 'class_counts': [counts[c] for c in range(NUM_CLASSES)],
        'groups': {'many': [c for c in range(NUM_CLASSES) if counts[c] > 100],
                   'medium': [c for c in range(NUM_CLASSES) if 20 <= counts[c] <= 100],
                   'few': [c for c in range(NUM_CLASSES) if counts[c] < 20]},
        'source_sha256': {'train2018.json': sha256_file(train_path), 'val2018.json': sha256_file(test_path), 'categories.json': sha256_file(args.categories)},
        'class_name_provenance': {'source_file': str(args.categories.resolve()), 'official_url': 'https://ml-inat-competition-datasets.s3.amazonaws.com/2018/categories.json.tar.gz', 'official_archive_sha256': OFFICIAL_CATEGORIES_ARCHIVE_SHA256, 'verification': 'canonical category content hash equals the official table'},
    }
    manifest['manifest_sha256'] = content_sha256(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(manifest, args.output)
    checked = load_manifest(args.output)
    print('manifest', checked['manifest_sha256'], 'sizes', {k: len(v) for k,v in checked['splits'].items()}, 'groups', {k:len(v) for k,v in checked['groups'].items()}, flush=True)


if __name__ == '__main__':
    main()
