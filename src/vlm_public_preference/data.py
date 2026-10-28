"""CPU-only provenance, identity and independent-split contracts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata

from PIL import Image


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def normalized_question(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def image_identity(data):
    """Match existing formal manifests: size string followed by decoded RGB bytes."""
    with Image.open(io.BytesIO(data)) as raw:
        image = raw.convert('RGB')
        image.load()
        return {'sha256': hashlib.sha256(data).hexdigest(),
                'pixel_sha256': hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                'width': image.width, 'height': image.height,
                'source_format': raw.format}


def parse_pair(row, repository, revision):
    """Preserve official human/AI distinction without converting preference labels."""
    if repository == 'openbmb/RLHF-V-Dataset':
        texts = json.loads(row['text'])
        label_source = 'official_human_segment_correction'
    elif repository == 'openbmb/RLAIF-V-Dataset':
        texts = row
        label_source = 'official_open_source_model_feedback'
    else:
        raise ValueError('Unsupported official preference repository')
    question, chosen, rejected = (texts.get(key) for key in ('question', 'chosen', 'rejected'))
    if not all(isinstance(value, str) and value.strip() for value in (question, chosen, rejected)):
        raise ValueError('Empty or nontext question/response')
    if chosen.strip() == rejected.strip():
        raise ValueError('Identical chosen/rejected strings')
    raw_origin = row['origin_split']
    try:
        origin = json.loads(raw_origin) if isinstance(raw_origin, str) else raw_origin
    except json.JSONDecodeError:
        origin = {'raw': raw_origin}
    return {'pair_id': f'{repository}@{revision}:{row["idx"]}',
            'question_id': f'{repository}:{row["idx"]}',
            'question_id_namespace': 'official_preference_row_idx;not_original_benchmark_qid',
            'question': question, 'chosen': chosen, 'rejected': rejected,
            'source_repository': repository, 'source_revision': revision, 'source_split': 'train',
            'source_idx': row['idx'], 'origin_dataset': row['origin_dataset'],
            'origin_split': origin, 'origin_split_raw': raw_origin,
            'source_image_path': row['image_path'], 'label_source': label_source,
            'prompt_suffix': '', 'chosen_words': len(chosen.split()),
            'rejected_words': len(rejected.split())}


def excluded_benchmark_origin(text):
    compact = re.sub(r'[^a-z0-9]', '', str(text).casefold())
    return any(token in compact for token in ('textvqa', 'docvqa', 'mmstar'))


def coco_source_id(path):
    """COCO 2014/2017 retain the same twelve-digit original image identifier."""
    if 'coco' not in str(path).casefold():
        return None
    match = re.search(r'(\d{12})\.[^. /]+$', str(path))
    return match.group(1) if match else None


def load_boundaries(manifest_paths):
    """Index every local train/dev/final and formal-transfer image, with source lock."""
    files, pixels, questions = defaultdict(list), defaultdict(list), defaultdict(list)
    raw_question_ids, lock = defaultdict(list), []
    for path in map(Path, manifest_paths):
        manifest = json.loads(path.read_text())
        expected = manifest['manifest_sha256']
        if content_sha({key: value for key, value in manifest.items()
                        if key != 'manifest_sha256'}) != expected:
            raise ValueError(f'Manifest content changed: {path}')
        if 'splits' in manifest:
            groups = {key: manifest['splits'][key] for key in ('train_all', 'dev', 'final')}
        else:
            groups = {manifest.get('task', 'formal'): manifest['records']}
        image_splits = defaultdict(set)
        for split, rows in groups.items():
            for row in rows:
                image_splits[row['image_id']].add(split)
                where = {'manifest': str(path), 'split': split,
                         'question_id': str(row['question_id']), 'image_id': row['image_id']}
                questions[normalized_question(row['question'])].append(where)
                raw_question_ids[str(row['question_id'])].append(where)
        for image_id, splits in image_splits.items():
            image = manifest['images'][image_id]
            where = {'manifest': str(path), 'image_id': image_id, 'splits': sorted(splits)}
            files[image['sha256']].append(where)
            pixels[image['pixel_sha256']].append(where)
        lock.append({'path': str(path), 'file_sha256': sha_file(path),
                     'manifest_sha256': expected, 'image_count': len(image_splits),
                     'question_counts': {split: len(rows) for split, rows in groups.items()}})
    return {'files': dict(files), 'pixels': dict(pixels), 'questions': dict(questions),
            'raw_question_ids': dict(raw_question_ids), 'lock': lock}


def exclusion_reasons(pair, image, boundary):
    reasons = []
    if excluded_benchmark_origin(pair['origin_dataset']) or excluded_benchmark_origin(pair['source_image_path']):
        reasons.append('benchmark_origin_or_path')
    if image['sha256'] in boundary['files']:
        reasons.append('existing_train_dev_final_or_formal_file_sha')
    if image['pixel_sha256'] in boundary['pixels']:
        reasons.append('existing_train_dev_final_or_formal_pixel_sha')
    return reasons


def one_pair_per_image(pairs, seed=42):
    """Select by immutable identity only; not answer scores or review judgments."""
    ordered = sorted(pairs, key=lambda pair: content_sha(['one_pair', seed, pair['pair_id']]))
    files, pixels, result, excluded = set(), set(), [], []
    for pair in ordered:
        if pair['image_sha256'] in files or pair['pixel_sha256'] in pixels:
            excluded.append(pair['pair_id'])
            continue
        files.add(pair['image_sha256'])
        pixels.add(pair['pixel_sha256'])
        result.append(pair)
    return result, excluded


def split_pairs(pairs, seed=42, heldout_fraction=0.1):
    if not 0 < heldout_fraction < 1:
        raise ValueError('heldout_fraction must lie strictly between 0 and 1')
    splits = {'preference_train': [], 'preference_heldout': []}
    for pair in pairs:
        # Pixel identity is the grouping key, so alternate encodings cannot cross splits.
        value = int(content_sha(['preference_split', seed, pair['pixel_sha256']]), 16) / 2 ** 256
        split = 'preference_heldout' if value < heldout_fraction else 'preference_train'
        splits[split].append({**pair, 'preference_split': split})
    return splits


def freeze_audit(training_pairs, count=64, seed=42):
    if len(training_pairs) < count:
        raise ValueError('Insufficient training images for the fixed semantic audit')
    return sorted(training_pairs, key=lambda pair: content_sha(['semantic_audit', seed,
                                                               pair['pair_id'], pair['pixel_sha256']]))[:count]


def summary_counts(pairs):
    return {'pairs': len(pairs), 'images_by_file_sha': len({r['image_sha256'] for r in pairs}),
            'images_by_pixel_sha': len({r['pixel_sha256'] for r in pairs}),
            'origin_datasets': dict(Counter(r['origin_dataset'] for r in pairs)),
            'label_sources': dict(Counter(r['label_source'] for r in pairs)),
            'prompt_suffix': '', 'preserve_official_response_text': True}
