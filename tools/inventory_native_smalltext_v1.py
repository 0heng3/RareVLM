"""Prediction-free inventory: native QA with literal OCR support and small text."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import time
import unicodedata

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, read_manifest, sha_file, write_json
from vlm_region_ocr_sft_v1.data import bounds, excludes_other_word_centers, region

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_native_smalltext_v1'


def literal(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold().strip()
    return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', value)


def main():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    if (ROOT / 'INVENTORY.json').exists():
        raise FileExistsError('Prediction-free inventory already exists')
    start = time.monotonic()
    manifest_path = BASE / 'vlm_deep_v2/prepared/manifest.json'
    source = read_manifest(manifest_path)
    old = json.loads((BASE / 'vlm_reference_replay_v1/DATA.json').read_text())
    previous = json.loads((BASE / 'vlm_native_ocr_correction_v1/revision2/DATA.json').read_text())
    config = json.loads((PROJECT_ROOT / 'configs/vlm_region_ocr_sft_v1.json').read_text())
    annotations_path = Path(config['annotations'])
    raw = json.loads(annotations_path.read_text())
    plan = json.loads((BASE / 'vlm_reference_replay_v1/PLAN.json').read_text())
    reference = json.loads(Path(plan['reference_config']).read_text())
    norm = lru_cache(maxsize=300000)(lambda s: NORMALIZER(s))
    frequency = Counter(norm(q['target']) for q in source['splits']['train_all'])
    blocked = {q['image_id'] for q in old['training_rows'] + previous['rows']}
    for rows in old['evaluations'].values():
        blocked.update(q['image_id'] for q in rows)
    for name in ('dev', 'final'):
        blocked.update(q['image_id'] for q in source['splits'][name])
    image_meta = {**source['images'], **old['images'], **previous['images']}

    def pixel(iid):
        return image_meta[iid].get('pixel_sha256', image_meta[iid]['sha256'])

    blocked_pixels = {pixel(i) for i in blocked if i in image_meta}
    by_image = defaultdict(list)
    counts = Counter()
    for q in source['splits']['train_all']:
        counts['source_qa'] += 1
        iid = q['image_id']
        if iid in blocked or pixel(iid) in blocked_pixels:
            continue
        target = literal(q['target'])
        if (len(q['target'].split()) != 1 or not re.fullmatch(r'[a-z0-9]{3,24}', target)
                or not re.search(r'[a-z]', target)):
            continue
        t = norm(q['target'])
        if sum(norm(a) == t for a in q['answers']) < 8 or frequency[t] > 20:
            continue
        question = q['question'].casefold()
        if not re.search(r'\b(brand|name|say|says|read|reads|written|word|words|text|logo|spell|letters|letter)\b', question):
            continue
        if re.search(r'\b(color|colour|time|many|much|price|language)\b', question):
            continue
        by_image[iid].append(q)
        counts['native_literal_consensus_rare_mid'] += 1

    policy = {**config['region_filter'], 'min_resized_width': 24, 'min_resized_height': 8, 'max_characters': 24}
    rows, used, targets = [], set(), Counter()
    for iid in sorted(by_image, key=lambda x: content_sha(['native_smalltext_v1:image', x])):
        if iid not in raw['imgs']:
            continue
        im = source['images'][iid]
        meta = raw['imgs'][iid]
        if (im['width'], im['height']) != (meta['width'], meta['height']):
            continue
        rh, rw = smart_resize(im['height'], im['width'], factor=28,
                              min_pixels=reference['min_pixels'], max_pixels=1024*784)
        groups = defaultdict(list)
        for aid in raw['imgToAnns'][iid]:
            a = raw['anns'][aid]
            groups[literal(a.get('utf8_string', ''))].append(a)
        for q in sorted(by_image[iid], key=lambda x: content_sha(['native_smalltext_v1:qa', x['question_id']])):
            matches = groups[literal(q['target'])]
            if len(matches) != 1:
                continue
            a = matches[0]
            value, reason = region(a, im['width'], im['height'], rw, rh, policy)
            if reason:
                counts[reason] += 1
                continue
            if not 8 <= value['effective_region_pixels'][1] < 16:
                counts['outside_small_text_band'] += 1
                continue
            other_boxes = []
            for aid in raw['imgToAnns'][iid]:
                if aid == a['id']:
                    continue
                try:
                    other_boxes.append(bounds(raw['anns'][aid]['points'], im['width'], im['height']))
                except (ValueError, KeyError, TypeError):
                    continue
            if not excludes_other_word_centers(value['box_xyxy_original_pixels'], other_boxes):
                counts['other_word_center'] += 1
                continue
            t = norm(q['target'])
            if targets[t] >= 8 or pixel(iid) in used:
                continue
            targets[t] += 1
            used.add(pixel(iid))
            f = frequency[t]
            rows.append({**q, 'stratum': 'singleword', 'answer_frequency': f,
                         'frequency_slice': 'rare_1_5' if f <= 5 else 'freq_6_20',
                         'ocr_evidence': value, 'ocr_annotation_id': a['id'],
                         'original_target_unchanged': True})
            break
    rows.sort(key=lambda q: content_sha(['native_smalltext_v1:execution', q['question_id']]))
    images = {q['image_id']: {**source['images'][q['image_id']],
              'path': str((manifest_path.parent/source['images'][q['image_id']]['path']).resolve())} for q in rows}
    write_json(ROOT / 'CANDIDATE_ROWS.json', {'rows': rows, 'images': images})
    result = {'status': 'completed_prediction_free_inventory', 'qa': len(rows), 'images': len(images),
              'unique_targets': len(targets), 'filter_counts': dict(counts), 'policy': policy,
              'height_band': '[8,16) at existing1024 policy; separate difficulty diagnostic, not old gate revision',
              'frequency': dict(Counter(q['frequency_slice'] for q in rows)),
              'model_forwards': 0, 'optimizer_updates': 0,
              'manifest_sha256': sha_file(manifest_path), 'annotation_sha256': sha_file(annotations_path),
              'data_sha256': sha_file(ROOT / 'CANDIDATE_ROWS.json'), 'active_seconds': time.monotonic()-start}
    write_json(ROOT / 'INVENTORY.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
