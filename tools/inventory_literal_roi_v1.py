"""Official given-word-region OCR supervision, prediction-free inventory only."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import re
import time
import unicodedata
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, read_manifest, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_region_ocr_sft_v1.data import bounds

BASE = PROJECT_ROOT.parents[1]/'outputs'
ROOT = BASE/'vlm_literal_roi_v1'


def canonical(s):
    return unicodedata.normalize('NFKC', str(s)).casefold().strip()


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def main():
    if (ROOT/'INVENTORY.json').exists():
        raise FileExistsError('Inventory exists')
    start = time.monotonic()
    path = BASE/'vlm_deep_v2/prepared/manifest.json'
    source = read_manifest(path)
    ann_path = BASE/'vlm_region_ocr_sft_v1/source_data/TextOCR_0.1_train.json'
    annotations = read(ann_path)
    previous = read(BASE/'vlm_reference_replay_v1/DATA.json')
    excluded = {q['image_id'] for q in previous['training_rows']}
    for rows in previous['evaluations'].values():
        excluded.update(q['image_id'] for q in rows)
    for study, name in [('vlm_native_ocr_correction_v1', 'revision2/DATA.json'), ('vlm_native_smalltext_v1', 'CANDIDATE_ROWS.json')]:
        excluded.update(q['image_id'] for q in read(BASE/study/name)['rows'])
    for fold in ('dev', 'final'):
        excluded.update(q['image_id'] for q in source['splits'][fold])
    meta = {**source['images'], **previous['images']}
    pixel = lambda iid: meta[iid].get('pixel_sha256', meta[iid]['sha256'])
    excluded_pixels = {pixel(i) for i in excluded if i in meta}
    frequency = Counter(w for q in source['splits']['train_all'] for w in re.findall(r'[a-z]+', canonical(q['target'])))
    candidates = defaultdict(list)
    counts = Counter()
    for iid in {q['image_id'] for q in source['splits']['train_all']} - excluded:
        im = source['images'][iid]
        if iid not in annotations['imgs'] or pixel(iid) in excluded_pixels:
            continue
        raw = annotations['imgs'][iid]
        if (im['width'], im['height']) != (raw['width'], raw['height']):
            continue
        boxes = []
        try:
            for aid in annotations['imgToAnns'][iid]:
                a = annotations['anns'][aid]
                boxes.append((a, bounds(a['points'], im['width'], im['height'])))
        except (ValueError, TypeError, KeyError):
            counts['image_with_invalid_annotation'] += 1
            continue
        for a, box in boxes:
            target = str(a.get('utf8_string', '')).strip()
            if not re.fullmatch(r'[A-Za-z]{3,20}', target) or frequency[canonical(target)] > 5:
                continue
            x0, y0, x1, y1 = box
            if x1-x0 < 48 or y1-y0 < 16:
                continue
            points = a['points']
            if abs(math.degrees(math.atan2(points[3]-points[1], points[2]-points[0]))) > 10:
                continue
            crop = [max(0, math.floor(x0)-2), max(0, math.floor(y0)-2), min(im['width'], math.ceil(x1)+2), min(im['height'], math.ceil(y1)+2)]
            if crop[0] == 0 or crop[1] == 0 or crop[2] == im['width'] or crop[3] == im['height']:
                continue
            if any(other['id'] != a['id'] and intersects(crop, obox) for other, obox in boxes):
                continue
            candidates[iid].append({'source_image_id': iid, 'annotation_id': a['id'], 'target': target,
                                    'crop_bounds_original_pixels': crop, 'word_bounds_original_pixels': box,
                                    'sft_answer_word_frequency': frequency[canonical(target)],
                                    'frequency_slice': 'zero_target_word_exposure' if not frequency[canonical(target)] else 'rare_1_5'})
            counts['eligible_word_regions'] += 1
    rows, targets, pixels = [], set(), set()
    for iid in sorted(candidates, key=lambda x: content_sha(['literal_roi_v1:image', x])):
        for q in sorted(candidates[iid], key=lambda q: content_sha(['literal_roi_v1:word', q['annotation_id']])):
            if canonical(q['target']) in targets or pixel(iid) in pixels:
                continue
            targets.add(canonical(q['target']))
            pixels.add(pixel(iid))
            rows.append(q)
            break
    rows.sort(key=lambda q: content_sha(['literal_roi_v1:execution', q['source_image_id'], q['annotation_id']]))
    images = {q['source_image_id']: {**source['images'][q['source_image_id']],
              'path': str((path.parent/source['images'][q['source_image_id']]['path']).resolve())} for q in rows}
    write_json(ROOT/'CANDIDATE_ROWS.json', {'rows': rows, 'source_images': images})
    result = {'status': 'completed_prediction_free_inventory', 'unique_source_images_and_words': len(rows),
              'counts': dict(counts), 'frequency': dict(Counter(q['frequency_slice'] for q in rows)),
              'manifest_sha256': sha_file(path), 'annotation_sha256': sha_file(ann_path),
              'given_word_region_task': True, 'new_model_calls': 0, 'optimizer_updates': 0,
              'active_seconds': time.monotonic()-start}
    write_json(ROOT/'INVENTORY.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
