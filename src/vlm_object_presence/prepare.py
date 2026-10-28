"""Prepare the complete eligible COCO pool without a model forward or training."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time
import zipfile

from PIL import Image, ImageDraw, ImageFont

from vlm_public_preference.data import load_boundaries
from .data import (PROMPT_SUFFIX, annotation_index, assert_disjoint, class_balanced_validation, content_sha,
                   corpus_summary, cpu_encode, definite_coco_source, frozen_audit,
                   image_identity, rows_for_image, sha_file, write_json)


def render_audit(root, rows, images):
    directory = root / 'semantic_audit'
    directory.mkdir(parents=True, exist_ok=True)
    text = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen COCO presence audit</title>',
            '<style>body{font:18px/1.5 sans-serif;max-width:1300px;margin:auto;padding:24px}article{border-top:2px solid #888;padding:24px}img{max-width:100%;max-height:700px}pre{white-space:pre-wrap}</style>',
            '<h1>Frozen 64-question COCO annotation audit</h1>',
            '<p>32 yes and 32 no fixed by salted SHA before review. Native source images are linked unchanged. Box displays are derived review overlays, not model input. Closed-world COCO annotation labels can miss visible objects and do not constitute human preferences or perfect visual truth. Negative questions have no boxes because their category is absent from all annotations including crowd. This audit checks a small frozen sample only.</p>']
    font = ImageFont.truetype('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf', 20)
    contacts = []
    for start in range(0, len(rows), 8):
        group = rows[start:start+8]
        sheet = Image.new('RGB', (1600, len(group)//2*640), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(group):
            number = start + offset + 1
            image = images[row['image_id']]
            source = Path(row['image_path'])
            relative = os.path.relpath(source, directory)
            with Image.open(source) as raw:
                display = raw.convert('RGB')
            overlay = display.copy()
            overlay_draw = ImageDraw.Draw(overlay)
            for box in row['bbox_evidence']:
                x, y, w, h = box['bbox_current_image_xywh']
                overlay_draw.rectangle((x, y, x+w, y+h), outline='#00ff00', width=3)
            overlay_name = f'evidence_{number:02d}.png'
            overlay.save(directory / overlay_name)
            ox, oy = offset % 2 * 800, offset//2 * 640
            overlay.thumbnail((776, 485))
            sheet.paste(overlay, (ox+(800-overlay.width)//2, oy+10))
            lines = [f'{number:02d} | {row["target"].upper()} | COCO {row["coco_image_id"]}',
                     row['question'], f'category {row["category_id"]}: {row["category_name"]}',
                     f'noncrowd boxes: {len(row["bbox_evidence"])}']
            for k, line in enumerate(lines):
                draw.text((ox+12, oy+505+k*28), line, font=font, fill='#111111')
            draw.rectangle((ox, oy, ox+799, oy+639), outline='#888888', width=2)
            text += [f'<article id="sample-{number:02d}"><h2>{number:02d} | {html.escape(row["question_id"])}</h2>',
                     f'<p><strong>{html.escape(row["question"])}</strong><br>Annotation target: {row["target"]}<br>Category: {row["category_id"]} {html.escape(row["category_name"])}</p>',
                     f'<p>Native source image ({image["width"]}×{image["height"]}), unchanged:</p><a href="{relative}"><img src="{relative}"></a>',
                     f'<p>Derived annotation overlay for review only:</p><img src="{overlay_name}">',
                     '<pre>' + html.escape(json.dumps({'label_rule': row['label_rule'],
                         'coco_width': image['coco_width'], 'coco_height': image['coco_height'],
                         'all_annotated_categories': image['annotated_any_including_crowd_category_ids'],
                         'bbox_evidence': row['bbox_evidence'], 'file_sha256': row['file_sha256'],
                         'pixel_sha256': row['pixel_sha256']}, indent=2)) + '</pre></article>']
        filename = f'contact_sheet_{start//8+1:02d}.jpg'
        sheet.save(directory / filename, quality=94)
        contacts.append(str(Path('semantic_audit') / filename))
    text.append('</html>')
    (directory / 'review_64.html').write_text('\n'.join(text))
    write_json(directory / 'CONTACT_SHEETS.json', {'display_only': True,
        'model_input_modified': False, 'native_image_paths_in_frozen_rows': True, 'files': contacts})


def extract_annotations(root):
    archive = root / 'annotations/annotations_trainval2017.zip'
    lock_path = root / 'annotations/DOWNLOAD_LOCK.json'
    lock = json.loads(lock_path.read_text())
    if archive.stat().st_size != 252907541 or sha_file(archive) != lock['sha256']:
        raise ValueError('Official annotation archive size/SHA changed')
    indexes, bindings = {}, []
    with zipfile.ZipFile(archive) as zipped:
        for split in ('train2017', 'val2017'):
            member = f'annotations/instances_{split}.json'
            destination = root / member
            if not destination.exists():
                destination.write_bytes(zipped.read(member))
            if hashlib.sha256(zipped.read(member)).hexdigest() != sha_file(destination):
                raise ValueError('Extracted official annotation JSON differs from archive')
            document = json.loads(destination.read_text())
            indexes[split] = annotation_index(document, split)
            bindings.append({'archive_member': member, 'path': str(destination),
                             'sha256': sha_file(destination), 'bytes': destination.stat().st_size,
                             'image_count': len(document['images']),
                             'annotation_count': len(document['annotations']),
                             'category_count': len(document['categories']), 'info': document['info']})
    if indexes['train2017']['categories'] != indexes['val2017']['categories']:
        raise ValueError('Official COCO category mappings differ by split')
    return indexes, {'download_lock_sha256': sha_file(lock_path), 'archive': lock, 'json_files': bindings}


def boundary_coco_ids(lock):
    ids = set()
    for binding in lock:
        manifest = json.loads(Path(binding['path']).read_text())
        for image in manifest['images'].values():
            if image.get('coco_image_id') is not None:
                ids.add(f'{int(image["coco_image_id"]):012d}')
            # Only explicit COCO prefixes or coco2017 paths count as source IDs.
            for key in ('path', 'source_image_path'):
                value = image.get(key, '')
                match = re.search(r'(?:COCO_(?:train|val)2014_|coco2017/(?:train|val)2017/)(\d{12})\.', value)
                if match:
                    ids.add(match.group(1))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--encoding-workers', type=int, default=8)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('All outputs must remain persistent')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'COMPLETION.json').exists():
        raise ValueError('Corpus is frozen and complete; use a separate version for changes')
    start = time.time()
    indexes, annotation_lock = extract_annotations(root)
    boundary_lock_path = args.source_root / 'BOUNDARY_LOCK.json'
    boundary_lock = json.loads(boundary_lock_path.read_text())
    for binding in boundary_lock:
        if sha_file(binding['path']) != binding['file_sha256']:
            raise ValueError('Prior boundary manifest file SHA changed')
    boundaries = load_boundaries([binding['path'] for binding in boundary_lock])
    blocked_ids = boundary_coco_ids(boundary_lock)
    external = args.source_root / 'EXTERNAL_COCO_ID_BOUNDARY.json'
    blocked_ids.update(json.loads(external.read_text())['coco_image_ids'])
    if len(blocked_ids) < 500:
        raise ValueError('Entire POPE 500-image source-ID boundary was not loaded')
    source_lock = {'annotation_lock': annotation_lock, 'source_manifests': {
        name: {'path': str(args.source_root / name), 'sha256': sha_file(args.source_root / name)}
        for name in ('TRAIN_MANIFEST.json', 'HELDOUT_MANIFEST.json')},
        'boundary_manifests': boundary_lock, 'blocked_coco_ids': sorted(blocked_ids),
        'external_coco_boundary': {'path': str(external), 'sha256': sha_file(external)},
        'reference_config': {'path': str(args.reference_config), 'sha256': sha_file(args.reference_config)},
        'model_root': str(args.model_root), 'processor_files': {
            path.name: sha_file(path) for path in args.model_root.iterdir()
            if path.is_file() and (path.suffix == '.json' or path.name in ('merges.txt', 'vocab.json'))},
        'audit_font': {'path': '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf',
            'sha256': sha_file('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf')},
        'code_files': {str(path): sha_file(path) for path in [
            Path(__file__), Path(__file__).with_name('data.py'),
            Path(__file__).parents[1] / 'vlm_deep/common.py',
            Path(__file__).parents[1] / 'vlm_probe/evaluate.py']}}
    source_lock['lock_sha256'] = content_sha(source_lock)
    previous_lock = root / 'SOURCE_LOCK.json'
    if previous_lock.exists() and json.loads(previous_lock.read_text()) != source_lock:
        raise ValueError('Preparation inputs/code changed; refusing to reuse partial outputs')
    write_json(previous_lock, source_lock)
    policy = {'schema_version': 'rarevlm.object_presence.policy.v1',
        'pool': 'all definite COCO IDs in previously frozen RLHF-V train/heldout image pools',
        'source_split_preserved': True, 'positive_policy': 'all unique noncrowd annotated categories per image; no size filtering',
        'negative_policy': 'equal count per image, absent from every annotation including crowd; lowest salted SHA category IDs',
        'seed': 42, 'question_template': 'Is there a {object} in the image?',
        'prompt_suffix': PROMPT_SUFFIX, 'labels': ['yes', 'no'], 'fabricated_human_answers': False,
        'annotation_scope': 'closed-world COCO 2017 instances; not perfect visual truth or human preference',
        'known_risks': ['missing annotations can create false negatives',
                        'category ontology may differ from natural-language visible-object judgments',
                        'resized public images checked by metadata/aspect ratio, not original-image byte identity',
                        'pretraining overlap and near duplicates unknown'],
        'selection_uses_model_predictions_or_prior_errors': False, 'full_eligible_inventory': True,
        'boundaries': 'all old Text train_all/dev/final, DocVQA, MMStar and all POPE500 by file/pixel SHA and known COCO source IDs',
        'sequence_limit': 2048, 'visual_token_budget': 1024, 'input_truncation': False,
        'audit': 'yes32 then no32 in salted SHA order, skipping previously selected pixel SHA across both labels, then independent SHA ordering; 64 distinct training images before review',
        'question_prior_control': 'VAL_CLASS_BALANCED_MANIFEST per-category yes/no min counts, salted SHA selected; grounding diagnostic only',
        'training_allowed': False, 'approval_note': 'Data preparation only; modeling protocol/admission is separate'}
    write_json(root / 'PREPARATION_POLICY.json', policy)
    records, images, inventory = {}, {}, []
    for split, filename in [('train', 'TRAIN_MANIFEST.json'), ('val', 'HELDOUT_MANIFEST.json')]:
        rows, image_map = [], {}
        for source in json.loads((args.source_root / filename).read_text())['rows']:
            item = {'source_pair_id': source['pair_id'], 'source_split': split,
                    'source_image_path': source['source_image_path'], 'image_path': source['image_path']}
            mapping = definite_coco_source(source['source_image_path'])
            if mapping is None:
                inventory.append({**item, 'admitted': False, 'reasons': ['not_a_definite_coco2017_source_path']})
                continue
            actual = image_identity(Path(source['image_path']).read_bytes())
            if (actual['sha256'] != source['image_sha256'] or actual['pixel_sha256'] != source['pixel_sha256']
                    or actual['width'] != source['image_width'] or actual['height'] != source['image_height']):
                raise ValueError('Existing source image bytes/pixels/size changed')
            reasons = []
            if actual['sha256'] in boundaries['files']:
                reasons.append('existing_boundary_file_sha')
            if actual['pixel_sha256'] in boundaries['pixels']:
                reasons.append('existing_boundary_pixel_sha')
            if f'{mapping[1]:012d}' in blocked_ids:
                reasons.append('existing_boundary_coco_source_id')
            if reasons:
                inventory.append({**item, **actual, 'admitted': False, 'reasons': reasons})
                continue
            created, image = rows_for_image(source, actual, indexes[mapping[0]], split)
            if not created:
                inventory.append({**item, **actual, 'admitted': False, 'reasons': ['no_noncrowd_positive_category']})
                continue
            image_id = created[0]['image_id']
            if image_id in image_map:
                raise ValueError('Duplicate source COCO image within a source split')
            image_map[image_id] = image
            rows += created
            inventory.append({**item, **actual, 'admitted': True, 'reasons': [],
                              'coco_image_id': image['coco_image_id'], 'questions': len(created),
                              'positive_categories': len(created)//2})
        records[split], images[split] = sorted(rows, key=lambda row: row['question_id']), image_map
    assert_disjoint(records['train'], records['val'])
    write_json(root / 'INVENTORY.json', {'rows': inventory,
        'reasons': dict(Counter(reason for item in inventory for reason in item['reasons']))})
    for split, filename in [('train', 'TRAIN_MANIFEST.json'), ('val', 'VAL_MANIFEST.json')]:
        manifest = {'schema_version': 'rarevlm.object_presence.manifest.v1', 'split': split,
                    'source_lock_sha256': source_lock['lock_sha256'], 'prompt_suffix': PROMPT_SUFFIX,
                    'training_requires_external_admission': True, 'label_source': policy['annotation_scope'],
                    'rows': records[split], 'images': images[split]}
        manifest['manifest_sha256'] = content_sha(manifest)
        write_json(root / filename, manifest)
    balanced, balance_counts = class_balanced_validation(records['val'])
    balanced_image_ids = {row['image_id'] for row in balanced}
    balanced_images = {key: value for key, value in images['val'].items() if key in balanced_image_ids}
    balanced_manifest = {'schema_version': 'rarevlm.object_presence.manifest.v1',
        'split': 'val_class_balanced', 'source_lock_sha256': source_lock['lock_sha256'],
        'prompt_suffix': PROMPT_SUFFIX, 'training_requires_external_admission': True,
        'diagnostic_only': True, 'scope': 'category-balanced grounding diagnostic; does not replace complete validation or POPE',
        'selection': 'within each category take min(Nyes,Nno) per label by fixed salted SHA; no model result used',
        'category_balance': balance_counts, 'rows': balanced, 'images': balanced_images}
    balanced_manifest['manifest_sha256'] = content_sha(balanced_manifest)
    write_json(root / 'VAL_CLASS_BALANCED_MANIFEST.json', balanced_manifest)
    summary = {split: corpus_summary(records[split], images[split]) for split in ('train', 'val')}
    summary['val_class_balanced'] = corpus_summary(balanced, balanced_images)
    summary['question_prior_bias_risk'] = 'Per-image balance does not balance each category. Complete train/val category label counts are provided; category-balanced validation is a separate grounding diagnostic.'
    summary.update(source_pool_images=len(inventory), excluded_images=sum(not row['admitted'] for row in inventory),
                   train_val_overlap_zero=['question_id', 'image_id', 'file_sha256', 'pixel_sha256', 'coco_image_id'],
                   external_boundary_overlap_zero=['file_sha256', 'pixel_sha256', 'known_coco_source_id'],
                   annotation_only_closed_world_labels=True, data_semantic_audit='pending')
    write_json(root / 'SUMMARY.json', summary)
    audit = frozen_audit(records['train'])
    audit_binding = {'schema_version': 'rarevlm.object_presence.frozen_audit.v1',
        'source_train_manifest_sha256': sha_file(root / 'TRAIN_MANIFEST.json'), 'seed': 42,
        'labels': {'yes': 32, 'no': 32}, 'unique_images_by_pixel_sha': 64,
        'selection': policy['audit'], 'review_state': 'pending', 'rows': audit}
    audit_binding['audit_sha256'] = content_sha(audit_binding)
    write_json(root / 'semantic_audit/FROZEN_64.json', audit_binding)
    render_audit(root, audit, images['train'])
    print('inventory_frozen', json.dumps(summary), flush=True)
    from vlm_probe.evaluate import processor_for
    import torch
    torch.set_num_threads(1)
    processor = processor_for(args.model_root, json.loads(args.reference_config.read_text()), 1024)
    all_rows = records['train'] + records['val']
    journal_path = root / 'ENCODING_CHECKS.jsonl'
    checks = {}
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            if line:
                check = json.loads(line)
                checks[check['question_id']] = check
    todo = [row for row in all_rows if row['question_id'] not in checks]
    with journal_path.open('a') as journal, ThreadPoolExecutor(max_workers=args.encoding_workers) as pool:
        for index, check in enumerate(pool.map(lambda row: cpu_encode(row, processor), todo), 1):
            checks[check['question_id']] = check
            journal.write(json.dumps(check, separators=(',', ':')) + '\n')
            if index % 128 == 0 or index == len(todo):
                journal.flush()
                write_json(root / 'STATUS.json', {'phase': 'cpu_encoding', 'complete': len(checks),
                    'total': len(all_rows), 'elapsed_seconds': time.time()-start,
                    'training_allowed': False, 'training_requires_external_admission': True})
                print('cpu_encoded', len(checks), '/', len(all_rows), flush=True)
    if set(checks) != {row['question_id'] for row in all_rows}:
        raise ValueError('Actual encoding inventory does not exactly match the corpus')
    sequence_lengths = [check['sequence_length'] for check in checks.values()]
    visual_tokens = [check['visual_tokens'] for check in checks.values()]
    encoding_summary = {'questions': len(checks), 'actual_frozen_cpu_encode': True,
        'model_forward_or_gpu_used': False, 'truncation': False, 'limit': 2048,
        'sequence_length_min': min(sequence_lengths), 'sequence_length_max': max(sequence_lengths),
        'visual_tokens_min': min(visual_tokens), 'visual_tokens_max': max(visual_tokens),
        'visual_token_budget': 1024, 'assistant_tokens': dict(Counter(
            check['assistant_tokens'] for check in checks.values())),
        'journal_sha256': sha_file(journal_path)}
    write_json(root / 'ENCODING_SUMMARY.json', encoding_summary)
    completion = {'complete': True, 'data_only': True, 'optimizer_updates': 0,
        'elapsed_seconds': time.time()-start, 'summary': summary, 'encoding': encoding_summary,
        'source_lock_sha256': source_lock['lock_sha256'], 'training_allowed': False,
        'artifacts_sha256': {name: sha_file(root / name) for name in [
            'TRAIN_MANIFEST.json', 'VAL_MANIFEST.json', 'INVENTORY.json', 'PREPARATION_POLICY.json',
            'VAL_CLASS_BALANCED_MANIFEST.json', 'SUMMARY.json', 'SOURCE_LOCK.json',
            'ENCODING_SUMMARY.json', 'semantic_audit/FROZEN_64.json']}}
    write_json(root / 'COMPLETION.json', completion)
    write_json(root / 'STATUS.json', {'phase': 'complete', 'complete': len(checks),
        'total': len(all_rows), 'elapsed_seconds': time.time()-start,
        'training_allowed': False, 'training_requires_external_admission': True})
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
