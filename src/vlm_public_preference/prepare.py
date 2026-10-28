"""Prepare a frozen public preference candidate and 64-image review; never train."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import inspect
import json
from pathlib import Path
import statistics
import textwrap

from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq

from .data import (coco_source_id, content_sha, exclusion_reasons, freeze_audit, image_identity,
                   load_boundaries, normalized_question, one_pair_per_image,
                   parse_pair, sha_file, split_pairs, summary_counts, write_json)


def length_check(pair, root, processor, limit):
    """Encode both actual completions without truncation via the frozen SFT encoder."""
    from vlm_deep.common import encode
    outputs = {}
    for completion in ('chosen', 'rejected'):
        sample = encode(processor, {**pair, 'target': pair[completion], 'prompt_suffix': ''},
                        root / pair['image_path'], True)
        length = int(sample['input_ids'].shape[1])
        labels = sample['labels']
        grid = sample['image_grid_thw'].tolist()
        visual_tokens = sum(int(t*h*w//4) for t, h, w in grid)
        if not 0 < visual_tokens <= 1024 or int((labels != -100).sum()) <= 0:
            raise ValueError('Invalid visual budget or empty assistant mask')
        outputs[completion] = {'sequence_length': length,
                               'assistant_tokens': int((labels != -100).sum()),
                               'image_grid_thw': grid, 'visual_tokens': visual_tokens,
                               'input_ids_sha256': hashlib.sha256(sample['input_ids'].numpy().tobytes()).hexdigest()}
    if outputs['chosen']['image_grid_thw'] != outputs['rejected']['image_grid_thw']:
        raise ValueError('Chosen and rejected visual preprocessing differ')
    return {'pair_id': pair['pair_id'], 'limit': limit, 'truncation': False,
            'admitted': all(r['sequence_length'] <= limit for r in outputs.values()), **outputs}


def render_audit(root, audit):
    review = root / 'semantic_audit'
    review.mkdir(parents=True, exist_ok=True)
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen 64 image preference audit</title>',
             '<style>body{font:18px/1.5 sans-serif;max-width:1400px;margin:auto;padding:24px}article{border-top:2px solid #888;padding:24px}img{max-width:100%;max-height:650px}pre{white-space:pre-wrap;font:inherit}.chosen{color:#145422}.rejected{color:#842121}</style>',
             '<h1>Frozen 64 image preference audit</h1><p>Official human corrections; candidate labels remain unverified. Display images are derived thumbnails, never model-input records. Judge whether chosen is clearly better for the original question using the actual image. Responses are unmodified. No training is allowed while review is pending.</p>']
    for index, pair in enumerate(audit, 1):
        relative = Path('..') / pair['image_path']
        lines += [f'<article id="sample-{index:02d}"><h2>{index:02d} | official idx {html.escape(str(pair["source_idx"]))}</h2>',
                  f'<p>{html.escape(pair["origin_dataset"])} | {html.escape(pair["label_source"])}</p>',
                  f'<a href="{relative}"><img src="{relative}" alt="actual source image {index}"></a>',
                  f'<h3>Question</h3><pre>{html.escape(pair["question"])}</pre>',
                  f'<h3 class="chosen">Chosen</h3><pre>{html.escape(pair["chosen"])}</pre>',
                  f'<h3 class="rejected">Rejected</h3><pre>{html.escape(pair["rejected"])}</pre>',
                  f'<p>pair_id: {html.escape(pair["pair_id"])}<br>file SHA: {pair["image_sha256"]}<br>RGB pixel SHA: {pair["pixel_sha256"]}</p></article>']
    lines.append('</html>')
    (review / 'review_64.html').write_text('\n'.join(lines))
    # Full original responses appear in every contact sheet, without text truncation.
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    except OSError:
        font = title_font = ImageFont.load_default()
    sheet_files, image_grids = [], []
    for start in range(0, len(audit), 8):
        group = audit[start:start + 8]
        tile_width = 1000
        text_rows = []
        for offset, pair in enumerate(group):
            sections = [f'{start+offset+1:02d} | idx {pair["source_idx"]} | {pair["origin_dataset"]}',
                        'Q: ' + pair['question'], 'CHOSEN: ' + pair['chosen'],
                        'REJECTED: ' + pair['rejected']]
            text_rows.append([line for section in sections
                              for paragraph in section.splitlines() or ['']
                              for line in (textwrap.wrap(paragraph, 79, break_long_words=True) or [''])])
        row_heights = [440 + max(len(text_rows[i]), len(text_rows[i+1]) if i+1<len(group) else 0) * 29
                       for i in range(0, len(group), 2)]
        sheet = Image.new('RGB', (2 * tile_width, sum(row_heights)), 'white')
        draw = ImageDraw.Draw(sheet)
        y = 0
        for index, pair in enumerate(group):
            if index and index % 2 == 0:
                y += row_heights[index // 2 - 1]
            x = index % 2 * tile_width
            with Image.open(root / pair['image_path']) as source:
                display = source.convert('RGB')
                display.thumbnail((tile_width - 40, 400))
                sheet.paste(display, (x + (tile_width-display.width)//2, y + 10))
            for line_index, line in enumerate(text_rows[index]):
                draw.text((x + 16, y + 420 + line_index * 29), line,
                          fill='#111111', font=title_font if line_index == 0 else font)
            draw.rectangle((x, y, x+tile_width-1, y+row_heights[index//2]-1), outline='#666666', width=2)
        path = review / f'contact_sheet_{start//8+1:02d}.jpg'
        sheet.save(path, quality=92)
        sheet_files.append(str(path.relative_to(root)))
        grid = Image.new('RGB', (1600, 2000), 'white')
        grid_draw = ImageDraw.Draw(grid)
        for offset, pair in enumerate(group):
            x, y = offset % 2 * 800, offset // 2 * 500
            with Image.open(root / pair['image_path']) as source:
                display = source.convert('RGB')
                display.thumbnail((776, 444))
                grid.paste(display, (x + (800-display.width)//2, y + (444-display.height)//2 + 8))
            grid_draw.text((x+16, y+461), f'{start+offset+1:02d} | official idx {pair["source_idx"]}',
                           fill='#111111', font=title_font)
            grid_draw.rectangle((x, y, x+799, y+499), outline='#666666', width=2)
        grid_path = review / f'image_grid_{start//8+1:02d}.jpg'
        grid.save(grid_path, quality=94)
        image_grids.append(str(grid_path.relative_to(root)))
    write_json(review / 'CONTACT_SHEETS.json', {'display_only': True, 'full_text_untruncated': True,
                                               'files': sheet_files,
                                               'image_only_index_grids': image_grids,
                                               'text_source': 'semantic_audit/FROZEN_64.json'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--boundary-manifest', type=Path, action='append', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--max-sequence-length', type=int, default=2048)
    parser.add_argument('--encoding-workers', type=int, default=8)
    parser.add_argument('--exclude-coco-ids', type=Path)
    parser.add_argument('--encoding-cache', type=Path)
    parser.add_argument('--encoding-only', action='store_true',
                        help='Complete CPU lengths but stop before candidate/split/audit freeze')
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('All preparation outputs must be persistent')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'CANDIDATE_MANIFEST.json').exists():
        raise ValueError('Candidate is frozen; use a distinct version instead of overwriting')
    download = json.loads((args.download_root / 'DOWNLOAD_LOCK.json').read_text())
    repository, revision = download['repository'], download['revision']
    boundaries = load_boundaries(args.boundary_manifest)
    from vlm_probe.evaluate import processor_for
    import torch
    torch.set_num_threads(1)
    reference = json.loads(args.reference_config.read_text())
    processor = processor_for(args.model_root, reference, 1024)
    coco_ids = set()
    if args.exclude_coco_ids:
        source_ids = json.loads(args.exclude_coco_ids.read_text())
        values = source_ids if isinstance(source_ids, list) else source_ids['coco_image_ids']
        coco_ids = {str(int(value)).zfill(12) for value in values}
        write_json(root / 'EXTERNAL_COCO_ID_BOUNDARY.json', {
            'source_path': str(args.exclude_coco_ids), 'file_sha256': sha_file(args.exclude_coco_ids),
            'coco_image_ids': sorted(coco_ids), 'source_metadata': source_ids if isinstance(source_ids, dict) else None})
    write_json(root / 'BOUNDARY_LOCK.json', boundaries['lock'])
    write_json(root / 'PREPARATION_POLICY.json', {
        'schema_version': 'rarevlm.public_preference.policy.v1', 'source_repository': repository,
        'source_revision': revision, 'source_split': 'train', 'selection_seed': args.seed,
        'source_exclusion': ['TextVQA', 'DocVQA', 'MMStar'],
        'technical_nonpreference_exclusion': 'empty/nontext question or response; chosen/rejected identical after stripping outer whitespace',
        'external_coco_id_exclusion': {'count': len(coco_ids),
            'source_sha256': sha_file(args.exclude_coco_ids) if args.exclude_coco_ids else None},
        'identity_exclusion': 'all existing TextVQA train_all/dev/final and formal DocVQA/MMStar file and RGB pixel SHA',
        'one_pair_per_image': 'minimum salted SHA of pair identity, grouping file and RGB pixel SHA',
        'split': '90/10 hash of pixel SHA, seed42; independent of old dev/test and answer quality',
        'audit': '64 minimum salted SHA from preference_train, frozen before any semantic judgment',
        'semantic_gate': {'clear_preference_fraction_at_least': 0.90,
                          'wilson_95_lower_bound_at_least': 0.80},
        'prompt_suffix': '', 'response_text': 'unaltered official strings; only technical total-sequence admission, no rewriting or answer-quality/length ranking',
        'length_admission': {'max_sequence_length': args.max_sequence_length,
                             'both_completions_actual_frozen_encode': True, 'truncation': False,
                             'visual_token_budget': 1024, 'min_pixels': reference['min_pixels'],
                             'model_root': str(args.model_root),
                             'model_id': reference['model_id'], 'model_revision': reference['model_revision'],
                             'reference_config_sha256': sha_file(args.reference_config)},
        'training_allowed': False, 'old_failed_749_and_199_used': False,
        'qid_policy': 'official preference idx is a new namespace, not original benchmark question_id; same question text is reported separately and image identity controls leakage',
        'pretraining_overlap': 'unknown; this check cannot establish pretraining non-overlap'})
    accepted, exclusions, identities, question_text_matches, numeric_idx_matches = [], [], {}, [], []
    counts, total = Counter(), 0
    for name, binding in download['files'].items():
        if not name.endswith('.parquet'):
            continue
        path = args.download_root / name
        if path.stat().st_size != binding['bytes'] or sha_file(path) != binding['sha256']:
            raise ValueError('Official download changed')
        if binding['official_lfs_sha256'] != binding['sha256']:
            raise ValueError('Official SHA not verified')
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                total += 1
                data = row['image']['bytes']
                if not data:
                    raise ValueError('Official embedded image bytes missing')
                identity = image_identity(data)
                try:
                    pair = parse_pair(row, repository, revision)
                except ValueError as error:
                    if str(error) not in ('Empty or nontext question/response', 'Identical chosen/rejected strings'):
                        raise
                    reason = 'empty_or_nontext_preference' if str(error).startswith('Empty') else 'identical_chosen_rejected'
                    exclusions.append({'pair_id': f'{repository}@{revision}:{row["idx"]}',
                                       'source_idx': row['idx'], 'origin_dataset': row['origin_dataset'],
                                       **identity, 'reasons': [reason]})
                    counts.update([reason])
                    continue
                reasons = exclusion_reasons(pair, identity, boundaries)
                if coco_source_id(pair['source_image_path']) in coco_ids:
                    reasons.append('external_pope_coco_source_id')
                record = {'pair_id': pair['pair_id'], 'source_idx': pair['source_idx'],
                          'origin_dataset': pair['origin_dataset'], **identity}
                question = normalized_question(pair['question'])
                if question in boundaries['questions']:
                    question_text_matches.append({'pair_id': pair['pair_id'], 'question': pair['question'],
                                                  'matches': boundaries['questions'][question],
                                                  'image_identity_overlap': bool(reasons and any('sha' in r for r in reasons))})
                if str(pair['source_idx']) in boundaries['raw_question_ids']:
                    numeric_idx_matches.append({'pair_id': pair['pair_id'], 'raw_idx': pair['source_idx'],
                                                'namespace_same': False,
                                                'existing_count': len(boundaries['raw_question_ids'][str(pair['source_idx'])])})
                if reasons:
                    exclusions.append({**record, 'reasons': reasons})
                    counts.update(reasons)
                    continue
                relative = f'images/{identity["sha256"]}.image'
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha_file(destination) != identity['sha256']:
                    raise ValueError('Prepared image collision')
                if not destination.exists():
                    destination.write_bytes(data)
                identities[identity['sha256']] = {**identity, 'path': relative}
                accepted.append({**pair, 'image_path': relative, 'image_sha256': identity['sha256'],
                                 'image_id': 'public_preference:' + identity['pixel_sha256'],
                                 'pixel_sha256': identity['pixel_sha256'],
                                 'image_width': identity['width'], 'image_height': identity['height']})
            if total % 512 == 0:
                print('decoded', total, 'candidate_rows', len(accepted), flush=True)
    cache = {}
    if args.encoding_cache:
        previous = json.loads(args.encoding_cache.read_text())
        binding = previous['binding']
        current_processor = {p.name: sha_file(p) for p in args.model_root.iterdir()
            if p.is_file() and (p.suffix == '.json' or p.name in ('merges.txt', 'vocab.json'))}
        if (binding['download_lock_sha256'] != sha_file(args.download_root / 'DOWNLOAD_LOCK.json')
                or binding['length_admission']['reference_config_sha256'] != sha_file(args.reference_config)
                or binding['length_admission']['max_sequence_length'] != args.max_sequence_length
                or binding['length_admission']['model_root'] != str(args.model_root)
                or binding['processor_files_sha256'] != current_processor
                or binding['length_check_sha256'] != hashlib.sha256(inspect.getsource(length_check).rstrip().encode()).hexdigest()
                or any(sha_file(path) != digest for path, digest in binding['frozen_sources'].items())):
            raise ValueError('CPU encoding cache input/encoder changed')
        cache = {record['pair_id']: record for record in previous['records']}
    encoded, length_admitted = [], []
    with ThreadPoolExecutor(max_workers=args.encoding_workers) as pool:
        checks = pool.map(lambda pair: cache.get(pair['pair_id']) or length_check(pair, root, processor, args.max_sequence_length), accepted)
        with (root / 'ENCODING_AUDIT.jsonl').open('w') as stream:
            for pair, check in zip(accepted, checks):
                stream.write(json.dumps(check, ensure_ascii=False) + '\n')
                encoded.append(check)
                if check['admitted']:
                    length_admitted.append({**pair, 'encoding': {key: check[key] for key in ('chosen', 'rejected')}})
                if len(encoded) % 256 == 0:
                    print('both_completions_encoded', len(encoded), '/', len(accepted), flush=True)
    encoding_lock = {
        'encoding_audit_sha256': sha_file(root / 'ENCODING_AUDIT.jsonl'),
        'both_completion_pairs_checked': len(encoded), 'admitted_pairs': len(length_admitted),
        'excluded_pair_ids': [r['pair_id'] for r in encoded if not r['admitted']],
        'max_sequence_length': args.max_sequence_length, 'truncation': False,
        'reused_cpu_encoded_pairs': sum(pair['pair_id'] in cache for pair in accepted),
        'encoding_cache_sha256': sha_file(args.encoding_cache) if args.encoding_cache else None,
        'source_sha256': {str(p): sha_file(p) for p in [Path(__file__),
            Path(__file__).parents[1] / 'vlm_deep/common.py',
            Path(__file__).parents[1] / 'vlm_sft/common.py',
            Path(__file__).parents[1] / 'vlm_probe/evaluate.py']},
        'processor_files_sha256': {p.name: sha_file(p) for p in args.model_root.iterdir()
            if p.is_file() and (p.suffix == '.json' or p.name in ('merges.txt', 'vocab.json'))}}
    write_json(root / 'ENCODING_LOCK.json', encoding_lock)
    if args.encoding_only:
        print(json.dumps({'state': 'length_encoding_complete_not_frozen',
                          'checked_pairs': len(encoded), 'admitted_pairs': len(length_admitted),
                          'exclusions': dict(counts), 'training_allowed': False}), flush=True)
        return
    unique, duplicates = one_pair_per_image(length_admitted, args.seed)
    splits = split_pairs(unique, args.seed)
    audit = freeze_audit(splits['preference_train'], 64, args.seed)
    manifest = {'schema_version': 'rarevlm.public_preference.candidate.v1',
                'source_repository': repository, 'source_revision': revision, 'source_split': 'train',
                'license': download['license'], 'download_lock_sha256': sha_file(args.download_root / 'DOWNLOAD_LOCK.json'),
                'boundary_lock_sha256': sha_file(root / 'BOUNDARY_LOCK.json'),
                'policy_sha256': sha_file(root / 'PREPARATION_POLICY.json'),
                'encoding_lock_sha256': sha_file(root / 'ENCODING_LOCK.json'),
                'external_coco_id_boundary_sha256': sha_file(root / 'EXTERNAL_COCO_ID_BOUNDARY.json') if coco_ids else None,
                'image_root': str(root), 'images': identities, 'splits': splits,
                'training_allowed': False, 'semantic_review_state': 'pending_frozen_64',
                'source_files_sha256': {str(p): sha_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    manifest['manifest_sha256'] = content_sha(manifest)
    write_json(root / 'CANDIDATE_MANIFEST.json', manifest)
    for split, name in [('preference_train', 'TRAIN_MANIFEST.json'), ('preference_heldout', 'HELDOUT_MANIFEST.json')]:
        exported = {'schema_version': 'rarevlm.public_preference.training_rows.v1',
                    'source_candidate_manifest_sha256': manifest['manifest_sha256'],
                    'split': split, 'prompt_suffix': '', 'training_allowed': False,
                    'rows': [{**pair, 'image_path_relative': pair['image_path'],
                              'image_path': str(root / pair['image_path'])} for pair in splits[split]]}
        write_json(root / name, exported)
    write_json(root / 'EXCLUSIONS.json', {'excluded_rows': exclusions, 'duplicate_image_pair_ids': duplicates})
    write_json(root / 'QUESTION_ID_TEXT_AUDIT.json', {'normalized_question_text_matches': question_text_matches,
                'raw_numeric_idx_collisions': numeric_idx_matches,
                'qid_namespace_note': 'preference row idx is not source benchmark qid; numeric equality is not identity',
                'question_text_note': 'generic question text can repeat across unrelated images; all image overlaps are excluded'})
    audit_payload = {'selection': 'minimum fixed salted SHA, preference_train only, before review',
                     'seed': args.seed, 'sample_count': 64,
                     'candidate_manifest_sha256': manifest['manifest_sha256'],
                     'judgments': 'pending', 'training_allowed': False, 'samples': audit}
    audit_payload['audit_sha256'] = content_sha(audit_payload)
    write_json(root / 'semantic_audit' / 'FROZEN_64.json', audit_payload)
    render_audit(root, audit)
    summary = {'state': 'prepared_pending_semantic_review', 'official_rows': total,
               'source_rows_after_origin_and_sha_exclusion': len(accepted),
               'origin_and_sha_excluded_rows': len(exclusions), 'multilabel_exclusion_reasons': dict(counts),
               'both_completion_length_checked_pairs': len(encoded),
               'both_completion_length_admitted_pairs': len(length_admitted),
               'length_excluded_pairs': len(encoded) - len(length_admitted),
               'reused_cpu_encoding_pairs': sum(pair['pair_id'] in cache for pair in accepted),
               'one_pair_per_image_excluded_rows': len(duplicates),
               'splits': {split: summary_counts(rows) for split, rows in splits.items()},
               'question_text_match_rows': len(question_text_matches),
               'question_text_and_image_identity_overlap_rows': sum(r['image_identity_overlap'] for r in question_text_matches),
               'raw_idx_collisions_different_namespace': len(numeric_idx_matches),
               'chosen_words': {'median': statistics.median(r['chosen_words'] for r in unique),
                                'max': max(r['chosen_words'] for r in unique)},
               'rejected_words': {'median': statistics.median(r['rejected_words'] for r in unique),
                                  'max': max(r['rejected_words'] for r in unique)},
               'training_launched': False, 'optimizer_updates': 0, 'gpu_used': False,
               'audit_sha256': audit_payload['audit_sha256'],
               'candidate_manifest_sha256': manifest['manifest_sha256']}
    summary['train_manifest_sha256'] = sha_file(root / 'TRAIN_MANIFEST.json')
    summary['heldout_manifest_sha256'] = sha_file(root / 'HELDOUT_MANIFEST.json')
    write_json(root / 'DATA_ADMISSION.json', {'status': 'pending_semantic_review',
        'train_manifest_sha256': summary['train_manifest_sha256'],
        'audit_sha256': audit_payload['audit_sha256'], 'training_allowed': False})
    write_json(root / 'PREPARATION_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
