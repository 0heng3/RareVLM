"""Independent fixed1536 supplement after the original pool's review shortfall."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import textwrap
import time

from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read
import run_native_correction_mining_v2 as parent

ROOT = parent.BASE / 'vlm_native_correction_supplement_v2'
MERGED = parent.BASE / 'vlm_native_correction_pool_v2'
PROTOCOL = parent.PROJECT_ROOT / 'docs/VLM_NATIVE_CORRECTION_V2_SUPPLEMENT.md'


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for, inference_row
    from vlm_deep.common import encode
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Supplement frozen')
    parent.verify(); first = read(parent.ROOT / 'DATA.json'); manifest_path = parent.BASE / 'vlm_deep_v2/prepared/manifest.json'; manifest = read(manifest_path)
    partial = read(parent.ROOT / 'VISUAL_REVIEW_PARTIAL.json')
    total_errors = read(parent.ROOT / 'COMPLETION.json')['candidate_errors']
    if total_errors - len(partial['not_admitted_ordinals']) >= 128:
        raise ValueError('First-pool shortfall not established')
    def walk(x):
        if isinstance(x, dict):
            if 'question_id' in x and 'image_id' in x: yield x
            else:
                for y in x.values(): yield from walk(y)
        elif isinstance(x, list):
            for y in x: yield from walk(y)
    exclusions = [parent.OLD / 'DATA.json', parent.BASE / 'vlm_replay_diversity_v1/DATA.json', parent.BASE / 'vlm_error_supervision_mining_v1/DATA.json', parent.BASE / 'vlm_native_ocr_correction_v1/revision2/DATA.json']
    blocked = {q['image_id'] for n in ('dev', 'final') for q in manifest['splits'][n]} | {q['image_id'] for q in first['rows']}
    for path in exclusions:
        blocked.update(q.get('source_image_id', q['image_id']) for q in walk(read(path)))
    pixels = {manifest['images'][i]['pixel_sha256'] for i in blocked if i in manifest['images']}
    freq = Counter(NORMALIZER(q['target']) for q in manifest['splits']['train_all']); seen_targets = Counter(NORMALIZER(q['target']) for q in first['rows']); groups = {'singleword': [], 'multiword': []}
    for q in manifest['splits']['train_all']:
        t = NORMALIZER(q['target']); votes = Counter(NORMALIZER(a) for a in q['answers'])
        if q['image_id'] in blocked or manifest['images'][q['image_id']]['pixel_sha256'] in pixels: continue
        if not t or t in {'yes', 'no', 'unknown', 'unanswerable', 'unreadable'} or len(q['target']) > 64 or len(q['target'].split()) > 6 or votes[t] < 8: continue
        stratum = 'singleword' if len(q['target'].split()) == 1 else 'multiword'
        groups[stratum].append({**q, 'stratum': stratum, 'answer_frequency': freq[t], 'target_consensus_votes': votes[t]})
    selected = []; used_ids = set(); used_pixels = set()
    for group in ('singleword', 'multiword'):
        count = 0
        for q in sorted(groups[group], key=lambda q: content_sha(['native_correction_v2:supplement', q['question_id']])):
            t = NORMALIZER(q['target']); pixel = manifest['images'][q['image_id']]['pixel_sha256']
            if q['image_id'] in used_ids or pixel in used_pixels or seen_targets[t] >= 2: continue
            selected.append(q); used_ids.add(q['image_id']); used_pixels.add(pixel); seen_targets[t] += 1; count += 1
            if count == 768: break
        if count != 768: raise ValueError('Insufficient fixed768 supplement stratum; no relaxation')
    images = {i: {**manifest['images'][i], 'path': str((manifest_path.parent / manifest['images'][i]['path']).resolve())} for i in used_ids}
    reference = read(parent.BASE / 'vlm_reference_replay_v1/PLAN.json'); torch.set_num_threads(2); processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    for i, q in enumerate(selected):
        sample = encode(processor, inference_row(q), Path(images[q['image_id']]['path']), False)
        q.update(expected_prefix_tokens=int(sample['input_ids'].shape[1]), expected_grid=sample['image_grid_thw'][0].tolist(), inference_input_sha256=content_sha(sample['input_ids'].tolist()))
        if (i + 1) % 256 == 0: write_json(ROOT / 'CPU_PROGRESS.json', {'completed': i + 1, 'expected': 1536})
    selected = [q for pair in zip(selected[:768], selected[768:]) for q in pair]; write_json(ROOT / 'DATA.json', {'rows': selected, 'images': images})
    paths = [Path(__file__), PROTOCOL, ROOT / 'DATA.json', parent.ROOT / 'VISUAL_REVIEW_PARTIAL.json', parent.ROOT / 'ERROR_CANDIDATES.json', parent.ROOT / 'COMPLETION.json', parent.ROOT / 'INITIALIZATION_GPU_PARITY.json']
    pp = read(parent.ROOT / 'PLAN.json'); files = dict(pp['files']); files.update({str(p): sha_file(p) for p in paths})
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'files': files, 'images': {v['path']: v['sha256'] for v in images.values()}, 'packages': pp['packages'],
               'base_weight_files': pp['base_weight_files'], 'qa': 1536, 'new_optimizer_updates': 0, 'initializer': read(parent.ROOT / 'INITIALIZATION.json'),
               'original_pool_max_admissible_after_partial': total_errors - len(partial['not_admitted_ordinals']), 'merged_review_cap': 256})
    print('SUPPLEMENT_PREPARED', {'qa': 1536, 'new_images': 1536, 'per_target_cap_joint_pools': 2}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    p = read(ROOT / 'PLAN.json')
    for path, digest in {**p['files'], **p['images']}.items():
        if sha_file(path) != digest: raise ValueError('Supplement input changed: ' + path)
    if package_lock() != p['packages']: raise ValueError('Runtime changed')
    return p


def worker(shard):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); model, processor, _ = actor(adapter=plan['initializer']['adapter']); before = lora_hash(model)
    if before != plan['initializer']['lora_sha256']: raise ValueError('Different OCR-enhanced initializer')
    model.eval(); model.config.use_cache = True; data = read(ROOT / 'DATA.json'); groups = [data['rows'][i:i + 4] for j, i in enumerate(range(0, 1536, 4)) if j % 2 == shard]
    rows = [q for chunk in groups for q in chunk]; directory = ROOT / f'shard_{shard}'; path = directory / 'PREDICTIONS.json'; preds = read(path) if path.exists() else []
    if len(preds) > 768 or len(preds) % 4 or [q['question_id'] for q in preds] != [q['question_id'] for q in rows[:len(preds)]]: raise ValueError('Resume differs')
    start = time.monotonic()
    for chunk in groups[len(preds) // 4:]:
        answers = generated(model, processor, chunk, data)
        if any(a['input_sha256'] != q['inference_input_sha256'] for a, q in zip(answers, chunk)): raise ValueError('Actual prefix differs')
        preds.extend(answers); tmp = path.with_suffix('.tmp'); write_json(tmp, preds); tmp.replace(path)
        write_json(directory / 'PROGRESS.json', {'status': 'mining', 'completed': len(preds), 'expected': 768, 'active_seconds': time.monotonic() - start})
        if time.monotonic() - start > 1200 or torch.cuda.max_memory_allocated() / 2**30 > 65: raise RuntimeError('Supplement cap')
    if lora_hash(model) != before: raise ValueError('Read-only weights changed')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'qa': 768, 'optimizer_updates': 0})


def analyze():
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    data = read(ROOT / 'DATA.json'); preds = {q['question_id']: q for s in (0, 1) for q in read(ROOT / f'shard_{s}/PREDICTIONS.json')}
    if len(preds) != 1536: raise ValueError('Incomplete supplement')
    errors = []; counts = Counter()
    for q in sorted(data['rows'], key=lambda q: content_sha(['native_correction_v2:supplement_errors', q['question_id']])):
        p = preds[q['question_id']]; target = NORMALIZER(q['target'])
        if p['vqa_score'] > .3 + 1e-8 or not p['terminated'] or target in NORMALIZER(p['prediction']) or counts[target] >= 2: continue
        errors.append({**q, 'initial_prediction': p}); counts[target] += 1
    write_json(ROOT / 'ERROR_CANDIDATES.json', {'rows': errors, 'images': data['images']})
    first = read(parent.ROOT / 'ERROR_CANDIDATES.json'); combined = first['rows'] + errors
    write_json(MERGED / 'ERROR_CANDIDATES.json', {'rows': combined, 'images': {**first['images'], **data['images']}})
    for leaf in ('INITIALIZATION.json', 'INITIALIZATION_GPU_PARITY.json'): write_json(MERGED / leaf, read(parent.ROOT / leaf))
    font = ImageFont.load_default(size=17); offset = len(first['rows'])
    for begin in range(0, min(len(errors), 256 - offset), 8):
        canvas = Image.new('RGB', (1200, 1440), '#edf2f8'); draw = ImageDraw.Draw(canvas)
        for j, q in enumerate(errors[begin:min(begin + 8, 256 - offset)]):
            x = (j % 2) * 600; y = (j // 2) * 360; im = ImageOps.contain(Image.open(data['images'][q['image_id']]['path']).convert('RGB'), (580, 245)); canvas.paste(im, (x + 10 + (580 - im.width) // 2, y + 4))
            text = f"#{offset+begin+j} q{q['question_id']} | {q['question']}\nGT: {q['target']}\nPred: {q['initial_prediction']['prediction']}"
            draw.text((x + 8, y + 251), '\n'.join(line for part in text.splitlines() for line in textwrap.wrap(part, 62)), fill='#172536', font=font)
        path = ROOT / 'review' / f'board_{begin // 8:02d}.png'; path.parent.mkdir(exist_ok=True); canvas.save(path)
    pp = read(ROOT / 'PLAN.json'); files = dict(pp['files'])
    for path in [parent.ROOT / 'PLAN.json', ROOT / 'PLAN.json', ROOT / 'ERROR_CANDIDATES.json', MERGED / 'ERROR_CANDIDATES.json', MERGED / 'INITIALIZATION.json', MERGED / 'INITIALIZATION_GPU_PARITY.json']:
        files[str(path)] = sha_file(path)
    write_json(MERGED / 'PLAN.json', {'status': 'prepared', 'files': files, 'packages': pp['packages'], 'base_weight_files': pp['base_weight_files'], 'source_pools': 2,
               'first_pool_candidates': len(first['rows']), 'supplement_candidates': len(errors), 'review_cap': 256, 'new_optimizer_updates': 0})
    result = {'status': 'completed', 'new_optimizer_updates': 0, 'mined_qa': 1536, 'candidate_errors': len(errors), 'merged_candidates': len(combined), 'review_cap': 256, 'training_admitted': False}
    write_json(ROOT / 'COMPLETION.json', result); print('SUPPLEMENT_COMPLETE', result, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepare', action='store_true'); p.add_argument('--worker', type=int); args = p.parse_args()
    if args.prepare: prepare(); return
    if args.worker is not None: worker(args.worker); return
    from research_runtime.queue import run_queue
    verify(); entry = str(Path(__file__).resolve()); run_queue(ROOT / 'queue', [{'key': str(s), 'command': [sys.executable, '-u', entry, '--worker', str(s)], 'completion_marker': str(ROOT / f'shard_{s}/COMPLETE.json')} for s in (0, 1)], {'plan_sha256': sha_file(ROOT / 'PLAN.json')}); analyze()


if __name__ == '__main__': main()
