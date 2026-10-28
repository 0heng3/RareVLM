"""Mine native question-conditioned errors from the OCR-improved policy."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import textwrap

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
OLD = BASE / 'vlm_iiit_scope_v1'
ROOT = BASE / 'vlm_native_correction_mining_v2'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_CORRECTION_MINING_V2_PROTOCOL.md'


def prepare():
    import torch
    from safetensors.torch import load_file, save_file
    from vlm_object_train.train import tensor_hash
    from vlm_probe.evaluate import processor_for, inference_row
    from vlm_deep.common import encode
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Mining frozen')
    torch.set_num_threads(2)
    manifest_path = BASE / 'vlm_deep_v2/prepared/manifest.json'
    manifest = read(manifest_path); parent = read(OLD / 'PLAN.json')
    old = read(OLD / 'DATA.json'); excluded = {q['image_id'] for s in ('dev', 'final') for q in manifest['splits'][s]}
    def walk(value):
        if isinstance(value, dict):
            if 'question_id' in value and 'image_id' in value:
                yield value
            else:
                for item in value.values():
                    yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
    exclusion_paths = [OLD / 'DATA.json', BASE / 'vlm_replay_diversity_v1/DATA.json',
                       BASE / 'vlm_error_supervision_mining_v1/DATA.json',
                       BASE / 'vlm_native_ocr_correction_v1/revision2/DATA.json']
    for path in exclusion_paths:
        excluded.update(q.get('source_image_id', q['image_id']) for q in walk(read(path)))
    held_pixels = {manifest['images'][i]['pixel_sha256'] for i in excluded if i in manifest['images']}
    freq = Counter(NORMALIZER(q['target']) for q in manifest['splits']['train_all'])
    groups = {'singleword': [], 'multiword': []}
    for q in manifest['splits']['train_all']:
        target = NORMALIZER(q['target'])
        if q['image_id'] in excluded or manifest['images'][q['image_id']]['pixel_sha256'] in held_pixels:
            continue
        if not target or target in {'yes', 'no', 'unknown', 'unanswerable', 'unreadable'} or len(q['target']) > 64 or len(q['target'].split()) > 6:
            continue
        votes = Counter(NORMALIZER(a) for a in q['answers'])
        if votes[target] < 8:
            continue
        stratum = 'singleword' if len(q['target'].split()) == 1 else 'multiword'
        groups[stratum].append({**q, 'stratum': stratum, 'answer_frequency': freq[target], 'target_consensus_votes': votes[target]})
    selected = []; used_images = set(); used_pixels = set(); targets = Counter()
    for stratum in ('singleword', 'multiword'):
        count = 0
        for q in sorted(groups[stratum], key=lambda q: content_sha(['native_correction_mining_v2', q['question_id']])):
            pixel = manifest['images'][q['image_id']]['pixel_sha256']; target = NORMALIZER(q['target'])
            if q['image_id'] in used_images or pixel in used_pixels or targets[target] >= 2:
                continue
            selected.append(q); used_images.add(q['image_id']); used_pixels.add(pixel); targets[target] += 1; count += 1
            if count == 1024:
                break
        if count != 1024:
            raise ValueError('Insufficient fixed1024 stratum; do not lower filters')
    images = {i: {**manifest['images'][i], 'path': str((manifest_path.parent / manifest['images'][i]['path']).resolve())} for i in used_images}
    result = read(OLD / 'runs/sparse_language/train/TRAIN_COMPLETE.json')
    source = Path(result['adapter']); weights = load_file(str(source / 'adapter_model.safetensors'))
    visual_b = [v for k, v in weights.items() if '.visual.' in k and '.lora_B.' in k]
    if len(visual_b) != 30 or any(torch.count_nonzero(v) for v in visual_b):
        raise ValueError('Only exactly zero-output visual adapters may be removed')
    language = {k: v for k, v in weights.items() if '.visual.' not in k}
    if len(language) != 144 or sum(v.numel() for v in language.values()) != 1843200 or any(v.dtype != torch.float32 for v in language.values()):
        raise ValueError('Original trained language factors differ')
    destination = ROOT / 'initial_adapter'; destination.mkdir(parents=True, exist_ok=True)
    config = read(Path(read(OLD / 'INITIALIZATION.json')['original_adapter']) / 'adapter_config.json')
    config.update(lora_dropout=0., inference_mode=False)
    write_json(destination / 'adapter_config.json', config)
    save_file(language, str(destination / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    ordered = []
    for layer in range(36):
        for projection in ('q_proj', 'v_proj'):
            for factor in ('A', 'B'):
                key = f'base_model.model.model.layers.{layer}.self_attn.{projection}.lora_{factor}.weight'
                ordered.append((key.replace('.weight', '.default.weight'), language[key]))
    language_hash = tensor_hash(ordered)
    reference = read(BASE / 'vlm_reference_replay_v1/PLAN.json')
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    for i, q in enumerate(selected):
        sample = encode(processor, inference_row(q), Path(images[q['image_id']]['path']), False)
        q.update(expected_prefix_tokens=int(sample['input_ids'].shape[1]), expected_grid=sample['image_grid_thw'][0].tolist(), inference_input_sha256=content_sha(sample['input_ids'].tolist()))
        if (i + 1) % 256 == 0:
            write_json(ROOT / 'CPU_PROGRESS.json', {'completed': i + 1, 'expected': 2048})
    selected = [q for pair in zip(selected[:1024], selected[1024:]) for q in pair]
    write_json(ROOT / 'DATA.json', {'rows': selected, 'images': images})
    write_json(ROOT / 'INITIALIZATION.json', {'adapter': str(destination), 'lora_sha256': language_hash, 'source': str(source),
               'source_composite_lora_sha256': result['final_lora_sha256'], 'language_factors_bitwise_preserved': True,
               'removed_visual_B_zero_factors': 30, 'expected_parameters': 1843200})
    paths = [Path(__file__), PROTOCOL, manifest_path, ROOT / 'DATA.json', ROOT / 'INITIALIZATION.json',
             destination / 'adapter_config.json', destination / 'adapter_model.safetensors', source / 'adapter_model.safetensors',
             OLD / 'runs/sparse_language/train/TRAIN_COMPLETE.json', OLD / 'runs/sparse_language/seen_validation/PREDICTIONS.json'] + exclusion_paths
    files = dict(parent['files']); files.update({str(p): sha_file(p) for p in paths})
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'files': files, 'packages': parent['packages'], 'base_weight_files': parent['base_weight_files'],
               'images': {v['path']: v['sha256'] for v in images.values()}, 'qa': 2048, 'singleword': 1024, 'multiword': 1024,
               'eligible_per_stratum': {k: len(v) for k, v in groups.items()}, 'initial_lora_sha256': language_hash,
               'new_optimizer_updates': 0, 'new_generations': 2056, 'minimum_reviewed_errors': 128, 'minimum_unique_targets': 96,
               'review_cap': 256, 'automatic_training': False, 'original_sft_train_all_exposure': True,
               'excluded_image_ids': len(excluded), 'task': 'Original full image and original native QA, not word crops.'})
    print('NATIVE_MINING_PREPARED', {'rows': 2048, 'eligible': {k: len(v) for k, v in groups.items()}, 'initial_language_hash': language_hash}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen mining input changed: ' + path)
    if package_lock() != plan['packages']:
        raise ValueError('Runtime changed')
    for path, meta in plan['base_weight_files'].items():
        stat = Path(path).stat()
        if stat.st_size != meta['bytes'] or stat.st_mtime_ns != meta['mtime_ns']:
            raise ValueError('Base identity changed')
    return plan


def worker(shard):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_fit_positive_control_v1.study import generated
    verify(); init = read(ROOT / 'INITIALIZATION.json'); model, processor, _ = actor(adapter=init['adapter'])
    before = lora_hash(model)
    if before != init['lora_sha256']:
        raise ValueError('Wrong OCR-improved language initializer')
    model.eval(); model.config.use_cache = True
    if shard == 0:
        data = read(OLD / 'DATA.json'); rows = data['evaluations']['seen_validation'][:8]
        answers = generated(model, processor, rows, data)
        expected = read(OLD / 'runs/sparse_language/seen_validation/PREDICTIONS.json')[:8]
        if [(q['question_id'], q['completion_ids'], q['input_sha256']) for q in answers] != [(q['question_id'], q['completion_ids'], q['input_sha256']) for q in expected]:
            raise ValueError('Zero-output adapter removal changed fixed8 parity')
        write_json(ROOT / 'INITIALIZATION_GPU_PARITY.json', {'status': 'passed', 'qa': 8, 'exact_completion_ids_and_prefixes': True, 'scope': 'Structural zero-B removal plus8 fixedQA spot check, not full output equality proof.'})
    data = read(ROOT / 'DATA.json'); groups = [data['rows'][i:i + 4] for j, i in enumerate(range(0, 2048, 4)) if j % 2 == shard]
    ordered = [q for chunk in groups for q in chunk]; directory = ROOT / ('shard_' + str(shard)); path = directory / 'PREDICTIONS.json'
    preds = read(path) if path.exists() else []
    if len(preds) > 1024 or len(preds) % 4 or [q['question_id'] for q in preds] != [q['question_id'] for q in ordered[:len(preds)]]:
        raise ValueError('Mining resume differs')
    start = time.monotonic()
    for chunk in groups[len(preds) // 4:]:
        answers = generated(model, processor, chunk, data)
        if any(a['input_sha256'] != q['inference_input_sha256'] for a, q in zip(answers, chunk)):
            raise ValueError('Mining prefix mismatch')
        preds.extend(answers); temp = path.with_suffix('.tmp'); write_json(temp, preds); temp.replace(path)
        write_json(directory / 'PROGRESS.json', {'status': 'mining', 'completed': len(preds), 'expected': 1024, 'active_seconds': time.monotonic() - start})
        if time.monotonic() - start > 1200 or torch.cuda.max_memory_allocated() / 2**30 > 65:
            raise RuntimeError('Mining budget exceeded')
    if lora_hash(model) != before:
        raise ValueError('Read-only mining changed model')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'qa': 1024, 'optimizer_updates': 0})


def analyze():
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    data = read(ROOT / 'DATA.json'); meta = {q['question_id']: q for q in data['rows']}
    preds = {q['question_id']: q for shard in (0, 1) for q in read(ROOT / f'shard_{shard}/PREDICTIONS.json')}
    if len(preds) != 2048:
        raise ValueError('Incomplete fixed mining pool')
    errors = []; targets = Counter()
    for qid in sorted(meta, key=lambda qid: content_sha(['native_correction_mining_v2:errors', qid])):
        q = meta[qid]; p = preds[qid]; target = NORMALIZER(q['target']); prediction = NORMALIZER(p['prediction'])
        if p['vqa_score'] > .3 + 1e-8 or not p['terminated'] or target in prediction or targets[target] >= 2:
            continue
        errors.append({**q, 'initial_prediction': p}); targets[target] += 1
    write_json(ROOT / 'ERROR_CANDIDATES.json', {'rows': errors, 'images': data['images']})
    font = ImageFont.load_default(size=17); boards = []
    for begin in range(0, min(256, len(errors)), 8):
        canvas = Image.new('RGB', (1200, 1440), '#edf2f8'); draw = ImageDraw.Draw(canvas)
        for j, q in enumerate(errors[begin:begin + 8]):
            x = (j % 2) * 600; y = (j // 2) * 360
            picture = ImageOps.contain(Image.open(data['images'][q['image_id']]['path']).convert('RGB'), (580, 245))
            canvas.paste(picture, (x + 10 + (580 - picture.width) // 2, y + 4))
            text = f"#{begin+j} q{q['question_id']} | {q['question']}\nGT: {q['target']}\nPred: {q['initial_prediction']['prediction']}"
            lines = '\n'.join(line for paragraph in text.splitlines() for line in textwrap.wrap(paragraph, 62))
            draw.text((x + 8, y + 251), lines, fill='#172536', font=font)
        path = ROOT / 'review' / f'board_{begin // 8:02d}.png'; path.parent.mkdir(exist_ok=True); canvas.save(path); boards.append(str(path))
    result = {'status': 'completed', 'training_updates': 0, 'mined_qa': 2048, 'initial_vqa_percent': 100 * sum(q['vqa_score'] for q in preds.values()) / 2048,
              'candidate_errors': len(errors), 'unique_candidate_targets': len(targets), 'boards': boards, 'visual_review_completed': False, 'training_admitted': False,
              'goal_complete': False, 'next': 'Review first256 maximum; need128 supported nativeQA errors/96 targets before separately frozen training.'}
    write_json(ROOT / 'COMPLETION.json', result); print('MINING_COMPLETE', {k: v for k, v in result.items() if k != 'boards'}, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker is not None:
        worker(args.worker); return
    from research_runtime.queue import run_queue
    verify(); identity = {'plan_sha256': sha_file(ROOT / 'PLAN.json')}; entry = str(Path(__file__).resolve())
    run_queue(ROOT / 'queue', [{'key': str(s), 'command': [sys.executable, '-u', entry, '--worker', str(s)], 'completion_marker': str(ROOT / f'shard_{s}/COMPLETE.json')} for s in (0, 1)], identity)
    analyze()


if __name__ == '__main__':
    main()
