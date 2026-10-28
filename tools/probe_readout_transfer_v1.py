"""Image-disjoint auxiliary-region transfer and direct readout ablation; zero updates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import encode, content_sha, sha_file, write_json

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_readout_transfer_v1'
PARENT = ROOT.parent / 'vlm_region_adapter_location_v1'
DATA_ROOT = ROOT.parent / 'vlm_region_preference_probe_v1'
POOL = ROOT.parent / 'vlm_region_binding_data_v1/PAIR_CANDIDATES.json'
ROLES = ('readout', 'readout_zero', 'initial', 'plain_sft')


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from vlm_probe.evaluate import processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already prepared; inspect current controller')
    pool = read(POOL)
    data = read(DATA_ROOT / 'data/MANIFEST.json')
    exclusion_sources = [DATA_ROOT / 'data/MANIFEST.json',
        ROOT.parent / 'vlm_region_binding_probe_v1/DATA.json',
        ROOT.parent / 'vlm_pixel_transfer_diagnosis_v1/DATA.json']
    excluded = {q['image_id'] for q in data['rows'] + data['retention_eval']}
    for path in exclusion_sources[1:]:
        old = read(path)
        excluded.update(old.get('images', {}))
        for q in old.get('rows', []):
            excluded.add(q.get('source_image_id', q['image_id']))
    def image_key(image):
        return image.get('pixel_sha256', image['sha256'])
    excluded_pixels = {image_key(im) for iid, im in pool['images'].items() if iid in excluded}
    excluded_pixels.update(image_key(data['images'][iid]) for iid in excluded if iid in data['images'])
    candidates = [pair for pair in pool['pairs'] if pair['source_image_id'] not in excluded
                  and image_key(pool['images'][pair['source_image_id']]) not in excluded_pixels]
    candidates.sort(key=lambda pair: hashlib.sha256(('readout_transfer_v1:' + image_key(pool['images'][pair['source_image_id']])).encode()).hexdigest())
    selected = []
    seen = set()
    for pair in candidates:
        key = image_key(pool['images'][pair['source_image_id']])
        if key not in seen:
            selected.append(pair)
            seen.add(key)
        if len(selected) == 128:
            break
    if len(selected) != 128:
        raise ValueError('Insufficient image-disjoint candidates; no fallback')
    c = read(PROJECT_ROOT / 'configs/vlm_region_adapter_location_v1.json')
    base = read(c['baseline_config'])
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    images = {q['source_image_id']: pool['images'][q['source_image_id']] for q in selected}
    rows = []
    for pair in selected:
        a, b = pair['a'], pair['b']
        if re.sub(r'\[[0-9, ]+\]', '[BOX]', a['question']) != re.sub(r'\[[0-9, ]+\]', '[BOX]', b['question']):
            raise ValueError('Pair differs beyond registered box coordinates')
        for side, q in (('a', a), ('b', b)):
            item = encode(processor, q, Path(images[q['image_id']]['path']), False)
            grid = item['image_grid_thw'][0].tolist()
            if [grid[2] * 14, grid[1] * 14] != q['actual_processed_dimensions']:
                raise ValueError('Pixel coordinate frame differs')
            rows.append({**q, 'pair_id': pair['pair_id'], 'side': side, 'expected_grid': grid,
                         'expected_tokens': int(item['input_ids'].shape[1]),
                         'prefix_sha256': content_sha(item['input_ids'].tolist())})
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': images})
    adapters = {'initial': c['initial_adapter'], 'plain_sft': str(DATA_ROOT / 'runs/sft/train/adapter'),
                'readout': str(PARENT / 'runs/readout/train/adapter')}
    initial_hash = read(DATA_ROOT / 'baseline/COMPLETE.json')['lora_sha256']
    plain_hash = read(DATA_ROOT / 'runs/sft/COMPLETE.json')['final_lora_sha256']
    trained = read(PARENT / 'runs/readout/train/TRAIN_COMPLETE.json')
    files = [Path(__file__).resolve(), ROOT / 'DATA.json', POOL, *exclusion_sources,
             PROJECT_ROOT / 'docs/VLM_READOUT_TRANSFER_V1_PROTOCOL.md',
             PARENT / 'COMPLETION.json', PARENT / 'runs/readout/train/TRAIN_COMPLETE.json',
             PARENT / 'runs/readout/hard/PREDICTIONS.json', Path(c['baseline_config']), Path(base['reference_config'])]
    files += [Path(adapter) / name for adapter in adapters.values() for name in ('adapter_config.json', 'adapter_model.safetensors')]
    files += [Path(adapters['readout']) / name for name in ('residual.safetensors', 'residual_config.json')]
    files += [SOURCE_ROOT / name for name in ('vlm_region_adapter_location_v1/model.py', 'vlm_fp32_readout_v1/model.py',
              'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py', 'vlm_probe/evaluate.py',
              'vlm_deep/common.py', 'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'training_updates': 0, 'pairs': 128, 'qa_each_role': 256,
               'scope': 'Post hoc readout-control follow-up; images disjoint from current adaptation and earlier named diagnostics, but initial SFT saw training whitelist. Not a blind or formal benchmark.',
               'eligible_pairs': len(candidates), 'excluded_ids': len(excluded), 'excluded_pixel_hashes': len(excluded_pixels),
               'adapters': adapters, 'base': base, 'active_cap_seconds': 300,
               'lora_sha256': {'initial': initial_hash, 'plain_sft': plain_hash, 'readout': trained['final_lora_sha256']},
               'residual_sha256': trained['residual']['residual_sha256'],
               'files': {str(path): sha_file(path) for path in files}, 'images': {im['path']: im['sha256'] for im in images.values()}})
    print({'status': 'prepared', 'pairs': 128, 'qa_each_role': 256, 'eligible_pairs': len(candidates),
           'actual_cpu_prefix_encodings': len(rows), 'optimizer_updates': 0})


def generate(model, processor, rows, images, directory, plan):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'PREDICTIONS.json'
    predictions = read(path) if path.exists() else []
    if len(predictions) % 4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Cached batch order differs')
    begun = time.monotonic()
    with torch.inference_mode():
        for start in range(len(predictions), len(rows), 4):
            chunk = rows[start:start + 4]
            answers, _ = evaluate_batch(model, processor, chunk, images, ROOT, next(model.parameters()).device,
                                       1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            for q, answer in zip(chunk, answers):
                if 'expected_grid' in q and (answer['image_grid_thw'] != q['expected_grid'] or answer['input_tokens'] != q['expected_tokens']):
                    raise ValueError('Actual inference frame or length differs')
                predictions.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                                    'pair_id': q.get('pair_id'), 'side': q.get('side'), **answer,
                                    'raw_exact': answer['prediction'] == q['target'],
                                    'casefold_exact': answer['prediction'].casefold() == q['target'].casefold()})
            write_json(path, predictions)
            write_json(directory / 'PROGRESS.json', {'status': 'generating', 'completed_qa': len(predictions),
                       'expected_qa': len(rows), 'active_seconds': time.monotonic() - begun})
            if time.monotonic() - begun > plan['active_cap_seconds']:
                raise TimeoutError('Read-only transfer generation budget reached')
    return predictions


def worker(role):
    import torch
    from vlm_region_ocr_sft_v1.study import actor
    from vlm_region_adapter_location_v1.model import load, residual_hash
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen transfer input changed: ' + path)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    begun = time.monotonic()
    extra_role = role.startswith('readout')
    key = 'readout' if extra_role else role
    if extra_role:
        model, binding = load(plan['base']['model_root'], plan['adapters'][key], torch.device('cuda:0'), role='readout', training=False)
        processor = processor_for(Path(plan['base']['model_root']), read(plan['base']['reference_config']), 1024)
        if not binding['residual_restored'] or residual_hash(model) != plan['residual_sha256']:
            raise ValueError('Full residual checkpoint was not restored')
    else:
        model, processor, binding = actor(plan['adapters'][key])
    before = lora_hash(model)
    if before != plan['lora_sha256'][key]:
        raise ValueError('Loaded LoRA identity differs')
    model.eval()
    model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    directory = ROOT / role
    if role == 'readout':
        old = read(DATA_ROOT / 'data/MANIFEST.json')
        reloaded = generate(model, processor, old['hard_eval'], old['images'], directory / 'reload28', plan)
        expected = read(PARENT / 'runs/readout/hard/PREDICTIONS.json')
        if [q['prediction'] for q in reloaded] != [q['prediction'] for q in expected]:
            raise ValueError('Reloaded complete model does not reproduce its28 original answers')
        write_json(directory / 'RELOAD_CHECK.json', {'status': 'passed', 'qa': 28, 'prediction_differences': 0,
                   'residual_restored': True, 'scope': 'Saved complete model reload, not a new gain'})
    backup = None
    if role == 'readout_zero':
        backup = model.region_residual.up.detach().clone()
        with torch.no_grad():
            model.region_residual.up.zero_()
    data = read(ROOT / 'DATA.json')
    try:
        predictions = generate(model, processor, data['rows'], data['images'], directory, plan)
    finally:
        if backup is not None:
            with torch.no_grad():
                model.region_residual.up.copy_(backup)
    if lora_hash(model) != before or (extra_role and residual_hash(model) != plan['residual_sha256']):
        raise ValueError('Read-only model state was not preserved/restored')
    pairs = []
    for start in range(0, len(predictions), 2):
        a, b = predictions[start:start + 2]
        if a['pair_id'] != b['pair_id'] or a['side'] != 'a' or b['side'] != 'b':
            raise ValueError('Pair output order differs')
        pairs.append({'question_id': a['pair_id'], 'image_id': a['image_id'],
                      'both_correct': a['casefold_exact'] and b['casefold_exact'],
                      'same_answer': a['prediction'].casefold() == b['prediction'].casefold()})
    write_json(directory / 'PAIRS.json', pairs)
    result = {'status': 'completed', 'training_updates': 0, 'qa': len(predictions), 'pairs': len(pairs),
              'qa_correct': sum(q['casefold_exact'] for q in predictions), 'raw_correct': sum(q['raw_exact'] for q in predictions),
              'both_correct': sum(q['both_correct'] for q in pairs), 'same_answer': sum(q['same_answer'] for q in pairs),
              'model_state_preserved_or_restored': True, 'lora_sha256': before, 'active_seconds': time.monotonic() - begun,
              'direct_ablation': 'up set to zero in memory then restored; q/v identical' if role == 'readout_zero' else None}
    write_json(directory / 'COMPLETE.json', result)


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    models = {role: read(ROOT / role / 'COMPLETE.json') for role in ROLES}
    predictions = {role: read(ROOT / role / 'PREDICTIONS.json') for role in ROLES}
    pairs = {role: read(ROOT / role / 'PAIRS.json') for role in ROLES}
    transform = lambda rows, field: [{'question_id': q['question_id'], 'image_id': q['image_id'], 'score': float(q[field])} for q in rows]
    comparisons = {}
    for before in ('initial', 'plain_sft', 'readout_zero'):
        comparisons[before] = {'qa': paired_interval(transform(predictions[before], 'casefold_exact'), transform(predictions['readout'], 'casefold_exact'), 6000, 20261020),
                               'pair': paired_interval(transform(pairs[before], 'both_correct'), transform(pairs['readout'], 'both_correct'), 6000, 20261020)}
    positive = lambda key: comparisons[key]['qa']['delta_pp'] >= 1. and comparisons[key]['qa']['image_cluster_95_ci_pp'][0] > 0
    transfer = positive('initial') and positive('plain_sft') and comparisons['plain_sft']['pair']['delta_pp'] >= 2.
    module = positive('readout_zero')
    result = {'status': 'completed', 'scope': read(ROOT / 'PLAN.json')['scope'], 'training_updates': 0,
              'new_transfer_predictions': 1024, 'reload_predictions': 28, 'models': models, 'comparisons': comparisons,
              'exploratory_transfer_passed': transfer, 'direct_residual_contribution_passed': module,
              'next': 'No automatic training/benchmark/CV change; interpret control selection and all uncertainty.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 读出模型图像分离迁移与残差消融', '', result['scope'], '', '|模型|区域QA正确|双框全对|同答|', '|---|---:|---:|---:|']
    lines += [f'|{role}|{v["qa_correct"]}/256|{v["both_correct"]}/128|{v["same_answer"]}/128|' for role, v in models.items()]
    lines += ['', f'探索迁移准入={transfer}；残差直接贡献准入={module}。0更新，不是独立正式benchmark或简历成果。']
    (PROJECT_ROOT / 'docs/VLM_READOUT_TRANSFER_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.role:
        try:
            worker(args.role)
        except Exception as error:
            write_json(ROOT / args.role / 'ERROR.json', {'status': 'failed', 'error': repr(error)})
            raise
    else:
        from research_runtime.queue import run_queue
        jobs = [{'key': role, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--role', role],
                 'completion_marker': str(ROOT / role / 'COMPLETE.json')} for role in ROLES]
        run_queue(ROOT / 'queue', jobs, {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        analyze()


if __name__ == '__main__':
    main()
