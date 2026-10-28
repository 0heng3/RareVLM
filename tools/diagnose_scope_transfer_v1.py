"""Frozen-model stratified512 multiword scope diagnostic plus initial78 training-signal audit."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import encode, content_sha, sha_file, write_json, read_manifest, NORMALIZER
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_scope_transfer_diagnosis_v1'
PARENT = ROOT.parent / 'vlm_scope_balanced_sft_v2'
PREP = ROOT.parent / 'vlm_scope_balanced_replay_v1'
ROLES = ('initial', 'empirical_mix', 'scope_balanced')


def prepare():
    from vlm_probe.evaluate import processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already prepared; inspect existing queue')
    c = read(PROJECT_ROOT / 'configs/vlm_scope_balanced_sft_v2.json')
    base = read(c['baseline_config'])
    source = read_manifest(c['source_manifest'])
    current = read(PARENT / 'data/empirical_mix/MANIFEST.json')
    excluded = {q['image_id'] for q in current['rows']}
    for rel in ('vlm_scope_balanced_sft_v2/data/EVAL.json', 'vlm_region_preference_probe_v1/data/MANIFEST.json'):
        excluded.update(q['image_id'] for q in read(ROOT.parent / rel)['retention_eval'])
    excluded.update(read(ROOT.parent / 'vlm_readout_transfer_v1/DATA.json')['images'])
    image_key = lambda im: im.get('pixel_sha256', im['sha256'])
    excluded_pixels = {image_key(source['images'][iid]) for iid in excluded if iid in source['images']}
    eligible = [q for q in source['splits']['train_all'] if q['image_id'] not in excluded
        and image_key(source['images'][q['image_id']]) not in excluded_pixels
        and 2 <= len(q['target'].split()) <= 6 and Counter(q['answers'])[q['target']] >= 8]
    eligible.sort(key=lambda q: hashlib.sha256(('scope_transfer_v1:' + q['image_id'] + ':' + q['question_id']).encode()).hexdigest())
    held = []
    counts = Counter()
    seen = set()
    for q in eligible:
        group = 'two_words' if len(q['target'].split()) == 2 else 'three_to_six'
        pixel = image_key(source['images'][q['image_id']])
        if counts[group] >= 256 or pixel in seen:
            continue
        held.append({**q, 'diagnostic_stratum': group})
        seen.add(pixel)
        counts[group] += 1
        if len(held) == 512:
            break
    if counts != {'two_words': 256, 'three_to_six': 256}:
        raise ValueError('Need fixed256+256 image-disjoint diagnostic, no fallback')
    views = read(PREP / 'MATCHED_REPLAY_VIEWS.json')
    changed = [{**b, 'diagnostic_stratum': 'extra_training_long'} for a, b in zip(views['views']['empirical_mix'], views['views']['scope_balanced']) if a['question_id'] != b['question_id']]
    if len(changed) != 78:
        raise ValueError('Expected78 actual substituted training QA')
    images = {}
    for q in held + changed:
        im = source['images'][q['image_id']]
        images[q['image_id']] = {**im, 'path': str((Path(c['source_manifest']).parent / im['path']).resolve())}
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    for q in held + changed:
        item = encode(processor, q, Path(images[q['image_id']]['path']), False)
        q['expected_tokens'] = int(item['input_ids'].shape[1])
        q['expected_grid'] = item['image_grid_thw'][0].tolist()
        q['prefix_sha256'] = content_sha(item['input_ids'].tolist())
    write_json(ROOT / 'DATA.json', {'held': held, 'training_signal': changed, 'images': images})
    adapters = {'initial': c['initial_adapter'], **{role: str(PARENT / 'runs' / role / 'train/adapter') for role in ROLES[1:]}}
    hashes = {'initial': read(PARENT / 'baseline/COMPLETE.json')['lora_sha256'],
              **{role: read(PARENT / 'runs' / role / 'COMPLETE.json')['final_lora_sha256'] for role in ROLES[1:]}}
    files = [Path(__file__).resolve(), ROOT / 'DATA.json', Path(c['source_manifest']),
             PROJECT_ROOT / 'docs/VLM_SCOPE_TRANSFER_DIAGNOSIS_V1_PROTOCOL.md',
             PARENT / 'COMPLETION.json', PARENT / 'POSTRUN_TREATMENT_REVIEW.json',
             PARENT / 'data/EVAL.json', PARENT / 'data/empirical_mix/MANIFEST.json',
             PREP / 'MATCHED_REPLAY_VIEWS.json', Path(c['baseline_config']), Path(base['reference_config'])]
    files += [Path(path) / name for path in adapters.values() for name in ('adapter_config.json', 'adapter_model.safetensors')]
    files += [SOURCE_ROOT / name for name in ('vlm_region_ocr_sft_v1/study.py', 'vlm_fp32_readout_v1/model.py',
              'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py', 'vlm_probe/evaluate.py',
              'vlm_deep/common.py', 'vlm_deep/evaluate.py', 'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'training_updates': 0, 'held_qa': 512, 'strata': dict(counts),
               'initial_training_signal_qa': 78, 'new_model_predictions': 1614,
               'scope': 'Diagnostic only: current-adaptation image-disjoint training-whitelist questions; initial SFT may have seen them, single seed, not main/official benchmark or a reversal of failed short gate.',
               'adapters': adapters, 'lora_sha256': hashes, 'active_cap_seconds_per_cohort': 600,
               'files': {str(path): sha_file(path) for path in files}, 'images': {im['path']: im['sha256'] for im in images.values()}})
    print({'status': 'prepared', 'held_images': 512, 'strata': dict(counts), 'actual_cpu_prefix_encodings': 590,
           'training_signal_qa': 78, 'new_predictions': 1614, 'new_optimizer_updates': 0})


def generate(model, processor, rows, images, directory, cap):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    path = directory / 'PREDICTIONS.json'
    predictions = read(path) if path.exists() else []
    if (len(predictions) != len(rows) and len(predictions) % 4) or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Cached diagnostic batch order differs')
    begun = time.monotonic()
    with torch.inference_mode():
        for start in range(len(predictions), len(rows), 4):
            chunk = rows[start:start + 4]
            answers, _ = evaluate_batch(model, processor, chunk, images, ROOT, next(model.parameters()).device,
                                       1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            for q, answer in zip(chunk, answers):
                if answer['image_grid_thw'] != q['expected_grid'] or answer['input_tokens'] != q['expected_tokens']:
                    raise ValueError('Actual diagnostic inference frame differs')
                p, target = NORMALIZER(answer['prediction']), NORMALIZER(q['target'])
                predictions.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                    'stratum': q['diagnostic_stratum'], **answer, 'vqa_score': score_prediction(answer['prediction'], q, 'textvqa'),
                    'casefold_exact': answer['prediction'].casefold() == q['target'].casefold(),
                    'prefix_truncated': bool(p) and target.startswith(p + ' ')})
            write_json(path, predictions)
            write_json(directory / 'PROGRESS.json', {'status': 'generating', 'completed_qa': len(predictions), 'expected_qa': len(rows),
                       'active_seconds': time.monotonic() - begun})
            if time.monotonic() - begun > cap:
                raise TimeoutError('Read-only diagnostic cohort budget reached')
    return predictions


def metrics(rows):
    return {'qa': len(rows), 'vqa_score_points': sum(q['vqa_score'] for q in rows),
            'vqa_score_percent': 100 * sum(q['vqa_score'] for q in rows) / len(rows),
            'casefold_exact': sum(q['casefold_exact'] for q in rows), 'full_vqa_score_qa': sum(q['vqa_score'] >= 1. for q in rows),
            'prefix_truncated': sum(q['prefix_truncated'] for q in rows)}


def worker(role):
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen diagnostic input changed: ' + path)
    model, processor, _ = actor(plan['adapters'][role])
    before = lora_hash(model)
    if before != plan['lora_sha256'][role]:
        raise ValueError('Diagnostic checkpoint differs')
    model.eval()
    model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    data = read(ROOT / 'DATA.json')
    signal = None
    if role == 'initial':
        rows = generate(model, processor, data['training_signal'], data['images'], ROOT / role / 'training_signal', plan['active_cap_seconds_per_cohort'])
        signal = metrics(rows)
    rows = generate(model, processor, data['held'], data['images'], ROOT / role / 'held', plan['active_cap_seconds_per_cohort'])
    if lora_hash(model) != before:
        raise ValueError('Read-only diagnostic changed adapter')
    write_json(ROOT / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 0, 'lora_sha256': before,
               'adapter_unchanged': True, 'metrics': metrics(rows),
               'strata': {name: metrics([q for q in rows if q['stratum'] == name]) for name in ('two_words', 'three_to_six')},
               'initial_training_signal': signal})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    models = {role: read(ROOT / role / 'COMPLETE.json') for role in ROLES}
    rows = {role: read(ROOT / role / 'held/PREDICTIONS.json') for role in ROLES}
    comparisons = {}
    for before in ('initial', 'empirical_mix'):
        a, b = rows[before], rows['scope_balanced']
        if [q['question_id'] for q in a] != [q['question_id'] for q in b]:
            raise ValueError('Diagnostic cohorts unpaired')
        values = {}
        for name in ('all', 'two_words', 'three_to_six'):
            transform = lambda qs: [{'question_id': q['question_id'], 'image_id': q['image_id'], 'score': q['vqa_score']}
                                    for q in qs if name == 'all' or q['stratum'] == name]
            values[name] = paired_interval(transform(a), transform(b), 6000, 20261020)
        comparisons[before] = values
    result = {'status': 'completed', 'scope': read(ROOT / 'PLAN.json')['scope'], 'new_training_updates': 0,
              'models': models, 'comparisons': comparisons,
              'next': 'No automatic training/main benchmark/ratio search/CV update; diagnosis cannot overwrite failed32-step gate.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 长答案分层迁移与新增监督信号诊断', '', result['scope'], '',
             '|模型|VQA得分|规范化前缀截短|大小写忽略精确正确|', '|---|---:|---:|---:|']
    lines += [f'|{role}|{v["metrics"]["vqa_score_percent"]:.6f}|{v["metrics"]["prefix_truncated"]}/512|{v["metrics"]["casefold_exact"]}/512|' for role, v in models.items()]
    lines += ['', '结果只诊断已保存模型，不将增加样本量改写成原短门槛通过。']
    (PROJECT_ROOT / 'docs/VLM_SCOPE_TRANSFER_DIAGNOSIS_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


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
