"""Three saved checkpoints on fresh same-image native question pairs; zero updates."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, read_manifest, sha_file, write_json
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_native_question_binding_v1'
OLD = ROOT.parent / 'vlm_error_supervision_sft_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_error_supervision_sft_v1.json'
ROLES = ('initial', 'sft', 'dpo_sft')
EXCLUSIONS = (
    'vlm_region_ocr_sft_v1/data/CONTROL_TRAIN.json',
    'vlm_scope_balanced_sft_v2/data/empirical_mix/MANIFEST.json',
    'vlm_scope_balanced_sft_v2/data/EVAL.json',
    'vlm_scope_transfer_diagnosis_v1/DATA.json',
    'vlm_scope_preference_mining_v1/DATA.json',
    'vlm_error_supervision_mining_v1/DATA.json',
    'vlm_error_supervision_sft_v1/data/MANIFEST.json',
    'vlm_region_preference_probe_v1/data/MANIFEST.json',
    'vlm_readout_transfer_v1/DATA.json',
    'vlm_pixel_transfer_diagnosis_v1/DATA.json',
)

def key(namespace, value):
    return hashlib.sha256((namespace + ':' + value).encode()).hexdigest()

def prepare():
    import torch
    from vlm_probe.evaluate import inference_row, processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen; inspect existing queue')
    c = read(CONFIG)
    base = read(c['baseline_config'])
    source = read_manifest(c['source_manifest'])
    excluded = {q['image_id'] for split in ('dev', 'final') for q in source['splits'][split]}
    for relative in EXCLUSIONS:
        data = read(ROOT.parent / relative)
        excluded.update(data.get('images', {}))
        for name in ('retention_eval', 'hard_eval', 'validation_eval'):
            excluded.update(q['image_id'] for q in data.get(name, []))
    pixel = lambda iid: source['images'][iid].get('pixel_sha256', source['images'][iid]['sha256'])
    excluded_pixels = {pixel(i) for i in excluded if i in source['images']}
    frequency = Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    by_image = defaultdict(list)
    generic = {'yes', 'no', 'unanswerable', 'nothing', 'none', 'cannot tell', 'not readable'}
    for q in source['splits']['train_all']:
        if q['image_id'] in excluded or pixel(q['image_id']) in excluded_pixels:
            continue
        if NORMALIZER(q['target']) not in generic and Counter(q['answers'])[q['target']] >= 8 and 1 <= len(q['target'].split()) <= 6:
            by_image[q['image_id']].append(q)
    candidates = []
    for iid, questions in by_image.items():
        pairs = []
        for i, a in enumerate(questions):
            for b in questions[i + 1:]:
                x, y = NORMALIZER(a['target']), NORMALIZER(b['target'])
                if a['question'] != b['question'] and x not in y and y not in x and min(frequency[x], frequency[y]) <= 5:
                    pairs.append(sorted((a, b), key=lambda q: key('native_pair_v1:question', q['question_id'])))
        if pairs:
            pairs.sort(key=lambda qs: key('native_pair_v1:pair', ':'.join(q['question_id'] for q in qs)))
            candidates.append((iid, pairs[0]))
    candidates.sort(key=lambda item: key('native_pair_v1:image', item[0]))
    selected, seen = [], set()
    for iid, pair in candidates:
        if pixel(iid) not in seen:
            seen.add(pixel(iid))
            selected.append((iid, pair))
        if len(selected) == 128:
            break
    if len(selected) != 128:
        raise ValueError('Need128 unique high-vote native pairs; no fallback')
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    rows, images = [], {}
    for iid, questions in selected:
        im = source['images'][iid]
        path = (Path(c['source_manifest']).parent / im['path']).resolve()
        images[iid] = {**im, 'path': str(path)}
        for side, q in zip(('A', 'B'), questions):
            sample = encode(processor, inference_row(q), path, False)
            rows.append({**q, 'pair_id': iid, 'side': side, 'answer_frequency': frequency[NORMALIZER(q['target'])],
                'expected_prefix_tokens': int(sample['input_ids'].shape[1]), 'expected_grid': sample['image_grid_thw'][0].tolist(),
                'inference_input_sha256': content_sha(sample['input_ids'].tolist())})
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': images})
    roles = {'initial': {'adapter': c['initial_adapter'],
        'lora_sha256': read(ROOT.parent / 'vlm_error_supervision_mining_v1/PLAN.json')['initial_lora_sha256']}}
    for role in ROLES[1:]:
        v = read(OLD / 'runs' / role / 'train/TRAIN_COMPLETE.json')
        roles[role] = {'adapter': v['adapter'], 'lora_sha256': v['final_lora_sha256']}
    paths = [Path(__file__).resolve(), CONFIG, ROOT / 'DATA.json', Path(c['source_manifest']),
        Path(c['baseline_config']), Path(base['reference_config']), PROJECT_ROOT / 'configs/vlm_region_ocr_sft_v1.json',
        PROJECT_ROOT / 'docs/VLM_NATIVE_QUESTION_BINDING_V1_PROTOCOL.md', OLD / 'COMPLETION.json']
    paths += [ROOT.parent / f for f in EXCLUSIONS]
    paths += [Path(v['adapter']) / f for v in roles.values() for f in ('adapter_model.safetensors', 'adapter_config.json')]
    paths += [SOURCE_ROOT / f for f in ('vlm_region_ocr_sft_v1/study.py', 'vlm_probe/evaluate.py',
        'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
        'vlm_attention_scope_repair_v1/generation.py', 'vlm_deep/common.py', 'vlm_deep/evaluate.py',
        'vlm_error_supervision_sft_v1/common.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'roles': roles, 'pairs': 128, 'qa_per_role': 256,
        'eligible_pair_images': len(candidates), 'actual_inference_cpu_encodings': 256,
        'worker_active_cap_seconds': 300, 'training_updates': 0, 'automatic_training': False,
        'scope': 'Native-question conditioning on new adaptation images from initial SFT training whitelist; not original benchmark gain.',
        'files': {str(f): sha_file(f) for f in paths}, 'images': {v['path']: v['sha256'] for v in images.values()}})
    print({'status': 'prepared', 'pairs': 128, 'actual_cpu_encodings': 256, 'eligible_pair_images': len(candidates)}, flush=True)

def verify():
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen native-question input changed: ' + path)
    return plan

def run(role):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = verify()
    model, processor, _ = actor(plan['roles'][role]['adapter'])
    if lora_hash(model) != plan['roles'][role]['lora_sha256']:
        raise ValueError('Wrong saved checkpoint')
    model.eval()
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    data = read(ROOT / 'DATA.json')
    path = ROOT / role / 'PREDICTIONS.json'
    records = read(path) if path.exists() else []
    if len(records) % 4 or [q['question_id'] for q in records] != [q['question_id'] for q in data['rows'][:len(records)]]:
        raise ValueError('Resume order or batch grouping differs')
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(len(records), 256, 4):
            chunk = data['rows'][start:start + 4]
            answers, resource = evaluate_batch(model, processor, chunk, data['images'], ROOT,
                next(model.parameters()).device, 1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            for q, answer in zip(chunk, answers):
                if answer['input_tokens'] != q['expected_prefix_tokens'] or answer['image_grid_thw'] != q['expected_grid']:
                    raise ValueError('Actual native inference frame differs')
                records.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'pair_id': q['pair_id'],
                    'side': q['side'], 'target': q['target'], 'answer_frequency': q['answer_frequency'],
                    'vqa_score': score_prediction(answer['prediction'], q, 'textvqa'), **answer, **resource})
            write_json(path, records)
            write_json(ROOT / role / 'PROGRESS.json', {'status': 'generating', 'completed_qa': len(records),
                'expected_qa': 256, 'training_updates': 0, 'active_seconds': time.monotonic() - started})
            if time.monotonic() - started > plan['worker_active_cap_seconds']:
                raise TimeoutError('Native-question generation time cap reached')
    if lora_hash(model) != plan['roles'][role]['lora_sha256']:
        raise ValueError('Read-only generation changed adapter')
    write_json(ROOT / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 0, 'qa': len(records),
        'adapter_unchanged': True, 'active_seconds': time.monotonic() - started})

def analyze():
    from vlm_error_supervision_sft_v1.common import paired
    records = {r: read(ROOT / r / 'PREDICTIONS.json') for r in ROLES}
    pairs, metrics = {}, {}
    for role, rows in records.items():
        pairs[role] = []
        for a, b in zip(rows[::2], rows[1::2]):
            if a['pair_id'] != b['pair_id'] or (a['side'], b['side']) != ('A', 'B'):
                raise ValueError('A/B group changed')
            pairs[role].append({'question_id': a['pair_id'], 'image_id': a['image_id'],
                'vqa_score': float(a['vqa_score'] >= .999999 and b['vqa_score'] >= .999999),
                'same_answer': NORMALIZER(a['prediction']) == NORMALIZER(b['prediction'])})
        rare = [q for q in rows if q['answer_frequency'] <= 5]
        metrics[role] = {'vqa_percent': 100 * sum(q['vqa_score'] for q in rows) / 256,
            'both_full_score_pairs': sum(q['vqa_score'] for q in pairs[role]),
            'same_answer_pairs': sum(q['same_answer'] for q in pairs[role]),
            'rare_proxy_qa': len(rare), 'rare_proxy_vqa_percent': 100 * sum(q['vqa_score'] for q in rare) / len(rare)}
        write_json(ROOT / role / 'PAIRS.json', pairs[role])
    comparisons = {}
    for a, b in [('initial', 'sft'), ('initial', 'dpo_sft'), ('sft', 'dpo_sft')]:
        comparisons[b + '_minus_' + a] = {'qa': paired(records[a], records[b]), 'both_correct': paired(pairs[a], pairs[b])}
    result = {'status': 'completed', 'training_updates': 0, 'metrics': metrics, 'comparisons': comparisons,
        'scope': read(ROOT / 'PLAN.json')['scope'], 'intervals': 'Pointwise exploratory, no familywise confirmation.',
        'old_512_gate_failure_unchanged': True, 'automatic_training': False, 'automatic_main_evaluation': False}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 同图原生双问：完整零更新结果', '', result['scope'], '',
        '|模型|256题VQA|128图两问均满分|两问同答|罕见监督代理VQA|', '|---|---:|---:|---:|---:|']
    lines += [f'|{r}|{m["vqa_percent"]:.6f}%|{m["both_full_score_pairs"]}/128|{m["same_answer_pairs"]}/128|{m["rare_proxy_vqa_percent"]:.6f}%|' for r, m in metrics.items()]
    lines += ['', '完整图像簇区间见对应输出COMPLETION.json；输出不同率本身不等于正确绑定。旧512门槛失败不回写，不自动训练或改简历。']
    (PROJECT_ROOT / 'docs/VLM_NATIVE_QUESTION_BINDING_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--role', choices=ROLES)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.role:
        run(a.role)
    else:
        from research_runtime.queue import run_queue
        verify()
        jobs = [{'key': r, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--role', r],
            'completion_marker': str(ROOT / r / 'COMPLETE.json')} for r in ROLES]
        run_queue(ROOT / 'queue', jobs, {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        analyze()

if __name__ == '__main__':
    main()

