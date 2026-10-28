"""Teacher-prefix EOS screening followed by real greedy verification, training pool only; no updates."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import encode, collate, content_sha, sha_file, write_json, read_manifest, NORMALIZER
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_scope_preference_mining_v1'
SOURCE = ROOT.parent / 'vlm_scope_balanced_sft_v2'
DIAG = ROOT.parent / 'vlm_scope_transfer_diagnosis_v1'


def prepare():
    from vlm_probe.evaluate import processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already prepared; inspect current controller')
    c = read(PROJECT_ROOT / 'configs/vlm_scope_balanced_sft_v2.json')
    base = read(c['baseline_config'])
    source = read_manifest(c['source_manifest'])
    frequency = Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    exclusion_files = [DIAG / 'DATA.json', SOURCE / 'data/EVAL.json', SOURCE / 'data/empirical_mix/MANIFEST.json',
                       ROOT.parent / 'vlm_region_preference_probe_v1/data/MANIFEST.json', ROOT.parent / 'vlm_readout_transfer_v1/DATA.json']
    excluded = set(read(DIAG / 'DATA.json')['images'])
    excluded.update(q['image_id'] for q in read(SOURCE / 'data/empirical_mix/MANIFEST.json')['rows'])
    for path in (SOURCE / 'data/EVAL.json', ROOT.parent / 'vlm_region_preference_probe_v1/data/MANIFEST.json'):
        excluded.update(q['image_id'] for q in read(path)['retention_eval'])
    excluded.update(read(ROOT.parent / 'vlm_readout_transfer_v1/DATA.json')['images'])
    excluded.update(q['image_id'] for split in ('dev', 'final') for q in source['splits'][split])
    image_key = lambda im: im.get('pixel_sha256', im['sha256'])
    excluded_pixels = {image_key(source['images'][iid]) for iid in excluded if iid in source['images']}
    candidates = [q for q in source['splits']['train_all'] if q['image_id'] not in excluded
        and image_key(source['images'][q['image_id']]) not in excluded_pixels
        and 2 <= len(q['target'].split()) <= 6 and Counter(q['answers'])[q['target']] >= 8
        and 1 <= frequency[NORMALIZER(q['target'])] <= 5]
    candidates.sort(key=lambda q: hashlib.sha256(('scope_pref_mining_v1:' + q['image_id'] + ':' + q['question_id']).encode()).hexdigest())
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    rows, images, seen = [], {}, set()
    for q in candidates:
        im = source['images'][q['image_id']]
        key = image_key(im)
        if key in seen:
            continue
        path = (Path(c['source_manifest']).parent / im['path']).resolve()
        item = encode(processor, q, path, True)
        labels = item['labels'][0][item['labels'][0] != -100].tolist()
        boundaries = []
        words = list(re.finditer(r'\S+', q['target']))
        for word_index, match in enumerate(words[:-1]):
            prefix = q['target'][:match.end()]
            ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
            if ids and labels[:len(ids)] == ids and len(ids) < len(labels):
                boundaries.append({'word_prefix': prefix, 'prefix_words': word_index + 1,
                    'answer_token_index': len(ids), 'correct_next_token': labels[len(ids)]})
        if not boundaries:
            continue
        prefix_length = int(item['input_ids'].shape[1]) - len(labels)
        rows.append({**q, 'answer_frequency': frequency[NORMALIZER(q['target'])], 'boundaries': boundaries,
            'supervised_input_sha256': content_sha(item['input_ids'].tolist()),
            'supervised_labels_sha256': content_sha(item['labels'].tolist()),
            'expected_prefix_tokens': prefix_length, 'expected_grid': item['image_grid_thw'][0].tolist()})
        images[q['image_id']] = {**im, 'path': str(path)}
        seen.add(key)
        if len(rows) == 512:
            break
    if len(rows) != 512:
        raise ValueError('Need512 rare high-vote aligned training candidates; no fallback')
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': images})
    trained = read(SOURCE / 'runs/empirical_mix/train/TRAIN_COMPLETE.json')
    files = [Path(__file__).resolve(), ROOT / 'DATA.json', Path(c['source_manifest']),
             PROJECT_ROOT / 'docs/VLM_SCOPE_PREFERENCE_MINING_V1_PROTOCOL.md', DIAG / 'COMPLETION.json',
             SOURCE / 'runs/empirical_mix/train/TRAIN_COMPLETE.json', *exclusion_files,
             Path(trained['adapter']) / 'adapter_config.json', Path(trained['adapter']) / 'adapter_model.safetensors']
    files += [SOURCE_ROOT / name for name in ('vlm_region_ocr_sft_v1/study.py', 'vlm_probe/evaluate.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_alignment_signal/loss.py', 'vlm_attention_scope_repair_v1/generation.py', 'vlm_deep/common.py',
              'vlm_deep/evaluate.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'training_updates': 0, 'candidate_qa': 512, 'teacher_shards': 2,
               'max_real_rollouts': 128, 'minimum_algorithmically_eligible_pairs_before_considering_training': 32,
               'adapter': trained['adapter'], 'lora_sha256': trained['final_lora_sha256'],
               'scope': 'Train-only rare-answer proxy mining, not heldout/model gain. Miner is the completed empirical-mixture model, not a promoted model or next training initialization.',
               'teacher_cap_seconds_per_shard': 600, 'rollout_cap_seconds': 300,
               'files': {str(path): sha_file(path) for path in files}, 'images': {im['path']: im['sha256'] for im in images.values()}})
    print({'status': 'prepared', 'candidate_qa': 512, 'actual_supervised_cpu_encodings': 512,
           'word_boundaries': sum(len(q['boundaries']) for q in rows), 'new_optimizer_updates': 0})


def load_miner():
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen training-mining input changed: ' + path)
    model, processor, _ = actor(plan['adapter'])
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Wrong mining checkpoint')
    model.eval()
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    return plan, model, processor


def screen(shard):
    import torch
    from vlm_alignment_signal.loss import completion_logps
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    plan, model, processor = load_miner()
    data = read(ROOT / 'DATA.json')
    rows = data['rows'][shard * 256:(shard + 1) * 256]
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    directory = ROOT / f'screen_{shard}'
    path = directory / 'RECORDS.json'
    records = read(path) if path.exists() else []
    if len(records) % 4 or [q['qid'] for q in records] != [q['question_id'] for q in rows[:len(records)]]:
        raise ValueError('Teacher-screen cache unpaired')
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(len(records), len(rows), 4):
            chunk = rows[start:start + 4]
            samples = [encode(processor, q, Path(data['images'][q['image_id']]['path']), True) for q in chunk]
            for sample, q in zip(samples, chunk):
                if content_sha(sample['input_ids'].tolist()) != q['supervised_input_sha256'] or content_sha(sample['labels'].tolist()) != q['supervised_labels_sha256']:
                    raise ValueError('Actual teacher input differs from CPU encoding')
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), next(model.parameters()).device)
            labels = batch['labels']
            keep = int((labels != -100).sum(1).max()) + 1
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
            try:
                output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
            finally:
                hook.remove()
            logp, count = completion_logps(output.logits, labels)
            for i, q in enumerate(chunk):
                first = int((labels[i] != -100).nonzero()[0])
                boundaries = []
                for boundary in q['boundaries']:
                    absolute = first + boundary['answer_token_index']
                    if int(labels[i, absolute]) != boundary['correct_next_token']:
                        raise ValueError('Boundary shifted from correct assistant token')
                    position = absolute - 1 - (labels.shape[1] - output.logits.shape[1])
                    logits = output.logits[i, position].float()
                    eos_values = logits[eos]
                    eos_logit, eos_index = eos_values.max(0)
                    non_eos = logits.clone()
                    non_eos[eos] = -torch.inf
                    boundaries.append({**boundary, 'eos_id': eos[int(eos_index)],
                        'correct_next_text': processor.tokenizer.decode([boundary['correct_next_token']]),
                        'eos_vs_best_non_eos': float(eos_logit - non_eos.max()),
                        'eos_vs_correct_continuation': float(eos_logit - logits[boundary['correct_next_token']])})
                records.append({'qid': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                    'boundaries': boundaries, 'mean_gt_ce': float(-logp[i] / count[i]),
                    'max_eos_stop_margin': max(x['eos_vs_best_non_eos'] for x in boundaries),
                    'teacher_prefix_eos_top1': any(x['eos_vs_best_non_eos'] >= 0 for x in boundaries)})
                if not math.isfinite(records[-1]['mean_gt_ce']) or any(not math.isfinite(v['eos_vs_best_non_eos']) or not math.isfinite(v['eos_vs_correct_continuation']) for v in boundaries):
                    raise FloatingPointError('Nonfinite EOS-screen readout')
            write_json(path, records)
            write_json(directory / 'PROGRESS.json', {'status': 'screening', 'completed_qa': len(records), 'expected_qa': 256,
                       'active_seconds': time.monotonic() - started})
            del samples, batch, output, labels, logp, count, logits, non_eos
            if time.monotonic() - started > plan['teacher_cap_seconds_per_shard']:
                raise TimeoutError('Teacher-prefix screening cap reached')
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Screening changed miner')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'training_updates': 0,
               'qa': len(records), 'adapter_unchanged': True, 'active_seconds': time.monotonic() - started})


def rank():
    records = read(ROOT / 'screen_0/RECORDS.json') + read(ROOT / 'screen_1/RECORDS.json')
    if len(records) != 512 or len({q['qid'] for q in records}) != 512:
        raise ValueError('Teacher coverage incomplete')
    records.sort(key=lambda q: (-q['max_eos_stop_margin'], q['qid']))
    data = read(ROOT / 'DATA.json')
    lookup = {q['question_id']: q for q in data['rows']}
    selected = [{**lookup[q['qid']], 'screen_rank': i + 1, 'teacher_eos_stop_margin': q['max_eos_stop_margin']}
                for i, q in enumerate(records[:128])]
    write_json(ROOT / 'ROLLOUT_PLAN.json', {'rows': selected, 'teacher_prefix_eos_top1_qa': sum(q['teacher_prefix_eos_top1'] for q in records),
               'note': 'Teacher-forced EOS ranking is NOT a confirmed greedy error or preference label.'})
    paths = [ROOT / 'PLAN.json', ROOT / 'ROLLOUT_PLAN.json', ROOT / 'screen_0/RECORDS.json', ROOT / 'screen_1/RECORDS.json']
    write_json(ROOT / 'ROLLOUT_INPUT_LOCK.json', {str(p): sha_file(p) for p in paths})


def validate():
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_human_sft.evaluate import lora_hash
    plan, model, processor = load_miner()
    for path, digest in read(ROOT / 'ROLLOUT_INPUT_LOCK.json').items():
        if sha_file(path) != digest:
            raise ValueError('Frozen rollout selection changed')
    data = read(ROOT / 'DATA.json')
    rows = read(ROOT / 'ROLLOUT_PLAN.json')['rows']
    path = ROOT / 'verification/PREDICTIONS.json'
    predictions = read(path) if path.exists() else []
    if len(predictions) % 4 or [q['qid'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Verification cache order differs')
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(len(predictions), len(rows), 4):
            chunk = rows[start:start + 4]
            answers, _ = evaluate_batch(model, processor, chunk, data['images'], ROOT, next(model.parameters()).device,
                                       1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            for q, answer in zip(chunk, answers):
                if answer['input_tokens'] != q['expected_prefix_tokens'] or answer['image_grid_thw'] != q['expected_grid']:
                    raise ValueError('Greedy verification prefix frame differs')
                rejected_score = score_prediction(answer['prediction'], q, 'textvqa')
                chosen_score = score_prediction(q['target'], q, 'textvqa')
                pred, target = NORMALIZER(answer['prediction']), NORMALIZER(q['target'])
                prefix = bool(pred) and target.startswith(pred + ' ')
                eligible = prefix and rejected_score <= .3000001 and chosen_score - rejected_score >= .6999999
                predictions.append({'qid': q['question_id'], 'image_id': q['image_id'], 'question': q['question'],
                    'chosen': q['target'], 'chosen_score': chosen_score, 'rejected': answer['prediction'], 'rejected_score': rejected_score,
                    'actual_prefix_truncated': prefix, 'algorithmically_eligible': eligible,
                    'answer_frequency': q['answer_frequency'], 'teacher_eos_stop_margin': q['teacher_eos_stop_margin']})
            write_json(path, predictions)
            write_json(ROOT / 'verification/PROGRESS.json', {'status': 'verifying_actual_generation', 'completed_qa': len(predictions),
                       'expected_qa': 128, 'active_seconds': time.monotonic() - started})
            if time.monotonic() - started > plan['rollout_cap_seconds']:
                raise TimeoutError('Actual verification generation cap reached')
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Verification changed miner')
    pairs = [q for q in predictions if q['algorithmically_eligible']]
    write_json(ROOT / 'PREFERENCE_CANDIDATES.json', {'status': 'algorithmic_candidates_not_visually_admitted', 'pairs': pairs,
               'no_training_performed': True, 'heldout_diagnostic_questions_used': 0})
    quantity = len(pairs) >= plan['minimum_algorithmically_eligible_pairs_before_considering_training']
    result = {'status': 'completed', 'training_updates': 0, 'teacher_screen_qa': 512, 'actual_greedy_verified_qa': 128,
              'teacher_prefix_eos_top1_qa': read(ROOT / 'ROLLOUT_PLAN.json')['teacher_prefix_eos_top1_qa'],
              'actual_prefix_truncations': sum(q['actual_prefix_truncated'] for q in predictions), 'eligible_preference_pairs': len(pairs),
              'minimum_quantity_passed': quantity, 'training_admitted': False, 'adapter_unchanged': True,
              'scope': plan['scope'], 'next': 'No automatic SFT/DPO. Quantity and visible-evidence review plus a matched protocol are required; mining itself is not model gain.'}
    write_json(ROOT / 'verification/COMPLETE.json', result)
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 真实截短偏好挖掘：结果', '', result['scope'], '',
             f'512训练候选teacher筛选，128实际greedy验证，前缀截短{result["actual_prefix_truncations"]}，质量差合格{len(pairs)}对。',
             f'数量门槛通过={quantity}；尚未人工可见证据准入或训练，0更新、无模型提升声明。']
    (PROJECT_ROOT / 'docs/VLM_SCOPE_PREFERENCE_MINING_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--screen', type=int, choices=(0, 1))
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.screen is not None:
        screen(args.screen)
    elif args.validate:
        validate()
    else:
        from research_runtime.queue import run_queue
        jobs = [{'key': f'screen_{i}', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--screen', str(i)],
                 'completion_marker': str(ROOT / f'screen_{i}/COMPLETE.json')} for i in (0, 1)]
        run_queue(ROOT / 'screen_queue', jobs, {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        rank()
        run_queue(ROOT / 'verification_queue', [{'key': 'verify', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--validate'],
                  'completion_marker': str(ROOT / 'verification/COMPLETE.json')}], {'rollout_lock_sha256': sha_file(ROOT / 'ROLLOUT_INPUT_LOCK.json')})


if __name__ == '__main__':
    main()
