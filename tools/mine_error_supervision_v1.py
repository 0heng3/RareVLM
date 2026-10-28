"""Mine actual errors of the original SFT, with a prospectively separate validation cohort."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, read_manifest, sha_file, write_json
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_error_supervision_mining_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_scope_balanced_sft_v2.json'
EXCLUSIONS = (
    'vlm_region_ocr_sft_v1/data/CONTROL_TRAIN.json',
    'vlm_scope_balanced_sft_v2/data/empirical_mix/MANIFEST.json',
    'vlm_scope_balanced_sft_v2/data/EVAL.json',
    'vlm_scope_transfer_diagnosis_v1/DATA.json',
    'vlm_scope_preference_mining_v1/DATA.json',
    'vlm_region_preference_probe_v1/data/MANIFEST.json',
    'vlm_readout_transfer_v1/DATA.json',
    'vlm_pixel_transfer_diagnosis_v1/DATA.json',
)


def ordering(namespace, row):
    return hashlib.sha256((namespace + ':' + row['image_id'] + ':' + row['question_id']).encode()).hexdigest()


def text_relation(chosen, rejected):
    target, pred = NORMALIZER(chosen), NORMALIZER(rejected)
    if not pred:
        return 'empty'
    if target.startswith(pred + ' '):
        return 'prefix_missing_words'
    def subsequence(a, b):
        stream = iter(b)
        return all(any(word == candidate for candidate in stream) for word in a)
    if len(pred.split()) < len(target.split()) and subsequence(pred.split(), target.split()):
        return 'other_missing_words'
    if len(target.split()) < len(pred.split()) and subsequence(target.split(), pred.split()):
        return 'extra_words_or_context'
    return 'different_text'


def prepare():
    import torch
    from vlm_probe.evaluate import inference_row, processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen: inspect existing controller instead of regenerating')
    c = read(CONFIG)
    base = read(c['baseline_config'])
    source = read_manifest(c['source_manifest'])
    excluded = {q['image_id'] for split in ('dev', 'final') for q in source['splits'][split]}
    for relative in EXCLUSIONS:
        data = read(ROOT.parent / relative)
        excluded.update(data.get('images', {}))
        for key in ('hard_eval', 'retention_eval'):
            excluded.update(q['image_id'] for q in data.get(key, []))
    pixel = lambda im: im.get('pixel_sha256', im['sha256'])
    excluded_pixels = {pixel(source['images'][iid]) for iid in excluded if iid in source['images']}
    frequency = Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    eligible = [q for q in source['splits']['train_all'] if q['image_id'] not in excluded
                and pixel(source['images'][q['image_id']]) not in excluded_pixels
                and Counter(q['answers'])[q['target']] >= 8
                and 1 <= len(q['target'].split()) <= 6
                and 1 <= frequency[NORMALIZER(q['target'])] <= 5]
    eligible.sort(key=lambda q: ordering('error_supervision_mining_v1:pool', q))
    # Select the scarcer multiword stratum first; this rule is fixed before outcomes.
    buckets, seen = {}, set()
    for name, predicate in [('multiword', lambda q: len(q['target'].split()) > 1),
                            ('singleword', lambda q: len(q['target'].split()) == 1)]:
        selected = []
        for q in eligible:
            key = pixel(source['images'][q['image_id']])
            if not predicate(q) or key in seen:
                continue
            selected.append({**q, 'stratum': name, 'answer_frequency': frequency[NORMALIZER(q['target'])],
                             'fold': 'training_pool' if len(selected) < 512 else 'validation'})
            seen.add(key)
            if len(selected) == 768:
                break
        if len(selected) != 768:
            raise ValueError('Insufficient unique high-consensus rows; no fallback or threshold change')
        buckets[name] = selected
    rows = sum(buckets.values(), [])
    rows.sort(key=lambda q: (q['fold'], ordering('error_supervision_mining_v1:execution', q)))
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    images = {}
    started = time.monotonic()
    for index, q in enumerate(rows):
        image = source['images'][q['image_id']]
        image_path = (Path(c['source_manifest']).parent / image['path']).resolve()
        item = encode(processor, inference_row(q), image_path, False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]),
                 expected_grid=item['image_grid_thw'][0].tolist(),
                 inference_input_sha256=content_sha(item['input_ids'].tolist()))
        images[q['image_id']] = {**image, 'path': str(image_path)}
        if (index + 1) % 64 == 0:
            write_json(ROOT / 'CPU_PROGRESS.json', {'status': 'encoding', 'completed': index + 1, 'expected': 1536,
                                                    'active_seconds': time.monotonic() - started})
    if len(images) != 1536 or len(seen) != 1536:
        raise ValueError('Image or pixel identity not unique')
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': images})
    paths = [Path(__file__).resolve(), CONFIG, Path(c['source_manifest']), ROOT / 'DATA.json',
             PROJECT_ROOT / 'docs/VLM_ERROR_SUPERVISION_MINING_V1_PROTOCOL.md',
             PROJECT_ROOT / 'configs/vlm_region_ocr_sft_v1.json', Path(c['baseline_config']), Path(base['reference_config'])]
    paths += [ROOT.parent / f for f in EXCLUSIONS]
    paths += [Path(c['initial_adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
    paths += [SOURCE_ROOT / f for f in ('vlm_region_ocr_sft_v1/study.py', 'vlm_probe/evaluate.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_attention_scope_repair_v1/generation.py', 'vlm_deep/common.py', 'vlm_deep/evaluate.py',
              'vlm_human_sft/evaluate.py', 'vlm_sft/common.py', 'research_runtime/queue.py')]
    initial = read(ROOT.parent / 'vlm_region_ocr_sft_v1/STUDY_INPUT_LOCK.json')['initial_lora_sha256']
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'training_updates': 0, 'automatic_training': False,
               'initial_adapter': c['initial_adapter'], 'initial_lora_sha256': initial,
               'training_pool_qa': 1024, 'validation_qa': 512, 'unique_images': 1536,
               'teacher_scoring_or_eos_ranking': False, 'review_cap': 64, 'minimum_visibly_supported_pairs': 32,
               'worker_active_cap_seconds': 900, 'selection_eligible_qa': len(eligible), 'excluded_images': len(excluded),
               'files': {str(path): sha_file(path) for path in paths},
               'images': {im['path']: im['sha256'] for im in images.values()},
               'scope': 'Initial SFT training-whitelist data; new adaptation-train/validation split, not initial-unseen or benchmark gain.'})
    print({'status': 'prepared', 'training_pool_qa': 1024, 'validation_qa': 512,
           'actual_inference_cpu_encodings': len(rows), 'cpu_seconds': time.monotonic() - started}, flush=True)


def verify():
    plan = read(ROOT / 'PLAN.json')
    for path, expected in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != expected:
            raise ValueError('Frozen actual-error mining input changed: ' + path)
    return plan


def shard(index):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = verify()
    model, processor, _ = actor(plan['initial_adapter'])
    if lora_hash(model) != plan['initial_lora_sha256']:
        raise ValueError('Miner must be the original FP32 SFT checkpoint')
    model.eval()
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    data = read(ROOT / 'DATA.json')
    rows = data['rows'][index::2]
    directory = ROOT / f'shard_{index}'
    path = directory / 'PREDICTIONS.json'
    predictions = read(path) if path.exists() else []
    if len(predictions) % 4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Actual-error mining resume order differs')
    initial_count = len(predictions)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(len(predictions), len(rows), 4):
            chunk = rows[start:start + 4]
            answers, resource = evaluate_batch(model, processor, chunk, data['images'], ROOT,
                          next(model.parameters()).device, 1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            for q, answer in zip(chunk, answers):
                if answer['input_tokens'] != q['expected_prefix_tokens'] or answer['image_grid_thw'] != q['expected_grid']:
                    raise ValueError('Actual generation input differs from registered CPU frame')
                score = score_prediction(answer['prediction'], q, 'textvqa')
                chosen_score = score_prediction(q['target'], q, 'textvqa')
                usable = bool(NORMALIZER(answer['prediction'])) and answer['generated_nonpad_tokens'] < 32
                eligible = usable and score <= .3000001 and chosen_score - score >= .6999999
                predictions.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'question': q['question'],
                    'target': q['target'], 'chosen_score': chosen_score, 'vqa_score': score,
                    'fold': q['fold'], 'stratum': q['stratum'], 'answer_frequency': q['answer_frequency'],
                    'text_relation': text_relation(q['target'], answer['prediction']),
                    'quality_gap_passed': eligible, 'training_candidate': q['fold'] == 'training_pool' and eligible,
                    **answer, 'batch_start': start, **resource})
            write_json(path, predictions)
            elapsed = time.monotonic() - started
            write_json(directory / 'PROGRESS.json', {'status': 'generating', 'completed_qa': len(predictions),
                       'expected_qa': len(rows), 'new_qa_this_session': len(predictions) - initial_count,
                       'active_seconds': elapsed, 'training_updates': 0})
            if elapsed > plan['worker_active_cap_seconds']:
                raise TimeoutError('Registered actual-generation time cap reached')
    if lora_hash(model) != plan['initial_lora_sha256']:
        raise ValueError('Read-only mining changed original SFT')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'qa': len(predictions), 'training_updates': 0,
                'adapter_unchanged': True, 'active_seconds': time.monotonic() - started})


def gather():
    plan = verify()
    rows = sum((read(ROOT / f'shard_{i}/PREDICTIONS.json') for i in (0, 1)), [])
    data = read(ROOT / 'DATA.json')
    if len(rows) != 1536 or len({q['question_id'] for q in rows}) != 1536:
        raise ValueError('Actual generation coverage incomplete')
    train = [q for q in rows if q['fold'] == 'training_pool']
    validation = [q for q in rows if q['fold'] == 'validation']
    candidates = sorted((q for q in train if q['training_candidate']),
                        key=lambda q: ordering('error_supervision_mining_v1:review', q))
    write_json(ROOT / 'TRAINING_CANDIDATES.json', {'status': 'not_visually_admitted', 'pairs': candidates,
                'review_first': [q['question_id'] for q in candidates[:plan['review_cap']]],
                'validation_qa_used_for_training': 0, 'training_admitted': False})
    result = {'status': 'completed', 'training_updates': 0, 'adapter_unchanged': True,
              'training_pool_qa': len(train), 'validation_qa': len(validation),
              'algorithmic_training_pairs': len(candidates),
              'candidate_text_relation_counts': dict(Counter(q['text_relation'] for q in candidates)),
              'candidate_stratum_counts': dict(Counter(q['stratum'] for q in candidates)),
              'validation_initial_vqa_percent': 100 * sum(q['vqa_score'] for q in validation) / len(validation),
              'visible_review_completed': False, 'training_admitted': False, 'scope': plan['scope'],
              'next': 'Review the fixed first64 training pairs; never mine the validation pool or automatically train.'}
    write_json(ROOT / 'COMPLETION.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--shard', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.shard is not None:
        shard(args.shard)
    else:
        from research_runtime.queue import run_queue
        verify()
        jobs = [{'key': f'shard_{i}', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--shard', str(i)],
                 'completion_marker': str(ROOT / f'shard_{i}/COMPLETE.json')} for i in (0, 1)]
        run_queue(ROOT / 'queue', jobs, {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        gather()


if __name__ == '__main__':
    main()
