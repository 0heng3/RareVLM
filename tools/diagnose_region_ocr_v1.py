"""Read-only, fixed-training-cohort diagnosis after the region SFT negative result."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json

PARENT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_ocr_sft_v1'
ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_ocr_diagnosis_v1'
ROLES = ('control', 'region', 'initial')


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from vlm_region_ocr_sft_v1.study import verify
    verify()
    if read(PARENT / 'COMPLETION.json')['passed']:
        raise ValueError('This diagnosis is registered for the completed negative experiment')
    probe = read(PARENT / 'data/REGION_TRAIN_PROBE.json')
    control = read(PARENT / 'data/CONTROL_TRAIN.json')
    images = [q['image_id'] for q in probe['qa']]
    by_image = {source: {q['image_id']: q for q in control['splits']['train_all']
                         if q['study_source'] == source} for source in ('common', 'control_aux')}
    cohorts = {'region': probe['qa'],
               **{source: [by_image[source][iid] for iid in images] for source in by_image}}
    data = {'cohorts': cohorts, 'images': probe['images']}
    if any(len(q) != 64 for q in cohorts.values()):
        raise ValueError('Expected three fixed, matched 64-question training cohorts')
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Diagnosis already registered; inspect its status before resuming')
    write_json(ROOT / 'DATA.json', data)
    changes = read(PARENT / 'SCREEN_CHANGED_CASES.json')
    prefix = {}
    for before in ('initial', 'control'):
        rows = changes[before]['text']
        shorter = [q for q in rows if q['after'].strip()
                   and q['before'].lower().startswith(q['after'].lower() + ' ')]
        prefix[before] = {'changed_predictions': len(rows), 'strict_prefix_predictions': len(shorter),
                          'prefix_net_score_points': sum(q['after_score']-q['before_score'] for q in shorter),
                          'all_net_score_points': sum(q['after_score']-q['before_score'] for q in rows),
                          'cases': shorter}
    lengths = {}
    for role in ('control', 'region'):
        rows = read(PARENT / f'data/{role.upper()}_TRAIN.json')['splits']['train_all']
        lengths[role] = {src: {'questions': len(values), 'mean_whitespace_words': statistics.mean(values),
                               'single_word_targets': sum(x == 1 for x in values)}
                         for src in sorted({q['study_source'] for q in rows})
                         for values in [[len(q['target'].split()) for q in rows if q['study_source'] == src]]}
    files = [Path(__file__).resolve(), ROOT / 'DATA.json', PARENT / 'COMPLETION.json',
             PARENT / 'initial/region_train_probe/PREDICTIONS.json',
             PARENT / 'data/CPU_INPUTS.jsonl', PARENT / 'PRELAUNCH_INPUT_LOCK.json']
    config = read(PROJECT_ROOT / 'configs/vlm_region_ocr_sft_v1.json')
    adapters = {role: config['initial_adapter'] if role == 'initial'
                else str(PARENT / f'runs/{role}/train/adapter') for role in ROLES}
    files += [Path(path) / name for path in adapters.values()
              for name in ('adapter_model.safetensors', 'adapter_config.json')]
    write_json(ROOT / 'PLAN.json', {
        'registered_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Fixed original training-only cohorts; no optimizer, heldout selection, or gain claim',
        'question_selection': 'Exactly the pre-training hash-fixed 64 region questions and their original QA siblings',
        'cohort_qa': {k: len(v) for k, v in cohorts.items()},
        'new_generation_qa': {'initial': 0, 'control': 64, 'region': 64},
        'teacher_forced_qa_per_model': 192,
        'teacher_metrics': 'Per-QA CE, answer-text NLL, EOS NLL and EOS-vs-GT continuation logit after first word',
        'adapters': adapters, 'frozen_files': {str(p): sha_file(p) for p in files},
        'parent_decoding_and_numerics': 'Unchanged FP32 adapter/head; explicit raw greedy; max32/batch4',
        'worker_active_cap_seconds': 600, 'training_target_lengths': lengths,
        'screen_prefix_observation': prefix,
        'causal_limit': 'Prefix association and teacher-forced margins do not establish a unique training cause'})


def teacher_batch(model, processor, rows, images, cpu_records):
    import torch
    import torch.nn.functional as F
    from vlm_deep.common import encode, collate
    from vlm_sft.common import to_device
    samples = [encode(processor, q, Path(images[q['image_id']]['path']), True) for q in rows]
    for q, sample in zip(rows, samples):
        old = cpu_records[q['question_id']]
        if any(content_sha(sample[k].tolist()) != old[k + '_sha256'] for k in ('input_ids', 'labels')):
            raise ValueError('Teacher-forced inputs do not match the frozen original training inputs')
    batch = to_device(collate(samples, processor.tokenizer.pad_token_id), next(model.parameters()).device)
    counts = batch['labels'].ne(-100).sum(1)
    keep = int(counts.max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
    try:
        out = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        hook.remove()
    targets = batch['labels'][:, -out.logits.shape[1]:][:, 1:]
    logits = out.logits[:, :-1].float()
    if not torch.equal(targets.ne(-100).sum(1), counts):
        raise ValueError('Causal shifting lost a target token')
    values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    result = []
    for i, q in enumerate(rows):
        positions = targets[i].ne(-100).nonzero().flatten()
        ids = targets[i, positions].tolist()
        losses = values[i, positions].tolist()
        eos_at = ids.index(eos)
        if eos_at == 0:
            raise ValueError('Expected nonempty transcription before EOS')
        record = {'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                  'qa_ce': statistics.mean(losses), 'answer_text_nll': statistics.mean(losses[:eos_at]),
                  'eos_nll': losses[eos_at], 'assistant_target_ids': ids,
                  'assistant_token_nll': losses, 'first_word_eos_minus_gt_logit': None}
        if len(q['target'].split()) > 1:
            first = q['target'].split()[0]
            boundaries = [n for n in range(1, eos_at)
                          if processor.tokenizer.decode(ids[:n]).strip() == first]
            if boundaries:
                n = max(boundaries)
                pos = int(positions[n])
                record['first_word_eos_minus_gt_logit'] = float(logits[i, pos, eos] - logits[i, pos, ids[n]])
        if not all(math_finite(x) for x in losses):
            raise FloatingPointError('Nonfinite teacher-forced token loss')
        result.append(record)
    return result


def math_finite(value):
    import math
    return math.isfinite(value)


def worker(role):
    import torch
    from vlm_region_ocr_sft_v1.study import actor
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_probe.evaluate import evaluate_batch
    from vlm_human_sft.evaluate import lora_hash
    plan = read(ROOT / 'PLAN.json')
    for path, expected in plan['frozen_files'].items():
        if sha_file(path) != expected:
            raise ValueError('Diagnostic input changed: ' + path)
    data = read(ROOT / 'DATA.json'); directory = ROOT / role
    directory.mkdir(exist_ok=True)
    cpu_records = {q['question_id']: q for q in
                   (json.loads(line) for line in (PARENT / 'data/CPU_INPUTS.jsonl').read_text().splitlines())}
    model, processor, _ = actor(plan['adapters'][role])
    initial_hash = lora_hash(model)
    model.eval(); model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    started = time.monotonic(); teacher = {}; predictions = []
    def progress(stage, done, total):
        elapsed = time.monotonic()-started
        write_json(directory / 'PROGRESS.json', {'status': 'running', 'role': role, 'stage': stage,
                   'completed_qa': done, 'expected_qa': total, 'worker_active_seconds': elapsed})
        if elapsed > plan['worker_active_cap_seconds']:
            raise TimeoutError('Registered read-only diagnostic budget exceeded')
    with torch.inference_mode():
        for cohort, rows in data['cohorts'].items():
            teacher[cohort] = []
            for start in range(0, len(rows), 4):
                teacher[cohort] += teacher_batch(model, processor, rows[start:start+4], data['images'], cpu_records)
                if start % 16 == 0:
                    progress('teacher_' + cohort, start+4, len(rows))
            write_json(directory / f'TEACHER_{cohort}.json', teacher[cohort])
        if role == 'initial':
            predictions = read(PARENT / 'initial/region_train_probe/PREDICTIONS.json')
        else:
            rows = data['cohorts']['region']
            for start in range(0, len(rows), 4):
                chunk = rows[start:start+4]
                answers, _ = evaluate_batch(model, processor, chunk, data['images'], PARENT / 'data',
                    next(model.parameters()).device, 1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
                predictions += [{'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                    **a, 'raw_exact': a['prediction'] == q['target'],
                    'normalized_correct': bool(a['prediction'].strip()) and NORMALIZER(a['prediction']) == NORMALIZER(q['target'])}
                    for q, a in zip(chunk, answers)]
                write_json(directory / 'PREDICTIONS.json', predictions)
                progress('region_generation', start+4, len(rows))
        if [q['question_id'] for q in predictions] != [q['question_id'] for q in data['cohorts']['region']]:
            raise ValueError('Region question order changed')
    if lora_hash(model) != initial_hash:
        raise ValueError('Read-only diagnosis changed model parameters')
    write_json(directory / 'PREDICTIONS.json', predictions)
    summary = {'status': 'completed', 'role': role, 'qa': 64,
        'normalized_correct': sum(q['normalized_correct'] for q in predictions),
        'raw_exact_correct': sum(q['raw_exact'] for q in predictions),
        'worker_active_seconds': time.monotonic()-started, 'training_updates': 0,
        'adapter_unchanged': True, 'lora_sha256': initial_hash, 'teacher': {}}
    for cohort, rows in teacher.items():
        summary['teacher'][cohort] = {key: statistics.mean(q[key] for q in rows if q[key] is not None)
            for key in ('qa_ce', 'answer_text_nll', 'eos_nll', 'first_word_eos_minus_gt_logit')
            if any(q[key] is not None for q in rows)}
        summary['teacher'][cohort]['multiword_margin_questions'] = sum(q['first_word_eos_minus_gt_logit'] is not None for q in rows)
    write_json(directory / 'COMPLETE.json', summary)
    write_json(directory / 'PROGRESS.json', summary)
    print(json.dumps(summary), flush=True)


def run():
    from research_runtime.queue import run_queue
    run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--role', role],
        'completion_marker': str(ROOT / role / 'COMPLETE.json')} for role in ROLES],
        {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    summaries = {role: read(ROOT / role / 'COMPLETE.json') for role in ROLES}
    changes = {}
    current = read(ROOT / 'region/PREDICTIONS.json')
    for role in ('initial', 'control'):
        before = read(ROOT / role / 'PREDICTIONS.json')
        changes[role] = {'corrected': sum(not a['normalized_correct'] and b['normalized_correct'] for a,b in zip(before,current)),
                         'broken': sum(a['normalized_correct'] and not b['normalized_correct'] for a,b in zip(before,current))}
    write_json(ROOT / 'RESULTS.json', {'status': 'completed', 'scope': 'Training-only diagnostic, not heldout capability',
        'models': summaries, 'region_corrected_broken': changes, 'training_updates': 0,
        'screen_observation': read(ROOT / 'PLAN.json')['screen_prefix_observation']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.role:
        worker(args.role)
    else:
        run()
