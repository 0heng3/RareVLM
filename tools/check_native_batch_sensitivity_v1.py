"""Zero-update reproduction of six observed same-policy/batch disagreements."""
import argparse
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_native_correction_sft_dpo_v2'
ROOT = BASE / 'vlm_native_batch_sensitivity_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_BATCH_SENSITIVITY_V1_PROTOCOL.md'


def prepare():
    if (ROOT / 'PLAN.json').exists(): raise FileExistsError('Diagnostic already frozen')
    data = read(PARENT / 'data/MANIFEST.json'); hard = data['evaluations']['native_hard']
    baseline = {q['question_id']: q for q in read(PARENT / 'baseline/native_hard/PREDICTIONS.json')}
    chosen = [q for q in hard if q['initial_prediction']['vqa_score'] <= .3 + 1e-8 and baseline[q['question_id']]['vqa_score'] >= .999999]
    if len(chosen) != 6: raise ValueError('Expected six observed discordant training QAs')
    sources = [BASE / name / 'DATA.json' for name in ('vlm_native_correction_mining_v2', 'vlm_native_correction_supplement_v2')]
    pools = [read(p) for p in sources]; rows = {}; images = {}; jobs = []
    for q in chosen:
        original_group = None
        for pool in pools:
            indexes = [i for i, row in enumerate(pool['rows']) if row['question_id'] == q['question_id']]
            if indexes:
                i = indexes[0]; original_group = pool['rows'][(i // 4) * 4:(i // 4) * 4 + 4]; images.update(pool['images']); break
        if original_group is None: raise ValueError('Original mining group missing')
        index = next(i for i, row in enumerate(hard) if row['question_id'] == q['question_id'])
        fixed_group = hard[(index // 4) * 4:(index // 4) * 4 + 4]
        for row in original_group + fixed_group: rows[row['question_id']] = row
        jobs.append({'qid': q['question_id'], 'mining': [row['question_id'] for row in original_group], 'fixed': [row['question_id'] for row in fixed_group],
                     'historical_mining': q['initial_prediction']['prediction'], 'historical_fixed': baseline[q['question_id']]['prediction']})
    images.update(data['images']); used = {q['image_id'] for q in rows.values()}
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': {i: images[i] for i in used}, 'jobs': jobs})
    initial = read(BASE / 'vlm_native_correction_mining_v2/INITIALIZATION.json'); parent = read(PARENT / 'STUDY_INPUT_LOCK.json')
    paths = [Path(__file__), PROTOCOL, ROOT / 'DATA.json', PARENT / 'data/MANIFEST.json', PARENT / 'baseline/native_hard/PREDICTIONS.json',
             BASE / 'vlm_native_correction_mining_v2/INITIALIZATION.json'] + sources
    files = dict(parent['files']); files.update({str(p): sha_file(p) for p in paths})
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'files': files, 'packages': parent['packages'], 'base_weight_files': parent['base_weight_files'],
               'images': {images[i]['path']: images[i]['sha256'] for i in used}, 'initializer': initial, 'target_qa': 6,
               'sequence_generations': 96, 'target_generations': 24, 'optimizer_updates': 0, 'order': ['mining1', 'fixed1', 'mining2', 'fixed2']})
    print('BATCH_DIAGNOSTIC_PREPARED', {'target_qa': 6, 'sequences': 96, 'optimizer_updates': 0}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest: raise ValueError('Frozen diagnostic source/input changed')
    if package_lock() != plan['packages']: raise ValueError('Environment changed')
    return plan


def worker():
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); data = read(ROOT / 'DATA.json'); model, processor, _ = actor(adapter=plan['initializer']['adapter'])
    before = lora_hash(model)
    if before != plan['initializer']['lora_sha256']: raise ValueError('Wrong same-policy diagnostic model')
    model.eval(); model.config.use_cache = True; results = []; started = time.monotonic()
    for case in data['jobs']:
        runs = {}
        for name in plan['order']:
            mode = 'mining' if name.startswith('mining') else 'fixed'; ids = case[mode]; slot = ids.index(case['qid']); chunk = [data['rows'][qid] for qid in ids]
            trace = {'positions': [], 'logits': []}; prefix = data['rows'][case['qid']]['expected_prefix_tokens']
            def positions(module, args, kwargs):
                pos = kwargs.get('position_ids'); mask = kwargs.get('attention_mask')
                if pos is None: trace['positions'].append({'available': False}); return
                if pos.ndim == 3: selected = pos[:, slot, -min(prefix, pos.shape[-1]):]
                elif pos.ndim == 2: selected = pos[slot, -min(prefix, pos.shape[-1]):]
                else: raise ValueError('Unexpected actual position_ids shape')
                value = selected.detach().cpu().tolist()
                trace['positions'].append({'available': True, 'shape': list(pos.shape), 'unpad_position_sha256': content_sha(value),
                                           'first_positions': [v[:4] if isinstance(v, list) else v for v in value] if pos.ndim == 3 else value[:4],
                                           'attention_mask_shape': list(mask.shape) if mask is not None else None})
            def logits(module, args, output):
                values, indexes = torch.topk(output[slot, -1].float(), 10)
                vals = values.detach().cpu().tolist(); tokens = indexes.detach().cpu().tolist()
                trace['logits'].append({'token_ids': tokens, 'logits': vals, 'top1_top2_margin': vals[0] - vals[1]})
            handles = [model.get_base_model().model.register_forward_pre_hook(positions, with_kwargs=True), model.get_output_embeddings().register_forward_hook(logits)]
            try: predictions = generated(model, processor, chunk, data)
            finally:
                for handle in handles: handle.remove()
            if any(a['input_sha256'] != q['inference_input_sha256'] for a, q in zip(predictions, chunk)): raise ValueError('Actual native prefix changed')
            target = predictions[slot]
            runs[name] = {'prediction': target['prediction'], 'completion_ids': target['completion_ids'], 'input_sha256': target['input_sha256'],
                          'question_ids': ids, 'actual_prefix_tokens': prefix, **trace}
            write_json(ROOT / 'PROGRESS.json', {'status': 'diagnosing', 'completed_sequences': (len(results) * 4 + len(runs)) * 4, 'expected_sequences': 96, 'optimizer_updates': 0, 'active_seconds': time.monotonic() - started})
            if time.monotonic() - started > 600 or torch.cuda.max_memory_allocated() / 2**30 > 65: raise RuntimeError('Diagnostic budget reached')
        a, b = runs['mining1'], runs['fixed1']; divergence = next((i for i, (u, v) in enumerate(zip(a['completion_ids'], b['completion_ids'])) if u != v), None)
        if divergence is None and len(a['completion_ids']) != len(b['completion_ids']): divergence = min(len(a['completion_ids']), len(b['completion_ids']))
        first = None
        if divergence is not None and divergence < min(len(a['logits']), len(b['logits'])):
            first = {'index': divergence, 'mining_top1_top2_margin': a['logits'][divergence]['top1_top2_margin'], 'fixed_top1_top2_margin': b['logits'][divergence]['top1_top2_margin'],
                     'mining_top_token': processor.tokenizer.decode([a['completion_ids'][divergence]]), 'fixed_top_token': processor.tokenizer.decode([b['completion_ids'][divergence]])}
        p1, p2 = a['positions'][0], b['positions'][0]
        result = {**case, 'runs': runs, 'same_group_mining_repeat': a['completion_ids'] == runs['mining2']['completion_ids'],
                  'same_group_fixed_repeat': b['completion_ids'] == runs['fixed2']['completion_ids'], 'batch_outputs_equal': a['completion_ids'] == b['completion_ids'],
                  'unpad_prefill_positions_equal': p1.get('unpad_position_sha256') == p2.get('unpad_position_sha256') if p1.get('available') and p2.get('available') else None,
                  'first_divergence': first}
        results.append(result); write_json(ROOT / 'RESULTS.json', results)
    if lora_hash(model) != before: raise ValueError('Read-only check changed adapter')
    report = {'status': 'completed', 'target_qa': 6, 'new_generations': 96, 'optimizer_updates': 0, 'model_hash_unchanged': True,
              'same_group_mining_stable': sum(r['same_group_mining_repeat'] for r in results), 'same_group_fixed_stable': sum(r['same_group_fixed_repeat'] for r in results),
              'batch_disagreements': sum(not r['batch_outputs_equal'] for r in results), 'unpad_position_mismatches': sum(r['unpad_prefill_positions_equal'] is False for r in results),
              'active_seconds': time.monotonic() - started, 'scope': 'Six posthoc selected training cases; not a population instability rate or a new quality score.'}
    write_json(ROOT / 'COMPLETION.json', report); print('BATCH_DIAGNOSTIC_COMPLETE', report, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare(); return
    if args.worker: worker(); return
    from research_runtime.queue import run_queue
    verify(); run_queue(ROOT / 'queue', [{'key': 'diagnostic', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], 'completion_marker': str(ROOT / 'COMPLETION.json')}], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})


if __name__ == '__main__': main()
