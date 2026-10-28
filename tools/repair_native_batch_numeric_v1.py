"""Frozen conditional numerical repair, selected on training stability only."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_native_batch_sensitivity_v1'
ROOT = BASE / 'vlm_batch_numeric_repair_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_BATCH_NUMERIC_REPAIR_V1_PROTOCOL.md'


def prepare():
    if (ROOT / 'PLAN.json').exists(): raise FileExistsError('Already frozen')
    plan = read(PARENT / 'PLAN.json'); data = read(PARENT / 'DATA.json'); original = read(BASE / 'vlm_native_correction_sft_dpo_v2/data/MANIFEST.json')
    selected = {q['qid'] for q in data['jobs']}
    extra = [q for q in original['evaluations']['native_hard'] if q['question_id'] not in selected][:8]
    for q in extra: data['rows'][q['question_id']] = q
    data['extra_ids'] = [q['question_id'] for q in extra]
    data['images'].update({q['image_id']: original['images'][q['image_id']] for q in extra})
    write_json(ROOT / 'DATA.json', data)
    files = dict(plan['files'])
    for path in [Path(__file__), PROTOCOL, ROOT / 'DATA.json', PROJECT_ROOT / 'src/vlm_batch_stable_v1/model.py', PROJECT_ROOT / 'src/vlm_batch_stable_v1/__init__.py', BASE / 'vlm_batch_origin_trace_v1/COMPLETION.json']:
        files[str(path)] = sha_file(path)
    images = {v['path']: v['sha256'] for v in data['images'].values()}
    write_json(ROOT / 'PLAN.json', {**plan, 'files': files, 'images': images, 'policies': ['imagewise', 'imagewise_fp32_language'],
                                  'generations_per_policy': 118, 'maximum_generations': 236, 'optimizer_updates': 0})


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest: raise ValueError('Frozen input changed: ' + path)
    if package_lock() != plan['packages']: raise ValueError('Package mismatch')
    return plan


def worker():
    import torch
    from vlm_batch_stable_v1.model import install
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); data = read(ROOT / 'DATA.json'); total = 0; summaries = []
    for policy in plan['policies']:
        model, processor, _ = actor(adapter=plan['initializer']['adapter']); before = lora_hash(model)
        if before != plan['initializer']['lora_sha256']: raise ValueError('Wrong policy')
        binding = install(model, policy); base = model.get_base_model(); started = time.monotonic(); cases = []; extras = []; torch.cuda.reset_peak_memory_stats()
        def run(ids, qid):
            nonlocal total
            rows = [data['rows'][i] for i in ids]; slot = ids.index(qid); counts = [int(torch.tensor(q['expected_grid']).prod()) // 4 for q in rows]; left = sum(counts[:slot]); trace = []
            def capture(module, args, output):
                t = output[left:left+counts[slot]].detach().cpu().contiguous()
                trace.append(hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
            handle = base.visual.register_forward_hook(capture)
            try: answers = generated(model, processor, rows, data)
            finally: handle.remove()
            if len(trace) != 1 or any(q['input_sha256'] != row['inference_input_sha256'] for q, row in zip(answers, rows)): raise ValueError('Input or visual trace mismatch')
            total += len(ids); write_json(ROOT / 'PROGRESS.json', {'status': 'repair_check', 'policy': policy, 'completed_sequences': total, 'maximum_sequences': 236, 'optimizer_updates': 0, 'active_seconds_this_policy': time.monotonic() - started})
            if time.monotonic() - started > 900 or torch.cuda.max_memory_allocated() / 2**30 > 65: raise RuntimeError('Repair diagnostic budget')
            return {**answers[slot], 'visual_sha256': trace[0], 'group_ids': ids}
        for q in data['jobs']:
            runs = {name: run(q['mining'] if name.startswith('mining') else q['fixed'], q['qid']) for name in ('mining1', 'fixed1', 'mining2', 'fixed2')}
            runs['singleton'] = run([q['qid']], q['qid'])
            cases.append({'qid': q['qid'], 'runs': runs, 'output_ids_equal': len({tuple(x['completion_ids']) for x in runs.values()}) == 1,
                          'visual_features_equal': len({x['visual_sha256'] for x in runs.values()}) == 1})
            write_json(ROOT / policy / 'CASES.json', cases)
        for start in (0, 4):
            ids = data['extra_ids'][start:start+4]
            # Every target is inspected while generating each batch once.
            feature_slices = []; rows = [data['rows'][qid] for qid in ids]
            def extra_capture(module, args, output):
                lengths = [int(torch.tensor(q['expected_grid']).prod()) // 4 for q in rows]
                for t in output.detach().cpu().split(lengths, dim=0): feature_slices.append(hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            handle = base.visual.register_forward_hook(extra_capture)
            try: batched = generated(model, processor, rows, data)
            finally: handle.remove()
            if len(feature_slices) != 4: raise ValueError('Extra visual boundary mismatch')
            total += 4
            for row, pred, feature in zip(rows, batched, feature_slices):
                single = run([row['question_id']], row['question_id'])
                if pred['input_sha256'] != row['inference_input_sha256']: raise ValueError('Extra batch prefix changed')
                extras.append({'qid': row['question_id'], 'batch': pred, 'singleton': single, 'output_ids_equal': pred['completion_ids'] == single['completion_ids'], 'visual_features_equal': feature == single['visual_sha256']})
        passed = all(q['output_ids_equal'] and q['visual_features_equal'] for q in cases + extras)
        if lora_hash(model) != before: raise ValueError('Weights changed')
        summary = {'policy': policy, 'passed': passed, 'selected_case_outputs_equal': sum(q['output_ids_equal'] for q in cases), 'extra_case_outputs_equal': sum(q['output_ids_equal'] for q in extras),
                   'all_visual_features_equal': all(q['visual_features_equal'] for q in cases + extras), 'generations': 118, 'active_seconds': time.monotonic() - started, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'binding': binding}
        summaries.append(summary); write_json(ROOT / policy / 'EXTRA.json', extras); write_json(ROOT / policy / 'SUMMARY.json', summary)
        import gc
        del model, processor, base, run
        gc.collect(); torch.cuda.empty_cache()
        if passed: break
    result = {'status': 'completed', 'repair_passed': summaries[-1]['passed'], 'selected_policy': summaries[-1]['policy'] if summaries[-1]['passed'] else None,
              'policies': summaries, 'new_generations': total, 'optimizer_updates': 0, 'scope': '14 fixed training cases, not global stability or ability improvement.'}
    write_json(ROOT / 'COMPLETION.json', result); print('REPAIR_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare(); return
    if args.worker: worker(); return
    from research_runtime.queue import run_queue
    verify(); run_queue(ROOT / 'queue', [{'key': 'repair', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], 'completion_marker': str(ROOT / 'COMPLETION.json')}], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})


if __name__ == '__main__': main()
