"""Two zero-update prefills locating the first cross-batch numerical change."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json, content_sha
from vlm_grpo_iteration_v1.common import actor, read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_native_batch_sensitivity_v1'
ROOT = BASE / 'vlm_batch_origin_trace_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_BATCH_ORIGIN_TRACE_V1_PROTOCOL.md'


def prepare():
    if (ROOT / 'PLAN.json').exists(): raise FileExistsError('Already frozen')
    plan = read(PARENT / 'PLAN.json')
    files = dict(plan['files'])
    for path in [Path(__file__), PROTOCOL, PARENT / 'DATA.json', PARENT / 'PLAN.json', PARENT / 'RESULTS.json']:
        files[str(path)] = sha_file(path)
    write_json(ROOT / 'PLAN.json', {**plan, 'files': files, 'anchor_qid': '17716', 'optimizer_updates': 0, 'forward_batches': 2,
                                  'sequence_generations': 0, 'order': ['mining', 'fixed']})


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest: raise ValueError('Frozen input changed: ' + path)
    if package_lock() != plan['packages']: raise ValueError('Package mismatch')
    return plan


def worker():
    import torch
    from vlm_deep.common import collate, encode
    from vlm_probe.evaluate import inference_row
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); data = read(PARENT / 'DATA.json'); case = next(q for q in data['jobs'] if q['qid'] == plan['anchor_qid'])
    model, processor, _ = actor(adapter=plan['initializer']['adapter']); model.eval()
    initial_hash = lora_hash(model)
    if initial_hash != plan['initializer']['lora_sha256']: raise ValueError('Wrong initializer')
    base = model.get_base_model(); reference = {}; records = []; runs = {}; started = time.monotonic()
    def digest(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    for mode in plan['order']:
        ids = case[mode]; slot = ids.index(case['qid']); rows = [data['rows'][qid] for qid in ids]
        items = [encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False) for q in rows]
        for item, row in zip(items, rows):
            if content_sha(item['input_ids'].tolist()) != row['inference_input_sha256']: raise ValueError('Prefix mismatch')
        batch = to_device(collate(items, processor.tokenizer.pad_token_id), next(model.parameters()).device)
        counts = batch['image_grid_thw'].prod(-1).tolist(); start = sum(counts[:slot]); stop = start + counts[slot]
        prefix = rows[slot]['expected_prefix_tokens']; handles = []; names = []
        def capture(name, tensor):
            tensor = tensor.detach().cpu().contiguous(); names.append(name)
            if mode == 'mining': reference[name] = tensor; return
            old = reference.pop(name)
            if old.shape != tensor.shape: raise ValueError('Target trace slice shape mismatch: ' + name)
            difference = tensor.float() - old.float()
            records.append({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'equal': torch.equal(old, tensor),
                            'mining_sha256': digest(old), 'fixed_sha256': digest(tensor), 'max_abs': float(difference.abs().max()),
                            'rms_abs': float(difference.square().mean().sqrt()), 'changed_elements': int(difference.ne(0).sum())})
        capture('processor.pixel_values', batch['pixel_values'][start:stop])
        handles.append(base.visual.register_forward_pre_hook(lambda mod, args, kwargs: capture('visual.actual_pixels', args[0][start:stop]), with_kwargs=True))
        handles.append(base.visual.patch_embed.register_forward_hook(lambda mod, args, output: capture('visual.patch_embed', output[start:stop])))
        for index, layer in enumerate(base.visual.blocks):
            handles.append(layer.register_forward_hook(lambda mod, args, output, index=index: capture(f'visual.block{index:02d}', output[start:stop])))
        handles.append(base.visual.merger.register_forward_hook(lambda mod, args, output: capture('visual.merger_window_order', output[start//4:stop//4])))
        handles.append(base.visual.register_forward_hook(lambda mod, args, output: capture('visual.output_original_order', output[start//4:stop//4])))
        def decoder(mod, args, kwargs):
            capture('decoder.inputs_embeds', kwargs['inputs_embeds'][slot, -prefix:])
            capture('decoder.position_ids', kwargs['position_ids'][:, slot, -prefix:])
        handles.append(base.model.register_forward_pre_hook(decoder, with_kwargs=True))
        for index, layer in enumerate(base.model.layers):
            handles.append(layer.register_forward_hook(lambda mod, args, output, index=index: capture(f'decoder.layer{index:02d}.last_token', (output[0] if isinstance(output, tuple) else output)[slot, -1])))
        handles.append(base.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],)))
        try:
            with torch.inference_mode(): output = model(**batch, use_cache=False)
            capture('head.last_logits', output.logits[slot, -1])
            values, tokens = output.logits[slot, -1].topk(10)
            runs[mode] = {'ids': ids, 'slot': slot, 'padded_tokens': batch['input_ids'].shape[1], 'target_prefix_tokens': prefix,
                          'grids': batch['image_grid_thw'].cpu().tolist(), 'top_tokens': tokens.cpu().tolist(), 'logits': values.cpu().tolist(),
                          'pieces': [processor.tokenizer.decode([token]) for token in tokens.cpu().tolist()], 'trace_order': names}
        finally:
            for handle in handles: handle.remove()
        del output, batch, items
        write_json(ROOT / 'PROGRESS.json', {'status': 'tracing', 'completed_forwards': len(runs), 'expected_forwards': 2, 'active_seconds': time.monotonic() - started, 'optimizer_updates': 0})
        if time.monotonic() - started > 180 or torch.cuda.max_memory_allocated() / 2**30 > 65: raise RuntimeError('Trace budget')
    if reference or runs['mining']['trace_order'] != runs['fixed']['trace_order']: raise ValueError('Incomplete trace correspondence')
    if lora_hash(model) != initial_hash: raise ValueError('Weights changed')
    write_json(ROOT / 'RESULTS.json', {'qid': case['qid'], 'runs': runs, 'comparisons': records, 'runtime': {'visual_attention': type(base.visual.blocks[0].attn).__name__, 'language_attention': type(base.model.layers[0].self_attn).__name__, 'tf32': torch.backends.cuda.matmul.allow_tf32}})
    result = {'status': 'completed', 'optimizer_updates': 0, 'forward_batches': 2, 'model_hash_unchanged': True,
              'first_difference': next((x for x in records if not x['equal']), None), 'active_seconds': time.monotonic() - started}
    write_json(ROOT / 'COMPLETION.json', result); print('TRACE_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare(); return
    if args.worker: worker(); return
    from research_runtime.queue import run_queue
    verify(); run_queue(ROOT / 'queue', [{'key': 'trace', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], 'completion_marker': str(ROOT / 'COMPLETION.json')}], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})


if __name__ == '__main__': main()
