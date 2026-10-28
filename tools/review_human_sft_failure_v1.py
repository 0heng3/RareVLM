"""Bounded, zero-optimizer-update diagnosis of a completed negative experiment."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, write_json

ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
OUT = ROOT/'vlm_human_sft_failure_review_v1'
STUDY = ROOT/'vlm_human_sft_repair_v1'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    manifest = read(ROOT/'vlm_human_vqa_v1/MANIFEST.json')
    selected, images = [], set()
    for label in ('yes', 'no'):
        rows = [row for row in manifest['splits']['train_all'] if row['target'] == label]
        count = 0
        for row in sorted(rows, key=lambda row: content_sha(['failure_v1_fixed_fit_sample', row['question_id']])):
            if row['image_id'] in images:
                continue
            selected.append(row['question_id']); images.add(row['image_id']); count += 1
            if count == 128:
                break
        if count != 128:
            raise ValueError('Insufficient image-distinct diagnostic training rows')
    checkpoints = {}
    for condition in ('control', 'human'):
        path, = (STUDY/'runs'/condition/'train').glob('checkpoint_000200_*')
        checkpoints[condition] = str(path/'adapter')
    plan = {'fit_sample_question_ids': selected, 'fit_sample': '256 training QA, balanced labels and distinct images; not heldout',
            'early_updates': 200, 'early_adapters': checkpoints,
            'heldout_qa': 474, 'benchmark_microbatches': [4, 8, 16], 'effective_batch': 16,
            'warmup_equivalent_updates_per_microbatch': 1, 'timed_repeats': 3,
            'optimizer_updates': 0, 'retrospective': True,
            'selection': 'fixed step and training-ID hash, before reading early checkpoint scores; no checkpoint selection/deployment'}
    if (OUT/'PLAN.json').exists() and read(OUT/'PLAN.json') != plan:
        raise ValueError('Diagnostic plan changed')
    write_json(OUT/'PLAN.json', plan)


def load_actor(adapter, training=False):
    import torch
    from vlm_fp32_adapter.model import load
    config = read(PROJECT_ROOT/'configs/vlm_human_sft_baseline_v1.json')
    model, _ = load(config['model_root'], adapter, torch.device('cuda:0'), training=training, dropout=0.)
    return model


def prefix_scores(model, processor, rows, manifest, tag):
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_probe.evaluate import inference_row
    path = OUT/(tag+'.json')
    if path.exists():
        return read(path)
    aliases = {}
    for label in ('yes', 'no'):
        ids = set()
        for word in (label, label.title(), label.upper(), ' '+label, ' '+label.title(), ' '+label.upper()):
            tokens = processor.tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) == 1:
                ids.add(tokens[0])
        aliases[label] = sorted(ids)
    if not aliases['yes'] or not aliases['no'] or set(aliases['yes']) & set(aliases['no']):
        raise ValueError('Binary first-token alias sets invalid')
    records, started = [], time.monotonic()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            chunk = rows[start:start+4]
            samples = [encode(processor, inference_row(row), Path(manifest['images'][row['image_id']]['path']), False) for row in chunk]
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), torch.device('cuda:0'))
            handle = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
            try:
                logits = model(**batch, use_cache=False).logits[:, -1, :].float()
            finally:
                handle.remove()
            margin = torch.logsumexp(logits[:, aliases['yes']], 1)-torch.logsumexp(logits[:, aliases['no']], 1)
            values = margin.cpu().tolist()
            for row, value in zip(chunk, values):
                records.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                                'target': row['target'], 'yes_minus_no_logit': value})
            del logits, batch
            if len(records)%128 == 0:
                print(json.dumps({'phase': tag, 'completed_qa': len(records), 'expected_qa': len(rows)}), flush=True)
    result = {'status': 'completed', 'records': records, 'aliases': aliases,
              'elapsed_seconds': time.monotonic()-started,
              'scope': 'conditional yes/no first-token readout, not the frozen generation accuracy'}
    write_json(path, result)
    return result


def benchmark(processor, model):
    import numpy as np
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_fp32_adapter.loss import answer_loss
    from vlm_human_sft.evaluate import lora_hash
    from vlm_human_sft.study import data, slots, update_plan
    config = read(PROJECT_ROOT/'configs/vlm_human_sft_repair_v1.json')
    reference, manifests = data(config)
    records = slots(manifests['text']['splits']['train_all'], manifests['human']['splits']['train_all'], 'human', 42)
    indices = update_plan(len(records), 16, 42)[0]
    samples = []; started = time.monotonic()
    for index in indices:
        row, source = records[index]; task = 'human' if source == 'human' else 'text'
        image = manifests[task]['images'][row['image_id']]
        path = Path(reference[task+'_manifest']).parent/image['path']
        samples.append(encode(processor, row, path, True))
    preprocessing = time.monotonic()-started
    before = lora_hash(model); plan = read(OUT/'PLAN.json'); results = {}; reference_grad = None
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for micro in plan['benchmark_microbatches']:
        times = []; losses_out = []
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for repeat in range(1+plan['timed_repeats']):
            model.train(); model.get_base_model().visual.eval(); model.zero_grad(set_to_none=True)
            total = 0.; torch.cuda.synchronize(); begun = time.monotonic()
            for start in range(0, 16, micro):
                batch = to_device(collate(samples[start:start+micro], processor.tokenizer.pad_token_id), torch.device('cuda:0'))
                values, _, _ = answer_loss(model, batch); loss = values.sum()/16
                loss.backward(); total += float(values.detach().sum())/16
                del values, loss, batch
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            torch.cuda.synchronize(); elapsed = time.monotonic()-begun
            if repeat:
                times.append(elapsed); losses_out.append(total)
        gradients = torch.cat([parameter.grad.detach().cpu().flatten() for parameter in parameters])
        if reference_grad is None:
            reference_grad = gradients.clone()
        relative = float((gradients-reference_grad).double().norm()/reference_grad.double().norm())
        results[str(micro)] = {'core_update_seconds': times, 'median_core_seconds': float(np.median(times)),
                              'mean_qa_ce': losses_out, 'gradient_relative_l2_vs_micro4': relative,
                              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                              'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
        print(json.dumps({'benchmark_microbatch': micro, **results[str(micro)]}), flush=True)
    if lora_hash(model) != before:
        raise ValueError('Zero-step benchmark unexpectedly changed weights')
    write_json(OUT/'MICROBATCH_BENCHMARK.json', {'status': 'completed', 'optimizer_updates': 0,
        'effective_batch': 16, 'fixed_input_source_counts': {'text': 13, 'human': 3},
        'preprocessing_once_seconds': preprocessing, 'results': results,
        'scope': 'fixed-batch forward/backward/clip/H2D core; excludes repeated image preprocessing, optimizer step and checkpoint IO; no convergence equivalence claim'})


def worker(role):
    import torch
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import evaluate
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(65*2**30/torch.cuda.get_device_properties(0).total_memory)
    config = read(PROJECT_ROOT/'configs/vlm_human_sft_baseline_v1.json')
    processor = processor_for(Path(config['model_root']), read(config['reference_config']), 1024)
    manifest = read(config['human_manifest']); plan = read(OUT/'PLAN.json')
    byid = {row['question_id']: row for row in manifest['splits']['train_all']}
    fit = [byid[qid] for qid in plan['fit_sample_question_ids']]
    if role == 'candidate':
        model = load_actor(config['initial_adapter'])
        prefix_scores(model, processor, manifest['splits']['dev'], manifest, 'initial_val')
        prefix_scores(model, processor, fit, manifest, 'initial_trainfit')
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_actor(plan['early_adapters']['human'])
        evaluate(model, processor, config, 'human', OUT/'early200/human')
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_actor(STUDY/'runs/human/train/adapter')
        prefix_scores(model, processor, manifest['splits']['dev'], manifest, 'candidate_val')
        prefix_scores(model, processor, fit, manifest, 'candidate_trainfit')
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_actor(config['initial_adapter'], training=True)
        benchmark(processor, model)
    else:
        model = load_actor(plan['early_adapters']['control'])
        evaluate(model, processor, config, 'human', OUT/'early200/control')
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_actor(STUDY/'runs/control/train/adapter')
        prefix_scores(model, processor, manifest['splits']['dev'], manifest, 'control_val')
    write_json(OUT/(role.upper()+'_COMPLETE.json'), {'status': 'completed', 'optimizer_updates': 0})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', choices=['candidate', 'control']); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker); return
    prepare()
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file
    jobs = [{'key': role, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', role],
             'completion_marker': str(OUT/(role.upper()+'_COMPLETE.json'))} for role in ('candidate', 'control')]
    run_queue(OUT/'queue', jobs, {'plan_sha256': sha_file(OUT/'PLAN.json'), 'script_sha256': sha_file(Path(__file__)),
              'optimizer_updates': 0, 'retrospective': True})
    write_json(OUT/'GPU_DIAGNOSTICS_COMPLETE.json', {'status': 'completed', 'optimizer_updates': 0})


if __name__ == '__main__':
    main()
