"""SFT entry preserving FP32 masters; historical numerical contracts are untouched.

This is a training implementation, not a claim of quality improvement. A study
must separately register its data/objective and use matched independent evaluation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import SOURCE_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def worker(config_path):
    import numpy as np
    import torch
    from vlm_deep.common import (collate, content_sha, encode, read_manifest,
                                 sha_file, to_device, write_json)
    from vlm_object_train.train import (learning_rate, lock_json, resume_checkpoint,
                                        save_checkpoint, tensor_hash, trainable_hash)
    from vlm_probe.evaluate import evaluate_batch, processor_for
    from .model import load
    from .loss import answer_loss

    config = read(config_path)
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'COMPLETE.json').exists():
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    manifest = read_manifest(config['manifest'])
    rows = manifest['splits']['train_all']
    if config.get('validation_only'):
        # Fixed hash selection; no score/loss/test-answer based selection.
        seen, selected = set(), []
        for row in sorted(rows, key=lambda row: content_sha(['fp32_repair_validation_v3', seed, row['question_id']])):
            if row['image_id'] not in seen:
                selected.append(row); seen.add(row['image_id'])
            if len(selected) == config['sample_questions']:
                break
        rows = selected
    elif config.get('sample_questions') or len(rows) != config['expected_questions']:
        raise ValueError('Full training cannot silently subsample the registered corpus')
    updates = []
    for epoch in range(config['epochs']):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        updates.extend(order[start:start + config['effective_batch']]
                       for start in range(0, len(order), config['effective_batch']))
    if not updates:
        raise ValueError('Empty training schedule')
    schedule_sha = content_sha(updates)
    source_paths = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('loss.py'),
                    Path(config_path), Path(config['manifest']), Path(config['reference_config']),
                    Path(config['initial_adapter']) / 'adapter_model.safetensors',
                    SOURCE_ROOT / 'vlm_deep/common.py', SOURCE_ROOT / 'vlm_object_train/train.py',
                    SOURCE_ROOT / 'vlm_probe/evaluate.py']
    lineage = {'files': {str(path): sha_file(path) for path in source_paths},
               'question_ids': [row['question_id'] for row in rows], 'schedule_sha256': schedule_sha,
               'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'numerics': {
                   'base_storage': 'bfloat16', 'adapter_master': 'float32',
                   'global_bf16_autocast': config['global_bf16_autocast'], 'ce': 'float32'}}
    lock_json(root / 'INPUT_LOCK.json', lineage)
    lineage_sha = content_sha(lineage)
    device = torch.device('cuda:0')
    model, binding = load(config['model_root'], config['initial_adapter'], device,
                          training=True, dropout=0., gradient_checkpointing=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != config['expected_trainable']:
        raise ValueError('Trainable parameter count differs')
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    processor = processor_for(Path(config['model_root']), read(config['reference_config']), config['train_budget'])
    image_root = Path(config['manifest']).parent

    def batch_for(chunk):
        encoded = [encode(processor, row, image_root / manifest['images'][row['image_id']]['path'], True)
                   for row in chunk]
        if any(item['input_ids'].shape[1] > config['max_sequence_length'] or
               int(item['image_grid_thw'].prod()) // 4 > config['train_budget'] for item in encoded):
            raise ValueError('Input exceeds declared budget; truncation is forbidden')
        return to_device(collate(encoded, processor.tokenizer.pad_token_id), device)

    completed, stats = 0, {'questions': 0, 'loss_sum': 0., 'active_seconds': 0.}
    if (root / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(root, model, optimizer, lineage_sha, schedule_sha, len(updates))
    start, previous_seconds = time.monotonic(), stats['active_seconds']
    for step in range(completed, len(updates)):
        model.train(); model.get_base_model().visual.eval()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, len(updates), config)
        indices, update_loss = updates[step], 0.
        for begin in range(0, len(indices), config['microbatch']):
            batch = batch_for([rows[index] for index in indices[begin:begin + config['microbatch']]])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['global_bf16_autocast']):
                losses, _, _ = answer_loss(model, batch)
                if losses.dtype != torch.float32 or not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError('Expected finite FP32 CE')
                loss = losses.sum() / len(indices)
            loss.backward(); update_loss += float(losses.detach().sum())
            del batch, losses, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError('Zero training gradient')
        optimizer.step()
        stats['questions'] += len(indices); stats['loss_sum'] += update_loss
        stats['active_seconds'] = previous_seconds + time.monotonic() - start
        progress = {'status': 'training', 'completed_updates': step + 1, 'planned_updates': len(updates),
                    'gradient_norm': float(norm), 'ce_dtype': 'float32', **stats}
        write_json(root / 'PROGRESS.json', progress)
        print(json.dumps(progress), flush=True)
        if (step + 1) % config['checkpoint_every_updates'] == 0 or step + 1 == len(updates):
            save_checkpoint(root, model, optimizer, step + 1, stats, lineage_sha, schedule_sha)
        if config.get('validation_only') and step == 0:
            # Exercise the real saved optimizer/FP32 tensor resume path once.
            before = trainable_hash(model)
            restored, restored_stats = resume_checkpoint(root, model, optimizer, lineage_sha, schedule_sha, len(updates))
            assert restored == step + 1 and restored_stats == stats and trainable_hash(model) == before

    model.save_pretrained(root / 'adapter', safe_serialization=True)
    final_hash = trainable_hash(model)
    check_rows = rows[:config['roundtrip_questions']]
    def snapshot(actor):
        actor.eval(); actor.config.use_cache = True
        with torch.inference_mode():
            losses, _, _ = answer_loss(actor, batch_for(check_rows))
            answers, _ = evaluate_batch(actor, processor, check_rows, manifest['images'], image_root, device,
                config['train_budget'], processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
        return {'ce': losses.cpu().tolist(), 'predictions': [answer['prediction'] for answer in answers]}
    before = snapshot(model)
    del model, optimizer, parameters
    gc.collect(); torch.cuda.empty_cache()
    model, _ = load(config['model_root'], root / 'adapter', device, training=False)
    after = snapshot(model)
    reloaded_hash = tensor_hash((name, parameter) for name, parameter in model.named_parameters() if 'lora_' in name)
    if reloaded_hash != final_hash or before != after:
        raise ValueError('Saving/reloading changed FP32 masters or fixed-profile outputs')
    result = {'status': 'completed', 'validation_only': bool(config.get('validation_only')),
              'updates': len(updates), **stats, 'binding': binding, 'numerics': lineage['numerics'],
              'adapter_sha256': final_hash, 'roundtrip': before, 'roundtrip_exact': True,
              'quality_gain_evaluated': False, 'goal_complete': False}
    # A completed run must not keep reporting "training" in its progress file.
    write_json(root / 'PROGRESS.json', result)
    write_json(root / 'COMPLETE.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.config.resolve())
        return
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file
    config = read(args.config)
    root = Path(config['output_root']).resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('Output must stay in persistent workspace')
    run_queue(root / 'queue', [{'key': 'fp32_sft', 'command': [sys.executable, '-u', '-m',
              'vlm_fp32_adapter.train', '--config', str(args.config.resolve()), '--worker'],
              'completion_marker': str(root / 'COMPLETE.json')}],
              {'config_sha256': sha_file(args.config), 'source_sha256': sha_file(Path(__file__))})


if __name__ == '__main__':
    main()
