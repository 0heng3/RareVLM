"""Matched plain CE with q/v or q/k/v/o trainable attention scope.

The study owns data, the fixed schedule, GPU admission, and evaluation. This
module performs only the registered actor updates: no reference model, discarded
preflight update, extra generation, or checkpoint-selection evaluation.
"""

import gc
import json
import math
import os
from pathlib import Path
import random
import time


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_schedule(updates, rows, config):
    """Validate only the new schedule-to-loss boundary, before loading the GPU."""
    if config['effective_batch'] != 16 or config['microbatch'] != 4 or not updates:
        raise ValueError('This study requires nonempty 16-QA updates and 4-QA micros')
    for update in updates:
        micros = update['micros']
        if len(micros) != 4:
            raise ValueError('Each update must contain exactly four microbatches')
        source_counts = {'text': 0, 'human': 0}
        pair_count = 0
        for micro in micros:
            source = micro['source']
            qids, pairs, pair_ids = micro['question_ids'], micro['pair_indices'], micro['pair_ids']
            if source not in source_counts or len(qids) != 4 or any(qid not in rows for qid in qids):
                raise ValueError('Schedule references an invalid source, microbatch, or training QA')
            source_counts[source] += len(qids)
            if source == 'text':
                if pairs or pair_ids:
                    raise ValueError('Text replay must have no complementary-pair loss')
                continue
            if len(pairs) != 2 or len(pair_ids) != 2:
                raise ValueError('A human microbatch must contain two complete image pairs')
            if (any(len(pair) != 2 or any(type(index) is not int for index in pair) for pair in pairs)
                    or sorted(index for pair in pairs for index in pair) != list(range(4))):
                raise ValueError('Human pair indices must cover each microbatch row exactly once')
            for positive, negative in pairs:
                yes, no = rows[qids[positive]], rows[qids[negative]]
                if yes['target'] != 'yes' or no['target'] != 'no':
                    raise ValueError('Complementary pairs must be ordered yes, then no')
                if (yes['question'] != no['question']
                        or yes.get('prompt_suffix') != no.get('prompt_suffix')
                        or yes['image_id'] == no['image_id']):
                    raise ValueError('A pair requires the exact same question/prompt and two different images')
            pair_count += len(pairs)
        if source_counts != {'text': 8, 'human': 8} or pair_count != 4:
            raise ValueError('Each update must match 8 Text QA and 4 complementary image pairs')


def train(config, role, directory):
    """Run/resume one short arm and return its ``TRAIN_COMPLETE.json`` record.

    The caller must already own the shared GPU UUID lock. ``pair_indices`` are
    microbatch-local, positive/negative row indices from the predeclared schedule.
    Checkpoint statistics contain both cumulative active time and trace bytes, so
    a restart cannot reset the training budget or keep rolled-back trace rows.
    """
    import numpy as np
    import torch

    from vlm_deep.common import (collate, content_sha, encode, read_manifest,
                                 sha_file, to_device, write_json)
    from vlm_runtime_metadata_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import (learning_rate, lock_json, resume_checkpoint,
                                        save_checkpoint)
    from vlm_probe.evaluate import processor_for

    from vlm_complementary_rank.loss import forward_pair_loss
    from .model import apply_scope

    directory = Path(directory).resolve()
    study_root = Path(config['output_root']).resolve()
    private_root = Path('/opt/data/private')
    if not directory.is_relative_to(private_root) or not study_root.is_relative_to(private_root):
        raise ValueError('Training output must stay in the persistent private workspace')
    directory.mkdir(parents=True, exist_ok=True)
    lambda_rank = float(config['lambda_by_role'][role])
    cap = float(config['training_cap_seconds'])
    if not math.isfinite(lambda_rank) or lambda_rank < 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Ranking weight and cumulative training-time cap must be finite and valid')
    reference = {**_read(config['baseline_config']), 'initial_adapter': config['initial_adapter']}
    manifest = read_manifest(config['manifest'])
    rows = {row['question_id']: row for row in manifest['splits']['train_all']}
    updates = _read(config['schedule_path'])['updates']
    _validate_schedule(updates, rows, config)
    if any(not Path(manifest['images'][row['image_id']]['path']).is_absolute() for row in rows.values()):
        raise ValueError('Training manifest image paths must be absolute')
    schedule_sha = sha_file(config['schedule_path'])
    study_lock = study_root / 'STUDY_INPUT_LOCK.json'
    lineage = {
        'study_input_lock_sha256': sha_file(study_lock),
        'schedule_sha256': schedule_sha,
        'manifest_sha256': sha_file(config['manifest']),
        'baseline_config_sha256': sha_file(config['baseline_config']),
        'config_sha256': content_sha(config),
        'train_source_sha256': sha_file(Path(__file__)),
        'loss_source_sha256': sha_file(Path(__file__).resolve().parents[1] / 'vlm_complementary_rank/loss.py'),
        'role': role, 'lambda_rank': lambda_rank, 'attention_scope': role,
        'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
        'planned_updates': len(updates), 'seed': config['seed'],
        'numerics': {'base_storage': 'bfloat16', 'adapter_master': 'float32',
                     'global_autocast': False, 'ce': 'float32', 'dropout': 0.,
                     'tf32': False, 'gradient_checkpointing': True},
    }
    lock_json(directory / 'INPUT_LOCK.json', lineage)
    lineage_sha = content_sha(lineage)
    complete_path = directory / 'TRAIN_COMPLETE.json'
    if complete_path.exists():
        result = _read(complete_path)
        if result['lineage_sha256'] != lineage_sha or result['updates'] != len(updates):
            raise ValueError('Completed training belongs to a different registered arm')
        return result

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0')
    model, binding = load(reference['model_root'], reference['initial_adapter'], device,
                          training=True, dropout=0., gradient_checkpointing=True)
    scope_binding = apply_scope(model, role)
    from vlm_object_train.train import tensor_hash
    ko_named = lambda: [(n, p) for n, p in model.named_parameters() if 'lora_' in n and ('.k_proj.' in n or '.o_proj.' in n)]
    ko_initial_sha = tensor_hash(ko_named())
    initial_hash = lora_hash(model)
    expected_initial = _read(study_lock).get('initial_lora_sha256')
    if expected_initial is not None and initial_hash != expected_initial:
        raise ValueError('Initial FP32 LoRA differs from the registered study initialization')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if (sum(parameter.numel() for parameter in parameters) != config['expected_trainable_by_role'][role]
            or any(parameter.dtype != torch.float32 for parameter in parameters)):
        raise ValueError('The registered FP32 trainable parameter budget differs')
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    processor = processor_for(Path(reference['model_root']), _read(reference['reference_config']),
                               config['train_budget'])
    token_ids = {answer: processor.tokenizer.encode(answer, add_special_tokens=False)
                 for answer in ('yes', 'no')}
    if (any(len(ids) != 1 for ids in token_ids.values())
            or token_ids['yes'] == token_ids['no']):
        raise ValueError('Canonical yes/no answers must be distinct single local-tokenizer tokens')
    yes_id, no_id = token_ids['yes'][0], token_ids['no'][0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    completed = 0
    stats = {
        'questions': 0, 'pairs': 0, 'loss_sum': 0., 'rank_loss_sum': 0.,
        'objective_sum': 0., 'active_seconds': 0., 'trace_bytes': 0,
        'input_tokens': 0, 'visual_tokens': 0, 'assistant_tokens': 0, 'padding_tokens': 0,
        'source_counts': {'text': 0, 'human': 0},
        'source_loss_sums': {'text': 0., 'human': 0.},
        'source_visual_tokens': {'text': 0, 'human': 0},
        'peak_allocated_gib': 0., 'peak_reserved_gib': 0.,
    }
    if (directory / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha,
                                              schedule_sha, len(updates))
    trace_path = directory / 'INPUT_TRACES.jsonl'
    trace_path.touch(exist_ok=True)
    if trace_path.stat().st_size < stats['trace_bytes']:
        raise ValueError('Actual input trace is shorter than its saved checkpoint boundary')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started, prior_seconds = time.monotonic(), stats['active_seconds']
    durations = []

    with trace_path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes'])
        trace.seek(0, os.SEEK_END)

        def stop_for_budget(completed_updates):
            stats['active_seconds'] = prior_seconds + time.monotonic() - started
            stats['trace_bytes'] = trace.tell()
            trace.flush()
            os.fsync(trace.fileno())
            save_checkpoint(directory, model, optimizer, completed_updates, stats,
                            lineage_sha, schedule_sha)
            stop = {'status': 'stopped_training_time_cap', 'role': role,
                    'completed_updates': completed_updates, 'planned_updates': len(updates),
                    'training_cap_seconds': cap, 'lineage_sha256': lineage_sha,
                    'schedule_sha256': schedule_sha, **stats}
            write_json(directory / 'BUDGET_STOP.json', stop)
            write_json(directory / 'PROGRESS.json', stop)
            print(json.dumps(stop, allow_nan=False), flush=True)
            raise TimeoutError('Cumulative training-time cap reached at a saved optimizer boundary')

        if prior_seconds >= cap:
            stop_for_budget(completed)
        for step in range(completed, len(updates)):
            step_started = time.monotonic()
            micros = updates[step]['micros']
            pair_count = sum(len(micro['pair_indices']) for micro in micros)
            model.train()
            model.get_base_model().visual.eval()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, len(updates), config)
            update_ce, update_rank = 0., 0.
            update_pair_records = []
            for micro_index, micro in enumerate(micros):
                source, qids = micro['source'], micro['question_ids']
                selected = [rows[qid] for qid in qids]
                pairs, pair_ids = micro['pair_indices'], micro['pair_ids']
                pair_by_row = {}
                for pair_id, (positive, negative) in zip(pair_ids, pairs):
                    pair_by_row[positive] = (pair_id, 'yes', qids[negative])
                    pair_by_row[negative] = (pair_id, 'no', qids[positive])
                encoded, evidence = [], []
                for index, row in enumerate(selected):
                    image = manifest['images'][row['image_id']]
                    sample = encode(processor, row, Path(image['path']), True)
                    grid = sample['image_grid_thw']
                    patches = int(grid.prod())
                    length = int(sample['input_ids'].shape[1])
                    targets = sample['labels'][0][sample['labels'][0].ne(-100)]
                    visual = patches // 4
                    counted = int(sample['input_ids'].eq(image_token_id).sum())
                    if (tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1 or patches % 4
                            or visual <= 0 or visual != counted
                            or int(sample['pixel_values'].shape[0]) != patches
                            or length > config['max_sequence_length']
                            or visual > config['train_budget'] or not targets.numel()):
                        raise ValueError('Actual training input violates image/token budget; no truncation allowed')
                    pair_id, pair_role, partner = pair_by_row.get(index, (None, None, None))
                    encoded.append(sample)
                    evidence.append({
                        'update': step + 1, 'microbatch_index': micro_index,
                        'microbatch_question_position': index, 'source': source,
                        'question_id': row['question_id'], 'image_id': row['image_id'],
                        'question_sha256': content_sha(row['question']),
                        'image_sha256': image['sha256'], 'image_path': image['path'],
                        'pair_id': pair_id, 'pair_role': pair_role, 'pair_partner_question_id': partner,
                        'target': row['target'], 'assistant_target_token_ids': targets.tolist(),
                        'sequence_length': length, 'input_tokens': int(sample['attention_mask'].sum()),
                        'image_grid_thw': grid.tolist(), 'pixel_values_shape': list(sample['pixel_values'].shape),
                        'premerge_patch_tokens': patches, 'visual_tokens': visual,
                        'counted_image_tokens': counted, 'assistant_tokens': int(targets.numel()),
                        'truncated': False,
                    })
                cpu_batch = collate(encoded, processor.tokenizer.pad_token_id)
                padded_length = int(cpu_batch['input_ids'].shape[1])
                for index, entry in enumerate(evidence):
                    first = int(cpu_batch['labels'][index].ne(-100).nonzero()[0, 0])
                    entry.update({
                        'padded_sequence_length': padded_length,
                        'left_padding_tokens': padded_length - entry['sequence_length'],
                        'input_ids': cpu_batch['input_ids'][index].tolist(),
                        'labels': cpu_batch['labels'][index].tolist(),
                        'attention_mask': cpu_batch['attention_mask'][index].tolist(),
                        'first_assistant_target_position': first,
                        'first_assistant_prediction_position': first - 1,
                    })
                batch = to_device(cpu_batch, device)
                # Explicitly disable an outer AMP context as well as not enabling
                # one here. Both arms take this identical single-forward path.
                with torch.autocast('cuda', enabled=False):
                    result = forward_pair_loss(model, batch, pairs, yes_id, no_id, tail=True)
                    if result['qa_ce'].dtype != torch.float32 or result['pair_rank'].dtype != torch.float32:
                        raise FloatingPointError('Expected FP32 CE and pair-rank reductions')
                    objective = (result['qa_ce'].sum() / config['effective_batch']
                                 + lambda_rank * result['pair_rank'].sum() / pair_count)
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError('Non-finite registered training objective')
                objective.backward()
                ce_values = result['qa_ce'].detach().cpu().tolist()
                margins = result['first_token_margin'].detach().cpu().tolist()
                first_ids = result['first_target_token_ids'].detach().cpu().tolist()
                rank_values = result['pair_rank'].detach().cpu().tolist()
                rank_by_row = {}
                for pair_id, (positive, negative), rank_value in zip(pair_ids, pairs, rank_values):
                    gap = margins[positive] - margins[negative]
                    pair_record = {'pair_id': pair_id, 'rank_loss': rank_value, 'margin_gap': gap,
                                   'yes_question_id': qids[positive], 'no_question_id': qids[negative]}
                    update_pair_records.append(pair_record)
                    rank_by_row[positive] = rank_by_row[negative] = pair_record
                for index, (entry, ce_value, margin, first_id) in enumerate(
                        zip(evidence, ce_values, margins, first_ids)):
                    pair_record = rank_by_row.get(index)
                    entry.update({'qa_ce': ce_value, 'first_yes_minus_no_margin': margin,
                                  'first_assistant_target_token_id': first_id,
                                  'pair_rank_loss': pair_record['rank_loss'] if pair_record else None,
                                  'pair_margin_gap': pair_record['margin_gap'] if pair_record else None})
                    trace.write((json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
                    stats['source_counts'][source] += 1
                    stats['source_loss_sums'][source] += ce_value
                    stats['source_visual_tokens'][source] += entry['visual_tokens']
                    stats['input_tokens'] += entry['input_tokens']
                    stats['visual_tokens'] += entry['visual_tokens']
                    stats['assistant_tokens'] += entry['assistant_tokens']
                    stats['padding_tokens'] += entry['left_padding_tokens']
                update_ce += sum(ce_values)
                update_rank += sum(rank_values)
                del result, objective, batch, cpu_batch, encoded
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'],
                                                 error_if_nonfinite=True)
            gradient_norm = float(norm)
            if gradient_norm <= 0:
                raise ValueError('The formal training update has zero gradient')
            ko_first_grad = {}
            if step == 0:
                for factor in ('A', 'B'):
                    parts = [p.grad.detach().float().reshape(-1) for n, p in ko_named() if 'lora_' + factor in n and p.grad is not None]
                    ko_first_grad[factor] = float(torch.cat(parts).norm()) if parts else 0.
                if role == 'qkvo' and not ko_first_grad['B'] > 0:
                    raise ValueError('New K/O B factors did not receive first-formal-update gradients')
                if role == 'qv' and any(p.grad is not None for _, p in ko_named()):
                    raise ValueError('Control K/O factors must stay frozen')
            optimizer.step()
            torch.cuda.synchronize(device)
            stats['questions'] += config['effective_batch']
            stats['pairs'] += pair_count
            stats['loss_sum'] += update_ce
            stats['rank_loss_sum'] += update_rank
            stats['objective_sum'] += update_ce / config['effective_batch'] + lambda_rank * update_rank / pair_count
            stats['trace_bytes'] = trace.tell()
            stats['active_seconds'] = prior_seconds + time.monotonic() - started
            stats['peak_allocated_gib'] = max(stats['peak_allocated_gib'],
                                               torch.cuda.max_memory_allocated(device) / 2**30)
            stats['peak_reserved_gib'] = max(stats['peak_reserved_gib'],
                                              torch.cuda.max_memory_reserved(device) / 2**30)
            if stats['active_seconds'] >= cap:
                stop_for_budget(step + 1)
            if (step == 0 or (step + 1) % config['checkpoint_every_updates'] == 0
                    or step + 1 == len(updates)):
                trace.flush()
                os.fsync(trace.fileno())
                save_checkpoint(directory, model, optimizer, step + 1, stats, lineage_sha, schedule_sha)
            durations.append(time.monotonic() - step_started)
            durations = durations[-50:]
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == len(updates):
                recent = sum(durations) / len(durations)
                progress = {
                    'status': 'training', 'role': role, 'completed_updates': step + 1,
                    'planned_updates': len(updates), 'lambda_rank': lambda_rank,
                    'recent_seconds_per_update': recent,
                    'remaining_training_seconds_estimate': recent * (len(updates) - step - 1),
                    'gradient_norm': gradient_norm, 'lr': optimizer.param_groups[0]['lr'],
                    'last_update_qa_ce_mean': update_ce / config['effective_batch'],
                    'last_update_pair_rank_mean': update_rank / pair_count,
                    'lineage_sha256': lineage_sha, 'schedule_sha256': schedule_sha,
                    'numerics': lineage['numerics'], **stats,
                }
                write_json(directory / 'PROGRESS.json', progress)
                print(json.dumps(progress, allow_nan=False), flush=True)
                if step == 0:
                    write_json(directory / 'FIRST_UPDATE.json', {
                        **progress, 'pairs_this_update': update_pair_records,
                        'initial_lora_sha256': initial_hash,
                        'canonical_answer_token_ids': token_ids,
                        'discarded_preflight_updates': 0, 'extra_reference_forwards': 0,
                        'scope': 'first formal update only; no quality conclusion',
                        'attention_scope_inventory': scope_binding, 'new_ko_gradient_norm': ko_first_grad,
                    })
        trace.flush()
        os.fsync(trace.fileno())

    expected_counts = {'text': len(updates) * 8, 'human': len(updates) * 8}
    if (stats['source_counts'] != expected_counts
            or stats['questions'] != len(updates) * config['effective_batch']
            or stats['pairs'] != len(updates) * 4):
        raise ValueError('Completed schedule exposure accounting differs')
    adapter = directory / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    final_hash = lora_hash(model)
    ko_final_sha = tensor_hash(ko_named())
    if role == 'qv' and ko_final_sha != ko_initial_sha:
        raise ValueError('Frozen control K/O weights changed')
    stats['active_seconds'] = prior_seconds + time.monotonic() - started
    result = {
        'status': 'completed', 'role': role, 'updates': len(updates),
        'lambda_rank': lambda_rank, 'adapter': str(adapter),
        'binding': binding, 'attention_scope_inventory': scope_binding,
        'ko_initial_sha256': ko_initial_sha, 'ko_final_sha256': ko_final_sha,
        'initial_lora_sha256': initial_hash, 'final_lora_sha256': final_hash,
        'lineage_sha256': lineage_sha, 'schedule_sha256': schedule_sha,
        'input_trace_sha256': sha_file(trace_path), 'canonical_answer_token_ids': token_ids,
        'numerics': lineage['numerics'], 'training_cap_seconds': cap,
        'quality_gain_evaluated': False, 'extra_reference_forwards': 0,
        'discarded_preflight_updates': 0, **stats,
    }
    write_json(complete_path, result)
    write_json(directory / 'PROGRESS.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    del model, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result
