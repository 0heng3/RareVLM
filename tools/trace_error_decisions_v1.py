"""Read-only production loss and actual greedy-token decisions on the reviewed32."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, write_json
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_error_decision_trace_v1'
OLD = ROOT.parent / 'vlm_error_supervision_sft_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_error_supervision_sft_v1.json'
ROLES = ('initial', 'sft', 'dpo_sft')

def prepare():
    import torch
    from vlm_probe.evaluate import inference_row, processor_for
    from vlm_object_train.train import learning_rate
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already prepared; inspect existing task')
    c = read(CONFIG)
    base = read(c['baseline_config'])
    data = read(OLD / 'data/MANIFEST.json')
    rows = data['hard_eval']
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    first = {}
    with (OLD / 'runs/sft/train/INPUT_TRACES.jsonl').open() as stream:
        for line in stream:
            import json
            q = json.loads(line)
            if q['source'] == 'preference' and q['question_id'] not in first:
                first[q['question_id']] = q
    oldpred = {q['question_id']: q for q in read(OLD / 'baseline/hard/PREDICTIONS.json')}
    review = []
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    for q in rows:
        trace = first[q['question_id']]
        labels = trace['labels']
        start = next(i for i, value in enumerate(labels) if value != -100)
        sample = encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False)
        if trace['input_ids'][:start] != sample['input_ids'][0].tolist():
            raise ValueError('Actual supervised prefix differs from inference')
        answer = [v for v in labels if v != -100]
        end = answer.index(eos)
        review.append({'question_id': q['question_id'], 'target': q['target'], 'label_ids': answer,
            'label_tokens': processor.tokenizer.convert_ids_to_tokens(answer),
            'eos_index': end, 'post_eos_tokens': answer[end + 1:],
            'post_eos_denominator_fraction': (len(answer) - end - 1) / len(answer),
            'initial_prediction': oldpred[q['question_id']]['prediction'], 'registered_rejected': q['rejected_target'],
            'rejected_matches_initial': q['rejected_target'].casefold() == oldpred[q['question_id']]['prediction'].casefold()})
    schedule = read(OLD / 'data/SCHEDULE.json')['updates']
    micros = [{'key': f'{step+1}:{micro}', 'ids': ids} for step, update in enumerate(schedule[:4])
              for micro, ids in enumerate(update['micros']) if micro in (1, 3)]
    if Counter(q for batch in micros for q in batch['ids']) != Counter(q['question_id'] for q in rows):
        raise ValueError('Need one actual first-exposure batch for each32')
    write_json(ROOT / 'DATA.json', {'rows': rows, 'images': {q['image_id']: data['images'][q['image_id']] for q in rows},
                                   'micros': micros, 'supervision': review})
    write_json(ROOT / 'CPU_FINDINGS.json', {'status': 'completed', 'same_actual_train_inference_prefixes': len(review),
        'post_eos_supervised_cases': sum(bool(q['post_eos_tokens']) for q in review),
        'mean_post_eos_denominator_fraction': sum(q['post_eos_denominator_fraction'] for q in review) / 32,
        'registered_rejected_matches_initial': sum(q['rejected_matches_initial'] for q in review),
        'configured_lr_sequence': [learning_rate(i, 32, c) for i in range(32)],
        'configured_cumulative_lr': sum(learning_rate(i, 32, c) for i in range(32)),
        'caveat': 'Token counts and configured step sizes are not gradient/Adam contributions or cause of capability failure.'})
    roles = {'initial': {'adapter': c['initial_adapter'], 'lora_sha256': read(OLD / 'baseline/COMPLETE.json')['lora_sha256']}}
    for role in ROLES[1:]:
        result = read(OLD / 'runs' / role / 'train/TRAIN_COMPLETE.json')
        roles[role] = {'adapter': result['adapter'], 'lora_sha256': result['final_lora_sha256']}
    files = [Path(__file__).resolve(), CONFIG, Path(c['baseline_config']), Path(base['reference_config']),
        PROJECT_ROOT / 'docs/VLM_ERROR_DECISION_TRACE_V1_PROTOCOL.md', ROOT / 'DATA.json',
        ROOT / 'CPU_FINDINGS.json', OLD / 'data/MANIFEST.json', OLD / 'data/CPU_INPUTS.json',
        OLD / 'data/SCHEDULE.json', OLD / 'reference/CACHE.json']
    files += [OLD / ('baseline' if r == 'initial' else 'runs/' + r) / 'hard/PREDICTIONS.json' for r in ROLES]
    files += [Path(v['adapter']) / f for v in roles.values() for f in ('adapter_config.json', 'adapter_model.safetensors')]
    files += [SOURCE_ROOT / f for f in ('vlm_region_preference_probe_v1/common.py', 'vlm_region_ocr_sft_v1/study.py',
        'vlm_error_supervision_sft_v1/train.py', 'vlm_alignment_signal/loss.py', 'vlm_deep/common.py',
        'vlm_sft/common.py', 'vlm_probe/evaluate.py', 'vlm_fp32_readout_v1/model.py',
        'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
        'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'training_updates': 0, 'roles': roles,
        'generation_qa_per_role': 32, 'teacher_sequences_per_role': 64, 'worker_active_cap_seconds': 300,
        'files': {str(f): sha_file(f) for f in files},
        'images': {im['path']: im['sha256'] for im in read(ROOT / 'DATA.json')['images'].values()}})
    print(read(ROOT / 'CPU_FINDINGS.json'), flush=True)

def verify(role=None):
    plan = read(ROOT / 'PLAN.json')
    files = {**plan['files'], **plan['images']}
    if role is not None and role != 'initial':
        files.update(read(ROOT / 'INITIAL_READOUT_LOCK.json'))
    for path, digest in files.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen decision-trace input changed: ' + path)
    return plan

def logit_info(logits, correct, wrong, tokenizer):
    import torch
    logprobs = logits.float().log_softmax(-1)
    top = logits.topk(5).indices.tolist()
    return {'correct_id': int(correct), 'correct_text': tokenizer.decode([correct]),
        'wrong_id': int(wrong), 'wrong_text': tokenizer.decode([wrong]),
        'correct_minus_wrong_logit': float(logits[correct] - logits[wrong]),
        'correct_rank': int((logits > logits[correct]).sum()) + 1,
        'correct_probability': float(logprobs[correct].exp()),
        'wrong_probability': float(logprobs[wrong].exp()),
        'top5': [{'id': k, 'text': tokenizer.decode([k]), 'logit': float(logits[k])} for k in top]}

def run(role):
    import torch
    from vlm_probe.evaluate import inference_row
    from vlm_sft.common import to_device
    from vlm_deep.evaluate import score_prediction
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_error_supervision_sft_v1.common import make_batch
    plan = verify(role)
    model, processor, _ = actor(plan['roles'][role]['adapter'])
    if lora_hash(model) != plan['roles'][role]['lora_sha256']:
        raise ValueError('Wrong existing adapter')
    model.eval()
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    data = read(ROOT / 'DATA.json')
    sup = {q['question_id']: q for q in data['supervision']}
    original = {q['question_id']: q for q in read(OLD / ('baseline' if role == 'initial' else 'runs/' + role) / 'hard/PREDICTIONS.json')}
    device = next(model.parameters()).device
    directory = ROOT / role
    generation = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, 32, 4):
            chunk = data['rows'][start:start + 4]
            samples = [encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False) for q in chunk]
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), device)
            out = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True,
                                 return_dict_in_generate=True, output_scores=True)
            padded = batch['input_ids'].shape[1]
            for i, q in enumerate(chunk):
                ids = out.sequences[i, padded:].tolist()
                end = next((j for j, token in enumerate(ids) if token in eos), len(ids)-1)
                ids = ids[:end+1]
                prediction = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                target = sup[q['question_id']]['label_ids'][:sup[q['question_id']]['eos_index']+1]
                k = next((j for j, (a, b) in enumerate(zip(target, ids)) if a != b), None)
                critical = None
                if k is not None:
                    critical = {'label_index': k, **logit_info(out.scores[k][i], target[k], ids[k], processor.tokenizer)}
                    if int(out.scores[k][i].argmax()) != ids[k]:
                        raise ValueError('Actual greedy ID not the recorded score argmax')
                generation.append({'question_id': q['question_id'], 'target': q['target'], 'prediction': prediction,
                    'vqa_score': score_prediction(prediction, q, 'textvqa'), 'generated_ids': ids,
                    'old_prediction': original[q['question_id']]['prediction'],
                    'replay_matches_saved_text': prediction == original[q['question_id']]['prediction'],
                    'critical': critical, 'eos_ids': eos})
            write_json(directory / 'GENERATION.json', generation)
            del out, batch, samples
    baseline = {q['question_id']: q for q in (generation if role == 'initial' else read(ROOT / 'initial/GENERATION.json'))}
    cpu = read(OLD / 'data/CPU_INPUTS.json')
    refs = read(OLD / 'reference/CACHE.json')['batches']
    lookup = {q['question_id']: q for q in data['rows']}
    teacher = []
    model.config.use_cache = False
    with torch.inference_mode():
        for micro in data['micros']:
            rows = [lookup[q] for q in micro['ids']]
            batch, binding, _, _ = make_batch(processor, data, rows, cpu)
            ref = refs[micro['key']]
            if binding != ref['binding']:
                raise ValueError('Teacher batch differs from actual training/reference')
            batch = to_device(batch, device)
            labels = batch['labels']
            keep = int((labels != -100).sum(1).max()) + 1
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
            try:
                out = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
            finally:
                hook.remove()
            targets = labels[:, -out.logits.shape[1]:][:, 1:]
            logits = out.logits[:, :-1, :].float()
            valid = targets != -100
            safe = targets.masked_fill(~valid, 0)
            logps = logits.log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            counts = valid.sum(1)
            if not bool((counts == (labels != -100).sum(1)).all()):
                raise ValueError('Aligned teacher readout lost tokens')
            totals = (logps * valid).sum(1)
            for i, q in enumerate(rows):
                positions = valid[i].nonzero().flatten().tolist()
                label_ids = targets[i, positions].tolist()
                if label_ids != sup[q['question_id']]['label_ids']:
                    raise ValueError('Actual teacher targets differ from saved training labels')
                stop = sup[q['question_id']]['eos_index']
                tokens = []
                for j, pos in enumerate(positions):
                    token = int(targets[i, pos])
                    tokens.append({'index': j, 'id': token, 'text': processor.tokenizer.decode([token]),
                        'kind': 'content' if j < stop else ('eos' if j == stop else 'post_eos'),
                        'ce': float(-logps[i, pos]),
                        'rank': int((logits[i, pos] > logits[i, pos, token]).sum())+1})
                basecritical = baseline[q['question_id']]['critical']
                decision = None
                if basecritical is not None:
                    j = basecritical['label_index']
                    decision = logit_info(logits[i, positions[j]], label_ids[j], basecritical['wrong_id'], processor.tokenizer)
                    decision['label_index'] = j
                chosen, rejected = float(totals[i]), float(totals[i+4])
                refc, refr = ref['chosen_logps'][i], ref['rejected_logps'][i]
                teacher.append({'question_id': q['question_id'], 'target': q['target'], 'tokens': tokens,
                    'chosen_logp': chosen, 'rejected_logp': rejected, 'mean_full_ce': -chosen / len(tokens),
                    'mean_content_ce': sum(t['ce'] for t in tokens if t['kind']=='content') / stop,
                    'absolute_chosen_minus_rejected_logp': chosen - rejected,
                    'relative_margin_vs_original_reference': (chosen-rejected)-(refc-refr),
                    'initial_reference_chosen_residual': chosen-refc if role=='initial' else None,
                    'initial_reference_rejected_residual': rejected-refr if role=='initial' else None,
                    'teacher_at_initial_critical_prefix': decision})
            write_json(directory / 'TEACHER.json', teacher)
            write_json(directory / 'PROGRESS.json', {'status': 'reading_teacher_and_critical_tokens',
                'generation_qa': len(generation), 'teacher_qa': len(teacher), 'expected_teacher_qa': 32,
                'training_updates': 0, 'active_seconds': time.monotonic()-started})
            del out, batch, labels, logits, targets, logps, totals
            if time.monotonic()-started > plan['worker_active_cap_seconds']:
                raise TimeoutError('Read-only decision-trace cap reached')
    if lora_hash(model) != plan['roles'][role]['lora_sha256']:
        raise ValueError('Decision trace changed adapter')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'training_updates': 0, 'generation_qa': 32,
        'teacher_sequences': 64, 'replay_equal_text_qa': sum(q['replay_matches_saved_text'] for q in generation),
        'adapter_unchanged': True, 'active_seconds': time.monotonic()-started})

def analyze():
    result = {'status': 'completed', 'training_updates': 0, 'cpu': read(ROOT / 'CPU_FINDINGS.json'), 'roles': {}}
    for role in ROLES:
        gen = read(ROOT / role / 'GENERATION.json')
        teacher = {q['question_id']: q for q in read(ROOT / role / 'TEACHER.json')}
        unresolved = [q for q in gen if q['vqa_score'] < .999999]
        active = [q for q in unresolved if q['critical'] is not None]
        ce = lambda kind: sum(sum(t['ce'] for t in q['tokens'] if t['kind']==kind) for q in teacher.values())
        total = sum(ce(kind) for kind in ('content', 'eos', 'post_eos'))
        result['roles'][role] = {'replay_equal_text_qa': sum(q['replay_matches_saved_text'] for q in gen),
            'full_score_qa': 32-len(unresolved), 'unresolved_qa': len(unresolved),
            'actual_greedy_critical_cases': len(active),
            'unresolved_chosen_sequence_still_below_registered_rejected': sum(teacher[q['question_id']]['absolute_chosen_minus_rejected_logp']<0 for q in unresolved),
            'content_ce_sum': ce('content'), 'eos_ce_sum': ce('eos'), 'post_eos_ce_sum': ce('post_eos'),
            'post_eos_fraction_of_ce_sum': ce('post_eos')/total,
            'mean_full_ce': sum(q['mean_full_ce'] for q in teacher.values())/32,
            'mean_content_ce': sum(q['mean_content_ce'] for q in teacher.values())/32}
        if role=='initial':
            result['roles'][role]['max_abs_original_reference_residual'] = max(abs(q[k]) for q in teacher.values()
                for k in ('initial_reference_chosen_residual','initial_reference_rejected_residual'))
    write_json(ROOT / 'COMPLETION.json', result)
    print(result, flush=True)

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
        job = lambda role: {'key': role, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--role', role],
                            'completion_marker': str(ROOT / role / 'COMPLETE.json')}
        run_queue(ROOT / 'initial_queue', [job('initial')], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        f = ROOT / 'initial/GENERATION.json'
        write_json(ROOT / 'INITIAL_READOUT_LOCK.json', {str(f): sha_file(f)})
        run_queue(ROOT / 'final_queue', [job(r) for r in ROLES[1:]], {'initial_sha256': sha_file(f)})
        analyze()

if __name__ == '__main__':
    main()

