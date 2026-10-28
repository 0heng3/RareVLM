"""Fixed-checkpoint sequence-versus-first-divergence diagnosis; no updates."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_visual_preference_v1'
ROOT = BASE / 'vlm_native_divergence_readout_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_DIVERGENCE_READOUT_V1_PROTOCOL.md'
ROLES = ('start', 'ce', 'ce_dpo')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def first_divergence(chosen, rejected):
    for index, (a, b) in enumerate(zip(chosen, rejected)):
        if a != b:
            return index
    return None


def token_stats(logits, labels, pairs):
    import torch
    from vlm_alignment_signal.loss import completion_logps
    sums, counts = completion_logps(logits, labels)
    targets = labels[:, -logits.shape[1]:][:, 1:]
    shifted = logits[:, :-1, :].float()
    result = []
    for i in range(pairs):
        left = (targets[i] != -100).nonzero().flatten()
        right = (targets[i+pairs] != -100).nonzero().flatten()
        chosen = targets[i, left].tolist()
        rejected = targets[i+pairs, right].tolist()
        index = first_divergence(chosen, rejected)
        record = {'chosen_ids': chosen, 'rejected_ids': rejected,
                  'chosen_sum_logp': float(sums[i]), 'rejected_sum_logp': float(sums[i+pairs]),
                  'chosen_nll_per_token': float(-sums[i]/counts[i]),
                  'rejected_nll_per_token': float(-sums[i+pairs]/counts[i+pairs]),
                  'sequence_margin': float(sums[i]-sums[i+pairs]),
                  'first_divergence_index': index}
        if index is not None:
            vector = shifted[i, left[index]]
            a, b = chosen[index], rejected[index]
            logp = vector.log_softmax(-1)
            record.update(first_chosen_id=a, first_rejected_id=b,
                          first_chosen_logp=float(logp[a]), first_rejected_logp=float(logp[b]),
                          first_margin=float(vector[a]-vector[b]),
                          first_chosen_rank=1+int((vector > vector[a]).sum()),
                          first_top1_id=int(vector.argmax()),
                          chosen_shifted_position=int(left[index]))
        result.append(record)
    if not bool(torch.isfinite(sums).all()):
        raise FloatingPointError('Nonfinite sequence scores')
    return result


def self_check():
    import torch
    # Deliberately make the correct sequence win while its first decision loses.
    labels = torch.tensor([[-100, -100, 1, 3, 4], [-100, -100, 2, 0, 4]])
    logits = torch.zeros(2, 5, 5, dtype=torch.float32)
    logits[0, 1, 1], logits[0, 1, 2] = 1., 2.
    logits[0, 2, 3], logits[0, 3, 4] = 6., 6.
    logits[1, 1] = logits[0, 1]
    logits[1, 2, 0], logits[1, 3, 4] = -6., -6.
    q = token_stats(logits, labels, 1)[0]
    assert q['sequence_margin'] > 0 and q['first_margin'] == -1.
    assert q['first_divergence_index'] == 0 and q['first_chosen_rank'] == 2
    # Masked/padding positions must not influence either result.
    other = logits.clone(); other[:, 0, :] = 1234.; other[:, 4, :] = -1234.
    assert token_stats(other, labels, 1) == [q]
    # A shared first answer token moves the comparison to the next position.
    labels2 = labels.clone(); labels2[1, 2] = 1
    q2 = token_stats(logits, labels2, 1)[0]
    assert q2['first_divergence_index'] == 1 and q2['first_chosen_id'] == 3
    # A common additive vocabulary offset leaves the local logP ratio invariant.
    offset = logits.clone(); offset[0, 1, :] += 20.
    assert token_stats(offset, labels, 1)[0]['first_margin'] == q['first_margin']
    assert first_divergence([1, 4], [1, 4]) is None
    assert first_divergence([1, 4], [1, 3, 4]) == 1
    return {'status': 'passed', 'scope': 'New divergence/alignment statistics only',
            'checks': ['sequence_win_local_loss', 'shift_and_vocabulary_axis', 'padding_ignored',
                       'shared_prefix', 'logit_offset_invariance', 'identical_and_eos_sequences']}


def prepare():
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Study already frozen')
    if read(PARENT / 'COMPLETION.json')['status'] != 'completed':
        raise ValueError('Require complete original pilot')
    models = {'start': {'adapter': read(PARENT/'CONFIG.json')['initial_adapter'],
                        'final_lora_sha256': read(PARENT/'STUDY_INPUT_LOCK.json')['initial_lora_sha256']}}
    for role in ROLES[1:]:
        models[role] = read(PARENT/'runs'/role/'train/TRAIN_COMPLETE.json')
    data = read(PARENT/'data/MANIFEST.json')
    rows = {q['question_id']: q for q in data['rows']}
    batches = []
    for update in read(PARENT/'data/SCHEDULE.json')['updates']:
        for ids in update['micros']:
            if rows[ids[0]]['study_source'] == 'preference' and ids not in batches:
                batches.append(ids)
    if len(batches) != 32 or len({q for batch in batches for q in batch}) != 128:
        raise ValueError('Require exact original 128 questions/32 batches')
    c = {'models': models, 'roles': list(ROLES), 'batches': batches,
         'data_manifest': str(PARENT/'data/MANIFEST.json'),
         'cpu_inputs': str(PARENT/'data/CPU_INPUTS.json'),
         'mechanism_minimum_count': 16, 'activity_cap_seconds': 900,
         'new_optimizer_updates': 0, 'new_generations': 0,
         'teacher_forced_sequences': 768, 'automatic_training': False}
    write_json(ROOT/'CPU_CHECK.json', self_check()); write_json(ROOT/'CONFIG.json', c)
    paths = [Path(__file__), PROTOCOL, ROOT/'CPU_CHECK.json', ROOT/'CONFIG.json',
             PARENT/'STUDY_INPUT_LOCK.json', PARENT/'COMPLETION.json',
             PARENT/'data/SCHEDULE.json', Path(c['data_manifest']), Path(c['cpu_inputs'])]
    paths += [PROJECT_ROOT/'src'/n for n in ['vlm_native_scope_v1/model.py',
               'vlm_batch_stable_v1/model.py', 'vlm_alignment_signal/loss.py',
               'vlm_literal_preference_v1/common.py', 'research_runtime/queue.py']]
    for role, m in models.items():
        paths += [Path(m['adapter'])/n for n in ['adapter_config.json', 'adapter_model.safetensors']]
        paths += [PARENT/('baseline/native_hard/PREDICTIONS.json' if role=='start' else f'runs/{role}/internal/native_hard/PREDICTIONS.json')]
    write_json(ROOT/'STUDY_INPUT_LOCK.json', {'status': 'frozen', 'frozen_at': now(),
               'files': {str(p): sha_file(p) for p in paths}, 'optimizer_updates': 0,
               'new_generations': 0, 'source_study_unchanged': True})


def verify():
    for path, digest in read(ROOT/'STUDY_INPUT_LOCK.json')['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen diagnostic input changed: '+path)


def worker(role):
    import torch
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_literal_preference_v1.common import make_batch
    from vlm_sft.common import to_device
    verify(); c = read(ROOT/'CONFIG.json'); started = time.monotonic()
    outdir = ROOT/'runs'/role
    old = read(outdir/'PROGRESS.json') if (outdir/'PROGRESS.json').exists() else {}
    prior = old.get('active_seconds', 0.)
    model, processor, _ = load(c['models'][role]['adapter'], 'visual', False)
    model.eval(); model.requires_grad_(False); before = lora_hash(model)
    if before != c['models'][role]['final_lora_sha256']:
        raise ValueError('Fixed checkpoint identity mismatch')
    data = read(c['data_manifest']); cpu = read(c['cpu_inputs'])
    rows = {q['question_id']: q for q in data['rows']}
    predictions = read(PARENT/('baseline/native_hard/PREDICTIONS.json' if role=='start' else f'runs/{role}/internal/native_hard/PREDICTIONS.json'))
    predictions = {q['question_id']: q for q in predictions}
    results = []; torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, ids in enumerate(c['batches']):
            destination = outdir/'batches'/f'{index:03d}.json'
            if destination.exists():
                saved = read(destination)
                if saved['question_ids'] != ids or saved['lora_sha256'] != before:
                    raise ValueError('Diagnostic resume differs')
                results.extend(saved['rows']); continue
            batch, binding, chosen, rejected = make_batch(processor, data, [rows[q] for q in ids], cpu)
            batch = to_device(batch, next(model.parameters()).device)
            labels = batch['labels']; keep = int((labels != -100).sum(1).max())+1
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda m, args: (args[0][:, -keep:, :],))
            try:
                out = model(**{k:v for k,v in batch.items() if k!='labels'}, use_cache=False)
            finally:
                hook.remove()
            stats = token_stats(out.logits, labels, len(ids))
            for qid, stat, a, b in zip(ids, stats, chosen, rejected):
                if stat['first_divergence_index'] is not None:
                    first_a = int((a['labels'][0] != -100).nonzero()[0])
                    first_b = int((b['labels'][0] != -100).nonzero()[0])
                    k = stat['first_divergence_index']
                    if not torch.equal(a['input_ids'][0, :first_a+k], b['input_ids'][0, :first_b+k]):
                        raise ValueError('Compared tokens do not share an actual prefix')
                    for key in ['first_chosen_id', 'first_rejected_id', 'first_top1_id']:
                        stat[key.replace('_id', '_text')] = processor.tokenizer.decode([stat[key]], skip_special_tokens=False)
                pred = predictions[qid]; q = rows[qid]
                stat.update(question_id=qid, image_id=q['image_id'], question=q['question'], target=q['target'],
                            rejected_target=q['rejected_target'], saved_prediction=pred['prediction'],
                            saved_vqa_score=pred['vqa_score'], saved_completion_ids=pred['completion_ids'],
                            original_negative_persists=pred['completion_ids']==q['rejected_completion_ids'])
            del out, batch
            torch.cuda.synchronize()
            if torch.cuda.max_memory_allocated()/2**30 > 65:
                raise RuntimeError('Diagnostic memory budget exceeded')
            write_json(destination, {'question_ids': ids, 'lora_sha256': before,
                       'binding': binding, 'rows': stats, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']})
            results.extend(stats)
            elapsed = prior+time.monotonic()-started
            write_json(outdir/'PROGRESS.json', {'status':'diagnostic','role':role,
                       'completed_questions':len(results),'planned_questions':128,
                       'teacher_forced_sequences':len(results)*2,'optimizer_updates':0,
                       'new_generations':0,'active_seconds':elapsed,
                       'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
            if elapsed > c['activity_cap_seconds']:
                raise RuntimeError('Diagnostic activity budget exceeded')
    if len(results) != 128 or lora_hash(model) != before:
        raise ValueError('Diagnostic count/checkpoint changed')
    write_json(outdir/'ROWS.json',results)
    done={'status':'completed','role':role,'questions':128,'teacher_forced_sequences':256,
          'new_generations':0,'optimizer_updates':0,'lora_unchanged':True,
          'lora_sha256':before,'active_seconds':prior+time.monotonic()-started,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'rows_sha256':sha_file(outdir/'ROWS.json')}
    write_json(outdir/'COMPLETE.json',done); write_json(outdir/'PROGRESS.json',done)


def analyze():
    verify(); records={r:read(ROOT/'runs'/r/'ROWS.json') for r in ROLES}
    summaries={}
    for role, rows in records.items():
        residual=[q for q in rows if q['saved_vqa_score']<1-1e-9]
        persistent=[q for q in residual if q['original_negative_persists']]
        mismatch=[q for q in persistent if q['sequence_margin']>0 and q.get('first_margin',0)<0]
        summaries[role]={'questions':128,'saved_full_score':128-len(residual),'residual':len(residual),
                         'persistent_negative_residual':len(persistent),
                         'sequence_positive':sum(q['sequence_margin']>0 for q in rows),
                         'first_divergence_positive':sum(q.get('first_margin',0)>0 for q in rows),
                         'sequence_win_local_loss_persistent':len(mismatch),
                         'mismatch_question_ids':[q['question_id'] for q in mismatch],
                         'residual_sequence_positive':sum(q['sequence_margin']>0 for q in residual),
                         'residual_first_positive':sum(q.get('first_margin',0)>0 for q in residual),
                         'missing_comparable_divergence':sum(q['first_divergence_index'] is None for q in rows)}
    count=summaries['ce_dpo']['sequence_win_local_loss_persistent']
    supported=count>=read(ROOT/'CONFIG.json')['mechanism_minimum_count']
    result={'status':'completed','completed_at':now(),'summaries':summaries,
            'mechanism_signal':supported,'phase':'local_decision_mismatch_supported' if supported else 'local_decision_mismatch_insufficient',
            'optimizer_updates':0,'new_generations':0,'teacher_forced_sequences':768,
            'automatic_training':False,'goal_complete':False,
            'scope':'Fixed checkpoints/train-only teacher-forced mechanism;not actual decoder trace, unique cause or task improvement.'}
    write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-run',action='store_true')
    parser.add_argument('--self-check',action='store_true');parser.add_argument('--worker',choices=ROLES)
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check()));return
    if args.worker:worker(args.worker);return
    if args.prepare_run and not (ROOT/'STUDY_INPUT_LOCK.json').exists():prepare()
    verify()
    from research_runtime.queue import run_queue
    jobs=[{'key':r,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',r],
           'completion_marker':str(ROOT/'runs'/r/'COMPLETE.json')} for r in ROLES]
    run_queue(ROOT/'queue',jobs,{'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')})
    analyze();print(json.dumps(read(ROOT/'COMPLETION.json'),ensure_ascii=False))


if __name__=='__main__':main()
