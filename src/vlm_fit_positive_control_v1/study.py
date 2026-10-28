import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_fit_positive_control_v1'
OLD = ROOT.parent / 'vlm_coverage_sft_v1'
ROLES = {'lr2e5': 2e-5, 'lr2e4': 2e-4}
POINTS = (0, 16, 32, 64)
PROTOCOL = PROJECT_ROOT / 'docs/VLM_FIT_POSITIVE_CONTROL_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_FIT_POSITIVE_CONTROL_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/16题拟合正控制_正文概率与自由生成.html'


def read(path):
    return json.loads(Path(path).read_text())


def token_statistics(logits, labels, eos):
    """One record per QA; exclude EOS explicitly from content diagnostics."""
    import torch
    import torch.nn.functional as F
    targets = labels[:, -logits.shape[1]:][:, 1:]
    valid = targets.ne(-100)
    if not torch.equal(valid.sum(1), labels.ne(-100).sum(1)):
        raise ValueError('Lost supervised actions in diagnostic readout')
    values = logits[:, :-1].float()
    records = []
    for index in range(len(labels)):
        ids = targets[index, valid[index]]
        scores = values[index, valid[index]]
        if int(ids[-1]) != eos or int(ids.eq(eos).sum()) != 1 or len(ids) < 2:
            raise ValueError('Require content followed by one actual EOS')
        gold = scores.gather(1, ids[:, None]).squeeze(1)
        tops, choices = scores.topk(2, dim=1)
        rivals = torch.where(choices[:, 0].eq(ids), choices[:, 1], choices[:, 0])
        rival_logits = torch.where(choices[:, 0].eq(ids), tops[:, 1], tops[:, 0])
        logps = F.log_softmax(scores, -1).gather(1, ids[:, None]).squeeze(1)
        records.append({
            'target_ids': ids.tolist(), 'target_logps': logps.tolist(),
            'target_ranks': (scores.gt(gold[:, None]).sum(-1) + 1).tolist(),
            'teacher_argmax_ids': choices[:, 0].tolist(), 'rival_ids': rivals.tolist(),
            'target_minus_best_rival': (gold - rival_logits).tolist(),
            'content_ce': float(-logps[:-1].mean()), 'eos_ce': float(-logps[-1]),
            'all_ce': float(-logps.mean()),
            'all_target_argmax': bool(choices[:, 0].eq(ids).all()),
            'content_target_argmax': bool(choices[:-1, 0].eq(ids[:-1]).all()),
        })
    return records


def previous_review():
    data = read(OLD / 'DATA.json')
    review = {'scope': 'Saved predictions and online per-question loss only;zero model calls', 'roles': {}}
    for role in ('repeat32', 'coverage128'):
        groups = defaultdict(list)
        for line in (OLD / f'runs/{role}/train/INPUT_TRACES.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['source'] == 'hard':
                groups[row['question_id']].append(row)
        review['roles'][role] = {
            'questions': len(groups), 'first_online_mean_ce': statistics.mean(v[0]['chosen_ce'] for v in groups.values()),
            'last_online_mean_ce': statistics.mean(v[-1]['chosen_ce'] for v in groups.values()),
            'last_ce_below_point2': sum(v[-1]['chosen_ce'] < .2 for v in groups.values()),
            'last_ce_above1': sum(v[-1]['chosen_ce'] > 1 for v in groups.values()),
        }
    review['old_analysis'] = read(OLD / 'ANALYSIS.json')
    review['online_loss_caveat'] = 'First/last appearances occur at different checkpoints and other-row padding contexts;not fixed-model paired CE.'
    write_json(ROOT / 'PREVIOUS_REVIEW.json', review)
    return data


def prepare():
    import torch
    from vlm_grpo_iteration_v1.common import chosen_sample
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Frozen study already exists')
    data = previous_review()
    cells = defaultdict(list)
    for row in data['hard_rows']:
        cells[(row['review_error_type'], row['frequency_slice'], row['stratum'])].append(row)
    rows = [min(cells[key], key=lambda q: hashlib.sha256(('fit_positive_control_v1:' + q['question_id']).encode()).hexdigest()) for key in sorted(cells)]
    if len(rows) != 16 or len({q['image_id'] for q in rows}) != 16:
        raise ValueError('Need one of each16 occupied strata')
    selected = {'rows': rows, 'retention': data['evaluations']['retention'], 'images': data['images']}
    write_json(ROOT / 'DATA.json', selected)
    old = read(OLD / 'PLAN.json')
    processor = processor_for(Path(old['model_root']), read(old['reference_config']), 1024)
    cpu = {}
    for row in rows:
        item = chosen_sample(processor, row, data['images'])
        cpu[row['question_id']] = {'input_ids': content_sha(item['input_ids'].tolist()), 'labels': content_sha(item['labels'].tolist())}
    write_json(ROOT / 'CPU_INPUTS.json', cpu)
    schedule = [[q['question_id'] for q in rows[(step % 4) * 4:(step % 4 + 1) * 4]] for step in range(64)]
    write_json(ROOT / 'SCHEDULE.json', schedule)
    paths = [Path(__file__), Path(__file__).parent / '__init__.py', PROTOCOL,
             PROJECT_ROOT / 'tools/run_fit_positive_control_v1.py', PROJECT_ROOT / 'tests/test_vlm_fit_positive_control_v1.py',
             ROOT / 'DATA.json', ROOT / 'CPU_INPUTS.json', ROOT / 'SCHEDULE.json', ROOT / 'PREVIOUS_REVIEW.json', OLD / 'PLAN.json']
    # The previous frozen manifest already binds its dependency sources and initial weights.
    files = {**old['files'], **{str(p): sha_file(p) for p in paths}}
    write_json(ROOT / 'PLAN.json', {'status': 'prepared', 'roles': ROLES, 'updates': 64, 'batch_size': 4,
               'unique_training': 16, 'exposures_per_question': 16, 'probe_points': list(POINTS),
               'generation_qa_per_role': 192, 'teacher_diagnostic_qa_per_role': 64,
               'files': files, 'initial_lora_sha256': old['lora_sha256'], 'packages': old['packages'],
               'base_weight_files': old['base_weight_files'], 'automatic_performance_study': False})
    RESULT.write_text('# 16题拟合正控制\n\n已准备，同16题/64更新/每题16次，两学习率2e−5与2e−4。尚无新结果；训练拟合不等于能力泛化。\n')
    render()
    print('PREPARED', {'questions': len(rows), 'strata': len(cells), 'generations_per_role': 192}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen source/input changed: ' + path)
    if package_lock() != plan['packages']:
        raise ValueError('Package versions changed')
    for path, value in plan['base_weight_files'].items():
        st = Path(path).stat()
        if value != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
            raise ValueError('Base storage changed')
    return plan


def generated(model, processor, rows, data):
    import torch
    from vlm_deep.common import encode, collate
    from vlm_sft.common import to_device
    from vlm_probe.evaluate import inference_row
    from vlm_deep.evaluate import score_prediction
    from vlm_sampled_signal_v1.trajectory import stop_at_eos
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    answers = []
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            chunk = rows[start:start + 4]
            items = [encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False) for q in chunk]
            batch = to_device(collate(items, processor.tokenizer.pad_token_id), next(model.parameters()).device)
            output = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True)
            for index, (q, item) in enumerate(zip(chunk, items)):
                ids, terminated = stop_at_eos(output[index, batch['input_ids'].shape[1]:].tolist(), model.generation_config.eos_token_id, 32)
                prediction = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                if item['input_ids'].shape[1] != q['expected_prefix_tokens'] or item['image_grid_thw'][0].tolist() != q['expected_grid']:
                    raise ValueError('Inference frame differs from audited data')
                answers.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                                'prediction': prediction, 'completion_ids': ids, 'terminated': terminated,
                                'input_sha256': content_sha(item['input_ids'].tolist()),
                                'visual_tokens': int(item['image_grid_thw'].prod(-1).sum()) // 4,
                                'vqa_score': score_prediction(prediction, q, 'textvqa')})
            del output, batch
    return answers


def probe(model, processor, data, cpu, role, step):
    import torch
    from vlm_grpo_iteration_v1.common import chosen_batch
    from vlm_sampled_signal_v1.trajectory import readout_tail_length
    from vlm_sft.common import to_device
    directory = ROOT / 'runs' / role / 'probes' / f'step{step:06d}'
    model.eval(); model.config.use_cache = True
    teacher = []
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    with torch.inference_mode():
        for start in range(0, 16, 4):
            chunk = data['rows'][start:start + 4]
            batch = to_device(chosen_batch(processor, chunk, data['images'], cpu), next(model.parameters()).device)
            keep = readout_tail_length(batch['labels'])
            handle = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
            try:
                output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
            finally:
                handle.remove()
            stats = token_statistics(output.logits, batch['labels'], eos)
            for row, record in zip(chunk, stats):
                teacher.append({'question_id': row['question_id'], **record,
                                'target_pieces': [processor.tokenizer.decode([x]) for x in record['target_ids']],
                                'rival_pieces': [processor.tokenizer.decode([x]) for x in record['rival_ids']]})
            del batch, output
    write_json(directory / 'TEACHER.json', teacher)
    preds = generated(model, processor, data['rows'], data)
    write_json(directory / 'PREDICTIONS.json', preds)
    if step in (0, 64):
        write_json(directory / 'RETENTION.json', generated(model, processor, data['retention'], data))
    result = {'status': 'completed', 'step': step, 'full': sum(q['vqa_score'] >= .999999 for q in preds),
              'vqa_percent': 100 * statistics.mean(q['vqa_score'] for q in preds),
              'mean_content_ce': statistics.mean(q['content_ce'] for q in teacher),
              'mean_eos_ce': statistics.mean(q['eos_ce'] for q in teacher),
              'teacher_all_target_argmax': sum(q['all_target_argmax'] for q in teacher),
              'teacher_content_target_argmax': sum(q['content_target_argmax'] for q in teacher)}
    write_json(directory / 'COMPLETE.json', result)
    return result


def run_role(role):
    import numpy as np
    import torch
    from vlm_grpo_iteration_v1.common import actor, chosen_batch
    from vlm_sft.common import to_device
    from vlm_sampled_signal_v1.trajectory import sequence_logps
    from vlm_object_train.train import learning_rate, save_checkpoint, resume_checkpoint, lock_json
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); data = read(ROOT / 'DATA.json'); cpu = read(ROOT / 'CPU_INPUTS.json'); schedule = read(ROOT / 'SCHEDULE.json')
    directory = ROOT / 'runs' / role; directory.mkdir(parents=True, exist_ok=True)
    lineage = {'plan_sha256': sha_file(ROOT / 'PLAN.json'), 'schedule_sha256': sha_file(ROOT / 'SCHEDULE.json'),
               'role': role, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory / 'INPUT_LOCK.json', lineage)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, processor, binding = actor(training=True)
    if lora_hash(model) != plan['initial_lora_sha256']:
        raise ValueError('Initial adapter differs')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 1843200 or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Unexpected trainables')
    optimizer = torch.optim.AdamW(parameters, lr=ROLES[role], weight_decay=0.)
    stats = {'training_active_seconds': 0., 'probe_seconds': 0., 'metrics_bytes': 0, 'exposures': 0}
    completed = 0
    if (directory / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, content_sha(lineage), lineage['schedule_sha256'], 64)
    session = time.monotonic()
    lookup = {q['question_id']: q for q in data['rows']}
    metric_path = directory / 'STEP_METRICS.jsonl'; metric_path.touch(exist_ok=True)
    def bounded_probe(step):
        start = time.monotonic(); value = probe(model, processor, data, cpu, role, step)
        stats['probe_seconds'] += time.monotonic() - start
        if stats['probe_seconds'] > 600:
            raise TimeoutError('600sec cumulative probe budget')
        print('PROBE', role, value, flush=True)
        return value
    if not (directory / 'probes/step000000/COMPLETE.json').exists():
        if completed:
            raise ValueError('Resumed run lacks initial probe')
        bounded_probe(0)
    with metric_path.open('r+b') as stream:
        if metric_path.stat().st_size < stats['metrics_bytes']:
            raise ValueError('Checkpoint metrics missing')
        stream.truncate(stats['metrics_bytes']); stream.seek(0, os.SEEK_END)
        def checkpoint(step):
            stream.flush(); os.fsync(stream.fileno()); stats['metrics_bytes'] = stream.tell()
            save_checkpoint(directory, model, optimizer, step, stats, content_sha(lineage), lineage['schedule_sha256'])
        for step in range(completed, 64):
            if stats['training_active_seconds'] >= 900:
                checkpoint(step); raise TimeoutError('900sec cumulative training budget')
            start = time.monotonic()
            model.train(); model.get_base_model().visual.eval(); model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, 64, {'lr': ROLES[role], 'warmup_ratio': .125})
            for group in optimizer.param_groups:
                group['lr'] = lr
            chunk = [lookup[qid] for qid in schedule[step]]
            batch = to_device(chosen_batch(processor, chunk, data['images'], cpu), next(model.parameters()).device)
            logps, counts = sequence_logps(model, batch); ces = -logps / counts
            if not bool(torch.isfinite(ces).all()):
                raise FloatingPointError('Nonfinite CE')
            ces.mean().backward()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            optimizer.step(); torch.cuda.synchronize()
            seconds = time.monotonic() - start
            stats['training_active_seconds'] += seconds; stats['exposures'] += 4
            record = {'step': step + 1, 'question_ids': schedule[step], 'chosen_ce': ces.detach().cpu().tolist(),
                      'gradient_norm': norm, 'lr': lr, 'step_seconds': seconds,
                      'training_active_seconds': stats['training_active_seconds'],
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            stream.write((json.dumps(record) + '\n').encode()); stream.flush()
            del batch, logps, counts, ces
            if record['peak_allocated_gib'] > 65:
                checkpoint(step + 1); raise RuntimeError('65GiB allocation budget exceeded')
            if step + 1 in POINTS:
                bounded_probe(step + 1)
            if step == 0 or step + 1 in POINTS:
                checkpoint(step + 1)
            write_json(directory / 'PROGRESS.json', {'status': 'training', 'completed_updates': step + 1,
                       'planned_updates': 64, **record, 'probe_seconds': stats['probe_seconds']})
    if stats['exposures'] != 256:
        raise ValueError('Exposure budget differs')
    model.save_pretrained(directory / 'adapter', safe_serialization=True)
    result = {'status': 'completed', 'role': role, 'completed_updates': 64, 'lora_sha256': lora_hash(model),
              'session_seconds_this_attempt': time.monotonic() - session, **stats}
    write_json(directory / 'COMPLETE.json', result)
    write_json(directory / 'PROGRESS.json', result)


def analyze():
    roles = {}
    for role in ROLES:
        probes = [read(ROOT / 'runs' / role / 'probes' / f'step{x:06d}' / 'COMPLETE.json') for x in POINTS]
        initial, final = probes[0], probes[-1]
        drop = 1 - final['mean_content_ce'] / initial['mean_content_ce']
        retention = {str(step): 100 * statistics.mean(q['vqa_score'] for q in read(ROOT / 'runs' / role / 'probes' / f'step{step:06d}' / 'RETENTION.json')) for step in (0, 64)}
        roles[role] = {'probes': probes, 'content_ce_relative_drop': drop, 'retention_percent': retention,
                       'fit_passed': final['full'] >= 13 and final['full'] - initial['full'] >= 10 and drop >= .5,
                       'cost': read(ROOT / 'runs' / role / 'COMPLETE.json')}
    rescued = (roles['lr2e4']['fit_passed'] and not roles['lr2e5']['fit_passed']
               and roles['lr2e4']['probes'][-1]['full'] - roles['lr2e5']['probes'][-1]['full'] >= 4)
    result = {'status': 'completed', 'scope': 'Exploratory train-set fit diagnostic;not generalization',
              'roles': roles, 'supports_lr_scale_bottleneck_on_these16': rescued, 'goal_complete': False,
              'next': 'No automatic further training/main/CV;use fixed-token evidence to preregister next performance study.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    table = '|组|初始→Final满分/16|正文CE初始→Final|拟合门槛|保持初始→Final%|\n|---|---:|---:|---|---:|\n'
    for role, v in roles.items():
        a, b = v['probes'][0], v['probes'][-1]
        table += f'|{role}|{a["full"]}→{b["full"]}|{a["mean_content_ce"]:.6f}→{b["mean_content_ce"]:.6f}|{v["fit_passed"]}|{v["retention_percent"]["0"]:.6f}→{v["retention_percent"]["64"]:.6f}|\n'
    RESULT.write_text('# 16题拟合正控制完成\n\n' + table + '\n支持本16题学习率尺度瓶颈=' + str(rescued) + '。仅训练拟合，不是验证提分；旧负结果和原门槛不改。\n\n[协议](VLM_FIT_POSITIVE_CONTROL_V1_PROTOCOL.md) · [原图/token/曲线](learning/16题拟合正控制_正文概率与自由生成.html)\n')
    render(result)
    write_json(ROOT / 'COMPLETION.json', result)
    return result


def render(result=None):
    data = read(ROOT / 'DATA.json')
    relative = lambda path: quote(os.path.relpath(path, PAGE.parent), safe='/._-')
    cards = []
    for row in data['rows']:
        content = ''
        for role in ROLES:
            for step in POINTS:
                folder = ROOT / 'runs' / role / 'probes' / f'step{step:06d}'
                if (folder / 'COMPLETE.json').exists():
                    pred = next(q for q in read(folder / 'PREDICTIONS.json') if q['question_id'] == row['question_id'])
                    teacher = next(q for q in read(folder / 'TEACHER.json') if q['question_id'] == row['question_id'])
                    content += '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in [role, step, pred['prediction'], pred['vqa_score'], round(teacher['content_ce'], 4), round(teacher['eos_ce'], 4), teacher['target_ranks'], [round(x, 3) for x in teacher['target_minus_best_rival']]]) + '</tr>'
        cards.append(f'<article><h2>q{row["question_id"]} · {html.escape(row["review_error_type"])}</h2><img loading="lazy" src="{relative(data["images"][row["image_id"]]["path"])}" alt="已审核原始训练图片"><p>{html.escape(row["question"])}</p><p>GT：{html.escape(row["target"])}；原审核：{html.escape(row["visible_evidence"])}</p><div class="scroll"><table><tr><th>组</th><th>step</th><th>真实生成</th><th>VQA</th><th>正文CE</th><th>EOS CE</th><th>GT token rank</th><th>GT−最强竞争logit</th></tr>{content}</table></div></article>')
    charts = ''
    if result:
        for field, title, ymax in [('full', '训练题满分数（不是泛化）', 16), ('mean_content_ce', '固定同帧正文CE', max(q['mean_content_ce'] for r in result['roles'].values() for q in r['probes']) * 1.1)]:
            lines = ''
            for role, color in [('lr2e5', '#2563eb'), ('lr2e4', '#d97706')]:
                coords = [(50 + q['step'] * 7, 200 - q[field] / max(ymax, 1e-12) * 170) for q in result['roles'][role]['probes']]
                lines += f'<polyline points="{" ".join(f"{x},{y}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="3"/>'
                lines += ''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/><text x="{x+4}" y="{y-5}" font-size="11">{q[field]:.2f}</text>' for (x,y),q in zip(coords,result['roles'][role]['probes']))
            charts += f'<h2>{title}</h2><svg viewBox="0 0 560 235" role="img" aria-label="{title}"><path d="M50 20V200H510" fill="none" stroke="#475569"/>{lines}<text x="45" y="225">0</text><text x="155" y="225">16</text><text x="267" y="225">32</text><text x="480" y="225">64 step</text></svg>'
    status = '准备完成，尚无新结果。' if result is None else '诊断完成；支持学习率尺度瓶颈=' + str(result['supports_lr_scale_bottleneck_on_these16']) + '；用户能力提升目标仍未完成。'
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>16题拟合正控制</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:18px;margin:20px 0;border:1px solid #cbd5e1}img{width:100%;max-height:440px;object-fit:contain}table{border-collapse:collapse;font-size:14px}td,th{padding:6px;border:1px solid #cbd5e1}.scroll{overflow:auto}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}svg{width:100%;max-width:650px}</style><h1>先验证能学会：正文概率与自由生成</h1><p>' + status + '</p><p>同16个已审样本，各层固定SHA选1；新同卡初始对照，唯一组间变化为学习率2e−5（蓝）/2e−4（橙）。旧128题负结果不被此诊断替换。</p><pre>CE_content = −Σ正文token log p(y_t|图像,问题,正确前缀) / 正文token数\nmargin_t = logit(正确token) − max(其余token logits)\nmargin&lt;0：正确前缀条件下，目标token仍不是greedy首选。\nEOS CE与正文CE单列；teacher强制前缀和自由生成不是同一条件。\n拟合：Final≥13/16且净增≥10，固定正文CE至少降50%。\n拟合通过仍不证明新图片/新问题泛化，保持64也不是新验证集。</pre>' + f'<p><a href="{relative(PROTOCOL)}">事前协议</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(Path(__file__))}">真实代码</a></p>' + charts + ''.join(cards) + '</html>')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.role:
        run_role(args.role)
    else:
        from research_runtime.queue import run_queue
        verify()
        run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_fit_positive_control_v1.study', '--role', role],
                   'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        print('ANALYSIS', analyze()['supports_lr_scale_bottleneck_on_these16'], flush=True)


if __name__ == '__main__':
    main()
