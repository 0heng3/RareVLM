import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import html
import math
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from .common import ROOT, OLD, MINING, ROLES, read, chosen_sample

PROTOCOL = PROJECT_ROOT / 'docs/VLM_LENGTH_REPLAY_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_LENGTH_REPLAY_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/多词答案保持_扩覆盖与重加权.html'


def order(salt, qid):
    return hashlib.sha256((salt + ':' + qid).encode()).hexdigest()


def cell(q):
    return q['frequency_slice'], q['stratum']


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen')
    if read(OLD / 'COMPLETION.json')['status'] != 'completed':
        raise ValueError('Previous study is incomplete')
    prior = read(OLD / 'PLAN.json'); data = read(OLD / 'DATA.json'); mine = read(MINING / 'DATA.json')
    predictions = {q['question_id']: q for q in read(MINING / 'PREDICTIONS.json')}
    hard = [{**q, 'training_weight': 1.} for q in data['hard_rows']]
    replay = [q for q in data['training_rows'] if q['training_source'] == 'replay']
    existing = {q['question_id'] for q in replay}
    donors = [q for q in mine['rows'] if q['fold'] == 'training' and q['stratum'] == 'multiword'
              and q['question_id'] not in existing and predictions[q['question_id']]['vqa_score'] >= .999999]
    donors.sort(key=lambda q: order('length_replay_v1:donor', q['question_id']))
    if len(donors) < 190 or sum(q['stratum'] == 'multiword' for q in replay) != 50:
        raise ValueError('Declared replay inventory differs')
    added = donors[:190]
    replacement = {}
    for frequency in ('rare_1_5', 'freq_6_20'):
        new = [q for q in added if q['frequency_slice'] == frequency]
        old = sorted([q for q in replay if q['frequency_slice'] == frequency and q['stratum'] == 'singleword'],
                     key=lambda q: order('length_replay_v1:slot', q['question_id']))
        if len(old) < len(new):
            raise ValueError('Insufficient same-frequency singleword slots')
        replacement.update({a['question_id']: b for a, b in zip(old, new)})
    expanded = [{**replacement.get(q['question_id'], q), 'training_source': 'replay', 'training_weight': 1.} for q in replay]
    before = Counter(cell(q) for q in replay); after = Counter(cell(q) for q in expanded)
    weighted = [{**q, 'training_weight': after[cell(q)] / before[cell(q)]} for q in replay]
    by_role = {'weighted50': hard + weighted, 'expanded240': hard + expanded}
    previous_schedule = read(OLD / 'SCHEDULE.json')['roles']['coverage128']
    schedules = {'roles': {}}
    for role in ROLES:
        schedules['roles'][role] = [{**block, 'micros': [[replacement[qid]['question_id'] if role == 'expanded240' and qid in replacement else qid for qid in micro] for micro in block['micros']]} for block in previous_schedule]
    merged = {**data['images'], **mine['images']}
    used = {q['image_id'] for rows in by_role.values() for q in rows} | {q['image_id'] for rows in data['evaluations'].values() for q in rows}
    images = {iid: merged[iid] for iid in used}
    for role, rows in by_role.items():
        lookup = {q['question_id']: q for q in rows}
        counts = Counter(qid for block in schedules['roles'][role] for micro in block['micros'] for qid in micro)
        if len(lookup) != 640 or any(counts[q['question_id']] != (4 if q['training_source'] == 'hard' else 1) for q in rows):
            raise ValueError('Unique/exposure schedule differs')
        train_pixels = {images[q['image_id']].get('pixel_sha256', images[q['image_id']]['sha256']) for q in rows}
        held_pixels = {images[q['image_id']].get('pixel_sha256', images[q['image_id']]['sha256']) for n in ('validation', 'retention') for q in data['evaluations'][n]}
        if train_pixels & held_pixels:
            raise ValueError('Train/heldout pixel overlap')
        if not all(q['training_weight'] > 0 and math.isfinite(q['training_weight']) for q in rows):
            raise ValueError('Invalid weights')
    mass = {c: sum(q['training_weight'] for q in weighted if cell(q) == c) for c in before}
    if any(abs(mass[c] - after[c]) > 1e-9 for c in after) or sum(after.values()) != 512 or sum(v for c, v in after.items() if c[1] == 'multiword') != 240:
        raise ValueError('Aggregate frequency/length supervision mass differs')
    for frequency in ('rare_1_5', 'freq_6_20'):
        if sum(v for c, v in before.items() if c[0] == frequency) != sum(v for c, v in after.items() if c[0] == frequency):
            raise ValueError('Frequency totals changed')
    prepared = {'training_rows': by_role, 'hard_rows': hard, 'images': images, 'evaluations': data['evaluations']}
    write_json(ROOT / 'DATA.json', prepared); write_json(ROOT / 'SCHEDULE.json', schedules)
    cpu = read(OLD / 'CPU_INPUTS.json')
    processor = processor_for(Path(prior['model_root']), read(prior['reference_config']), 1024)
    for q in added:
        item = chosen_sample(processor, q, images)
        cpu[q['question_id']] = {'input_ids': content_sha(item['input_ids'].tolist()), 'labels': content_sha(item['labels'].tolist()),
                                'visual_tokens': int(item['image_grid_thw'].prod(-1).sum()) // 4}
    if len(cpu) != 830:
        raise ValueError('Expected640 reused plus190 new CPU inputs')
    write_json(ROOT / 'CPU_INPUTS.json', cpu)
    for cohort in prepared['evaluations']:
        directory = ROOT / 'initial' / cohort; directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD / 'initial' / cohort / 'PREDICTIONS.json', directory / 'PREDICTIONS.json')
    receipt = {'status': 'passed', 'new_encoded': 190, 'reused_encoded': 640, 'unique_inputs': 830,
               'new_baseline_model_calls': 0, 'new_donors': [q['question_id'] for q in added],
               'replacements': {k: v['question_id'] for k, v in replacement.items()},
               'cells': [{'cell': c, 'original_questions': before[c], 'expanded_questions': after[c], 'weighted50_per_qa_weight': after[c] / before[c], 'weighted50_mass': mass[c]} for c in sorted(before)],
               'actual_multi_exposures': {'weighted50': 78, 'expanded240': 268}, 'multi_supervision_weight_both': 268,
               'per_step_mass_equal': False, 'initial_predictions_reused': True}
    # Frozen generation function is copied verbatim into this namespace.
    old_tree = ast.parse((SOURCE_ROOT / 'vlm_coverage_sft_v2/evaluate.py').read_text())
    new_tree = ast.parse((SOURCE_ROOT / 'vlm_length_replay_v1/evaluate.py').read_text())
    old_eval = next(x for x in old_tree.body if isinstance(x, ast.FunctionDef))
    new_eval = next(x for x in new_tree.body if isinstance(x, ast.FunctionDef))
    if ast.dump(old_eval, include_attributes=False) != ast.dump(new_eval, include_attributes=False):
        raise ValueError('Evaluation function changed')
    receipt['evaluation_function_ast_identical'] = True
    write_json(ROOT / 'PREPARE_RECEIPT.json', receipt)
    paths = [PROTOCOL, PROJECT_ROOT / 'tools/run_length_replay_v1.py', OLD / 'PLAN.json', OLD / 'COMPLETION.json', OLD / 'POSTRUN_TRANSFER_REVIEW.json']
    paths += list((SOURCE_ROOT / 'vlm_length_replay_v1').glob('*.py'))
    paths += [ROOT / n for n in ('DATA.json', 'SCHEDULE.json', 'CPU_INPUTS.json', 'PREPARE_RECEIPT.json')]
    paths += list((ROOT / 'initial').glob('*/PREDICTIONS.json'))
    paths += list((OLD / 'runs/coverage128').glob('*/PREDICTIONS.json'))
    plan = {k: prior[k] for k in ('adapter', 'lora_sha256', 'model_root', 'reference_config', 'packages', 'base_weight_files')}
    plan.update(status='prepared', roles=list(ROLES), updates_per_role=64, generation_qa_per_role=1044,
                files={**prior['files'], **{str(path): sha_file(path) for path in paths}}, images={v['path']: v['sha256'] for v in images.values()},
                automatic_more_training=False, automatic_main_evaluation=False)
    write_json(ROOT / 'PLAN.json', plan)
    RESULT.write_text('# 多词回放准备完成\n\n同128困难题、64步，扩覆盖240对原50等分层监督权重控制；新190/复用640真实CPU合同与调度质量检查通过。尚无新训练或能力结论。\n')
    render()
    print('PREPARED_LENGTH_REPLAY', receipt, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen input/source changed: ' + path)
    if package_lock() != plan['packages']:
        raise ValueError('Package versions changed')
    for path, value in plan['base_weight_files'].items():
        st = Path(path).stat()
        if value != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
            raise ValueError('Base storage changed')
    return plan


def condition(role):
    from .train import train
    from .evaluate import evaluate
    verify(); model, processor, result = train(role)
    evaluate(model, processor, role)
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'train': result})
    write_json(ROOT / 'runs' / role / 'train/PROGRESS.json', {'status': 'completed', 'role': role, 'completed_updates': 64})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    records = {}
    for name, location in [('initial', ROOT / 'initial'), ('plain128', OLD / 'runs/coverage128'),
                           ('weighted50', ROOT / 'runs/weighted50'), ('expanded240', ROOT / 'runs/expanded240')]:
        records[name] = {cohort: [q for q in read(location / cohort / 'PREDICTIONS.json') if q['evaluation_scored']] for cohort in ('training_context', 'validation', 'retention')}
    percent = lambda rows: 100 * sum(q['vqa_score'] for q in rows) / len(rows)
    scores = {name: {cohort: percent(rows) for cohort, rows in parts.items()} for name, parts in records.items()}
    candidate = records['expanded240']; primary = {name: paired(records[name]['validation'], candidate['validation']) for name in ('initial', 'plain128', 'weighted50')}
    gates = {name + ':overall': value['delta_pp'] >= .5 and value['ci95'][0] > 0 for name, value in primary.items()}
    slices = {}
    for name in ('initial', 'weighted50'):
        for key, field in [('singleword', 'stratum'), ('multiword', 'stratum'), ('rare_1_5', 'frequency_slice'), ('freq_6_20', 'frequency_slice')]:
            value = paired([q for q in records[name]['validation'] if q[field] == key], [q for q in candidate['validation'] if q[field] == key])
            slices[name + ':' + key] = value
            gates[name + ':' + key] = value['delta_pp'] >= (.5 if name == 'initial' and key == 'rare_1_5' else -.5)
        gates[name + ':retention'] = scores['expanded240']['retention'] - scores[name]['retention'] >= -.5
    repair = paired([q for q in records['plain128']['validation'] if q['stratum'] == 'multiword'], [q for q in candidate['validation'] if q['stratum'] == 'multiword'])
    gates['multiword_repair_vs_plain'] = repair['delta_pp'] >= .5
    protected = {q['question_id'] for q in records['initial']['validation'] if q['stratum'] == 'multiword' and q['vqa_score'] >= .999999}
    keep = paired([q for q in records['initial']['validation'] if q['question_id'] in protected], [q for q in candidate['validation'] if q['question_id'] in protected])
    gates['initial_full_multiword_retention'] = keep['delta_pp'] >= -.5
    full = {name: sum(q['vqa_score'] >= .999999 for q in parts['training_context']) for name, parts in records.items()}
    gates['training_net32'] = full['expanded240'] - full['initial'] >= 32
    result = {'status': 'completed', 'passed': all(gates.values()), 'scores_percent': scores, 'primary': primary,
              'slices': slices, 'multiword_repair': repair, 'previously_full_multiword': keep, 'gates': gates, 'training_full': full,
              'goal_complete': False, 'next': 'No automatic more steps/seeds/main/CV;confirmation only after joint pass.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    table = '|组|训练128满分|验证512 VQA%|保持64 VQA%|\n|---|---:|---:|---:|\n'
    for name, value in scores.items():
        table += f'|{name}|{full[name]}|{value["validation"]:.6f}|{value["retention"]:.6f}|\n'
    RESULT.write_text('# 多词回放完整结果\n\n' + table + '\n联合门槛=' + str(result['passed']) + '；主配对=' + str(primary) + '\n\n只作单seed42内部探索，条件区间不含种子方差，不改旧负结果或简历。\n\n[协议](VLM_LENGTH_REPLAY_V1_PROTOCOL.md) · [原图与原理](learning/多词答案保持_扩覆盖与重加权.html)\n')
    render(result); write_json(ROOT / 'COMPLETION.json', result)
    return result


def render(result=None):
    data = read(ROOT / 'DATA.json'); previous = read(OLD / 'POSTRUN_TRANSFER_REVIEW.json')
    relative = lambda path: quote(os.path.relpath(path, PAGE.parent), safe='/._-')
    locations = [('initial', ROOT / 'initial'), ('plain128', OLD / 'runs/coverage128'), ('weighted50', ROOT / 'runs/weighted50'), ('expanded240', ROOT / 'runs/expanded240')]
    saved = {name: {q['question_id']: q for q in read(path / 'validation/PREDICTIONS.json')} for name, path in locations if (path / 'validation/PREDICTIONS.json').exists()}
    cards = []
    for case in previous['changes']:
        rows = ''
        for name, mapping in saved.items():
            if case['question_id'] in mapping:
                pred = mapping[case['question_id']]
                rows += '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in [name, pred['prediction'], pred['vqa_score']]) + '</tr>'
        boundary = '左端文字被指针遮挡，语义优劣不确定，原VQA评分保留。' if case['question_id'] == '29126' else '这是旧结果事后教学病例；没有据此把验证题加入训练或重写评分。'
        cards.append(f'<article><h2>q{case["question_id"]}</h2><img loading="lazy" src="{relative(case["image"])}" alt="旧验证病例原图"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><p>{boundary}</p><table><tr><th>组</th><th>保存的实际生成</th><th>VQA</th></tr>{rows}</table></article>')
    bars = ''.join(f'<text x="5" y="{35+i*50}" font-size="13">{name}</text><rect x="240" y="{18+i*50}" width="{ratio*4}" height="24" fill="{color}"/><text x="{245+ratio*4}" y="{35+i*50}" font-size="13">{ratio:.2f}%</text>' for i, (name, ratio, color) in enumerate([('原训练：真实多词曝光', 78/1024*100, '#64748b'), ('扩覆盖：真实多词曝光', 268/1024*100, '#2563eb'), ('重加权：多词loss权重占比', 268/1024*100, '#d97706'), ('验证：真实多词QA占比', 50., '#059669')]))
    status = '准备/运行中，尚无本轮能力结论。' if result is None else html.escape(str(result['scores_percent'])) + '；联合门槛=' + str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多词答案保持</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:18px;margin:20px 0;border:1px solid #cbd5e1}img{width:100%;max-height:450px;object-fit:contain}td,th{padding:7px;border:1px solid #cbd5e1}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}svg{width:100%;max-width:800px}</style><h1>纠错收益为什么被原答案退化抵消</h1><p>' + status + '</p><p>旧高lr候选验证4升(+3.7)/5降(−4.4)，训练满分56/128。三个完整品牌/地名被缩短有原图支持，长度错配尚是待检验假设。回放64保持持平不代表所有原正确问题保持。</p><svg viewBox="0 0 560 220" role="img" aria-label="多词真实曝光与监督权重占比">' + bars + '</svg><pre>L = [Σhard CE_i + Σreplay w_i CE_i] / 16\nexpanded240：240不同多词+272单词，w=1\nweighted50：原50不同多词+462单词，w=目标cell计数/原cell计数\ncell=(监督频次桶,单词/多词)，累计分层权重总量相等，总回放512\n相同权重总量不等于相同不同样本信息、每步权重、梯度方差或算力。\n两组同128困难题/64更新/512hard+512replay/lr2e−4。\n只编码新增190题，旧640输入与全部初始预测复用。\n保护：原初始满分多词243题另列，同时保留整体/rare/原保持门槛。</pre>' + f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_length_replay_v1/train.py")}">真实训练代码</a></p>' + ''.join(cards) + '</html>')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.role:
        condition(args.role)
    else:
        from research_runtime.queue import run_queue
        verify()
        run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_length_replay_v1.study', '--role', role],
                   'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        print('COMPLETED_LENGTH_REPLAY', analyze()['passed'], flush=True)


if __name__ == '__main__':
    main()
