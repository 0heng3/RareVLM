import argparse
import ast
import html
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import sha_file, write_json
from .common import ROOT, OLD, FIT, ROLES, read

PROTOCOL = PROJECT_ROOT / 'docs/VLM_COVERAGE_SFT_V2_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_COVERAGE_SFT_V2_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/学习率与覆盖_四格真实结果.html'


def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Study already frozen')
    fit = read(FIT / 'COMPLETION.json')
    if fit['status'] != 'completed' or not fit['supports_lr_scale_bottleneck_on_these16']:
        raise ValueError('Completed fit diagnostic does not support this next step')
    previous = read(OLD / 'PLAN.json')
    original = SOURCE_ROOT / 'vlm_coverage_sft_v1/train.py'
    current = SOURCE_ROOT / 'vlm_coverage_sft_v2/train.py'
    class NormalizeLR(ast.NodeTransformer):
        changes = 0
        def visit_Constant(self, node):
            if isinstance(node.value, float) and node.value == 2e-4:
                self.changes += 1
                return ast.copy_location(ast.Constant(value=2e-5), node)
            return node
    normalizer = NormalizeLR()
    same = ast.dump(normalizer.visit(ast.parse(current.read_text())), include_attributes=False) == ast.dump(ast.parse(original.read_text()), include_attributes=False)
    if not same or normalizer.changes != 2:
        raise ValueError('Training differs beyond the two declared LR constants')
    write_json(ROOT / 'SINGLE_FACTOR_SOURCE_CHECK.json', {'status': 'passed', 'only_changes': '2e-5 to2e-4 at optimizer and schedule',
               'changed_constants': 2, 'v1_sha256': sha_file(original), 'v2_sha256': sha_file(current), 'normalized_ast_identical': same})
    for name in ('DATA.json', 'SCHEDULE.json', 'CPU_INPUTS.json', 'BASELINE_REUSE.json'):
        shutil.copyfile(OLD / name, ROOT / name)
    for cohort in ('training_context', 'validation', 'retention'):
        directory = ROOT / 'initial' / cohort; directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD / 'initial' / cohort / 'PREDICTIONS.json', directory / 'PREDICTIONS.json')
    paths = [PROTOCOL, PROJECT_ROOT / 'tools/run_coverage_sft_v2.py', FIT / 'COMPLETION.json', FIT / 'PLAN.json',
             FIT / 'INITIAL_MATCHED_TOKEN_CHECK.json', OLD / 'PLAN.json', OLD / 'ANALYSIS.json', ROOT / 'SINGLE_FACTOR_SOURCE_CHECK.json']
    paths += list((SOURCE_ROOT / 'vlm_coverage_sft_v2').glob('*.py'))
    paths += [ROOT / n for n in ('DATA.json', 'SCHEDULE.json', 'CPU_INPUTS.json', 'BASELINE_REUSE.json')]
    paths += list((ROOT / 'initial').glob('*/PREDICTIONS.json'))
    for role in ROLES:
        paths += list((OLD / 'runs' / role).glob('*/PREDICTIONS.json'))
        paths += [OLD / 'runs' / role / 'train' / 'TRAIN_COMPLETE.json']
    plan = {k: v for k, v in previous.items() if k != 'files'}
    plan.update(peak_lr=2e-4, reuse='v1 exact data/schedule/CPU inputs/initial predictions;no new baseline calls',
                files={**previous['files'], **{str(path): sha_file(path) for path in paths}})
    write_json(ROOT / 'PLAN.json', plan)
    RESULT.write_text('# 高学习率覆盖研究\n\n准备完成，尚无新训练结果。新高lr两臂沿用同DATA/SCHEDULE/CPU输入/初始预测，旧低lr结果只读复用。\n')
    render()
    print('PREPARED_HIGH_LR', {'normalized_train_ast_identical': same, 'changed_constants': 2, 'new_training_roles': 2}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen study source/input changed: ' + path)
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
    from vlm_human_sft.evaluate import lora_hash
    verify()
    model, processor, result = train(role)
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Final adapter identity differs')
    evaluate(model, processor, role)
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'train': result})
    write_json(ROOT / 'runs' / role / 'train/PROGRESS.json', {'status': 'completed', 'completed_updates': 64, 'role': role})


def scored(records):
    return [q for q in records if q['evaluation_scored']]


def percent(records):
    return 100 * sum(q['vqa_score'] for q in records) / len(records)


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    records = {}
    for name, location in [('initial', ROOT / 'initial'), ('low_repeat32', OLD / 'runs/repeat32'),
                           ('low_coverage128', OLD / 'runs/coverage128'), ('high_repeat32', ROOT / 'runs/repeat32'),
                           ('high_coverage128', ROOT / 'runs/coverage128')]:
        records[name] = {cohort: scored(read(location / cohort / 'PREDICTIONS.json')) for cohort in ('training_context', 'validation', 'retention')}
    candidate = records['high_coverage128']
    scores = {name: {cohort: percent(rows) for cohort, rows in parts.items()} for name, parts in records.items()}
    primary = {name: paired(records[name]['validation'], candidate['validation']) for name in ('initial', 'high_repeat32', 'low_coverage128')}
    gates = {name + ':overall': v['delta_pp'] >= .5 and v['ci95'][0] > 0 for name, v in primary.items()}
    slices = {}
    for name in ('initial', 'high_repeat32'):
        for key, field in [('singleword', 'stratum'), ('multiword', 'stratum'), ('rare_1_5', 'frequency_slice'), ('freq_6_20', 'frequency_slice')]:
            value = paired([q for q in records[name]['validation'] if q[field] == key], [q for q in candidate['validation'] if q[field] == key])
            slices[name + ':' + key] = value
            gates[name + ':' + key] = value['delta_pp'] >= (.5 if name == 'initial' and key == 'rare_1_5' else -.5)
        gates[name + ':retention'] = scores['high_coverage128']['retention'] - scores[name]['retention'] >= -.5
    full = {name: sum(q['vqa_score'] >= .999999 for q in parts['training_context']) for name, parts in records.items()}
    gates['training_net32'] = full['high_coverage128'] - full['initial'] >= 32
    effects = {
        'lr_with_coverage128': primary['low_coverage128'],
        'lr_with_repeat32': paired(records['low_repeat32']['validation'], records['high_repeat32']['validation']),
        'coverage_at_low_lr': paired(records['low_repeat32']['validation'], records['low_coverage128']['validation']),
        'coverage_at_high_lr': primary['high_repeat32'],
    }
    result = {'status': 'completed', 'passed': all(gates.values()), 'scores_percent': scores, 'primary': primary,
              'slices': slices, 'gates': gates, 'training_full': full, 'factor_effects': effects,
              'goal_complete': False, 'next': 'Separate confirmation only if all gates pass;no automatic extra steps/seeds/main/CV.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    table = '|单元|训练128满分|验证512 VQA%|保持64 VQA%|\n|---|---:|---:|---:|\n'
    for name, v in scores.items():
        table += f'|{name}|{full[name]}|{v["validation"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 高学习率覆盖研究完成\n\n' + table + '\n联合门槛=' + str(result['passed']) + '。\n\n主配对：' + str(primary) + '\n\n完整控制/切片/条件效应见ANALYSIS；单seed42内部探索、历史低lr单元复用、非盲测。训练拟合不替代验证能力，原门槛不改。\n\n[协议](VLM_COVERAGE_SFT_V2_PROTOCOL.md) · [原图/四格答案](learning/学习率与覆盖_四格真实结果.html)\n')
    render(result)
    write_json(ROOT / 'COMPLETION.json', result)
    return result


def render(result=None):
    data = read(ROOT / 'DATA.json')
    relative = lambda path: quote(os.path.relpath(path, PAGE.parent), safe='/._-')
    locations = [('initial', ROOT / 'initial'), ('low_repeat32', OLD / 'runs/repeat32'), ('low_coverage128', OLD / 'runs/coverage128'),
                 ('high_repeat32', ROOT / 'runs/repeat32'), ('high_coverage128', ROOT / 'runs/coverage128')]
    saved = {name: {q['question_id']: q for q in read(path / 'training_context/PREDICTIONS.json')} for name, path in locations if (path / 'training_context/PREDICTIONS.json').exists()}
    cards = []
    for q in data['hard_rows']:
        rows = ''
        for name, mapping in saved.items():
            if q['question_id'] in mapping:
                pred = mapping[q['question_id']]
                rows += '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in [name, pred['prediction'], pred['vqa_score']]) + '</tr>'
        cards.append(f'<article><h2>q{q["question_id"]} · {html.escape(q["review_error_type"])}</h2><img loading="lazy" src="{relative(data["images"][q["image_id"]]["path"])}" alt="已审核原始训练图"><p>{html.escape(q["question"])}</p><p>原GT：{html.escape(q["target"])}；{html.escape(q["visible_evidence"])}</p><table><tr><th>单元</th><th>实际保存生成</th><th>VQA</th></tr>{rows}</table></article>')
    status = '高lr两臂准备/运行中，无本轮能力提升结论。' if result is None else html.escape(str(result['scores_percent'])) + '；联合门槛=' + str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>学习率与覆盖四格</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:18px;margin:20px 0;border:1px solid #cbd5e1}img{width:100%;max-height:440px;object-fit:contain}td,th{padding:7px;border:1px solid #cbd5e1}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}</style><h1>从能学会，到能迁移：学习率 × 错误覆盖</h1><p>' + status + '</p><pre>已有低lr：32/128困难覆盖，lr=2e−5\n新增高lr：同32/128覆盖，lr=2e−4\n四格其余训练schedule、真实输入与初始模型一致。\n同覆盖高−低：检验更新尺度；同lr 128−32：检验覆盖策略。\n16题正控制：高lr16/16，低lr8/16，只是拟合，不是新QA泛化。\n主要验收：高128对初始、高32、低128均+.5pp/正95%配对下界。\n训练128净32满分及原切片/保持保护，不挑checkpoint或control挽救。</pre>' + f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="16题拟合正控制_正文概率与自由生成.html">先前16题概率曲线</a></p>' + ''.join(cards) + '</html>')


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
        run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_coverage_sft_v2.study', '--role', role],
                   'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        print('COMPLETED_HIGH_LR', analyze()['passed'], flush=True)


if __name__ == '__main__':
    main()
