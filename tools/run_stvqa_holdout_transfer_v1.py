"""Fixed Final-only source-holdout comparison; reuse the frozen probe loop."""
import argparse
from collections import Counter
from datetime import datetime
import html
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_stvqa_holdout_transfer_v1'
SOURCE = BASE / 'vlm_stvqa_headroom_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md'
ROLES = ('old_ce', 'old_dpo', 'new_ce', 'new_dpo')
LABELS = {'start': '共同起点（复用）', 'old_ce': '旧裁词CE', 'old_dpo': '旧裁词CE+DPO',
          'new_ce': '新整图SFT', 'new_dpo': '新整图SFT+DPO'}


def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Fixed source comparison already registered')
    source = read(SOURCE / 'DATA.json')
    rows = [q for q in source['rows'] if q['probe_cohort'] == 'source_holdout']
    ids = {q['question_id'] for q in rows}
    groups = [g for g in source['groups'] if set(g) <= ids]
    if len(rows) != 256 or len({q['image_id'] for q in rows}) != 256 or len(groups) != 64:
        raise ValueError('Require the complete frozen256 scenes and64 original groups')
    if Counter(q for g in groups for q in g) != Counter(ids):
        raise ValueError('Holdout grouping differs')
    images = {q['image_id']: source['images'][q['image_id']] for q in rows}
    pixels = {m['pixel_sha256'] for m in images.values()}
    raw = {m['sha256'] for m in images.values()}
    if len(pixels) != 256:
        raise ValueError('Holdout contains repeated decoded scenes')
    models = {}
    training_manifests = []
    for label, parent_name in [('old', 'vlm_visual_preference_v1'), ('new', 'vlm_stvqa_replay_v1')]:
        parent = BASE / parent_name
        manifest_path = parent / 'data/MANIFEST.json'
        training_manifests.append(manifest_path)
        manifest = read(manifest_path)
        train_images = [manifest['images'][q['image_id']] for q in manifest['rows']]
        if pixels & {m['pixel_sha256'] for m in train_images} or raw & {m['sha256'] for m in train_images}:
            raise ValueError('Source holdout overlaps actual training scenes')
        for local, name in [('ce', 'ce'), ('ce_dpo', 'dpo')]:
            complete = parent / 'runs' / local / 'train/TRAIN_COMPLETE.json'
            trained = read(complete)
            if trained['status'] != 'completed' or trained['updates'] != 32:
                raise ValueError('Require the preregistered completed Final32')
            models[label + '_' + name] = {'adapter': trained['adapter'],
                'lora_sha256': trained['final_lora_sha256'], 'training_completion': str(complete)}
    data = {'rows': rows, 'images': images, 'groups': groups}
    write_json(ROOT / 'DATA.json', data)
    baseline_sources = [SOURCE / f'runs/shard_{s}/PREDICTIONS.json' for s in (0, 1)]
    lookup = {q['question_id']: q for path in baseline_sources for q in read(path)
              if q['probe_cohort'] == 'source_holdout'}
    baseline = [lookup[q['question_id']] for q in rows]
    if len(baseline) != 256 or any(p['input_sha256'] != q['inference_input_sha256'] for p, q in zip(baseline, rows)):
        raise ValueError('Reused initial prediction inputs differ')
    write_json(ROOT / 'baseline/PREDICTIONS.json', baseline)
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'DATA.json', ROOT / 'baseline/PREDICTIONS.json',
                    SOURCE / 'PLAN.json', SOURCE / 'SOURCE_INFERENCE_ADAPTER_PLAN.json', SOURCE / 'DATA.json',
                    PROJECT_ROOT / 'tools/run_stvqa_headroom_v1.py',
                    PROJECT_ROOT / 'tools/run_stvqa_headroom_source_adapter_v1.py',
                    PROJECT_ROOT / 'src/vlm_native_scope_v1/model.py',
                    PROJECT_ROOT / 'src/vlm_batch_stable_v1/model.py',
                    PROJECT_ROOT / 'src/vlm_fit_positive_control_v1/study.py',
                    PROJECT_ROOT / 'src/research_runtime/queue.py'] + training_manifests + baseline_sources
    dependencies += [Path(p) for p in read(SOURCE / 'SOURCE_INFERENCE_ADAPTER_PLAN.json')['files']]
    dependencies += [Path(m['path']) for m in images.values()]
    shared = {str(p): sha_file(p) for p in dependencies}
    role_plans = {}
    for role, model in models.items():
        directory = ROOT / 'roles' / role
        write_json(directory / 'DATA.json', data)
        files = [directory / 'DATA.json', Path(model['training_completion'])]
        files += [Path(model['adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
        write_json(directory / 'PLAN.json', {'status': 'frozen', 'model': model,
                   'files': {**shared, **{str(p): sha_file(p) for p in files}}})
        role_plans[role] = sha_file(directory / 'PLAN.json')
    write_json(ROOT / 'PLAN.json', {'status': 'frozen', 'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'files': shared, 'models': models, 'role_plan_sha256': role_plans,
               'qa_per_role': 256, 'roles': list(ROLES), 'jobs': 8,
               'new_generations': 1024, 'teacher_forced_sequences': 1024, 'optimizer_updates': 0,
               'reused_baseline_generations': 256, 'train_scene_pixel_overlap': 0,
               'automatic_training': False, 'learning_case_ids': [q['question_id'] for q in rows[:4]],
               'original_training_stop_unchanged': True})
    learning(data, {'start': baseline})
    print('PREPARED256_HOLDOUT_4_FINALS_ZERO_UPDATES', flush=True)


def worker(argument):
    role, shard = argument.split(':')
    if role not in ROLES or shard not in ('0', '1'):
        raise ValueError('Unregistered model/shard')
    plan = read(ROOT / 'PLAN.json')
    directory = ROOT / 'roles' / role
    if sha_file(directory / 'PLAN.json') != plan['role_plan_sha256'][role]:
        raise ValueError('Role plan changed')
    import run_stvqa_headroom_v1 as engine
    import vlm_fit_positive_control_v1.study as generation_provider
    from run_stvqa_headroom_source_adapter_v1 import source_generated
    engine.ROOT = directory
    generation_provider.generated = source_generated
    engine.worker(int(shard))


def contrast(records, weights):
    import numpy as np
    before = records['start']
    clusters = {}
    for i, base in enumerate(before):
        value = sum(w * float(records[role][i]['source_literal_exact']) for role, w in weights.items())
        item = clusters.setdefault(base['image_id'], [0., 0])
        item[0] += value; item[1] += 1
    values = np.array(list(clusters.values()), dtype=np.float64)
    index = np.random.default_rng(20261021).integers(0, len(values), size=(6000, len(values)))
    bootstrap = 100 * values[index, 0].sum(1) / values[index, 1].sum(1)
    return {'delta_pp': float(100 * values[:, 0].sum() / values[:, 1].sum()),
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'ci98_75_four_comparisons': np.quantile(bootstrap, [.00625, .99375]).tolist(),
            'image_clusters': len(values), 'scope': 'Fixed-model source holdout only; no training-seed uncertainty or all-history correction.'}


def learning(data, records, result=None):
    by_id = {role: {q['question_id']: q for q in rows} for role, rows in records.items()}
    articles = []
    for q in data['rows'][:4]:
        path = Path(data['images'][q['image_id']]['path'])
        url = '../' + str(path.relative_to(BASE))
        answers = '\n'.join(f"{LABELS[role]}: {by_id[role][q['question_id']]['prediction']}"
                            f" | exact={by_id[role][q['question_id']]['source_literal_exact']}"
                            f" | bodyNLL={by_id[role][q['question_id']]['body_nll']:.4f}" for role in records)
        articles.append(f'<article><h2>{html.escape(q["question_id"])}</h2><img src="{html.escape(url)}"><p>{html.escape(q["question"])}</p><p>原训练源参考：{html.escape(q["target"])}</p><pre>{html.escape(answers)}</pre></article>')
    import json
    summary = html.escape(json.dumps(result, ensure_ascii=False, indent=2)) if result else '四Final评测待完成；以下目前只有复用起点，不填缺失分数。'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>来源保持泛化验证</title><style>body{max-width:1350px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}article,section{background:white;padding:20px;margin:22px 0}img{max-width:100%;max-height:500px}pre{white-space:pre-wrap}</style><h1>新监督学到了什么：固定Final的256图来源保持验证</h1><section><p>所有模型权重冻结、0更新。保持集未参与训练，但起点分数此前已知；不是盲测、官方ANLS或原native VQA。评测源内泛化，不能替代完整任务/三seed/外部验证。</p><p>CE提高GT token概率；DPO提高chosen/rejected相对冻结reference的序列似然差。正文NLL下降与自由生成答对是两个指标。本轮不训练，不改变这两个损失；仅比较保存Final。原代码直接复用run_stvqa_headroom_v1.worker与source_generated，逐题绑定同prefix/grid/input。</p><p><a href="../../packages/RareVLM_minimal_20261001/docs/VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md">事前固定协议</a> · <a href="../../packages/RareVLM_minimal_20261001/tools/run_stvqa_holdout_transfer_v1.py">实际角色绑定与统计代码</a></p><pre>' + summary + '</pre></section>' + ''.join(articles) + '</html>'
    (ROOT / '来源保持_原理代码与实际回答.html').write_text(page)


def analyze():
    import statistics
    data = read(ROOT / 'DATA.json')
    order = [q['question_id'] for q in data['rows']]
    records = {'start': read(ROOT / 'baseline/PREDICTIONS.json')}
    for role in ROLES:
        lookup = {q['question_id']: q for s in (0, 1)
                  for q in read(ROOT / f'roles/{role}/runs/shard_{s}/PREDICTIONS.json')}
        if set(lookup) != set(order):
            raise ValueError('Incomplete Final source comparison')
        records[role] = [lookup[qid] for qid in order]
        write_json(ROOT / 'results' / role / 'PREDICTIONS.json', records[role])
    identity = lambda rows: [(q['question_id'], q['image_id'], q['input_sha256']) for q in rows]
    if any(identity(rows) != identity(records['start']) for rows in records.values()):
        raise ValueError('Fixed-model inference frames differ')
    summaries = {role: {'qa': len(rows), 'literal_correct': sum(q['source_literal_exact'] for q in rows),
                       'exact_percent': 100 * sum(q['source_literal_exact'] for q in rows) / len(rows),
                       'mean_body_nll': statistics.mean(q['body_nll'] for q in rows),
                       'mean_eos_nll': statistics.mean(q['eos_nll'] for q in rows),
                       'length_censored': sum(not q['terminated'] for q in rows)} for role, rows in records.items()}
    contrasts = {'new_ce_vs_start': {'new_ce': 1, 'start': -1},
                 'new_ce_vs_old_ce': {'new_ce': 1, 'old_ce': -1},
                 'new_dpo_vs_new_ce': {'new_dpo': 1, 'new_ce': -1},
                 'data_loss_interaction': {'new_dpo': 1, 'new_ce': -1, 'old_dpo': -1, 'old_ce': 1}}
    comparisons = {name: contrast(records, weights) for name, weights in contrasts.items()}
    signal = comparisons['new_ce_vs_old_ce']['ci98_75_four_comparisons'][0] > 0
    harm = comparisons['new_dpo_vs_new_ce']['ci98_75_four_comparisons'][1] < 0
    result = {'status': 'completed', 'phase': 'fixed_final_source_diagnosis_complete', 'summaries': summaries,
              'contrasts': comparisons, 'source_data_effect_supported': signal,
              'source_dpo_harm_supported': harm, 'new_generations': 1024,
              'teacher_forced_sequences': 1024, 'optimizer_updates': 0,
              'goal_complete': False, 'original_training_stop_unchanged': True,
              'scope': 'Internal source holdout, fixed single-seed Finals; source exact, not ANLS/native VQA or independent external confirmation.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'diagnostic_complete_no_training_or_tuning',
               'source_data_effect_supported': signal, 'source_dpo_harm_supported': harm,
               'automatic_training': False, 'original_training_stop_unchanged': True})
    learning(data, records, result)
    write_json(ROOT / 'COMPLETION.json', result)
    print('SOURCE_HOLDOUT_DIAGNOSIS_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen source comparison changed: ' + path)
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve())
    run_queue(ROOT / 'queue', [{'key': role + '_' + str(s),
        'command': [sys.executable, '-u', entry, '--worker', role + ':' + str(s)],
        'completion_marker': str(ROOT / f'roles/{role}/runs/shard_{s}/COMPLETE.json')}
        for role in ROLES for s in (0, 1)], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    analyze()


if __name__ == '__main__':
    main()
