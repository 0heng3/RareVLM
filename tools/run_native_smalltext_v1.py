"""Matched full-image scale intervention, fixed model, zero training."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_native_smalltext_v1'
PARENT = BASE / 'vlm_native_ocr_correction_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_SMALLTEXT_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_NATIVE_SMALLTEXT_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/小字问答_实际分辨率与语言先验.html'
POLICIES = {'native1024': {'min_pixels': 50176, 'max_tokens': 1024},
            'enlarge3072': {'min_pixels': 3072*784, 'max_tokens': 4096}}


def processor(plan, policy):
    from vlm_probe.evaluate import processor_for
    config = {**read(plan['reference_config']), 'min_pixels': POLICIES[policy]['min_pixels']}
    return processor_for(Path(plan['model_root']), config, POLICIES[policy]['max_tokens'])


def prepare():
    import torch
    from vlm_deep.common import encode
    from vlm_probe.evaluate import inference_row
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Study frozen')
    start = time.monotonic()
    torch.set_num_threads(2)
    small = read(ROOT / 'CANDIDATE_ROWS.json')
    old = read(PARENT / 'revision2/DATA.json')
    if len(small['rows']) != 104:
        raise ValueError('Registered inventory differs')
    available = {q['question_id']: q for q in old['rows']
                 if q['actual_evidence_region_pixels'][1] >= 24 and q['answer_frequency'] <= 20}
    rows, matches = [], []
    for q in small['rows']:
        eligible = [r for r in available.values() if r['frequency_slice'] == q['frequency_slice']]
        control = min(eligible, key=lambda r: (abs(len(r['target'])-len(q['target'])),
                      abs(r['answer_frequency']-q['answer_frequency']),
                      content_sha(['smalltext_match_v1', q['question_id'], r['question_id']])))
        del available[control['question_id']]
        rows.extend([{**q, 'cohort': 'small_text'}, {**control, 'cohort': 'large_text'}])
        matches.append([q['question_id'], control['question_id']])
    images = {**small['images'], **{q['image_id']: old['images'][q['image_id']]
                                  for q in rows if q['cohort'] == 'large_text'}}
    if len(rows) != 208 or len({images[q['image_id']].get('pixel_sha256', images[q['image_id']]['sha256']) for q in rows}) != 208:
        raise ValueError('Need208 distinct images/pixels')
    previous = read(BASE / 'vlm_reference_replay_v1/PLAN.json')
    data = {'rows': rows, 'images': images, 'matched_question_ids': matches, 'policies': {}}
    for policy in POLICIES:
        proc = processor(previous, policy)
        encoded = []
        for q in rows:
            item = encode(proc, inference_row(q), Path(images[q['image_id']]['path']), False)
            grid = item['image_grid_thw'][0].tolist()
            box = q['ocr_evidence']['box_xyxy_original_pixels']
            im = images[q['image_id']]
            encoded.append({**q, 'expected_prefix_tokens': int(item['input_ids'].shape[1]),
                            'expected_grid': grid, 'inference_input_sha256': content_sha(item['input_ids'].tolist()),
                            'actual_evidence_region_pixels': [(box[2]-box[0])*grid[2]*14/im['width'],
                                                             (box[3]-box[1])*grid[1]*14/im['height']],
                            'actual_visual_tokens': grid[0]*grid[1]*grid[2]//4})
        data['policies'][policy] = encoded
    for q, high in zip(data['policies']['native1024'], data['policies']['enlarge3072']):
        if q['question_id'] != high['question_id'] or high['actual_visual_tokens'] <= q['actual_visual_tokens']:
            raise ValueError('Scale intervention must actually enlarge every image')
    write_json(ROOT / 'DATA.json', data)
    audit = {'scope': 'Only four original images opened by assistant; remaining38 unreviewed',
             'cases': [
                 {'question_id': '32542', 'decision': 'candidate_supported', 'reason': 'Visible sign RAINDEER; reindeer normalizes a real visual misspelling.'},
                 {'question_id': '6892', 'decision': 'exclude_pending_brand_spacing', 'reason': 'Stylized PurNatur/pur natur word boundary; do not treat as definite recognition failure.'},
                 {'question_id': '796', 'decision': 'exclude_truncated_word', 'reason': 'Right image boundary cuts hanger lettering; complete brand/name not recoverable.'},
                 {'question_id': '24461', 'decision': 'exclude_semantic_name_boundary', 'reason': 'Visible citi logo; question asks bank and citibank is a plausible full name.'}],
             'not_expert_relabeling': True, 'old_targets_scores_unchanged': True}
    write_json(ROOT / 'FOUR_CASE_REVIEW.json', audit)
    files = {**read(PARENT / 'PLAN.json')['files']}
    for path in (Path(__file__), PROJECT_ROOT/'tools/inventory_native_smalltext_v1.py', PROTOCOL,
                 ROOT/'CANDIDATE_ROWS.json', ROOT/'INVENTORY.json', ROOT/'DATA.json', ROOT/'FOUR_CASE_REVIEW.json'):
        files[str(path)] = sha_file(path)
    plan = {'status': 'prepared', 'model_root': previous['model_root'], 'reference_config': previous['reference_config'],
            'initial_adapter': previous['initial_adapter'], 'lora_sha256': read(PARENT/'PLAN.json')['models']['initial']['lora_sha256'],
            'files': files, 'image_files': {im['path']: im['sha256'] for im in images.values()},
            'base_weight_files': previous['base_weight_files'], 'packages': previous['packages'],
            'policies': POLICIES, 'qa_per_policy': 208, 'new_generations': 416, 'new_optimizer_updates': 0,
            'automatic_training': False, 'generation_seconds_per_shard': 900, 'active_prepare_seconds': time.monotonic()-start,
            'preview_questions': [q['question_id'] for q in sorted(small['rows'], key=lambda q: content_sha(['smalltext_preview', q['question_id']]))[:8]]}
    write_json(ROOT / 'PLAN.json', plan)
    RESULT.write_text('# 原生小字输入尺度诊断\n\n已准备104小字+104大字匹配对照，各两种整图尺度，共416生成，0训练。未有新能力结果。\n')
    render()
    print('PREPARED', {k: plan[k] for k in ('qa_per_policy', 'new_generations', 'active_prepare_seconds')}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen file differs: '+path)
    for path, digest in plan['image_files'].items():
        if sha_file(path) != digest:
            raise ValueError('Original image differs: '+path)
    if package_lock() != plan['packages']:
        raise ValueError('Package versions differ')
    for path, stats in plan['base_weight_files'].items():
        st = Path(path).stat()
        if stats != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
            raise ValueError('Base weight storage differs')
    return plan


def worker(shard):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan = verify()
    data = read(ROOT / 'DATA.json')
    model, _, _ = actor(adapter=plan['initial_adapter'])
    model.eval()
    model.config.use_cache = True
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Wrong initial model')
    processors = {p: processor(plan, p) for p in POLICIES}
    indexed = {p: {q['question_id']: q for q in data['policies'][p]} for p in POLICIES}
    groups = [data['rows'][i:i+4] for i in range(0, len(data['rows']), 4)][shard::2]
    directory = ROOT/'runs'/f'shard_{shard}'
    path = directory/'BATCHES.json'
    done = read(path) if path.exists() else []
    if len(done) > len(groups):
        raise ValueError('Extra cached batches')
    for batch, expected in zip(done, groups):
        if batch['question_ids'] != [q['question_id'] for q in expected] or set(batch['policies']) != set(POLICIES):
            raise ValueError('Partial or different cached batch')
    prior = read(directory/'PROGRESS.json').get('active_seconds', 0.) if (directory/'PROGRESS.json').exists() else 0.
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for batch_index in range(len(done), len(groups)):
        group = groups[batch_index]
        batch = {'question_ids': [q['question_id'] for q in group], 'policies': {}}
        order = list(POLICIES) if batch_index % 2 == 0 else list(reversed(POLICIES))
        for policy in order:
            rows = [indexed[policy][q['question_id']] for q in group]
            start = time.monotonic()
            answers = generated(model, processors[policy], rows, data)
            for q, answer in zip(rows, answers):
                if answer['input_sha256'] != q['inference_input_sha256']:
                    raise ValueError('Actual input changed')
                answer.update(cohort=q['cohort'], frequency_slice=q['frequency_slice'],
                              actual_grid=q['expected_grid'], evidence_region_pixels=q['actual_evidence_region_pixels'])
            batch['policies'][policy] = {'answers': answers, 'generation_with_preprocessing_seconds': time.monotonic()-start}
        done.append(batch)
        temp = path.with_suffix('.tmp')
        write_json(temp, done)
        temp.replace(path)
        active = prior+time.monotonic()-started
        peak = torch.cuda.max_memory_allocated()/2**30
        write_json(directory/'PROGRESS.json', {'status': 'fixed_model_generation', 'completed_qa': len(done)*8,
                   'expected_qa': len(groups)*8, 'active_seconds': active, 'peak_allocated_gib': peak,
                   'optimizer_updates': 0})
        if active > plan['generation_seconds_per_shard'] or peak > 65:
            raise RuntimeError('Short diagnostic budget exceeded')
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Read-only model changed')
    write_json(directory/'COMPLETE.json', {'status': 'completed', 'generated_qa': len(done)*8,
               'active_seconds': prior+time.monotonic()-started, 'optimizer_updates': 0})


def summarize(rows):
    return {'vqa_percent': 100*sum(q['vqa_score'] for q in rows)/len(rows),
            'full': sum(q['vqa_score'] >= .999999 for q in rows), 'qa': len(rows),
            'natural_eos': sum(q['terminated'] for q in rows),
            'mean_visual_tokens': sum(q['visual_tokens'] for q in rows)/len(rows)}


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT/'DATA.json')
    records = {p: {} for p in POLICIES}
    for shard in (0, 1):
        for batch in read(ROOT/'runs'/f'shard_{shard}/BATCHES.json'):
            for policy in POLICIES:
                for answer in batch['policies'][policy]['answers']:
                    if answer['question_id'] in records[policy]:
                        raise ValueError('Duplicate question')
                    records[policy][answer['question_id']] = answer
    predictions = {p: [records[p][q['question_id']] for q in data['rows']] for p in POLICIES}
    for p, rows in predictions.items():
        write_json(ROOT/'runs'/p/'PREDICTIONS.json', rows)
    scores, comparisons, slices = {}, {}, {}
    for cohort in ('small_text', 'large_text'):
        groups = {p: [q for q in predictions[p] if q['cohort'] == cohort] for p in POLICIES}
        scores[cohort] = {p: summarize(rows) for p, rows in groups.items()}
        comparisons[cohort] = paired(groups['native1024'], groups['enlarge3072'])
        slices[cohort] = {}
        for frequency in ('rare_1_5', 'freq_6_20'):
            low = [q for q in groups['native1024'] if q['frequency_slice'] == frequency]
            high = [q for q in groups['enlarge3072'] if q['frequency_slice'] == frequency]
            slices[cohort][frequency] = {'native1024': summarize(low), 'enlarge3072': summarize(high), 'comparison': paired(low, high)}
    signal = (comparisons['small_text']['delta_pp'] >= 5 and comparisons['small_text']['ci95'][0] > 0
              and comparisons['large_text']['delta_pp'] >= -2)
    candidates, targets = [], Counter()
    raw = {q['question_id']: q for q in data['rows'] if q['cohort'] == 'small_text'}
    for q in sorted(predictions['native1024'], key=lambda q: content_sha(['smalltext_errors', q['question_id']])):
        if q['cohort'] != 'small_text':
            continue
        target = NORMALIZER(q['target'])
        if q['vqa_score'] > .3+1e-8 or not q['terminated'] or target in NORMALIZER(q['prediction']).split() or targets[target] >= 2:
            continue
        targets[target] += 1
        candidates.append({**raw[q['question_id']], 'initial_prediction': q,
                           'enlarged_prediction': records['enlarge3072'][q['question_id']]})
    write_json(ROOT/'ERROR_CANDIDATES.json', {'rows': candidates, 'images': data['images'], 'not_semantically_reviewed': True})
    result = {'status': 'completed', 'scores': scores, 'comparisons': comparisons, 'frequency_slices': slices,
              'exploratory_scale_signal': signal, 'candidate_errors': len(candidates), 'unique_error_targets': len(targets),
              'training_pool_count_gate_passed': len(candidates) >= 128 and len(targets) >= 96,
              'new_optimizer_updates': 0, 'new_adapters': 0, 'automatic_training': False, 'goal_complete': False,
              'scope': 'Fixed-model paired input intervention on internal original-SFT-seen images; not new trained capability.'}
    write_json(ROOT/'ANALYSIS.json', result)
    table = '|层|原输入VQA%|放大VQA%|差值pp|95%区间|\n|---|---:|---:|---:|---|\n'
    for cohort in scores:
        c = comparisons[cohort]
        table += f'|{cohort}|{scores[cohort]["native1024"]["vqa_percent"]:.6f}|{scores[cohort]["enlarge3072"]["vqa_percent"]:.6f}|{c["delta_pp"]:.6f}|{c["ci95"]}|\n'
    RESULT.write_text('# 原生小字：完整输入尺度诊断\n\n'+table+f'\n探索尺度信号={signal}；新小字错误候选{len(candidates)}、不同target{len(targets)}。0训练/新adapter，未过128图/96target训练数量门槛。输入干预不能当新权重能力提升。\n\n[协议](VLM_NATIVE_SMALLTEXT_V1_PROTOCOL.md) · [真实原图与回答](learning/小字问答_实际分辨率与语言先验.html)\n')
    render(result)
    write_json(ROOT/'COMPLETION.json', result)
    print('SMALLTEXT_COMPLETE', json.dumps(result, ensure_ascii=False), flush=True)


def render(result=None):
    data = read(ROOT/'DATA.json')
    plan = read(ROOT/'PLAN.json')
    relative = lambda p: quote(os.path.relpath(p, PAGE.parent), safe='/._-')
    lookups = {p: {q['question_id']: q for q in read(ROOT/'runs'/p/'PREDICTIONS.json')} for p in POLICIES
               if (ROOT/'runs'/p/'PREDICTIONS.json').exists()}
    cards = []
    for qid in plan['preview_questions']:
        q = next(q for q in data['rows'] if q['question_id'] == qid)
        im = data['images'][q['image_id']]
        x0, y0, x1, y1 = q['ocr_evidence']['box_xyxy_original_pixels']
        overlay = f'left:{100*x0/im["width"]}%;top:{100*y0/im["height"]}%;width:{100*(x1-x0)/im["width"]}%;height:{100*(y1-y0)/im["height"]}%'
        body = ''
        for p in POLICIES:
            encoded = next(v for v in data['policies'][p] if v['question_id'] == qid)
            answer = lookups.get(p, {}).get(qid)
            body += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
                p, encoded['expected_grid'], encoded['actual_visual_tokens'],
                [round(v, 2) for v in encoded['actual_evidence_region_pixels']],
                answer['prediction'] if answer else '待生成', answer['vqa_score'] if answer else '—'))+'</tr>'
        cards.append(f'<article><h2>q{qid}</h2><figure><img loading="lazy" src="{relative(im["path"])}" alt="原图"><span class="box" style="{overlay}"></span></figure><p>{html.escape(q["question"])}</p><p>原目标 {html.escape(q["target"])}；官方文字 {html.escape(q["ocr_evidence"]["original_transcription"])}</p><table><tr><th>策略</th><th>实际grid</th><th>视觉token</th><th>证据宽高像素</th><th>真实回答</th><th>VQA</th></tr>{body}</table></article>')
    status = '准备/运行中，无新质量结果。' if result is None else html.escape(json.dumps(result['comparisons'], ensure_ascii=False))
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小字与实际分辨率</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{padding:18px;margin:20px 0;background:white;border:1px solid #cbd5e1}figure{position:relative;margin:0}img{display:block;width:100%;height:auto}.box{position:absolute;border:2px solid #ef4444;box-sizing:border-box;pointer-events:none}table{border-collapse:collapse}td,th{border:1px solid #cbd5e1;padding:6px}pre{white-space:pre-wrap}</style><h1>小字问答：实际输入尺度与语言先验</h1><p>'+status+'</p><p>104小字+104较大文字匹配对照，各两种整图尺度；模型权重不变，0训练。8例在推理前固定。框只在网页显示，不是模型输入或注意力；图为原图，非实际processor图。插值放大没有新增原生信息。</p><h2>为什么提高上限常常没有改变输入？</h2><p>原1269张图中，1067张只把max_pixels从1024×784提高到4096×784，smart_resize仍不变。本轮enlarge3072另设最小面积3072×784，才真正放大整图。既有14×14 patch与2×2 merger给出 visual_tokens=(grid_h×grid_w)/4；框高变成 h_box×grid_h×14/h_original。看清与选对是不同问题，字词边界和完整名称也会影响官方短答分。</p><h2>源码与判断条件</h2><pre>processor_for(model_root, {**config, min_pixels:3072*784}, 4096)\nanswers = generated(fixed_model, processor, original_questions, original_images)\n# target/OCR/bbox不进入生成提示；生成后才计算官方VQA</pre><p>预固定探索条件：小字差≥5pp且95%区间下界&gt;0、大字点差≥−2pp。即便有信号，也不是新增训练能力，不能写简历提分。错误候选须另过128图/96target与语义复核；本批104小字不可能单独过数量门槛。</p><p><a href="'+relative(PROTOCOL)+'">事前协议</a> · <a href="'+relative(Path(__file__).resolve())+'">真实执行代码</a></p>'+''.join(cards)+'</html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        worker(args.worker)
    else:
        from research_runtime.queue import run_queue
        verify()
        jobs = [{'key': f'shard_{s}', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(s)],
                 'completion_marker': str(ROOT/'runs'/f'shard_{s}/COMPLETE.json')} for s in (0, 1)]
        run_queue(ROOT/'queue', jobs, {'plan_sha256': sha_file(ROOT/'PLAN.json')})
        analyze()


if __name__ == '__main__':
    main()
