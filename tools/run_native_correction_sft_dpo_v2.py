"""Question-conditioned correction, using the verified32-update CE/DPO loop."""
import argparse
from collections import Counter
import copy
import html
import importlib
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read
from vlm_literal_preference_v1.common import sample, make_batch

BASE = PROJECT_ROOT.parents[1] / 'outputs'
MINING = BASE / 'vlm_native_correction_pool_v2'
OLD = BASE / 'vlm_iiit_scope_v1'
ROOT = BASE / 'vlm_native_correction_sft_dpo_v2'
CONFIG = ROOT / 'CONFIG.json'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_CORRECTION_SFT_DPO_V2_PROTOCOL.md'
PAGE = PROJECT_ROOT / 'docs/learning/真实问答纠错_SFT与DPO同数据对照.html'
INTERNAL = ('native_hard', 'literal_validation', 'seen_validation', 'retention')
ROLES = ('sft', 'dpo_sft')


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    from vlm_iiit_scope_v1.data import canonical
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Training study frozen')
    review = read(MINING / 'REVIEWED_ERRORS.json'); candidates = read(MINING / 'ERROR_CANDIDATES.json')
    if review['status'] != 'passed' or len(review['accepted_question_ids']) < 128:
        raise ValueError('Need128 visually supported native errors')
    selected_ids = review['accepted_question_ids'][:128]
    lookup = {q['question_id']: q for q in candidates['rows'][:256]}; hard = [copy.deepcopy(lookup[qid]) for qid in selected_ids]
    if len(set(selected_ids)) != 128 or len({q['image_id'] for q in hard}) != 128 or len({NORMALIZER(q['target']) for q in hard}) < 96:
        raise ValueError('Insufficient unique nativeQA images/targets')
    for q in hard:
        p = q['initial_prediction']
        if p['vqa_score'] > .3 + 1e-8 or not p['terminated']:
            raise ValueError('Invalid complete negative')
        q.update(study_source='preference', rejected_target=p['prediction'], rejected_completion_ids=p['completion_ids'])
    old = read(OLD / 'DATA.json'); sparse = set(old['sparse_question_ids'])
    ocr = [{**q, 'source_target': q['target'], 'target': canonical(q['target']), 'study_source': 'common'} for q in old['training_rows'] if q['question_id'] in sparse]
    previous = read(BASE / 'vlm_iiit_replay_repair_v1/units/repair42/DATA.json')
    native = sorted([q for q in previous['training_rows'] if q['training_domain'] == 'native'], key=lambda q: content_sha(['native_correction_v2:replay', q['question_id']]))[:128]
    native = [{**q, 'study_source': 'common'} for q in native]
    common = [q for step in range(32) for chunk in (ocr[4 * step:4 * step + 4], native[4 * step:4 * step + 4]) for q in chunk]
    transfer = read(BASE / 'vlm_iiit_native_transfer_v1/DATA.json')
    evaluations = {'native_hard': hard, **copy.deepcopy(old['evaluations']), **copy.deepcopy(transfer['evaluations'])}
    evaluations.pop('hard')
    images = {**previous['images'], **candidates['images'], **transfer['images']}
    baseline = OLD / 'runs/sparse_language'
    retention = {q['question_id']: q for q in read(baseline / 'retention/PREDICTIONS.json')}
    for q in evaluations['retention']:
        q['inference_input_sha256'] = retention[q['question_id']]['input_sha256']
    held = {q['image_id'] for n in ('literal_validation', 'seen_validation', 'retention', 'transfer_dev') for q in evaluations[n]}
    pixels = {images[i]['pixel_sha256'] for i in held}
    if any(q['image_id'] in held or images[q['image_id']]['pixel_sha256'] in pixels for q in hard + common):
        raise ValueError('Train/evaluation image overlap')
    data = {'rows': common + hard, 'images': images, 'evaluations': evaluations}
    parent = read(MINING / 'PLAN.json'); init = read(MINING / 'INITIALIZATION.json')
    c = {**read(PROJECT_ROOT / 'configs/vlm_error_supervision_sft_v1.json'), 'output_root': str(ROOT),
         'initial_adapter': init['adapter'], 'lr': 2e-4, 'training_cap_seconds': 1800, 'evaluation_cap_seconds': 1500,
         'automatic_more_training': False, 'automatic_main_evaluation': True}
    write_json(CONFIG, c)
    reference = read(BASE / 'vlm_reference_replay_v1/PLAN.json'); torch.set_num_threads(2)
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024); cpu = {}
    for q in data['rows']:
        for kind in ('chosen', 'rejected') if q['study_source'] == 'preference' else ('chosen',):
            item = sample(processor, data, q, kind == 'rejected')
            if item['input_ids'].shape[1] > 2048:
                raise ValueError('Sequence exceeds fixed limit')
            cpu[q['question_id'] + ':' + kind] = {k + '_sha256': content_sha(item[k].tolist()) for k in ('input_ids', 'labels')}
    h = [q['question_id'] for q in hard]; r = [q['question_id'] for q in common]; updates = []
    for step in range(32):
        j = (8 * step) % 128
        updates.append({'micros': [r[8 * step:8 * step + 4], h[j:j + 4], r[8 * step + 4:8 * step + 8], h[j + 4:j + 8]]})
    counts = Counter(qid for u in updates for m in u['micros'] for qid in m)
    if len(counts) != 384 or any(counts[qid] != 2 for qid in h) or any(counts[qid] != 1 for qid in r):
        raise ValueError('Source exposure budget differs')
    write_json(ROOT / 'data/MANIFEST.json', data); write_json(ROOT / 'data/CPU_INPUTS.json', cpu); write_json(ROOT / 'data/SCHEDULE.json', {'updates': updates})
    baseline_paths = []
    for name in ('literal_validation', 'seen_validation', 'retention', 'transfer_dev'):
        path = BASE / 'vlm_iiit_native_transfer_v1/runs/sparse_language/PREDICTIONS.json' if name == 'transfer_dev' else baseline / name / 'PREDICTIONS.json'
        preds = read(path); rows = evaluations[name]
        if [(q['question_id'], q['input_sha256']) for q in preds] != [(q['question_id'], q['inference_input_sha256']) for q in rows]:
            raise ValueError('Reused starting model inputs differ')
        target = ROOT / 'baseline' / name / 'PREDICTIONS.json'; write_json(target, preds); baseline_paths += [path, target]
    overlap = sorted({canonical(q['target']) for q in evaluations['literal_validation']} & {canonical(word) for q in hard + native for word in q['target'].split()})
    write_json(ROOT / 'DATA_RECEIPT.json', {'status': 'passed', 'hard128_distinct_targets': len({NORMALIZER(q['target']) for q in hard}),
               'hard_length_strata': dict(Counter(q['stratum'] for q in hard)), 'native_replay': 128, 'ocr_replay': 128,
               'train_eval_image_pixel_overlap': 0, 'native_supervision_words_overlapping_ocr_dev': overlap,
               'retention64_metadata_prefixes_added_from_frozen_baseline': True, 'native_images_original_sft_seen': True,
               'ocr_images_entered_in_parent_iiit_training': True})
    paths = [Path(__file__), PROTOCOL, PROJECT_ROOT / 'docs/VLM_NATIVE_CORRECTION_V2_SUPPLEMENT.md', CONFIG, Path(c['baseline_config']), MINING / 'REVIEWED_ERRORS.json', MINING / 'INITIALIZATION_GPU_PARITY.json',
             ROOT / 'DATA_RECEIPT.json', ROOT / 'data/MANIFEST.json', ROOT / 'data/CPU_INPUTS.json', ROOT / 'data/SCHEDULE.json',
             PROJECT_ROOT / 'src/vlm_literal_preference_v1/train.py', PROJECT_ROOT / 'src/vlm_literal_preference_v1/common.py',
             PROJECT_ROOT / 'src/vlm_alignment_signal/loss.py'] + baseline_paths
    for path, digest in review.get('source_review_files', {}).items():
        if sha_file(path) != digest:
            raise ValueError('Source visual review changed')
        paths.append(Path(path))
    files = dict(parent['files']); files.update({str(p): sha_file(p) for p in paths})
    used = {q['image_id'] for q in data['rows'] + [q for rows in evaluations.values() for q in rows]}
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'status': 'prepared', 'files': files, 'packages': parent['packages'], 'base_weight_files': parent['base_weight_files'],
               'images': {images[i]['path']: images[i]['sha256'] for i in used},
               'initial_lora_sha256': init['lora_sha256'], 'new_updates': 64, 'updates_per_role': 32, 'qa_exposures_per_role': 512,
               'qa_exposures': {'preference': 256, 'native_replay': 128, 'ocr_replay': 128}, 'new_baseline_generations': 128,
               'internal_generations_per_role': 960, 'conditional_native1617_generations': 3234, 'conditional_repeated_test_generations': 6000,
               'same_data_sft_control': True, 'independent_seed_replication': False, 'automatic_more_training': False})
    render(); print('NATIVE_CORRECTION_TRAIN_PREPARED', {'hard': 128, 'native_replay': 128, 'ocr_replay': 128, 'updates_per_role': 32}, flush=True)


def verify(reference=False):
    from vlm_object_train.train import package_lock
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen training input changed: ' + path)
    if package_lock() != lock['packages']:
        raise ValueError('Runtime changed')
    for path, meta in lock['base_weight_files'].items():
        stat = Path(path).stat()
        if stat.st_size != meta['bytes'] or stat.st_mtime_ns != meta['mtime_ns']:
            raise ValueError('Base model identity changed')
    if reference:
        for path, digest in read(ROOT / 'reference/INPUT_LOCK.json').items():
            if sha_file(path) != digest:
                raise ValueError('Reference cache changed')
    return lock


def evaluate(model, processor, directory, names, data=None):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    from vlm_fit_positive_control_v1.study import generated
    from run_literal_tile_v1 import generate
    from run_iiit_scope_v1 import alphanumeric
    data = read(ROOT / 'data/MANIFEST.json') if data is None else data
    model.eval(); model.config.use_cache = True; before = lora_hash(model)
    lock_json(directory / 'INPUT_LOCK.json', {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json'), 'model_lora_sha256': before, 'data_sha256': content_sha(data), 'cohorts': list(names)})
    started = time.monotonic(); cap = 2400 if 'official_test' in names else 1500
    for name in names:
        rows = data['evaluations'][name]; path = directory / name / 'PREDICTIONS.json'; preds = read(path) if path.exists() else []
        if len(preds) > len(rows) or (len(preds) % 4 and len(preds) != len(rows)) or [q['question_id'] for q in preds] != [q['question_id'] for q in rows[:len(preds)]]:
            raise ValueError('Whole-batch resume differs')
        for start in range(len(preds), len(rows), 4):
            chunk = rows[start:start + 4]; ocr = name in ('literal_validation', 'official_test')
            answers = generate(model, processor, chunk, data) if ocr else generated(model, processor, chunk, data)
            for q, a in zip(chunk, answers):
                if a['input_sha256'] != q['inference_input_sha256']:
                    raise ValueError('Actual generated prefix differs')
                if ocr:
                    a.update(iiit_alnum_exact=alphanumeric(a['prediction']) == alphanumeric(q['target']), filename_family=q['filename_family'])
            preds.extend(answers); temporary = path.with_suffix('.tmp'); write_json(temporary, preds); temporary.replace(path)
            write_json(directory / 'PROGRESS.json', {'status': 'evaluating', 'cohort': name, 'completed': len(preds), 'expected': len(rows), 'active_seconds': time.monotonic() - started})
            if time.monotonic() - started > cap or torch.cuda.max_memory_allocated() / 2**30 > 65:
                raise RuntimeError('Evaluation cap reached')
    if lora_hash(model) != before:
        raise ValueError('Read-only evaluation changed model')


def reference():
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_sft.common import to_device
    lock = verify(); model, processor, _ = actor(adapter=read(CONFIG)['initial_adapter'])
    if lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Reference must be OCR-enhanced policy')
    evaluate(model, processor, ROOT / 'baseline_hard', ('native_hard',))
    path = ROOT / 'baseline/native_hard/PREDICTIONS.json'; write_json(path, read(ROOT / 'baseline_hard/native_hard/PREDICTIONS.json'))
    model.eval(); model.config.use_cache = False; data = read(ROOT / 'data/MANIFEST.json'); rows = {q['question_id']: q for q in data['rows']}; cpu = read(ROOT / 'data/CPU_INPUTS.json')
    cache = {'reference_lora_sha256': lock['initial_lora_sha256'], 'batches': {}}; memo = {}; started = time.monotonic()
    with torch.inference_mode():
        for step, update in enumerate(read(ROOT / 'data/SCHEDULE.json')['updates']):
            for micro, ids in enumerate(update['micros']):
                chunk = [rows[qid] for qid in ids]
                if chunk[0]['study_source'] != 'preference':
                    continue
                batch, binding, _, _ = make_batch(processor, data, chunk, cpu); key = content_sha(binding)
                if key not in memo:
                    batch = to_device(batch, next(model.parameters()).device); logps, counts = sequence_logps(model, batch)
                    if not bool(torch.isfinite(logps).all()):
                        raise ValueError('Nonfinite reference log probabilities')
                    memo[key] = {'binding': binding, 'chosen_logps': logps[:4].cpu().tolist(), 'rejected_logps': logps[4:].cpu().tolist(), 'assistant_counts': counts.cpu().tolist()}
                    del batch, logps, counts
                cache['batches'][f'{step + 1}:{micro}'] = memo[key]
                write_json(ROOT / 'reference/PROGRESS.json', {'status': 'reference_cache', 'completed_batches': len(cache['batches']), 'expected': 64, 'unique_forward_batches': len(memo), 'active_seconds': time.monotonic() - started})
                if time.monotonic() - started > 600:
                    raise TimeoutError('Reference cap')
    if len(cache['batches']) != 64 or lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Reference identity/count mismatch')
    write_json(ROOT / 'reference/CACHE.json', cache)
    write_json(ROOT / 'reference/INPUT_LOCK.json', {str(p): sha_file(p) for p in (ROOT / 'reference/CACHE.json', path)})
    write_json(ROOT / 'reference/COMPLETE.json', {'status': 'completed', 'forward_batches': len(memo), 'new_baseline_qa': 128, 'optimizer_updates': 0})


def condition(role, mode='internal'):
    from vlm_human_sft.evaluate import lora_hash
    verify(reference=True); directory = ROOT / 'runs' / role
    if mode == 'internal':
        trainer = importlib.import_module('vlm_literal_preference_v1.train'); trainer.ROOT = ROOT; trainer.CONFIG = CONFIG
        model, processor, result = trainer.train(role)
        if lora_hash(model) != result['final_lora_sha256']:
            raise ValueError('Wrong Final32')
        evaluate(model, processor, directory / 'internal', INTERNAL)
    else:
        result = read(directory / 'train/TRAIN_COMPLETE.json'); model, processor, _ = actor(adapter=result['adapter'])
        if lora_hash(model) != result['final_lora_sha256']:
            raise ValueError('Wrong frozen Final32')
        if mode == 'transfer':
            evaluate(model, processor, directory / 'transfer', ('transfer_dev',))
        elif mode == 'test':
            binding = read(OLD / 'TEST_PLAN.json')
            if sha_file(OLD / 'TEST_DATA.json') != binding['data_sha256']:
                raise ValueError('Repeated test metadata changed')
            for path, digest in binding['test_images_sha256'].items():
                if sha_file(path) != digest:
                    raise ValueError('Repeated test image changed')
            evaluate(model, processor, directory / 'test', ('official_test',), read(OLD / 'TEST_DATA.json'))
        else:
            raise ValueError('Unknown mode')
    write_json(directory / (mode + '_COMPLETE.json'), {'status': 'completed', 'role': role, 'mode': mode, 'updates': 32, 'model_lora_sha256': result['final_lora_sha256']})


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT / 'data/MANIFEST.json'); names = list(INTERNAL) + (['transfer_dev'] if transfer else [])
    records = {'start': {n: read(ROOT / 'baseline' / n / 'PREDICTIONS.json') for n in names}}
    for role in ROLES:
        records[role] = {n: read(ROOT / 'runs' / role / ('transfer' if n == 'transfer_dev' else 'internal') / n / 'PREDICTIONS.json') for n in names}
    meta = data['evaluations']['seen_validation']
    for rec in records.values():
        for group in ('singleword', 'multiword', 'rare_1_5'):
            ids = {q['question_id'] for q in meta if (q['stratum'] == group if group != 'rare_1_5' else 1 <= q['answer_frequency'] <= 5)}
            rec[group] = [q for q in rec['seen_validation'] if q['question_id'] in ids]
    def norm(rows, n):
        return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows] if n == 'literal_validation' else rows
    metrics = {r: {n: 100 * sum(q['vqa_score'] for q in norm(rows, n)) / len(rows) for n, rows in rec.items()} for r, rec in records.items()}
    comparisons = {r: {n: paired(norm(records['start'][n], n), norm(rows, n)) for n, rows in records[r].items()} for r in ROLES}
    gates = {}; checks = {}
    for r in ROLES:
        changes = comparisons[r]
        checks[r] = {'hard_learning': sum(q['vqa_score'] >= .999999 for q in records[r]['native_hard']) >= 32 and changes['native_hard']['delta_pp'] >= 20,
                     'ocr_point_protection': changes['literal_validation']['delta_pp'] >= -.5,
                     'native_point_protections': all(changes[n]['delta_pp'] >= -.5 for n in ('seen_validation', 'retention', 'singleword', 'multiword', 'rare_1_5'))}
        gates[r] = all(checks[r].values())
    result = {'status': 'completed', 'metrics': metrics, 'vs_start': comparisons, 'screen_checks': checks, 'screen_passed': gates,
              'same_data_dpo_vs_sft': {n: paired(norm(records['sft'][n], n), norm(records['dpo_sft'][n], n)) for n in records['sft']},
              'goal_complete': False, 'scope': 'Fixed single-seed32-update pilot; native training images originally SFT-seen, IIIT images trained in parent study; historically used internal evaluation.'}
    if transfer:
        result['native_positive_signal'] = {r: gates[r] and comparisons[r]['transfer_dev']['delta_pp'] >= .5 and comparisons[r]['transfer_dev']['ci95'][0] > 0 for r in ROLES}
        original = read(BASE / 'vlm_iiit_native_transfer_v1/runs/initial/PREDICTIONS.json')
        result['transfer_vs_original_sft'] = {r: paired(original, records[r]['transfer_dev']) for r in ROLES}
        result['dpo_native_advantage_over_sft'] = result['same_data_dpo_vs_sft']['transfer_dev']['delta_pp'] >= .5 and result['same_data_dpo_vs_sft']['transfer_dev']['ci95'][0] > 0
    write_json(ROOT / ('TRANSFER_ANALYSIS.json' if transfer else 'SCREEN_ANALYSIS.json'), result); render(result); return result


def render(result=None):
    data = read(ROOT / 'data/MANIFEST.json'); rel = lambda p: quote(os.path.relpath(p, PAGE.parent), safe='/._-'); cards = []
    for q in data['evaluations']['native_hard'][:8]:
        answers = ''
        for role in ROLES:
            path = ROOT / 'runs' / role / 'internal/native_hard/PREDICTIONS.json'
            if path.exists():
                a = next(a for a in read(path) if a['question_id'] == q['question_id']); answers += f'<p>{role}: {html.escape(a["prediction"])} / VQA {a["vqa_score"]}</p>'
        cards.append(f'<article><img src="{rel(data["images"][q["image_id"]]["path"])}" alt="原生问答训练图"><p>{html.escape(q["question"])}</p><p>正确监督：{html.escape(q["target"])}；起点实际错误：{html.escape(q["rejected_target"])}</p>{answers}</article>')
    status = '尚无候选结果；8图为训练病例，不能当泛化。' if result is None else html.escape(str(result['metrics']))
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实问答SFT与DPO</title><style>body{max-width:1050px;margin:30px auto;font:17px/1.7 system-ui;padding:0 20px;background:#f4f7fb;color:#182638}article{background:white;padding:18px;margin:18px 0;border:1px solid #cbd5e1}img{max-width:100%;max-height:500px}pre{white-space:pre-wrap;background:#e5edf6;padding:15px}</style><h1>用真实问答纠错比较SFT与DPO</h1><p>'+status+'</p><pre>L_SFT = 逐QA的有效assistant token平均负log概率\nΔ = [logπθ(y+|x) − logπθ(y−|x)] − [logπref(y+|x) − logπref(y−|x)]\nL_DPO = −log sigmoid(0.1 × Δ)\n候选：相同CE + 偏好QA上的DPO；控制：相同CE\n每步8纠错QA +4 native回放+4 OCR回放；32步、512 chosen曝光。</pre><p>正负序列log概率求sum，CE按token平均。参考固定为OCR增强起点；负例保留生成token IDs，含真实EOS，不解码再分词。DPO仍依赖偏好可靠性，不保证提高视觉能力。两个模型都有相同负序列forward，避免把额外算力误称算法优势。训练原图曾在SFT中曝光，当前内部评测历史用过，不能称盲测。</p><p><a href="'+rel(PROTOCOL)+'">冻结协议</a> · <a href="'+rel(Path(__file__))+'">源码</a></p>'+''.join(cards)+'</html>')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        if args.worker == 'reference': reference()
        else:
            mode, role = args.worker.split(':'); condition(role, mode)
        return
    from research_runtime.queue import run_queue
    verify(); identity = {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')}; entry = str(Path(__file__).resolve())
    def job(mode, role):
        return {'key': role, 'command': [sys.executable, '-u', entry, '--worker', mode + ':' + role], 'completion_marker': str(ROOT / 'runs' / role / (mode + '_COMPLETE.json'))}
    run_queue(ROOT / 'reference_queue', [{'key': 'reference', 'command': [sys.executable, '-u', entry, '--worker', 'reference'], 'completion_marker': str(ROOT / 'reference/COMPLETE.json')}], identity)
    run_queue(ROOT / 'training_queue', [job('internal', r) for r in ROLES], identity); screen = analyze()
    if not any(screen['screen_passed'].values()):
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'screen_negative_no_expansion', 'new_updates': 64, 'new_generations': 2048, 'goal_complete': False}); return
    run_queue(ROOT / 'transfer_queue', [job('transfer', r) for r in ROLES], identity); result = analyze(transfer=True)
    passing = [r for r, ok in result['native_positive_signal'].items() if ok and result['transfer_vs_original_sft'][r]['delta_pp'] >= .5 and result['transfer_vs_original_sft'][r]['ci95'][0] > 0]
    if passing:
        run_queue(ROOT / 'repeated_test_queue', [job('test', r) for r in passing], identity)
        from vlm_grpo_iteration_v1.study import paired
        before = read(OLD / 'official_test/sparse_language/official_test/PREDICTIONS.json')
        norm = lambda rows: [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows]
        result['repeated_ocr_test'] = {r: paired(norm(before), norm(read(ROOT / 'runs' / r / 'test/official_test/PREDICTIONS.json'))) for r in passing}
        result['local_native_gain_and_ocr_noninferiority'] = {r: p['delta_pp'] >= -.5 and p['ci95'][0] > -.5 for r, p in result['repeated_ocr_test'].items()}
    write_json(ROOT / 'COMPLETION.json', {**result, 'phase': 'pilot_complete', 'new_updates': 64, 'new_generations': 5282 + 3000 * len(passing), 'automatic_more_training': False})


if __name__ == '__main__':
    main()
