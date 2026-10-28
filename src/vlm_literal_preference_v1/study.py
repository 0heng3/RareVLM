import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor
from .common import ROOT, CONFIG, read, sample, make_batch

BASE = ROOT.parent
MINING = BASE/'vlm_literal_roi_v1'
PROTOCOL = PROJECT_ROOT/'docs/VLM_LITERAL_PREFERENCE_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT/'docs/VLM_LITERAL_PREFERENCE_V1_RESULTS.md'
ROLES = ('sft', 'dpo_sft')


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Study frozen')
    review_path = MINING/'REVIEWED_ERRORS.json'
    review = read(review_path)
    if review.get('status') != 'passed' or len(review['accepted_question_ids']) < 128:
        raise ValueError('Need128 visually reviewed official word errors')
    raw = read(MINING/'DATA.json')
    candidates = {q['question_id']: q for q in read(MINING/'ERROR_CANDIDATES.json')['rows']}
    selected = [candidates[qid] for qid in review['accepted_question_ids'][:128]]
    if len({q['source_image_id'] for q in selected}) != 128 or len({q['target'].casefold() for q in selected}) != 128:
        raise ValueError('128 distinct source images/words required')
    hard = [{**q, 'study_source': 'preference', 'rejected_target': q['initial_prediction']['prediction'],
             'rejected_completion_ids': q['initial_prediction']['completion_ids']} for q in selected]
    old = read(BASE/'vlm_reference_replay_v1/DATA.json')
    hard_ids = {q['question_id'] for q in old['hard_rows']}
    replay = [q for q in old['training_rows'] if q['question_id'] not in hard_ids]
    if len(replay) != 512:
        raise ValueError('Established512 replay rows differ')
    replay = [{**q, 'study_source': 'common'} for q in sorted(replay, key=lambda q: content_sha(['literal_preference:replay', q['question_id']]))[:256]]
    evaluations = {name: old['evaluations'][name] for name in ('seen_validation', 'retention', 'transfer_dev')}
    evaluations.update(hard=hard, literal_validation=[q for q in raw['rows'] if q['fold'] == 'validation'])
    data = {'rows': replay+hard, 'images': {**old['images'], **raw['images']}, 'evaluations': evaluations}
    parent = read(BASE/'vlm_reference_replay_v1/PLAN.json')
    c = {**read(PROJECT_ROOT/'configs/vlm_error_supervision_sft_v1.json'), 'output_root': str(ROOT),
         'lr': 2e-4, 'training_cap_seconds': 1800, 'evaluation_cap_seconds': 1500,
         'maximum_reference_seconds': 600, 'automatic_main_evaluation': True}
    write_json(CONFIG, c)
    torch.set_num_threads(2)
    processor = processor_for(Path(parent['model_root']), read(parent['reference_config']), 1024)
    cpu = {}
    for q in data['rows']:
        for kind in ('chosen', 'rejected') if q['study_source'] == 'preference' else ('chosen',):
            item = sample(processor, data, q, kind == 'rejected')
            if item['input_ids'].shape[1] > 2048:
                raise ValueError('Training sequence above fixed limit')
            cpu[q['question_id']+':'+kind] = {key+'_sha256': content_sha(item[key].tolist()) for key in ('input_ids', 'labels')}
    updates = []
    h = [q['question_id'] for q in hard]
    r = [q['question_id'] for q in replay]
    for step in range(32):
        j = (8*step)%128
        updates.append({'micros': [r[8*step:8*step+4], h[j:j+4], r[8*step+4:8*step+8], h[j+4:j+8]]})
    counts = Counter(q for update in updates for micro in update['micros'] for q in micro)
    if any(counts[qid] != 2 for qid in h) or any(counts[qid] != 1 for qid in r):
        raise ValueError('Source exposure schedule differs')
    write_json(ROOT/'data/MANIFEST.json', data)
    write_json(ROOT/'data/CPU_INPUTS.json', cpu)
    write_json(ROOT/'data/SCHEDULE.json', {'updates': updates})
    lookup = {q['question_id']: q for q in read(MINING/'PREDICTIONS.json')}
    write_json(ROOT/'baseline/literal_validation/PREDICTIONS.json', [lookup[q['question_id']] for q in evaluations['literal_validation']])
    for name in ('seen_validation', 'retention'):
        before = read(BASE/'vlm_reference_replay_v1/runs/initial'/name/'PREDICTIONS.json')
        if [q['question_id'] for q in before] != [q['question_id'] for q in evaluations[name]]:
            raise ValueError('Initial reused VQA frame/order differs')
        write_json(ROOT/'baseline'/name/'PREDICTIONS.json', before)
    paths = [CONFIG, PROTOCOL, review_path, ROOT/'data/MANIFEST.json', ROOT/'data/CPU_INPUTS.json', ROOT/'data/SCHEDULE.json',
             PROJECT_ROOT/'tools/run_literal_preference_v1.py']+sorted(Path(__file__).parent.glob('*.py'))
    files = {**read(MINING/'PLAN.json')['files'], **{str(p): sha_file(p) for p in paths}}
    write_json(ROOT/'STUDY_INPUT_LOCK.json', {'status': 'prepared', 'files': files,
               'initial_lora_sha256': read(MINING/'PLAN.json')['lora_sha256'], 'packages': parent['packages'],
               'base_weight_files': parent['base_weight_files'], 'training_qa': 384, 'updates_per_role': 32,
               'stage1_generation_per_role': 960, 'reference_initial_hard_generation': 128,
               'baseline_reuse': {'literal_validation': 256, 'seen_validation': 512, 'retention': 64},
               'new_training_images_seen_in_original_sft': True})
    RESULT.write_text('# 完整单词SFT与DPO\n\n已准备128可靠词纠错+256回放，同起点32步两臂。尚无新训练能力结果。\n')
    print('PREPARED_LITERAL_PREFERENCE', {'hard': 128, 'replay': 256, 'updates_per_role': 32}, flush=True)


def verify(reference=False):
    from vlm_object_train.train import package_lock
    lock = read(ROOT/'STUDY_INPUT_LOCK.json')
    for path, digest in lock['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen input/source differs: '+path)
    if package_lock() != lock['packages']:
        raise ValueError('Package versions differ')
    for path, expected in lock['base_weight_files'].items():
        st = Path(path).stat()
        if expected != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
            raise ValueError('Base storage differs')
    if reference:
        for path, digest in read(ROOT/'reference/INPUT_LOCK.json').items():
            if sha_file(path) != digest:
                raise ValueError('Reference/baseline changed')
    return lock


def evaluate(model, processor, directory, names=('hard', 'literal_validation', 'seen_validation', 'retention')):
    from run_literal_tile_v1 import generate
    from vlm_fit_positive_control_v1.study import generated
    data = read(ROOT/'data/MANIFEST.json')
    model.eval()
    model.config.use_cache = True
    started = time.monotonic()
    for name in names:
        rows = data['evaluations'][name]
        path = directory/name/'PREDICTIONS.json'
        predictions = read(path) if path.exists() else []
        if (len(predictions) > len(rows) or (len(predictions)%4 and len(predictions) != len(rows))
                or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]):
            raise ValueError('Evaluation whole-batch cache differs')
        for begin in range(len(predictions), len(rows), 4):
            chunk = rows[begin:begin+4]
            answers = generate(model, processor, chunk, data) if name in ('hard', 'literal_validation') else generated(model, processor, chunk, data)
            predictions.extend(answers)
            temp = path.with_suffix('.tmp')
            write_json(temp, predictions)
            temp.replace(path)
            write_json(directory/'EVAL_PROGRESS.json', {'status': 'evaluating', 'cohort': name,
                       'completed_qa': len(predictions), 'expected_qa': len(rows), 'active_seconds': time.monotonic()-started})
            if time.monotonic()-started > 1500:
                raise TimeoutError('Evaluation cumulative active budget exceeded')


def reference():
    import torch
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sft.common import to_device
    lock = verify()
    model, processor, _ = actor(adapter=read(CONFIG)['initial_adapter'])
    if lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Reference must be exact original SFT')
    evaluate(model, processor, ROOT/'baseline', names=('hard',))
    model.eval()
    model.config.use_cache = False
    data = read(ROOT/'data/MANIFEST.json')
    lookup = {q['question_id']: q for q in data['rows']}
    cpu = read(ROOT/'data/CPU_INPUTS.json')
    cache = {'reference_lora_sha256': lock['initial_lora_sha256'], 'batches': {}}
    memo = {}
    started = time.monotonic()
    with torch.inference_mode():
        for step, update in enumerate(read(ROOT/'data/SCHEDULE.json')['updates']):
            for micro, ids in enumerate(update['micros']):
                rows = [lookup[qid] for qid in ids]
                if rows[0]['study_source'] != 'preference':
                    continue
                batch, binding, _, _ = make_batch(processor, data, rows, cpu)
                key = content_sha(binding)
                if key not in memo:
                    gpu = to_device(batch, next(model.parameters()).device)
                    values, counts = sequence_logps(model, gpu)
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError('Nonfinite reference logps')
                    memo[key] = {'binding': binding, 'chosen_logps': values[:4].cpu().tolist(),
                                 'rejected_logps': values[4:].cpu().tolist(), 'assistant_counts': counts.cpu().tolist()}
                    del gpu, values, counts
                cache['batches'][f'{step+1}:{micro}'] = memo[key]
                write_json(ROOT/'reference/CACHE.json', cache)
                write_json(ROOT/'reference/PROGRESS.json', {'status': 'reference_cache', 'completed_batches': len(cache['batches']),
                           'expected_batches': 64, 'unique_forward_batches': len(memo), 'active_seconds': time.monotonic()-started})
                if time.monotonic()-started > 600:
                    raise TimeoutError('Reference cache budget')
    if len(cache['batches']) != 64 or lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Reference coverage/weights differ')
    write_json(ROOT/'reference/COMPLETE.json', {'status': 'completed', 'cache_batches': 64,
               'actual_unique_forward_batches': len(memo), 'actual_reference_sequences': len(memo)*8,
               'active_seconds': time.monotonic()-started, 'new_optimizer_updates': 0})
    paths = [ROOT/'reference/CACHE.json', ROOT/'reference/COMPLETE.json']
    paths += [ROOT/'baseline'/n/'PREDICTIONS.json' for n in ('hard', 'literal_validation', 'seen_validation', 'retention')]
    write_json(ROOT/'reference/INPUT_LOCK.json', {str(p): sha_file(p) for p in paths})


def condition(role, transfer=False):
    from vlm_human_sft.evaluate import lora_hash
    verify(True)
    if transfer:
        result = read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json')
        model, processor, _ = actor(adapter=result['adapter'])
        evaluate(model, processor, ROOT/'runs'/role, names=('transfer_dev',))
    else:
        from .train import train
        model, processor, result = train(role)
        evaluate(model, processor, ROOT/'runs'/role)
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Evaluation changed Final')
    write_json(ROOT/'runs'/role/('TRANSFER_COMPLETE.json' if transfer else 'COMPLETE.json'),
               {'status': 'completed', 'updates': 32, 'final_lora_sha256': result['final_lora_sha256']})


def score(rows, literal=False):
    key = 'literal_exact' if literal else 'vqa_score'
    return {'qa': len(rows), 'score_percent': 100*sum(float(q[key]) for q in rows)/len(rows),
            'full': sum(float(q[key]) >= .999999 for q in rows)}


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    names = ('hard', 'literal_validation', 'seen_validation', 'retention')
    records = {}
    metrics = {}
    for role in ('initial',)+ROLES:
        directory = ROOT/'baseline' if role == 'initial' else ROOT/'runs'/role
        records[role] = {n: read(directory/n/'PREDICTIONS.json') for n in names}
        metrics[role] = {n: score(rows, n in ('hard', 'literal_validation')) for n, rows in records[role].items()}
    def delta(a, b, name):
        def normalized(rows):
            return [{**q, 'vqa_score': float(q['literal_exact']) if name in ('hard', 'literal_validation') else q['vqa_score']} for q in rows]
        return paired(normalized(records[a][name]), normalized(records[b][name]))
    checks = {}
    comparisons = {}
    data = read(ROOT/'data/MANIFEST.json')
    seen_meta = {q['question_id']: q for q in data['evaluations']['seen_validation']}
    for role in ROLES:
        changes = {n: delta('initial', role, n) for n in names}
        protections = {}
        for stratum in ('singleword', 'multiword'):
            before = [q for q in records['initial']['seen_validation'] if seen_meta[q['question_id']]['stratum'] == stratum]
            after = [q for q in records[role]['seen_validation'] if seen_meta[q['question_id']]['stratum'] == stratum]
            protections[stratum] = paired(before, after)
        rare_ids = {qid for qid, q in seen_meta.items() if q.get('frequency_slice') == 'rare_1_5' or q.get('answer_frequency', 100) <= 5}
        before = [q for q in records['initial']['seen_validation'] if q['question_id'] in rare_ids]
        after = [q for q in records[role]['seen_validation'] if q['question_id'] in rare_ids]
        if not before:
            raise ValueError('Missing registered native rare slice')
        protections['rare_1_5'] = paired(before, after)
        c = {'literal_gain': changes['literal_validation']['delta_pp'] >= 3 and changes['literal_validation']['ci95'][0] > 0,
             'native_gain': changes['seen_validation']['delta_pp'] >= .5 and changes['seen_validation']['ci95'][0] > 0,
             'retention': changes['retention']['delta_pp'] >= -.5,
             'training_net32': metrics[role]['hard']['full']-metrics['initial']['hard']['full'] >= 32,
             'native_slices': all(v['delta_pp'] >= -.5 for v in protections.values())}
        checks[role] = c
        comparisons[role] = {'changes': changes, 'native_protection_slices': protections}
    passed = {r: all(c.values()) for r, c in checks.items()}
    extra = {n: delta('sft', 'dpo_sft', n) for n in ('literal_validation', 'seen_validation')}
    dpo_extra = passed['dpo_sft'] and all(v['delta_pp'] >= .5 and v['ci95'][0] > 0 for v in extra.values())
    result = {'status': 'completed', 'metrics': metrics, 'comparisons': comparisons, 'checks': checks,
              'stage1_passed': passed, 'dpo_vs_sft': extra, 'dpo_extra_advantage': dpo_extra,
              'scope': 'Single-seed given-region OCR/full-image internal VQA screen; no seed stability or main-task success.',
              'goal_complete': False, 'optimizer_updates': 64, 'automatic_more_training': False}
    if transfer:
        result['transfer_comparisons'] = {}
        initial = read(BASE/'vlm_reference_replay_v1/runs/initial/transfer_dev/PREDICTIONS.json')
        strong = read(BASE/'vlm_reference_replay_v1/runs/hard_replay/transfer_dev/PREDICTIONS.json')
        finals = {r: read(ROOT/'runs'/r/'transfer_dev/PREDICTIONS.json') for r in ROLES}
        for role in ROLES:
            result['transfer_comparisons'][role] = {'vs_initial': paired(initial, finals[role]), 'vs_strong_qv_replay': paired(strong, finals[role])}
        result['transfer_dpo_vs_sft'] = paired(finals['sft'], finals['dpo_sft'])
        result['next'] = 'Report all original main-task/rare protections and matched/strong baselines before any seed expansion.'
    else:
        result['next'] = 'Original1617 transfer admitted only if a stage1 arm passes; otherwise stop both configurations.'
    write_json(ROOT/('ANALYSIS.json' if transfer else 'SCREEN.json'), result)
    table = '|模型|128训练满分|256词验证exact%|512原生VQA%|64保持VQA%|\n|---|---:|---:|---:|---:|\n'
    for role, m in metrics.items():
        table += f'|{role}|{m["hard"]["full"]}|{m["literal_validation"]["score_percent"]:.6f}|{m["seen_validation"]["score_percent"]:.6f}|{m["retention"]["score_percent"]:.6f}|\n'
    RESULT.write_text('# 完整单词SFT与DPO：结果\n\n'+table+'\n阶段一门槛：'+str(passed)+'；DPO对同数据SFT额外优势='+str(dpo_extra)+'。\n\n'+result['scope']+'\n\n[协议](VLM_LITERAL_PREFERENCE_V1_PROTOCOL.md)\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.worker:
        if args.worker == 'reference':
            reference()
        elif args.worker.startswith('transfer:'):
            condition(args.worker.split(':')[1], True)
        elif args.worker in ROLES:
            condition(args.worker)
        else:
            raise ValueError('Unknown worker')
        return
    from research_runtime.queue import run_queue
    verify()
    entry = PROJECT_ROOT/'tools/run_literal_preference_v1.py'
    def job(key, arg, marker):
        return {'key': key, 'command': [sys.executable, '-u', str(entry), '--worker', arg], 'completion_marker': str(marker)}
    run_queue(ROOT/'reference_queue', [job('reference', 'reference', ROOT/'reference/COMPLETE.json')],
              {'study_sha256': sha_file(ROOT/'STUDY_INPUT_LOCK.json')})
    run_queue(ROOT/'training_queue', [job(r, r, ROOT/'runs'/r/'COMPLETE.json') for r in ROLES],
              {'study_sha256': sha_file(ROOT/'STUDY_INPUT_LOCK.json'), 'reference_sha256': sha_file(ROOT/'reference/INPUT_LOCK.json')})
    result = analyze()
    if any(result['stage1_passed'].values()):
        run_queue(ROOT/'transfer_queue', [job('transfer_'+r, 'transfer:'+r, ROOT/'runs'/r/'TRANSFER_COMPLETE.json') for r in ROLES],
                  {'study_sha256': sha_file(ROOT/'STUDY_INPUT_LOCK.json'), 'screen_sha256': sha_file(ROOT/'SCREEN.json')})
        result = analyze(True)
    write_json(ROOT/'COMPLETION.json', result)
    print('LITERAL_PREFERENCE_COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
