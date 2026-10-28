"""Test task-local forgetting before one fixed-budget replay-distribution SFT."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, chosen_sample, read
from run_literal_tile_v1 import canonical, generate
import run_literal_case_sft_v1 as core

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_literal_task_replay_v1'
OLD = BASE / 'vlm_literal_case_sft_v1'
ROI = BASE / 'vlm_literal_roi_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_LITERAL_TASK_REPLAY_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_LITERAL_TASK_REPLAY_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/任务匹配回放_保住已有读词能力.html'
ROLE = 'canonical_case'


def training_row(q, role):
    return {**q, 'target': canonical(q['target'])} if q['study_source'] == 'preference' or q.get('task_replay') else q


# Reuse the frozen, tested loop; only the declared data root and label policy bind.
core.ROOT = ROOT
core.training_row = training_row


def audit():
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    if (ROOT / 'AUDIT_CANDIDATES.json').exists():
        raise FileExistsError('Audit already frozen')
    data = read(ROI / 'DATA.json')
    predictions = {q['question_id']: q for q in read(ROI / 'PREDICTIONS.json')}
    eligible = [q for q in data['rows'] if q['fold'] == 'training_pool'
                and predictions[q['question_id']]['literal_exact'] and predictions[q['question_id']]['terminated']
                and len(q['target']) >= 5 and q['original_crop_size'][0] >= 60 and q['original_crop_size'][1] >= 24]
    eligible.sort(key=lambda q: content_sha(['literal_task_replay_v1:audit', q['source_image_id'], q['annotation_id']]))
    if len(eligible) < 192:
        raise ValueError('Need192 training-only correct candidates')
    records = [{**q, 'ordinal': i, 'image': data['images'][q['image_id']]} for i, q in enumerate(eligible[:192])]
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    boards = []
    for number in range(12):
        canvas = Image.new('RGB', (1040, 960), '#eef2f6'); draw = ImageDraw.Draw(canvas)
        for j, q in enumerate(records[number * 16:number * 16 + 16]):
            x = (j % 2) * 520; y = (j // 2) * 120
            draw.text((x + 8, y + 4), f'#{q["ordinal"]} official: {q["target"]}', font=font, fill='black')
            if sha_file(q['image']['path']) != q['image']['sha256']:
                raise ValueError('Source crop changed')
            image = ImageOps.contain(Image.open(q['image']['path']).convert('RGB'), (500, 80), Image.Resampling.BICUBIC)
            canvas.paste(image, (x + 8 + (500 - image.width) // 2, y + 32 + (80 - image.height) // 2))
        path = ROOT / 'audit' / f'board_{number:02d}.png'; path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path); boards.append(str(path))
    write_json(ROOT / 'AUDIT_CANDIDATES.json', {'records': records, 'boards': boards, 'eligible': len(eligible),
        'initial_correctness_used': True, 'old_final_predictions_used': False, 'protocol_sha256': sha_file(PROTOCOL)})
    print('AUDIT_READY', len(eligible), flush=True)


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Plan already frozen')
    started = time.monotonic(); torch.set_num_threads(2)
    candidates = read(ROOT / 'AUDIT_CANDIDATES.json'); review = read(ROOT / 'AUDIT_REVIEW.json')
    if review['status'] != 'passed' or len(review['selected_ordinals']) != 128 or review['old_final_predictions_used']:
        raise ValueError('Need128 reviewed training-only words')
    selected = [candidates['records'][i] for i in review['selected_ordinals']]
    data = read(OLD / 'DATA.json'); old_rows = {q['question_id']: q for q in data['rows']}
    # The second native micro was a fixed, disjoint half of the256 replay catalog.
    schedule = read(OLD / 'SCHEDULE.json')
    removed = list(dict.fromkeys(qid for u in schedule['updates'] for qid in u['micros'][2]))
    if len(removed) != 128 or any(old_rows[qid]['study_source'] != 'common' for qid in removed):
        raise ValueError('Expected128 distinct second-native-micro questions')
    replacement = dict(zip(removed, [q['question_id'] for q in selected]))
    retained = [q for q in data['rows'] if q['question_id'] not in replacement]
    replay = []
    for q in selected:
        q = dict(q); im = q.pop('image'); q.update(study_source='common', task_replay=True, fold='training_replay')
        data['images'][q['image_id']] = im; replay.append(q)
    blocked = data['evaluations']['hard'] + data['evaluations']['clean_literal'] + data['evaluations']['literal_validation']
    for key in ('source_image_id', 'source_pixel_sha256'):
        if {q[key] for q in replay} & {q[key] for q in blocked}:
            raise ValueError('Replay source overlap with hard/validation')
    if {canonical(q['target']) for q in replay} & {canonical(q['target']) for q in blocked}:
        raise ValueError('Replay word overlap with hard/validation')
    data['rows'] = retained + replay; data['evaluations']['replay_literal'] = replay
    schedule['updates'] = [{'micros': [[replacement.get(qid, qid) for qid in micro] for micro in u['micros']]} for u in schedule['updates']]
    exposures = Counter(qid for u in schedule['updates'] for micro in u['micros'] for qid in micro)
    if len(exposures) != 384 or any(exposures[q['question_id']] != (8 if q['study_source'] == 'preference' else 4) for q in data['rows']):
        raise ValueError('Per-question budget changed')
    original_cpu = read(OLD / 'CPU_INPUTS.json')[ROLE]
    cpu = {q['question_id']: original_cpu[q['question_id']] for q in retained}
    parent = read(BASE / 'vlm_reference_replay_v1/PLAN.json')
    processor = processor_for(Path(parent['model_root']), read(parent['reference_config']), 1024)
    for q in replay:
        item = chosen_sample(processor, training_row(q, ROLE), data['images'])
        if item['input_ids'].shape[1] > 2048 or content_sha(item['input_ids'][:, :q['expected_prefix_tokens']].tolist()) != q['inference_input_sha256']:
            raise ValueError('New chosen sample has changed ROI prefix or exceeds cap')
        cpu[q['question_id']] = {k: content_sha(item[k].tolist()) for k in ('input_ids', 'labels')}
        cpu[q['question_id']]['assistant_ids'] = item['labels'][item['labels'].ne(-100)].tolist()
    initial = {q['question_id']: q for q in read(ROI / 'PREDICTIONS.json')}
    write_json(ROOT / 'DATA.json', data); write_json(ROOT / 'CPU_INPUTS.json', {ROLE: cpu}); write_json(ROOT / 'SCHEDULE.json', schedule)
    write_json(ROOT / 'REPLACEMENTS.json', replacement)
    write_json(ROOT / 'baseline/replay_literal/PREDICTIONS.json', [initial[q['question_id']] for q in replay])
    for name in ('hard', 'clean_literal', 'literal_validation', 'seen_validation', 'retention'):
        write_json(ROOT / 'baseline' / name / 'PREDICTIONS.json', read(OLD / 'baseline' / name / 'PREDICTIONS.json'))
    plan = read(OLD / 'PLAN.json'); files = dict(plan['files'])
    paths = [Path(__file__), PROTOCOL, ROOT / 'AUDIT_CANDIDATES.json', ROOT / 'AUDIT_REVIEW.json',
             ROOT / 'DATA.json', ROOT / 'CPU_INPUTS.json', ROOT / 'SCHEDULE.json', ROOT / 'REPLACEMENTS.json',
             ROI / 'DATA.json', ROI / 'PREDICTIONS.json', OLD / 'PLAN.json', OLD / 'runs' / ROLE / 'train/TRAIN_COMPLETE.json']
    paths += list((ROOT / 'baseline').glob('*/PREDICTIONS.json'))
    paths += list((OLD / 'runs' / ROLE).glob('*/PREDICTIONS.json'))
    for path in paths:
        files[str(path)] = sha_file(path)
    plan.update(files=files, new_image_files={**plan['new_image_files'], **{data['images'][q['image_id']]['path']: data['images'][q['image_id']]['sha256'] for q in replay}},
        roles=[ROLE], active_prepare_seconds=time.monotonic()-started, new_initial_generations=0,
        generation_per_role=1344, diagnostic_generations=128, diagnostic_required_losses=4,
        assistant_tokens_per_unique_training_catalog={domain: sum(len(cpu[q['question_id']]['assistant_ids']) for q in data['rows'] if ('literal' if q.get('task_replay') else q['study_source']) == domain) for domain in ('literal','common','preference')},
        matched_control=str(OLD / 'runs' / ROLE), candidate='task_matched_replay', original_sft_images_seen=True)
    write_json(ROOT / 'PLAN.json', plan); render(); print('PREPARED', {'retained_exact_cpu': len(retained), 'new_crop_replay': len(replay), 'updates': 128, 'seconds': plan['active_prepare_seconds']}, flush=True)


def eval_replay(model, processor, directory):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    data = read(ROOT / 'DATA.json'); rows = data['evaluations']['replay_literal']; model.eval(); model.config.use_cache = True
    lock_json(directory / 'INPUT_LOCK.json', {'model_lora_sha256': lora_hash(model), 'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    path = directory / 'PREDICTIONS.json'; preds = read(path) if path.exists() else []
    if len(preds) > 128 or len(preds) % 4 or [q['question_id'] for q in preds] != [q['question_id'] for q in rows[:len(preds)]]:
        raise ValueError('Replay evaluation resume changed')
    started = time.monotonic()
    for begin in range(len(preds), len(rows), 4):
        preds.extend(generate(model, processor, rows[begin:begin+4], data))
        tmp = path.with_suffix('.tmp'); write_json(tmp, preds); tmp.replace(path)
        write_json(directory / 'PROGRESS.json', {'status': 'evaluating', 'completed_qa': len(preds), 'expected_qa': 128, 'active_seconds': time.monotonic()-started})
        if time.monotonic()-started > 300 or torch.cuda.max_memory_allocated()/2**30 > 65:
            raise RuntimeError('Replay evaluation cap')
    return preds


def diagnostic():
    from vlm_human_sft.evaluate import lora_hash
    core.verify(); old = read(OLD / 'runs' / ROLE / 'train/TRAIN_COMPLETE.json')
    model, processor, _ = actor(adapter=old['adapter'])
    if lora_hash(model) != old['final_lora_sha256']:
        raise ValueError('Wrong fixed canonical control')
    preds = eval_replay(model, processor, ROOT / 'diagnostic')
    losses = sum(not q['literal_exact'] for q in preds)
    if lora_hash(model) != old['final_lora_sha256']:
        raise ValueError('Read-only diagnostic changed model')
    result = {'status': 'completed', 'qa': 128, 'initial_correct': 128, 'control_correct': 128-losses,
              'lost_correct': losses, 'training_admitted': losses >= 4, 'optimizer_updates': 0,
              'scope': 'Training replay construction/forgetting diagnostic, not capability validation'}
    write_json(ROOT / 'diagnostic/COMPLETE.json', result); print('FORGETTING_DIAGNOSTIC', result, flush=True)


def condition(transfer=False):
    from vlm_human_sft.evaluate import lora_hash
    core.verify()
    if transfer:
        result = read(ROOT / 'runs' / ROLE / 'train/TRAIN_COMPLETE.json'); model, processor, _ = actor(adapter=result['adapter'])
        core.evaluate(model, processor, ROOT / 'runs' / ROLE, ('transfer_dev',))
    else:
        model, processor, result = core.train(ROLE)
        core.evaluate(model, processor, ROOT / 'runs' / ROLE, ('hard','clean_literal','literal_validation','seen_validation','retention'))
        eval_replay(model, processor, ROOT / 'runs' / ROLE / 'replay_literal')
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Final weights changed in evaluation')
    write_json(ROOT / 'runs' / ROLE / ('TRANSFER_COMPLETE.json' if transfer else 'COMPLETE.json'), {'status': 'completed', 'updates': 128})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    names = ('hard','clean_literal','literal_validation','seen_validation','retention','replay_literal')
    literal = set(names)-{'seen_validation','retention'}
    records = {role: {name: read(path/name/'PREDICTIONS.json') for name in names if (path/name/'PREDICTIONS.json').exists()}
               for role,path in [('initial',ROOT/'baseline'),('control',OLD/'runs'/ROLE),('candidate',ROOT/'runs'/ROLE)]}
    records['control']['replay_literal'] = read(ROOT/'diagnostic/PREDICTIONS.json')
    metrics = {role: {name: {'qa':len(rows), 'full':sum(float(q['literal_exact']) if name in literal else q['vqa_score']>=.999999 for q in rows),
                            'score_percent':100*sum(float(q['literal_exact']) if name in literal else q['vqa_score'] for q in rows)/len(rows)} for name,rows in group.items()} for role,group in records.items()}
    def pair(a,b,name):
        normalize = lambda xs: [{**q,'vqa_score':float(q['literal_exact']) if name in literal else q['vqa_score']} for q in xs]
        return paired(normalize(records[a][name]),normalize(records[b][name]))
    changes = {n:pair('initial','candidate',n) for n in names}; matched = {n:pair('control','candidate',n) for n in names}
    seenmeta = {q['question_id']:q for q in read(ROOT/'DATA.json')['evaluations']['seen_validation']}; slices = {}
    for group in ('singleword','multiword','rare_1_5'):
        ids = {qid for qid,q in seenmeta.items() if (q['stratum']==group if group!='rare_1_5' else q.get('answer_frequency',100)<=5)}
        slices[group] = paired([q for q in records['initial']['seen_validation'] if q['question_id'] in ids], [q for q in records['candidate']['seen_validation'] if q['question_id'] in ids])
    checks = {'fitting_learnable':metrics['candidate']['hard']['full']>=96,
              'new_word_gain':changes['clean_literal']['delta_pp']>=3 and changes['clean_literal']['ci95'][0]>0,
              'native_gain':changes['seen_validation']['delta_pp']>=.5 and changes['seen_validation']['ci95'][0]>0,
              'old_word_protection':changes['literal_validation']['delta_pp']>=-.5,
              'retention':changes['retention']['delta_pp']>=-.5,'native_slices':all(x['delta_pp']>=-.5 for x in slices.values()),
              'matched_word_advantage':matched['clean_literal']['delta_pp']>=.5 and matched['clean_literal']['ci95'][0]>0}
    result = {'status':'completed','metrics':metrics,'vs_initial':changes,'vs_control':matched,'native_slices':slices,'checks':checks,
              'screen_passed':all(checks.values()),'optimizer_updates':128,'goal_complete':False,
              'scope':'Single-seed internal given-word OCR replay contrast; replay_literal is training, not ability evidence; original-SFT-seen images; conditional image-cluster CI only.'}
    write_json(ROOT/'SCREEN.json',result); render(result); return result


def render(result=None):
    data = read(ROOT/'DATA.json'); link = lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    state = '已准备，尚无新能力结果。'; diagnostic_path = ROOT/'diagnostic/COMPLETE.json'
    if diagnostic_path.exists():
        d = read(diagnostic_path); state = f'原模型128正确；旧控制保留{d["control_correct"]}，丢失{d["lost_correct"]}；训练准入={d["training_admitted"]}。该128是训练回放，不是验证。'
    table = ''
    if result:
        table = '|模型|训练词/128|新验证词%|旧验证词%|native512%|保持64%|训练回放%|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for role,m in result['metrics'].items():
            table += '|'+role+'|'+str(m['hard']['full'])+'|'+ '|'.join(f'{m[n]["score_percent"]:.6f}' for n in ('clean_literal','literal_validation','seen_validation','retention','replay_literal'))+'|\n'
        state += f' 联合能力门槛通过={result["screen_passed"]}；训练保持不能替代泛化。'
    RESULT.write_text('# 任务匹配回放：结果与状态\n\n'+state+'\n\n'+table+'\n[协议](VLM_LITERAL_TASK_REPLAY_V1_PROTOCOL.md) · [真实图片与源码](learning/任务匹配回放_保住已有读词能力.html)\n')
    cards = []
    for q in data['evaluations']['replay_literal'][:8]:
        outputs = []
        for title,path in [('原模型',ROOT/'baseline/replay_literal/PREDICTIONS.json'),('旧控制',ROOT/'diagnostic/PREDICTIONS.json'),('新候选',ROOT/'runs'/ROLE/'replay_literal/PREDICTIONS.json')]:
            if path.exists():
                pred = next((x for x in read(path) if x['question_id']==q['question_id']),None)
                if pred: outputs.append(title+'：'+html.escape(pred['prediction'])+' / '+str(pred['literal_exact']))
        cards.append('<article><h2>'+html.escape(q['target'])+'</h2><img src="'+link(data['images'][q['image_id']]['path'])+'" alt="实际词输入"><p>训练目标 '+html.escape(canonical(q['target']))+'；原答案词曝光 '+str(q['sft_answer_word_frequency'])+'。</p><p>'+'；'.join(outputs)+'</p></article>')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>任务匹配回放</title><style>body{max-width:1000px;margin:32px auto;padding:0 18px;font:17px/1.7 system-ui}article{border:1px solid #cbd5e1;padding:16px;margin:18px 0}img{max-width:100%;width:460px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style><h1>纠正错词时，如何保住已有的读词能力？</h1><p>'+html.escape(state)+'</p><p>旧控制的回放是整图VQA。新候选保留128纠错词和一半native回放，把另一半替换为原本答对的完整词区域。算法仍为SFT；改的是回放分布，检验相同任务的正确监督能否减少字母替换。只看训练loss或回放正确率不能说明能力提升。</p><pre>每步16QA = hard8 + native4 + literal_replay4\nCE(q) = -sum(log p(assistant token | image, question, prefix))/token_count\nL = (sum CE(hard8) + sum CE(native4) + sum CE(literal4))/16\n# 128词各8次；128native各4次；128正确词各4次\n# 与旧控制的区别：literal4替换第二个native4；其他预算相同</pre><p>以下8词在新输出前固定；图片是实际ROI输入，页面缩放不代表模型分辨率。它们进入训练，不能作为独立泛化证据。所有原图曾进原SFT；助手审核不等于专家标注。</p><p><a href="'+link(PROTOCOL)+'">完整协议</a> · <a href="'+link(Path(__file__).resolve())+'">数据替换与分析源码</a> · <a href="'+link(Path(core.__file__))+'">复用的训练循环</a></p>'+''.join(cards)+'<pre>'+html.escape(table)+'</pre></html>')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--audit',action='store_true'); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.audit: audit(); return
    if args.prepare: prepare(); return
    if args.worker:
        if args.worker=='diagnostic': diagnostic()
        elif args.worker=='candidate': condition()
        elif args.worker=='transfer': condition(True)
        else: raise ValueError('Unknown worker')
        return
    from research_runtime.queue import run_queue
    core.verify(); identity = {'plan_sha256':sha_file(ROOT/'PLAN.json')}; entry = str(Path(__file__).resolve())
    job = lambda key,marker:{'key':key,'command':[sys.executable,'-u',entry,'--worker',key],'completion_marker':str(marker)}
    run_queue(ROOT/'diagnostic_queue',[job('diagnostic',ROOT/'diagnostic/COMPLETE.json')],identity)
    d = read(ROOT/'diagnostic/COMPLETE.json'); render()
    if not d['training_admitted']:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','outcome':'hypothesis_not_supported_no_training','diagnostic':d,'optimizer_updates':0,'goal_complete':False})
        print('STOP_NO_TASK_FORGETTING_SIGNAL',d,flush=True); return
    run_queue(ROOT/'training_queue',[job('candidate',ROOT/'runs'/ROLE/'COMPLETE.json')],identity)
    result = analyze()
    if result['screen_passed']:
        from vlm_grpo_iteration_v1.study import paired
        run_queue(ROOT/'transfer_queue',[job('transfer',ROOT/'runs'/ROLE/'TRANSFER_COMPLETE.json')],identity)
        pred = read(ROOT/'runs'/ROLE/'transfer_dev/PREDICTIONS.json')
        result['transfer'] = {role:paired(read(BASE/'vlm_reference_replay_v1/runs'/role/'transfer_dev/PREDICTIONS.json'),pred) for role in ('initial','hard_replay')}
        write_json(ROOT/'ANALYSIS.json',result)
    write_json(ROOT/'COMPLETION.json',result); print('TASK_REPLAY_COMPLETE',result,flush=True)


if __name__=='__main__':
    main()
