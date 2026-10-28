"""Official given-region word reading; reuse exact literal-generation worker."""
import argparse
from collections import Counter
import html
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote

import run_literal_tile_v1 as core
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, encode, sha_file, write_json
from vlm_grpo_iteration_v1.common import read

ROOT = PROJECT_ROOT.parents[1]/'outputs/vlm_literal_roi_v1'
BASE = ROOT.parent
PROTOCOL = PROJECT_ROOT/'docs/VLM_LITERAL_ROI_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT/'docs/VLM_LITERAL_ROI_V1_RESULTS.md'
PAGE = PROJECT_ROOT/'docs/learning/完整单词转写_SFT与DPO监督.html'
QUESTION = 'Transcribe the complete word in the image. Return only that word.'
core.ROOT = ROOT


def prepare():
    import torch
    from PIL import Image
    import hashlib
    from vlm_probe.evaluate import inference_row, processor_for
    if (ROOT/'PLAN.json').exists():
        raise FileExistsError('Study frozen')
    start = time.monotonic()
    torch.set_num_threads(2)
    source = read(ROOT/'CANDIDATE_ROWS.json')
    if len(source['rows']) < 2304:
        raise ValueError('Source pool below registered2304')
    parent = read(BASE/'vlm_reference_replay_v1/PLAN.json')
    proc = processor_for(Path(parent['model_root']), read(parent['reference_config']), 1024)
    rows, images = [], {}
    for index, q in enumerate(source['rows'][:2304]):
        im = source['source_images'][q['source_image_id']]
        if sha_file(im['path']) != im['sha256']:
            raise ValueError('Original image changed')
        destination = ROOT/'data/images'/(q['annotation_id']+'.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(im['path']) as original:
            image = original.convert('RGB').crop(q['crop_bounds_original_pixels'])
            original_crop_size = list(image.size)
            # The pinned Qwen smart_resize rejects either edge below factor28.
            # Preserve aspect and all pixels; this is interpolation, not new detail.
            if min(image.size) < 28:
                scale = 28/min(image.size)
                image = image.resize((math.ceil(image.width*scale), math.ceil(image.height*scale)), Image.Resampling.BICUBIC)
            image.save(destination)
            pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        q = {**q, 'question_id': 'roi:'+q['annotation_id'], 'image_id': 'roi:'+q['source_image_id'],
             'question': QUESTION, 'fold': 'training_pool' if index < 2048 else 'validation',
             'source_image_path': im['path'], 'source_image_sha256': im['sha256'],
             'source_pixel_sha256': im.get('pixel_sha256', im['sha256']),
             'original_crop_size': original_crop_size, 'factor28_compatible_size': list(image.size)}
        images[q['image_id']] = {'path': str(destination), 'sha256': sha_file(destination),
                                 'pixel_sha256': pixel_sha, 'width': image.width, 'height': image.height}
        item = encode(proc, inference_row(q), destination, False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]), expected_grid=item['image_grid_thw'][0].tolist(),
                 inference_input_sha256=content_sha(item['input_ids'].tolist()))
        rows.append(q)
        if (index+1)%128 == 0:
            write_json(ROOT/'CPU_PROGRESS.json', {'status': 'preparing_given_regions', 'completed': index+1,
                       'expected': 2304, 'active_seconds': time.monotonic()-start})
    if len({images[q['image_id']]['pixel_sha256'] for q in rows}) != len(rows):
        raise ValueError('Actual crop pixel duplication')
    train, val = rows[:2048], rows[2048:]
    for key in ('source_image_id', 'source_pixel_sha256'):
        if {q[key] for q in train}&{q[key] for q in val}:
            raise ValueError('Source image/pixel overlap')
    if {core.canonical(q['target']) for q in train}&{core.canonical(q['target']) for q in val}:
        raise ValueError('Target word overlap')
    write_json(ROOT/'DATA.json', {'rows': rows, 'images': images})
    files = {**read(BASE/'vlm_native_ocr_correction_v1/PLAN.json')['files']}
    for path in (Path(__file__), Path(core.__file__), PROJECT_ROOT/'tools/inventory_literal_roi_v1.py',
                 PROTOCOL, ROOT/'CANDIDATE_ROWS.json', ROOT/'INVENTORY.json', ROOT/'DATA.json',
                 BASE/'cocotext_source_v1/parseq_textocr_converter.py'):
        files[str(path)] = sha_file(path)
    plan = {'status': 'prepared', 'files': files, 'packages': parent['packages'], 'base_weight_files': parent['base_weight_files'],
            'initial_adapter': parent['initial_adapter'], 'lora_sha256': read(BASE/'vlm_native_ocr_correction_v1/PLAN.json')['models']['initial']['lora_sha256'],
            'qa': 2304, 'training_pool': 2048, 'validation': 256, 'new_optimizer_updates': 0, 'automatic_training': False,
            'active_prepare_seconds': time.monotonic()-start,
            'frequency': {fold: dict(Counter(q['frequency_slice'] for q in rows if q['fold'] == fold)) for fold in ('training_pool', 'validation')},
            'preview_questions': [q['question_id'] for q in sorted(rows, key=lambda q: content_sha(['literal_roi_preview', q['question_id']]))[:8]]}
    write_json(ROOT/'PLAN.json', plan)
    RESULT.write_text('# 给定单词区域完整转写\n\n已准备2048训练候选+256固定转写验证，0训练/新adapter，尚无新能力结果。\n')
    render()
    print('PREPARED', {k: plan[k] for k in ('qa', 'active_prepare_seconds', 'frequency')}, flush=True)


def analyze():
    data = read(ROOT/'DATA.json')
    lookup = {q['question_id']: q for shard in (0, 1) for q in read(ROOT/'runs'/f'shard_{shard}/PREDICTIONS.json')}
    rows = [lookup[q['question_id']] for q in data['rows']]
    write_json(ROOT/'PREDICTIONS.json', rows)
    raw = {q['question_id']: q for q in data['rows']}
    scores = {}
    for fold in ('training_pool', 'validation'):
        group = [q for q in rows if q['fold'] == fold]
        scores[fold] = {'literal_exact_percent': 100*sum(q['literal_exact'] for q in group)/len(group),
                        'qa': len(group), 'full': sum(q['literal_exact'] for q in group)}
        scores[fold]['frequency'] = {}
        for frequency in ('zero_target_word_exposure', 'rare_1_5'):
            sub = [q for q in group if raw[q['question_id']]['frequency_slice'] == frequency]
            scores[fold]['frequency'][frequency] = {'qa': len(sub), 'literal_exact_percent': 100*sum(q['literal_exact'] for q in sub)/len(sub)}
    errors = []
    for q in sorted(rows, key=lambda q: content_sha(['literal_roi_errors', q['question_id']])):
        target, pred = core.canonical(q['target']), core.canonical(q['prediction'])
        if q['fold'] != 'training_pool' or q['literal_exact'] or not q['terminated'] or target in pred.split() or re.sub(r'\s+', '', pred) == target:
            continue
        errors.append({**raw[q['question_id']], 'initial_prediction': q})
    write_json(ROOT/'ERROR_CANDIDATES.json', {'rows': errors, 'images': data['images'], 'visual_review_required': True})
    result = {'status': 'completed', 'scores': scores, 'candidate_errors': len(errors),
              'unique_error_targets': len({core.canonical(q['target']) for q in errors}), 'count_gate_passed': len(errors) >= 128,
              'visual_review_passed': False, 'automatic_training': False, 'new_optimizer_updates': 0, 'new_adapters': 0, 'goal_complete': False,
              'scope': 'Initial given-official-word-region transcription; original-SFT-seen images, not new trained or whole-image capability.'}
    write_json(ROOT/'ANALYSIS.json', result)
    RESULT.write_text('# 给定单词区域完整转写：初始结果\n\n'+str(scores)+f'\n\n训练候选错误{len(errors)}，数量门槛={len(errors)>=128}，仍需实际区域图视觉复核与独立训练协议。0训练/新adapter，不是整图定位或原SFT未见图证据。\n\n[协议](VLM_LITERAL_ROI_V1_PROTOCOL.md) · [实际区域图、原图与回答](learning/完整单词转写_SFT与DPO监督.html)\n')
    render()
    write_json(ROOT/'COMPLETION.json', result)
    print('LITERAL_ROI_COMPLETE', result, flush=True)


def render():
    data = read(ROOT/'DATA.json')
    plan = read(ROOT/'PLAN.json')
    lookup = {q['question_id']: q for q in read(ROOT/'PREDICTIONS.json')} if (ROOT/'PREDICTIONS.json').exists() else {}
    relative = lambda path: quote(os.path.relpath(path, PAGE.parent), safe='/._-')
    cards = []
    for qid in plan['preview_questions']:
        q = next(q for q in data['rows'] if q['question_id'] == qid)
        a = lookup.get(qid)
        cards.append('<article><h2>'+html.escape(qid)+'</h2><img src="'+relative(data['images'][q['image_id']]['path'])+'" alt="实际给定词区域输入"><p>官方完整词 '+html.escape(q['target'])+'；已记录SFT答案词曝光'+str(q['sft_answer_word_frequency'])+'</p><p>实际回答 '+html.escape(a['prediction'] if a else '待生成')+'；完整字面exact='+str(a['literal_exact'] if a else None)+'</p><p><a href="'+relative(q['source_image_path'])+'">原图</a>；给定区域'+str(q['crop_bounds_original_pixels'])+'</p></article>')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整单词监督与偏好</title><style>body{max-width:1050px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui}article{border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{width:500px;max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style><h1>完整单词转写：先让监督回答同一个问题</h1><p>实际given-ROI图像与预固定8例。不是整图定位，原生VQA评测不按答案裁剪。官方单一转写不伪装10人共识；所有原图曾进原SFT，监督词曝光0或1–5不表示预训练未见。</p><p>PARSeq官方TextOCR转换器同样构建给定单词区域训练图；本轮另排与其他词相交的框、边缘截断与极小/大倾角。目标与指令都是完整单词，不再从品牌问法推断系列名。</p><pre>现阶段：2048训练候选+256固定验证，初始模型2304生成，0训练。\n后续需>=128视觉支持错题/96target，再另登记：\nL_SFT = mean_QA mean_token[-log p(y+)]\nL_DPO = -log sigmoid(.1*((log pθ(y+) - log pθ(y-)) - (log pref(y+) - log pref(y-))))\n候选 L=L_SFT+L_DPO；控制 L=L_SFT；同数据/起点/32步。\nDPO使用序列SUM和真实EOS；SFT用逐题token均值。\n转写提升与整图VQA/保持分列，不用拟合替代能力。</pre><p><a href="'+relative(PROTOCOL)+'">协议</a> · <a href="'+relative(Path(__file__).resolve())+'">执行代码</a> · <a href="https://github.com/baudm/parseq/blob/main/tools/textocr_converter.py">官方预处理参考</a></p>'+''.join(cards)+'</html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        core.worker(args.worker)
    else:
        from research_runtime.queue import run_queue
        core.verify()
        jobs = [{'key': f'shard_{s}', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(s)],
                 'completion_marker': str(ROOT/'runs'/f'shard_{s}/COMPLETE.json')} for s in (0, 1)]
        run_queue(ROOT/'queue', jobs, {'plan_sha256': sha_file(ROOT/'PLAN.json')})
        analyze()


if __name__ == '__main__':
    main()
