"""Given fixed-grid image tiles, read one complete official OCR word."""
import argparse
from collections import Counter, defaultdict
import html
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, read_manifest, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, read
from vlm_region_ocr_sft_v1.data import bounds

BASE = PROJECT_ROOT.parents[1]/'outputs'
ROOT = BASE/'vlm_literal_tile_v1'
PARENT = BASE/'vlm_native_ocr_correction_v1'
PROTOCOL = PROJECT_ROOT/'docs/VLM_LITERAL_TILE_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT/'docs/VLM_LITERAL_TILE_V1_RESULTS.md'
PAGE = PROJECT_ROOT/'docs/learning/固定网格完整转写_监督与偏好.html'
QUESTION = 'Transcribe the complete word in the image. Return only that word.'


def canonical(value):
    return unicodedata.normalize('NFKC', str(value)).casefold().strip()


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def prepare():
    import torch
    from PIL import Image
    from vlm_probe.evaluate import inference_row, processor_for
    if (ROOT/'INVENTORY.json').exists() or (ROOT/'PLAN.json').exists():
        raise FileExistsError('Inventory or study already prepared')
    start = time.monotonic()
    torch.set_num_threads(2)
    path = BASE/'vlm_deep_v2/prepared/manifest.json'
    source = read_manifest(path)
    previous = read(BASE/'vlm_reference_replay_v1/DATA.json')
    old_clear = read(PARENT/'revision2/DATA.json')
    small = read(BASE/'vlm_native_smalltext_v1/CANDIDATE_ROWS.json')
    annotation_path = BASE/'vlm_region_ocr_sft_v1/source_data/TextOCR_0.1_train.json'
    raw = read(annotation_path)
    blocked = {q['image_id'] for q in previous['training_rows']+old_clear['rows']+small['rows']}
    for rows in previous['evaluations'].values():
        blocked.update(q['image_id'] for q in rows)
    for name in ('dev', 'final'):
        blocked.update(q['image_id'] for q in source['splits'][name])
    meta = {**source['images'], **previous['images']}
    pixel = lambda iid: meta[iid].get('pixel_sha256', meta[iid]['sha256'])
    blocked_pixels = {pixel(i) for i in blocked if i in meta}
    frequency = Counter(word for q in source['splits']['train_all'] for word in NORMALIZER(q['target']).split())
    candidates = defaultdict(list)
    counts = Counter()
    train_ids = {q['image_id'] for q in source['splits']['train_all']}
    for iid in sorted(train_ids-blocked):
        im = source['images'][iid]
        if iid not in raw['imgs'] or pixel(iid) in blocked_pixels:
            continue
        if (im['width'], im['height']) != (raw['imgs'][iid]['width'], raw['imgs'][iid]['height']):
            continue
        anns = []
        invalid = False
        for aid in raw['imgToAnns'][iid]:
            try:
                anns.append((raw['anns'][aid], bounds(raw['anns'][aid]['points'], im['width'], im['height'])))
            except (ValueError, TypeError, KeyError):
                invalid = True
                break
        if invalid:
            counts['image_invalid_polygon'] += 1
            continue
        for row in range(4):
            for col in range(4):
                tile = [col*im['width']//4, row*im['height']//4, (col+1)*im['width']//4, (row+1)*im['height']//4]
                inside = [(a, box) for a, box in anns if intersects(tile, box)]
                if len(inside) != 1:
                    continue
                a, box = inside[0]
                x0, y0, x1, y1 = box
                if x0 < tile[0]+4 or y0 < tile[1]+4 or x1 > tile[2]-4 or y1 > tile[3]-4:
                    continue
                counts['single_complete_annotation_tile'] += 1
                target = str(a.get('utf8_string', '')).strip()
                if not re.fullmatch(r'[a-zA-Z]{6,16}', target):
                    continue
                if frequency[NORMALIZER(target)] != 0:
                    continue
                if x1-x0 < 48 or y1-y0 < 16:
                    continue
                points = a['points']
                angle = math.degrees(math.atan2(points[3]-points[1], points[2]-points[0]))
                if abs(angle) > 10:
                    continue
                candidates[iid].append({'source_image_id': iid, 'annotation_id': a['id'],
                                        'target': target, 'tile_grid': 4, 'tile_row': row, 'tile_col': col,
                                        'tile_bounds_original_pixels': tile, 'word_bounds_original_pixels': box,
                                        'sft_target_word_frequency': 0})
    rows, targets, pixels = [], set(), set()
    for iid in sorted(candidates, key=lambda x: content_sha(['literal_tile_v1:image', x])):
        for q in sorted(candidates[iid], key=lambda q: content_sha(['literal_tile_v1:tile', q['annotation_id'], q['tile_row'], q['tile_col']])):
            t = canonical(q['target'])
            if t in targets or pixel(iid) in pixels:
                continue
            targets.add(t)
            pixels.add(pixel(iid))
            rows.append(q)
            break
    rows.sort(key=lambda q: content_sha(['literal_tile_v1:execution', q['source_image_id'], q['annotation_id']]))
    inventory = {'status': 'prediction_free_inventory', 'eligible_unique_source_images_words': len(rows),
                 'selected': min(2304, len(rows)), 'required': 2304, 'filter_counts': dict(counts),
                 'official_single_annotation_per_target': True, 'original_sft_images_seen': True,
                 'sft_target_word_frequency': 0, 'new_optimizer_updates': 0,
                 'annotation_sha256': sha_file(annotation_path), 'manifest_sha256': sha_file(path)}
    write_json(ROOT/'INVENTORY.json', inventory)
    if len(rows) < 2304:
        write_json(ROOT/'COMPLETION.json', {**inventory, 'status': 'insufficient_source_pool', 'goal_complete': False})
        print('STOP_INSUFFICIENT_SOURCE', inventory, flush=True)
        return
    rows = rows[:2304]
    old_plan = read(BASE/'vlm_reference_replay_v1/PLAN.json')
    proc = processor_for(Path(old_plan['model_root']), read(old_plan['reference_config']), 1024)
    images = {}
    for index, q in enumerate(rows):
        iid = q['source_image_id']
        im = source['images'][iid]
        original = (path.parent/im['path']).resolve()
        if sha_file(original) != im['sha256']:
            raise ValueError('Original source image changed')
        crop_path = ROOT/'data/images'/f'{iid}_{q["tile_row"]}_{q["tile_col"]}.png'
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as raw_image:
            image = raw_image.convert('RGB').crop(q['tile_bounds_original_pixels'])
            image.save(crop_path)
            pixel_digest = __import__('hashlib').sha256(image.tobytes()).hexdigest()
        q.update(question_id='tile:'+q['annotation_id'], image_id='tile:'+iid,
                 question=QUESTION, fold='training_pool' if index < 2048 else 'validation',
                 source_image_path=str(original), source_image_sha256=im['sha256'],
                 source_pixel_sha256=im.get('pixel_sha256', im['sha256']))
        images[q['image_id']] = {'path': str(crop_path), 'sha256': sha_file(crop_path), 'pixel_sha256': pixel_digest,
                                  'width': image.width, 'height': image.height}
        item = encode(proc, inference_row(q), crop_path, False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]), expected_grid=item['image_grid_thw'][0].tolist(),
                 inference_input_sha256=content_sha(item['input_ids'].tolist()))
        if (index+1) % 128 == 0:
            write_json(ROOT/'CPU_PROGRESS.json', {'status': 'preparing_tiles', 'completed': index+1, 'expected': 2304,
                       'active_seconds': time.monotonic()-start})
    if len({images[q['image_id']]['pixel_sha256'] for q in rows}) != len(rows):
        raise ValueError('Duplicate actual tile pixels')
    write_json(ROOT/'DATA.json', {'rows': rows, 'images': images})
    files = {**read(PARENT/'PLAN.json')['files']}
    for file in (Path(__file__), PROTOCOL, ROOT/'INVENTORY.json', ROOT/'DATA.json'):
        files[str(file)] = sha_file(file)
    plan = {'status': 'prepared', 'files': files, 'packages': old_plan['packages'], 'base_weight_files': old_plan['base_weight_files'],
            'initial_adapter': old_plan['initial_adapter'], 'lora_sha256': read(PARENT/'PLAN.json')['models']['initial']['lora_sha256'],
            'qa': 2304, 'training_pool': 2048, 'validation': 256, 'new_optimizer_updates': 0, 'automatic_training': False,
            'active_prepare_seconds': time.monotonic()-start,
            'preview_questions': [q['question_id'] for q in sorted(rows, key=lambda q: content_sha(['literal_tile_preview', q['question_id']]))[:8]]}
    write_json(ROOT/'PLAN.json', plan)
    RESULT.write_text('# 固定网格完整转写\n\n已准备2048训练候选+256固定转写验证；0训练，尚无新能力结果。\n')
    render()
    print('PREPARED', {'qa': 2304, 'active_seconds': plan['active_prepare_seconds']}, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT/'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen file changed: '+path)
    if package_lock() != plan['packages']:
        raise ValueError('Package versions changed')
    for path, stats in plan['base_weight_files'].items():
        st = Path(path).stat()
        if stats != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
            raise ValueError('Base storage changed')
    return plan


def generate(model, proc, rows, data):
    import torch
    from vlm_deep.common import collate
    from vlm_sft.common import to_device
    from vlm_probe.evaluate import inference_row
    from vlm_sampled_signal_v1.trajectory import stop_at_eos
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    answers = []
    with torch.inference_mode():
        items = [encode(proc, inference_row(q), Path(data['images'][q['image_id']]['path']), False) for q in rows]
        batch = to_device(collate(items, proc.tokenizer.pad_token_id), next(model.parameters()).device)
        output = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True)
        for index, (q, item) in enumerate(zip(rows, items)):
            if (int(item['input_ids'].shape[1]) != q['expected_prefix_tokens']
                    or item['image_grid_thw'][0].tolist() != q['expected_grid']
                    or content_sha(item['input_ids'].tolist()) != q['inference_input_sha256']):
                raise ValueError('Actual inference frame changed')
            ids, terminated = stop_at_eos(output[index, batch['input_ids'].shape[1]:].tolist(), model.generation_config.eos_token_id, 32)
            prediction = proc.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            answers.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'fold': q['fold'], 'target': q['target'],
                            'prediction': prediction, 'completion_ids': ids, 'terminated': terminated,
                            'literal_exact': canonical(prediction) == canonical(q['target']),
                            'input_sha256': q['inference_input_sha256'], 'visual_tokens': int(item['image_grid_thw'].prod())//4})
    return answers


def worker(shard):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    plan = verify()
    data = read(ROOT/'DATA.json')
    model, proc, _ = actor(adapter=plan['initial_adapter'])
    model.eval()
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Wrong initial model')
    groups = [data['rows'][i:i+4] for i in range(0, len(data['rows']), 4)][shard::2]
    flat = [q for group in groups for q in group]
    directory = ROOT/'runs'/f'shard_{shard}'
    path = directory/'PREDICTIONS.json'
    preds = read(path) if path.exists() else []
    if len(preds) % 4 or [q['question_id'] for q in preds] != [q['question_id'] for q in flat[:len(preds)]]:
        raise ValueError('Whole-batch resume changed')
    prior = read(directory/'PROGRESS.json').get('active_seconds', 0.) if (directory/'PROGRESS.json').exists() else 0.
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for chunk in groups[len(preds)//4:]:
        for q in chunk:
            im = data['images'][q['image_id']]
            if sha_file(im['path']) != im['sha256']:
                raise ValueError('Actual tile changed')
        preds.extend(generate(model, proc, chunk, data))
        temp = path.with_suffix('.tmp')
        write_json(temp, preds)
        temp.replace(path)
        active = prior+time.monotonic()-start
        peak = torch.cuda.max_memory_allocated()/2**30
        write_json(directory/'PROGRESS.json', {'status': 'initial_literal_transcription', 'completed_qa': len(preds),
                   'expected_qa': len(flat), 'active_seconds': active, 'peak_allocated_gib': peak, 'optimizer_updates': 0})
        if active > 900 or peak > 65:
            raise RuntimeError('Initial-readout budget exceeded')
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Read-only model changed')
    write_json(directory/'COMPLETE.json', {'status': 'completed', 'qa': len(preds), 'active_seconds': prior+time.monotonic()-start})


def analyze():
    data = read(ROOT/'DATA.json')
    lookup = {q['question_id']: q for shard in (0, 1) for q in read(ROOT/'runs'/f'shard_{shard}/PREDICTIONS.json')}
    rows = [lookup[q['question_id']] for q in data['rows']]
    write_json(ROOT/'PREDICTIONS.json', rows)
    scores = {}
    for fold in ('training_pool', 'validation'):
        group = [q for q in rows if q['fold'] == fold]
        scores[fold] = {'literal_exact_percent': 100*sum(q['literal_exact'] for q in group)/len(group),
                        'qa': len(group), 'full': sum(q['literal_exact'] for q in group)}
    raw = {q['question_id']: q for q in data['rows']}
    errors = []
    for q in sorted(rows, key=lambda q: content_sha(['literal_tile_errors', q['question_id']])):
        target, pred = canonical(q['target']), canonical(q['prediction'])
        if q['fold'] != 'training_pool' or q['literal_exact'] or not q['terminated'] or target in pred.split() or re.sub(r'\s+', '', pred) == target:
            continue
        errors.append({**raw[q['question_id']], 'initial_prediction': q})
    write_json(ROOT/'ERROR_CANDIDATES.json', {'rows': errors, 'images': data['images'], 'visual_review_required': True})
    result = {'status': 'completed', 'scores': scores, 'candidate_errors': len(errors), 'unique_error_targets': len({canonical(q['target']) for q in errors}),
              'count_gate_passed': len(errors) >= 128, 'visual_review_passed': False, 'automatic_training': False,
              'new_optimizer_updates': 0, 'new_adapters': 0, 'goal_complete': False,
              'scope': 'Initial given-fixed-tile transcription headroom, not new trained capability or unseen-SFT image performance.'}
    write_json(ROOT/'ANALYSIS.json', result)
    RESULT.write_text('# 固定网格完整转写：初始结果\n\n'+str(scores)+f'\n\n训练候选错误{len(errors)}，数量门槛={len(errors)>=128}，仍需视觉复核与独立训练协议。0训练/新adapter；所有图曾进原SFT，完整目标词未进记录答案词。\n\n[协议](VLM_LITERAL_TILE_V1_PROTOCOL.md) · [原图块与实际回答](learning/固定网格完整转写_监督与偏好.html)\n')
    render()
    write_json(ROOT/'COMPLETION.json', result)
    print('LITERAL_TILE_COMPLETE', result, flush=True)


def render():
    data = read(ROOT/'DATA.json')
    plan = read(ROOT/'PLAN.json')
    lookup = {q['question_id']: q for q in read(ROOT/'PREDICTIONS.json')} if (ROOT/'PREDICTIONS.json').exists() else {}
    relative = lambda p: quote(os.path.relpath(p, PAGE.parent), safe='/._-')
    cards = []
    for qid in plan['preview_questions']:
        q = next(q for q in data['rows'] if q['question_id'] == qid)
        a = lookup.get(qid)
        cards.append('<article><h2>'+html.escape(qid)+'</h2><img src="'+relative(data['images'][q['image_id']]['path'])+'" alt="实际固定网格图块"><p>官方完整词：'+html.escape(q['target'])+'；原图网格(row,col)='+str((q['tile_row'], q['tile_col']))+'；固定边界'+str(q['tile_bounds_original_pixels'])+'</p><p>真实回答：'+html.escape(a['prediction'] if a else '待生成')+'；完整字面exact='+str(a['literal_exact'] if a else None)+'</p><p><a href="'+relative(q['source_image_path'])+'">原图</a></p></article>')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定网格完整转写</title><style>body{max-width:1000px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui}article{border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{max-width:100%;width:500px;image-rendering:auto}</style><h1>完整单词监督：先修复题意，再检验偏好学习</h1><p>8个例子在推理前固定；图片为模型实际接收的4×4固定网格图块，网页缩放仅展示。网格边界不按答案改变；只保留包含一个完整官方OCR标注且没有其他词相交的图块。目标为单一官方标注，不伪造10人共识。</p><p>本研究为给定图块转写，不是无框定位。原图曾进原SFT；标签词未出现在已记录SFT答案分词中，但预训练曝光未知。训练/验证原图、图块像素与文字隔离。品牌问题与系列名不再混为同一任务。</p><p>后续候选：L_SFT=mean_QA mean_token[−log p(y)]；L_DPO=−log sigmoid(β[(log pθ(y+)−log pθ(y−))−(log pref(y+)−log pref(y−))])。DPO序列SUM与SFT token均值分开，参考是原SFT且chosen/rejected用相同图块问题。只有可靠错题数量与视觉复核通过后，另登记SFT对DPO+SFT，本轮尚无训练。</p><p><a href="'+relative(PROTOCOL)+'">协议</a> · <a href="'+relative(Path(__file__).resolve())+'">真实数据与推理源码</a></p>'+''.join(cards)+'</html>')


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
