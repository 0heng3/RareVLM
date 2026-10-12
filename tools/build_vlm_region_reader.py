"""Offline learning reader using the locked, actual region-study input PNGs."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path('/opt/data/private')
CONDITIONS = ('whole', 'double_whole', 'text_region', 'random_matched')
LABELS = {'whole': '单整图', 'double_whole': '双整图',
          'text_region': '整图＋预测文字区域', 'random_matched': '整图＋随机同形区域'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embed(path):
    path = Path(path).resolve()
    if PRIVATE not in path.parents:
        raise ValueError('Inputs must be persistent workspace files')
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def esc(value):
    return html.escape(str(value))


def excerpt(relative, start, end):
    path = REPO / relative
    lines = path.read_text().splitlines()
    code = '\n'.join(f'{index+1:>3} {line}' for index, line in enumerate(lines)
                     if start <= index+1 <= end)
    return f'<details><summary>{esc(relative)}:{start} · SHA {sha(path)[:12]}</summary><pre>{esc(code)}</pre></details>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['output_root']).resolve()
    if PRIVATE not in root.parents:
        raise ValueError('Output must be persistent')
    lock_path, contract_path = root / 'VIEW_LOCK.json', root / 'INPUT_CONTRACT.json'
    lock, contract = json.loads(lock_path.read_text()), json.loads(contract_path.read_text())
    if contract['status'] != 'passed' or contract['view_lock_sha256'] != sha(lock_path):
        raise ValueError('Real CPU input contract missing or changed')
    manifest_path = Path(config['manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    vision = json.loads((Path(config['model_root']) / 'config.json').read_text())['vision_config']
    if (vision['patch_size'], vision['spatial_merge_size'], vision['fullatt_block_indexes']) != (14, 2, [7, 15, 23, 31]):
        raise ValueError('Teaching explanation differs from actual vision architecture')
    runtime_path = Path(importlib.util.find_spec('transformers').origin).parent / 'models/qwen2_5_vl/modeling_qwen2_5_vl.py'
    runtime_lines = runtime_path.read_text().splitlines()
    runtime_ranges = []
    for index, line in enumerate(runtime_lines):
        if 'cu_seqlens = torch.repeat_interleave' in line or 'if layer_num in self.fullatt_block_indexes' in line:
            runtime_ranges.append('\n'.join(f'{j+1:>3} {runtime_lines[j]}' for j in range(max(0,index-1), min(len(runtime_lines),index+7))))
    runtime_code = f'<details><summary>实际 Transformers 视觉编码器 · SHA {sha(runtime_path)[:12]}</summary><p>{esc(runtime_path)}</p><pre>{esc(chr(10).join(runtime_ranges))}</pre></details>'
    db_path = PRIVATE / '.venvs/rarevlm-ocr/lib/python3.10/site-packages/paddlex/inference/models/text_detection/processors.py'
    db_code = excerpt(str(db_path), 394, 425) + excerpt(str(db_path), 453, 466) + excerpt(str(db_path), 515, 534)
    details = {(record['condition'], record['question_id']): record for record in contract['records']}
    rows, seen = [], set()
    for row in sorted(manifest['splits']['dev'], key=lambda item: int(item['question_id'])):
        if row['image_id'] not in seen:
            seen.add(row['image_id'])
            rows.append(row)
        if len(rows) == 6:
            break
    predictions = {}
    for condition in CONDITIONS:
        path = root / 'runs' / condition / 'dev' / 'predictions.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                predictions[condition, str(record['question_id'])] = record
    cards = []
    for row in rows:
        qid, iid = str(row['question_id']), str(row['image_id'])
        record = lock['images'][iid]
        raw = manifest['images'][iid]
        w, h = record['raw_size_wh']
        region = record['conditions']['text_region']['selected_region']
        random = record['conditions']['random_matched']['random_region']
        rectangles = []
        for selection, color in ((region, '#167d62'), (random, '#ab4f00')):
            x1, y1, x2, y2 = selection['bbox_xyxy']
            rectangles.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="{max(w,h)/160}"/>')
        original = f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="原图上的真实裁剪坐标"><image width="{w}" height="{h}" href="{embed(manifest_path.parent / raw["path"])}"/>{"".join(rectangles)}</svg>'
        columns = []
        for condition in CONDITIONS:
            views = record['conditions'][condition]['views']
            detail = details[condition, qid]
            inputs = []
            for view in views:
                if sha(Path(view['path'])) != view['png_sha256']:
                    raise ValueError('Locked actual PNG changed')
                inputs.append(f'<img src="{embed(view["path"])}" alt="实际模型输入 {esc(view["key"])}"><small>{esc(view["size_wh"])} px · crop {esc(view["bbox_xyxy"])}</small>')
            pred = predictions.get((condition, qid))
            result = '<p>生成结果尚待完成。</p>' if pred is None else f'<p>预测：<b>{esc(pred["prediction"])}</b>；共识分 {float(pred["score"]):.3f}</p>'
            columns.append(f'<div class="condition"><h3>{LABELS[condition]}</h3>{"".join(inputs)}<p>grid={esc(detail["image_grid_thw"])}<br>视觉 token={detail["visual_tokens"]}；总输入 token={detail["input_tokens"]}<br>pixel_values={esc(detail["pixel_values_shape"])}</p>{result}</div>')
        cards.append(f'<article><h2>题 {qid} · 图 {esc(iid)}</h2><p>{esc(row["question"])}</p><div class="original">{original}</div><p>上图为<b>原图加坐标标注，仅用于讲解</b>。绿色＝检测区域；橙色＝随机同形区域。下方是保存的实际输入 PNG，CSS 仅调整显示尺寸。没有展示注意力图。</p><p>选中区域 confidence={esc(region["score"])}；priority={esc(region["priority"])}；fallback={region["fallback"]}；候选数={region["eligible_candidates"]}。</p><div class="grid">{"".join(columns)}</div></article>')
    code = db_code + excerpt('vlm_region/views.py', 78, 143) + excerpt('vlm_region/views.py', 146, 160) + excerpt('vlm_region/data.py', 60, 112) + runtime_code
    body = '''<h1>同样 1,024 个视觉 token，信息如何分配？</h1>
<p>这个实验固定普通 CE 的 seed42 最终适配器，只有图像视图分配发生变化。4 条件 × 1,617 问题＝6,468 条预测；新增训练为 0。它是探索性内部 dev 研究。</p>
<h2>先看计算链条</h2>
<p>实际检测器使用 DB 后处理：像素文字热图 <code>P &gt; 0.3</code> 得到二值掩膜，通过轮廓寻找候选文字块；候选旋转矩形内的热图平均值需至少 0.6。轮廓外扩距离为 <code>area × 1.5 / perimeter</code>，所以 unclip=1.5 不是宽高各扩大 1.5 倍。整个过程只检测文字位置，不识别文字内容。</p>
<p>CPU 检测器只输出文字多边形与置信度，不识别或向 VLM 提供文字答案。对合法矩形计算 <code>priority = confidence × √(area / image_area)</code>，选最大项，每边扩展 10%。这是固定的几何启发式，不保证选中问题真正需要的文字。置信度是检测区域的分数，不是答题正确概率；未检测到区域也不代表图片确实没有文字。</p>
<p>裁剪后保持宽高比，用双三次插值缩放并补灰边。Qwen 把图片分成 14×14 patch，再以 2×2 合并。896×896 得到 64×64/4＝1,024 个视觉 token；每张 896×448 得到 64×32/4＝512。两个半预算视图合计 1,024。</p>
<p>随机区域保持预测区域的宽、高与面积，只改变合法位置。文字区域−随机区域检验位置选择；文字区域−双整图检验局部信息分配；单整图是新 letterbox 协议的参考。三个双视图条件的提示和非图像 token 相同。</p>
<p><b>等视觉 token 不代表等信息、总输入长度、FLOPs 或耗时。</b>补边占用预算，重复整图也占用预算；区域放大可能提高字形可读性，也可能丢失上下文。旧 native-resize 87.112 分与本实验输入不同，不能直接充当同协议基线。</p>
<p>实际视觉编码器有 32 层，第 7/15/23/31 层采用图片内部全局注意力（从 0 编号）。合并前，单图有 4,096 个 patch，两半图各 2,048。只看这些层的注意力矩阵元素数代理：单图为 4,096²＝16,777,216，双图合计 2×2,048²＝8,388,608。其余窗口注意力、投影、语言生成、预处理还需另算；这个 2 倍关系<b>不是实测速度或整模型 FLOPs</b>。三个双视图条件则拥有相同的分块形状。</p>
<h2>如何读结果</h2>
<p>先比较文字区域与随机同形区域，再看双整图对照。报告整体、未见/罕见/中频/高频监督答案切片及按图像整簇的配对区间。单 checkpoint 的点式探索区间不代表三种子稳定性或等价。推进门槛为相对随机和双整图各至少 +0.5 pp，高频相对双整图至少 −0.5 pp；四组都完整运行，不根据中途分数换规则。</p>
<p>检测、裁剪是缓存准备成本，单独统计；模型预处理、H2D、生成及完整壁钟时间也各自报告。缓存复用耗时不是线上服务延迟。</p>
<h2>对应真实代码</h2>'''
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>区域信息分配实验学习页</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 20px;color:#20313b;background:#f6f8fa;line-height:1.65}}h1,h2,h3{{line-height:1.3}}article{{background:white;border:1px solid #ccd5df;border-radius:10px;padding:20px;margin:28px 0}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.condition img{{width:100%;height:auto;display:block;border:1px solid #ddd}}.condition p,small{{overflow-wrap:anywhere}}.original{{max-width:580px}}svg{{width:100%;height:auto}}pre{{overflow:auto;background:#e9eef3;padding:15px}}details{{margin:10px 0}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}</style>{body}{code}<h2>6 个真实样本：按题号排序取首 6 张不同图片</h2><p>样本选择不依据答案或模型得分；展示规模仅为教学，研究评测覆盖全部 1,617 QA。</p>{''.join(cards)}<p>来源：VIEW_LOCK SHA {sha(lock_path)}；INPUT_CONTRACT SHA {sha(contract_path)}；配置 SHA {sha(args.config)}。直接保存的原图/输入文件，未使用生成式图像。</p></html>'''
    path = root / 'learning' / '学习页.html'
    path.parent.mkdir(exist_ok=True)
    path.write_text(page)
    provenance = {'schema_version':'rarevlm.region_reader.v1','view_lock_sha256':sha(lock_path),
                  'input_contract_sha256':sha(contract_path),'config_sha256':sha(args.config),
                  'reader_source_sha256':sha(Path(__file__)), 'questions':[str(row['question_id']) for row in rows],
                  'vision_config':vision,'runtime_vision_source_path':str(runtime_path),'runtime_vision_source_sha256':sha(runtime_path),
                  'db_postprocess_source_path':str(db_path),'db_postprocess_source_sha256':sha(db_path),
                  'actual_saved_inputs':True,'overlay_is_teaching_only':True,'new_predictions':0,
                  'research_total_predictions':6468,'page_sha256':sha(path)}
    (path.parent / 'PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'page':str(path),'bytes':path.stat().st_size,'cards':len(rows)},ensure_ascii=False))


if __name__ == '__main__':
    main()
