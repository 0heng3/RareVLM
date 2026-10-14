"""Build an offline reader of the independent compact, cached-OCR followup."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import struct

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path('/opt/data/private')
CONDITIONS = ('baseline', 'empty_template', 'predicted_ocr', 'shuffled_ocr')
LABELS = {'baseline': '原生整图强基线', 'empty_template': '空 OCR 模板',
          'predicted_ocr': '预测 OCR 上下文', 'shuffled_ocr': '不同图 OCR 控制'}
CASE_IDS = ('1939', '6712', '4908', '6268', '5058', '3517', '3856', '6094')


def private_path(value):
    path = Path(value).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError('reader paths must stay under persistent workspace')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value))


def embed(path):
    path = private_path(path)
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()


def excerpt(relative, names):
    path = REPO / relative
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            selected.append('\n'.join(f'{index+1:>3} {lines[index]}' for index in
                                      range(node.lineno-1, min(node.end_lineno, node.lineno+78))))
    return (f'<details><summary>{esc(relative)} · SHA {sha(path)[:12]}</summary>'
            f'<pre>{esc(chr(10).join(selected))}</pre></details>')


def predictions(root):
    values, files = {}, {}
    for condition in CONDITIONS:
        path = root/'runs'/condition/'dev/predictions.jsonl'
        if not path.exists():
            continue
        files[condition] = sha(path)
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            values[condition, str(record['question_id'])] = record
    return values, files


def lora_binding(config):
    directory = private_path(config['fixed_adapter_dir'])
    config_path, weights_path = directory/'adapter_config.json', directory/'adapter_model.safetensors'
    adapter = json.loads(config_path.read_text())
    if (adapter['r'], adapter['lora_alpha'], set(adapter['target_modules'])) != (8, 16, {'q_proj', 'v_proj'}):
        raise ValueError('existing LoRA checkpoint differs from the teaching formulas')
    with weights_path.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    shapes = {key: value['shape'] for key, value in header.items() if key != '__metadata__'}
    expected = {'q_proj.lora_A.weight': [8, 2048], 'q_proj.lora_B.weight': [2048, 8],
                'v_proj.lora_A.weight': [8, 2048], 'v_proj.lora_B.weight': [256, 8]}
    layers = set()
    for name, shape in shapes.items():
        match = re.search(r'\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight$', name)
        if match is None or expected[f'{match[2]}.lora_{match[3]}.weight'] != shape:
            raise ValueError('existing LoRA checkpoint contains an unexpected tensor')
        layers.add(int(match[1]))
    total = sum(math.prod(shape) for shape in shapes.values())
    if layers != set(range(36)) or len(shapes) != 144 or total != 1843200:
        raise ValueError('existing LoRA layer/parameter totals differ')
    return {'rank': 8, 'alpha': 16, 'scale': 2, 'layers': 36, 'parameters': total,
            'adapter_config_sha256': sha(config_path), 'adapter_weights_sha256': sha(weights_path),
            'tensor_shapes': shapes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = private_path(args.config)
    config = json.loads(config_path.read_text())
    root = private_path(config['output_root'])
    lock_path = private_path(config['context_lock_path'])
    contract_path = private_path(config['input_contract_path'])
    raw_path = private_path(config['ocr_raw_path'])
    lora = lora_binding(config)
    vision_path = private_path(Path(config['model_root'])/'config.json')
    vision = json.loads(vision_path.read_text())['vision_config']
    if (vision['patch_size'], vision['temporal_patch_size'], vision['spatial_merge_size']) != (14, 2, 2):
        raise ValueError('Qwen patch/grid explanation differs from the real model config')
    lock, contract, raw = [json.loads(path.read_text()) for path in (lock_path, contract_path, raw_path)]
    import yaml
    model_yaml_path = private_path(Path(config['recognizer_model_dir'])/'inference.yml')
    model_yaml = yaml.safe_load(model_yaml_path.read_text())
    transform_ops = model_yaml['PreProcess']['transform_ops']
    rec_shape = next(op['RecResizeImg']['image_shape'] for op in transform_ops if 'RecResizeImg' in op)
    if rec_shape != [3, 48, 320] or model_yaml['PostProcess']['name'] != 'CTCLabelDecode':
        raise ValueError('recognizer configuration differs from the explained shape/decode path')
    processor_source_path = PRIVATE/'.venvs/rarevlm-ocr/lib/python3.10/site-packages/paddlex/inference/models/text_recognition/processors.py'
    if 'self.max_imgW = 3200' not in processor_source_path.read_text():
        raise ValueError('official recognizer resize bound changed')
    if (lock['status'] != 'passed' or contract['status'] != 'passed'
            or contract['context_lock_sha256'] != sha(lock_path)
            or lock['ocr_raw_file_sha256'] != sha(raw_path)
            or contract['config_sha256'] != sha(config_path)):
        raise ValueError('locked real OCR/input contract missing or changed')
    if tuple(config['learning_question_ids']) != CASE_IDS:
        raise ValueError('fixed teaching examples changed')
    manifest_path = private_path(config['manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    rows = {str(row['question_id']): row for row in manifest['splits']['dev']}
    details = {(item['condition'], str(item['question_id'])): item for item in contract['records']}
    preds, prediction_files = predictions(root)
    cards = []
    for qid in CASE_IDS:
        row = rows[qid]
        iid = str(row['image_id'])
        image = manifest['images'][iid]
        source = manifest_path.parent/image['path']
        if sha(source) != image['sha256']:
            raise ValueError('original teaching image changed')
        w, h = int(image['width']), int(image['height'])
        context = lock['images'][iid]
        retained = {item['detector_index'] for item in context['context_items']}
        overflow = set(context['skipped_by_token_cap_detector_indices'])
        duplicate = set(context.get('duplicate_detector_indices', []))
        polygons, table = [], []
        for item in raw['images'][iid]['items']:
            index = int(item['detector_index'])
            color = '#167d62' if index in retained else ('#b56500' if index in overflow else '#687480')
            points = ' '.join(f'{float(x):.2f},{float(y):.2f}' for x, y in item['polygon'])
            polygons.append(f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="{max(w,h)/260}"/><text x="{float(item["polygon"][0][0]):.2f}" y="{float(item["polygon"][0][1]):.2f}" fill="{color}" font-size="{max(w,h)/55}">{index}</text>')
            reason = ('输入上下文' if index in retained else '完整行超 token 上限' if index in overflow
                      else '重复文字' if index in duplicate else '低置信度或空文字')
            crop_h, crop_w, channels = item['crop_shape']
            norm_w = min(int(48 * max(320/48, crop_w/crop_h)), 3200)
            table.append(f'<tr><td>{index}</td><td>{esc(item["rec_text"])}</td><td>{float(item["rec_score"]):.4f}</td><td>{float(item["det_score"]):.4f}</td><td>{esc(item["crop_shape"])} → [3,48,{norm_w}]</td><td>{reason}</td></tr>')
        original = f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="原图与预测检测框"><image width="{w}" height="{h}" href="{embed(source)}"/>{"".join(polygons)}</svg>'
        columns = []
        for condition in CONDITIONS:
            detail = details[condition, qid]
            donor = detail.get('shuffle_source_image_id')
            text = (lock['images'][donor]['context'] if condition == 'shuffled_ocr' else
                    context['context'] if condition == 'predicted_ocr' else '')
            prefix = '' if condition == 'baseline' else config['ocr_prefix_template'].replace('{context}', text)
            actual_prompt = prefix + row['question'] + lock['prompt_suffix']
            pred = preds.get((condition, qid))
            result = '<p class="pending">完整评测尚待完成，未填入模拟预测。</p>' if pred is None else f'<p>实际预测：<b>{esc(pred["prediction"])}</b><br>共识分 {float(pred["score"]):.3f}</p>'
            columns.append(f'<div class="condition"><h3>{LABELS[condition]}</h3><p>图像：同一完整原图；native resized={detail["resized_width"]}×{detail["resized_height"]}<br>grid={esc(detail["image_grid_thw"])}；视觉 token={detail["visual_tokens"]}<br>OCR 内容 token={detail["ocr_text_tokens"]}；前缀含内容 token={detail["ocr_prefix_tokens"]}<br>总输入 token={detail["input_tokens"]}<br>pixel_values={esc(detail["pixel_values_shape"])}<br>CPU float32 SHA={detail["pixel_values_sha256"][:12]}<br>OCR donor={esc(donor)}</p><details><summary>实际文字提示（无答案输入）</summary><pre>{esc(actual_prompt)}</pre></details>{result}</div>')
        cards.append(f'<article><h2>题 {qid} · 图 {esc(iid)}</h2><p>{esc(row["question"])}</p><div class="original">{original}</div><p><b>上图为原图加预测框的教学标注</b>，框、编号与颜色没有输入VLM。实际图像仍为同一完整原图，由原生比例processor编码；颜色区分绿色保留、橙色token预算跳过、灰色其他。图片仅按CSS显示尺寸缩放。</p><p>raw文字行={context["raw_items_count"]}；通过置信度且非空={context["eligible_items_count"]}；最终完整行={context["retained_items_count"]}；OCR文字 token={context["ocr_text_tokens"]}。</p><details><summary>全部真实识别输出、crop形状和筛选原因</summary><p>crop H×W×3来自保存记录；箭头右侧是按锁定官方预处理公式推算的单crop C×48×动态宽度，不是另存的实际识别张量。batch堆叠时还可能按该batch最大宽度补零。</p><table><thead><tr><th>检测索引</th><th>原始预测文字</th><th>识别分</th><th>检测分</th><th>记录crop→预处理形状推算</th><th>处理</th></tr></thead><tbody>{"".join(table)}</tbody></table></details><details open><summary>实际预测 OCR 上下文</summary><pre>{esc(context["context"])}</pre></details><div class="grid">{"".join(columns)}</div></article>')
    sources = (excerpt('src/vlm_ocr/recognize.py', ['recognize_image'])
               + excerpt(str(processor_source_path), ['OCRReisizeNormImg', 'BaseRecLabelDecode', 'CTCLabelDecode', 'ToBatch'])
               + excerpt('src/vlm_deep/model.py', ['load'])
               + excerpt('src/vlm_precision/evaluate.py', ['load_path'])
               + excerpt('src/vlm_ocr_compact/prepare.py', ['bind_upstream', 'build_image_context', 'prepare_contexts'])
               + excerpt('src/vlm_ocr/data.py', ['normalize_ocr_text', 'sattolo_sources', 'build_ocr_batch', '_check_group']))
    analysis_path = root/'compact_analysis/ANALYSIS.json'
    analysis_block = '<p>整体与切片结果尚待完整四组完成；下面的单例不能替代完整评测。</p>'
    if analysis_path.exists():
        analysis_block = '<details open><summary>完整分析原始记录</summary><pre>' + esc(analysis_path.read_text()) + '</pre></details>'
    introduction = '''<h1>紧凑 OCR 文字：复用完整识别，再做独立四组对照</h1>
<p>固定普通 CE seed42 最终适配器、原权重 BF16、SDPA、batch4、greedy32，新增训练为0。四条件×1,617问题＝6,468条预测，研究覆盖完整内部dev。下面8题取自此前已核对的失败/改善案例，只用于教学；不是研究子集或新选点。</p>
<p>前一方案把全图视觉预算分为两个512视图，文字区域整体79.035低于单整图84.886。当前方案让四组共享原生比例完整图像：上限1,024个视觉token，实际逐题记录；不会补到固定1,024，也不会增加letterbox灰边。历史native1024的87.112只是参考，本轮重新运行同协议强基线。</p>
<h2>为什么只推进这一个完整后续实验？</h2><p>上一轮完整四组：native87.112、空模板84.960、预测OCR85.733、置换OCR85.615。预测OCR−native为−1.379个百分点，95%区间[−2.468,−0.308]；预测OCR−置换仅+0.117且区间跨零。原规则因256token上限跳过1,893行，复杂前缀与坐标也占用输入。</p><p>本轮一次性固定短前缀＋不显式输入坐标＋纯文字完整行整理，仍运行完整1,617题和四组，不挑“可救回”的部分题。<b>这是联合变化</b>：即使更好，也不能把结果单独归因于移除坐标；文字覆盖、总输入长度、提示和信息都可能变化。它不等于等信息或等计算实验。</p><h2>算法怎样连接到真实代码？</h2>
<p>本轮只读复用上一轮完整 OCR_RAW/来源/模型/输入锁，<b>0次新训练、0次新识别，也没有提出新的OCR算法</b>。检测与识别链条是上游已完成工作；本轮只重新整理预测文字并评测。</p>
<p>完整链条：<code>detector四边形 → 官方透视矫正BGR uint8 crop → 识别器FP32缩放/归一化 → CTC greedy → confidence过滤/NFC去重/阅读顺序/纯文字完整行token packing → Qwen原生visual＋新增text</code>。官方透视裁剪对细长竖向crop还有旋转规则；Qwen原图使用RGB，这与OCR裁剪的BGR接口分开。</p>
<p>真实模型YAML声明<code>RecResizeImg.image_shape=[3,48,320]</code>。高度为48，单crop归一化宽度为<code>min(int(48×max(320/48, crop_width/crop_height)),3200)</code>；长文字可能超过320，不能写成所有crop固定48×320。crop从记录的H×W×3转为C×48×W的FP32，像素按<code>(pixel/255−0.5)/0.5</code>归一化并补零，批处理还会按该batch最大宽度对齐。</p>
<p>CTC greedy的例子：时间帧<code>A,A,blank,A,B,B → A,blank,A,B → AAB</code>，<b>先合并原序列中连续重复，再移除blank</b>；blank分隔的两个A不会被合为一个。这个序列仅解释解码，不是保存的某个案例输出。</p>
<p>Qwen真实配置采用14×14视觉patch，静态图像按2帧temporal patch展开，单patch特征维度<code>3×2×14×14=1176</code>；2×2空间合并后4个patch得到1个视觉token。下方<code>pixel_values=[premerge_patches,1176]</code>与grid来自真实CPU编码，不是示意张量。实际视觉token不一定达到上限1024。</p>
<p>文字检测器先给出四边形和检测分。官方四点透视裁剪将文字块转为识别输入，CPU的英文PP-OCRv4_mobile_rec识别器预测字符序列。CTC解码沿时间帧选择字符，合并连续重复字符并移除blank；识别分是选中字符帧置信度均值，不是答题正确概率。检测分也不是答案置信度。</p>
<p>输入构造保持预测识别分≥0.80、NFC与空白整理、阅读顺序去重；每行只输入预测原文字，不再输入坐标数值。几何信息仍决定阅读顺序，并保留在锁及教学标注中，不进入上下文字符串。上游裁剪与识别不查看问题、参考答案、训练目标或历史错误。</p>
<p>按框y_min、x_min、检测索引排序，大小写折叠只用于去重，原预测大小写保留。依次加入完整行：若候选整体超过256个文字token，跳过该行再尝试下一行，不截断识别字符串。完整预测文字行计入256；新前缀固定为<code>OCR text:\n{context}\n\n</code>，前缀加内容最多320。保留token界符仅作字面转义，防止OCR改变聊天/图像标记结构，不作拼写纠正。</p>
<p>VLM仍从完整图像视觉特征和问题生成答案，OCR只是有错误风险的附加文字信息。推理时不执行参考答案归一化；TextVQA共识分只在生成完成后计算。OCR可能纠正罕见名字拼写，也可能引导到错误对象，真实收益必须超过同协议native强基线。</p>
<h2>四组对照与工程边界</h2>
<p>baseline保留原问题＋原后缀；empty_template仅加入空OCR模板，排查提示改变；predicted_ocr是唯一候选；shuffled_ocr将同样规则得到的上下文换为另一张图。Sattolo在空/非空组内生成无固定点置换，每张不同图文字用一次。独图上下文集合保持，多个问题的图像重复次数不同，因此每题输入长度与QA加权算力不保证相同。</p>
<p>四组逐题比较原图SHA、RGB SHA、CPU float32 pixel_values SHA、grid和视觉token，严格相同；总输入token因OCR变化，不能称等计算实验。检测/识别准备按独图计算，本轮没有再次识别，检测/识别时间引用上游并明确为复用成本；重新整理文字和缓存VLM吞吐单列，不能冒充线上服务加速。首次32题×4条件做真实CPU输入预检，完整6,468次编码合同通过后才进入GPU研究。</p>
<p>主比较只有predicted_ocr−baseline：整体至少+0.5个百分点，并报告配对图像簇95%和97.5%区间，以97.5%区间下界>0作保守推进筛选；这不消除历史自适应选择偏差，也不是全项目家族错误率证明；rare点值不下降、frequent点值最多−0.5。点值保护不证明非劣，单checkpoint、已探索内部dev、次要和切片点式区间都要保留限制，不能把整体提升自动称作解决长尾。</p>
<h2>已有LoRA接点：这轮没有重新训练</h2>
<p>真实adapter配置和safetensors头已核对：只在36层语言self-attention的q_proj/v_proj接入rank8，alpha16。矩阵增量为<code>ΔW=(alpha/r)BA=2BA</code>。q：<code>A[8,2048], B[2048,8]</code>；v：<code>A[8,2048], B[256,8]</code>。B的输出维度不同来自q/v投影本身不同，不能把v参数量按2048输出计算。</p>
<p>总参数为<code>36×[8×(2048+2048)+8×(2048+256)]=1,843,200</code>。这些是既有普通CE训练得到并保存的最终adapter；本轮只更改推理输入上下文，复用同一权重并以BF16推理，不能称为新增LoRA训练或BF16训练加速。</p><h2>对应源码</h2>'''
    links = []
    for relative, label in [('src/vlm_deep/model.py', '既有LoRA创建与加载'),
                            ('src/vlm_precision/evaluate.py', '本轮复用的BF16加载路径'),
                            ('src/vlm_ocr_compact/prepare.py', '本轮紧凑文字整理与完整CPU合同'),
                            ('src/vlm_ocr/data.py', '冻结的原生图像编码器复用')]:
        href = os.path.relpath(REPO/relative, root/'learning')
        links.append(f'<a href="{esc(href)}">{label}：{esc(relative)}</a>')
    introduction += '<p>' + ' · '.join(links) + '</p>'
    coverage = lock['coverage_comparison_v1']
    introduction += f'<p>真实完整缓存整理：v1保留{coverage["old_retained_lines"]}行；compact保留{coverage["new_retained_lines"]}行；新加入{coverage["newly_admitted_lines"]}行，原保留而本轮排除{coverage["old_admitted_now_excluded_lines"]}行。覆盖增加本身不是答题收益。</p>'
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>紧凑OCR受控实验学习页</title><style>body{{font-family:system-ui,sans-serif;max-width:1440px;margin:25px auto;padding:0 20px;line-height:1.65;color:#21313b;background:#f5f7fa}}article{{background:#fff;border:1px solid #ced7df;border-radius:10px;padding:20px;margin:26px 0}}.original{{max-width:660px}}svg{{width:100%;height:auto}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px}}pre{{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:13px}}details{{margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ccd6df;text-align:left;padding:8px;overflow-wrap:anywhere}}.condition p{{overflow-wrap:anywhere}}.pending{{color:#865100}}@media(max-width:960px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:580px){{.grid{{grid-template-columns:1fr}}}}</style>{introduction}{sources}<h2>完整研究结果</h2>{analysis_block}<h2>八个真实教学样本</h2>{''.join(cards)}<p>OCR_CONTEXT_LOCK SHA {sha(lock_path)}<br>INPUT_CONTRACT SHA {sha(contract_path)}<br>配置 SHA {sha(config_path)}</p></html>'''
    output = root/'learning/学习页.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    provenance = {'schema_version': 'rarevlm.ocr_compact.reader.v1',
                  'reader_source_sha256': sha(Path(__file__)), 'config_sha256': sha(config_path),
                  'context_lock_sha256': sha(lock_path), 'input_contract_sha256': sha(contract_path),
                  'raw_ocr_sha256': sha(raw_path), 'prediction_file_sha256': prediction_files,
                  'recognizer_yaml_sha256': sha(model_yaml_path), 'recognizer_image_shape': rec_shape,
                  'recognizer_processor_source_sha256': sha(processor_source_path),
                  'existing_lora': lora, 'qwen_vision_config_sha256': sha(vision_path),
                  'analysis_sha256': sha(analysis_path) if analysis_path.exists() else None,
                  'question_ids': list(CASE_IDS), 'teaching_selection': 'eight previously diagnosed examples; never an evaluation subset',
                  'research_questions': 1617, 'new_training_runs': 0, 'new_recognition_calls': 0, 'new_predictions_by_reader': 0,
                  'upstream_v1': lock['upstream_v1'], 'coverage_comparison_v1': lock['coverage_comparison_v1'],
                  'overlays_are_teaching_only': True, 'original_images_embedded': True,
                  'page_sha256': sha(output), 'page_bytes': output.stat().st_size}
    (output.parent/'PROVENANCE.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'page': str(output), 'cards': len(cards), 'bytes': output.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
