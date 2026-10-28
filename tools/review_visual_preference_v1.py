"""Saved-score review of the completed visual CE/DPO pilot; no inference."""
from collections import defaultdict
import datetime
import html
import os
from pathlib import Path
import statistics
import textwrap

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_visual_preference_v1'


def main():
    manifest = read(ROOT/'data/MANIFEST.json'); meta = {q['question_id']:q for q in manifest['evaluations']['transfer_dev']}
    paths = {'start':ROOT/'baseline/transfer_dev/PREDICTIONS.json',**{r:ROOT/f'runs/{r}/transfer/transfer_dev/PREDICTIONS.json' for r in ('ce','ce_dpo')}}
    predictions = {r:{q['question_id']:q for q in read(p)} for r,p in paths.items()}
    analysis = read(ROOT/'TRANSFER_ANALYSIS.json')
    result = {'status':'completed','observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope':'Posthoc saved-score internal review, single checkpoint; no new inference, rescore, intervals or changed gates.',
        'files_sha256':{str(p):sha_file(p) for p in [*paths.values(),ROOT/'TRANSFER_ANALYSIS.json',ROOT/'data/MANIFEST.json']},'comparisons':{},'training_residuals':{}}
    for name,base,role in [('ce_vs_start','start','ce'),('dpo_vs_start','start','ce_dpo'),('dpo_vs_ce','ce','ce_dpo')]:
        a = predictions[base]; b = predictions[role]
        if set(a)!=set(meta) or set(b)!=set(meta) or any(a[q]['input_sha256']!=b[q]['input_sha256'] for q in meta):
            raise ValueError('Prediction completeness/prefix differs')
        deltas = {q:b[q]['vqa_score']-a[q]['vqa_score'] for q in meta}
        changes = [{**meta[q],'before':a[q]['prediction'],'after':b[q]['prediction'],'scores':[a[q]['vqa_score'],b[q]['vqa_score']],'delta':v} for q,v in deltas.items() if abs(v)>1e-9]
        strata = {}
        for field in ('stratum','frequency_slice'):
            groups = defaultdict(list)
            for q,row in meta.items():
                groups[row[field]].append(q)
            strata[field] = {k:{'n':len(ids),'start_percent':100*statistics.mean(a[q]['vqa_score'] for q in ids),'after_percent':100*statistics.mean(b[q]['vqa_score'] for q in ids),'delta_pp':100*statistics.mean(deltas[q] for q in ids),'ci_computed':False} for k,ids in groups.items()}
        result['comparisons'][name] = {'gain_count':sum(q['delta']>0 for q in changes),'loss_count':sum(q['delta']<0 for q in changes),'net_score':sum(deltas.values()),'delta_pp':100*statistics.mean(deltas.values()),'strata':strata,'changes':changes}
    hard = {q['question_id']:q for q in manifest['rows'] if q['study_source']=='preference'}
    for role in ('ce','ce_dpo'):
        rows = read(ROOT/f'runs/{role}/internal/native_hard/PREDICTIONS.json'); residual = [q for q in rows if q['vqa_score']<1-1e-9]
        changed = [q for q in residual if q['completion_ids']!=hard[q['question_id']]['rejected_completion_ids']]
        result['training_residuals'][role] = {'full_score':128-len(residual),'residual':len(residual),'unchanged_original_negative':len(residual)-len(changed),'changed_answer':len(changed),'scope':'Exact observed completion IDs; not a semantic error taxonomy.'}
    queues = [read(ROOT/name/'STATUS.json') for name in ('reference_queue','training_queue','transfer_queue')]
    jobs = [j for q in queues for j in q['jobs'].values()]
    result['execution'] = {'worker_jobs':len(jobs),'all_exit0':all(j.get('exit_code')==0 for j in jobs),'contention_observations':sum(j.get('contention_observations',0) for j in jobs),'completed_at':queues[-1]['updated_at'],'new_updates':64,'new_generations':5154}
    write_json(ROOT/'RESULT_REVIEW.json',result)
    selected = ['32904','27323','11435']
    notes = {'32904':'机翼可见BREITLING，brilliant→breitling有原图支持；不能由单例外推普遍识字提升。',
        '27323':'后部标志可见The Old Tramshed。另一题完整编号LAK 309G却变成309G（1→0.3）；其route提问有范围风险，不改原分数。',
        '11435':'盒上可见LUCKY STRIKE，lucia缺乏该盒文字支持。是真实品牌误读，内部原因仍未确定。'}
    e = lambda x:html.escape(str(x)); cards = []
    for qid in selected:
        q = meta[qid]; im = Path(manifest['images'][q['image_id']]['path'])
        answers = '<br>'.join(f'{e(label)}：{e(predictions[role][qid]["prediction"])}（{predictions[role][qid]["vqa_score"]:.1f}）' for role,label in [('start','起点'),('ce','CE'),('ce_dpo','CE+DPO')])
        cards.append(f'<article><h3>QA {e(qid)}</h3><img src="{e(os.path.relpath(im,ROOT))}"><p>{e(q["question"])}</p><p>{answers}</p><p>{e(notes[qid])}</p></article>')
    table = ''.join(f'<tr><td>{e(name)}</td><td>{e(q["question_id"])}</td><td>{e(q["question"])}</td><td>{e(q["before"])}</td><td>{e(q["after"])}</td><td>{q["scores"][0]:.2f}→{q["scores"][1]:.2f}</td></tr>' for name,rec in result['comparisons'].items() for q in rec['changes'])
    intro = '<h1>视觉CE+DPO：正向点值，未通过能力门槛</h1><p>完整1617：起点87.198516%、CE87.303649%、CE+DPO87.631416%。候选对起点+0.432900pp，95%[−0.056866,+0.957398]；对匹配CE+0.327767pp，[−0.104362,+0.798247]，两者均未达到+0.5pp及正区间要求。全部5个worker exit0、0争用，64更新/5154新生成，13:03北京时间结束；没有运行故障证据。</p><p>对起点19题升/11题降、净得分+7；完整1617 rare342 −0.497076pp、多词469 −0.191898pp为事后无切片区间点值。原512初筛rare366 +0.546448pp来自不同cohort，不能拼成已改善长尾；不据跨0区间宣称无损。</p>'
    learning = '<h2>从公式到实际结果</h2><pre>L = CE(chosen) + λ × [−log sigmoid(β × ((logπθ(chosen)−logπθ(rejected)) − (logπref(chosen)−logπref(rejected))))]\nλ = 0 / 1，β = 0.1；两组均计算同样的4chosen+4negative。\n视觉LoRA 60因子 / 1843200参数训练；语言144因子冻结；同32步/512chosen+256negative。</pre><p>DPO直接改变相对reference的答案偏好，不直接保证新图像问答正确。训练满分CE37/128、DPO43/128；DPO残错85中65仍为原负答案、20换了错误回答。因此不能把静态负例过时称为主要病因。相对CE的点值增加不等于已经证明策略更优。</p><h2>下一步：相同曝光预算下的数据覆盖</h2><p>先筛选3072条新的训练池原生QA，至少8/10共识、排旧筛选/训练/评测图像与像素、排当前训练target，每target最多两图。原图审查≥64可信新错例与64新target后，才另行冻结96独立纠错各2次 vs 192各1次的匹配对照；现在只筛选、0新更新，不从验证错例补训练，也不承诺数据是唯一瓶颈。</p>'
    links = f'<p><a href="{e(os.path.relpath(PROJECT_ROOT/"tools/run_visual_preference_v1.py",ROOT))}">真实训练源码</a> · <a href="{e(os.path.relpath(PROJECT_ROOT/"docs/VLM_NATIVE_DIVERSITY_MINING_V1_PROTOCOL.md",ROOT))}">新筛选协议</a> · <a href="RESULT_REVIEW.json">完整结果JSON</a> · <a href="../rarevlm_monitor/状态看板.html">看板</a></p>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视觉DPO完整结果与数据覆盖</title><style>body{font:16px/1.8 system-ui;max-width:1440px;margin:30px auto;padding:20px;color:#173044;background:#f4f6f8}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:16px}img{width:100%;height:370px;object-fit:contain}pre{white-space:pre-wrap;background:#e8eff3;padding:20px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccd5dd}details{overflow:auto}@media(max-width:800px){.cards{grid-template-columns:1fr}}</style>'+intro+'<div class="cards">'+''.join(cards)+'</div><p>原图等比缩放展示，非模型注意力；助手非专家事后复核，不修改原标签/预测/评分。</p>'+learning+links+'<details><summary>完整升降：两控制比较均保留</summary><table><tr><th>比较</th><th>QA</th><th>问题</th><th>之前</th><th>之后</th><th>分数</th></tr>'+table+'</table></details></html>'
    (ROOT/'完整结果与原图.html').write_text(page)
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    fontpath = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'; title = ImageFont.truetype(fontpath,29); font = ImageFont.truetype(fontpath,20); small = ImageFont.truetype(fontpath,18)
    canvas = Image.new('RGB',(1600,1040),'#f4f6f8'); draw = ImageDraw.Draw(canvas)
    draw.text((25,18),'视觉DPO：1617题 +0.433pp，区间跨0，未通过原能力门槛',fill='#173044',font=title)
    draw.text((25,67),'19题升 / 11题降；起点87.1985% → CE87.3036% → CE+DPO87.6314%',fill='#475569',font=font)
    for column,qid in enumerate(selected):
        q = meta[qid]; x = 20+530*column; y = 120; draw.rounded_rectangle((x,y,x+510,990),radius=10,fill='white'); draw.text((x+14,y+12),'QA '+qid,fill='#173044',font=font)
        im = ImageOps.contain(Image.open(manifest['images'][q['image_id']]['path']).convert('RGB'),(480,400)); canvas.paste(im,(x+(510-im.width)//2,y+48+(400-im.height)//2)); ty = y+468
        lines = textwrap.wrap(q['question'],width=46)+['']
        for role,label in [('start','起点'),('ce','CE'),('ce_dpo','CE+DPO')]:
            pred = predictions[role][qid]; lines += [label+'：'+pred['prediction']+f' ({pred["vqa_score"]:.1f})']
        lines += ['']+[notes[qid][i:i+23] for i in range(0,len(notes[qid]),23)]
        for line in lines:
            draw.text((x+14,ty),line,fill='#173044',font=small); ty += 26
    draw.text((25,1005),'原图缩放与实际保存回答；单seed内部事后解释，不改GT或分数，不冒充模型输入裁图/注意力。',fill='#475569',font=small)
    canvas.save(ROOT/'完整结果与真实案例.png')
    print({k:{a:b for a,b in v.items() if a not in ('changes','strata')} for k,v in result['comparisons'].items()})


if __name__ == '__main__':
    main()
