"""Explain completed, saved divergence statistics; no model/scoring operations."""
import html
import math
import os
from pathlib import Path
import textwrap

from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import read

ROOT = PROJECT_ROOT.parents[1]/'outputs/vlm_native_divergence_readout_v1'
PARENT = ROOT.parent/'vlm_visual_preference_v1'
LABELS = {'start':'共同起点', 'ce':'视觉CE', 'ce_dpo':'视觉CE+DPO'}


def main():
    done = read(ROOT/'COMPLETION.json')
    rows = {r:{q['question_id']:q for q in read(ROOT/'runs'/r/'ROWS.json')} for r in LABELS}
    data = read(PARENT/'data/MANIFEST.json')
    dpo = rows['ce_dpo']
    mismatch = done['summaries']['ce_dpo']['mismatch_question_ids']
    persistent = [q for q in dpo.values() if q['saved_vqa_score']<1-1e-9 and q['original_negative_persists']]
    sequence_losses = sorted([q for q in persistent if q['sequence_margin']<0],key=lambda q:q['sequence_margin'],reverse=True)
    selected = list(mismatch[:1])
    selected += [q['question_id'] for q in sequence_losses if q['question_id'] not in selected][:2-len(selected)]
    selected += [q['question_id'] for q in persistent if q['question_id'] not in selected][:2-len(selected)]
    selected = selected[:2]
    e=lambda s:html.escape(str(s))
    title = '首分歧线索达到门槛，尚未训练' if done['mechanism_signal'] else '首分歧线索不足：不追加该训练'
    count=done['summaries']['ce_dpo']['sequence_win_local_loss_persistent']
    intro=f'<h1>{e(title)}</h1><p>{e(done["completed_at"])} · 三个固定模型/128训练QA/768 teacher-forced序列/0更新/0新生成。</p><p>DPO持续原负例残错中，整段胜出却首分歧选错：{count}条，事前准入门槛16条。机制线索不能当完整任务提分或唯一因果；原1617和96/192失败结论不变。</p>'
    formula='<h2>两个排序为什么可以不同？</h2><pre>整段差 Δseq = Σ log P(正确序列token | 其前缀) − Σ log P(错误序列token | 其前缀)\n首分歧差 Δfirst = log P(正确词元 | 共享前缀) − log P(错误词元 | 同一共享前缀)\nΔseq > 0，并不保证 Δfirst > 0。后续token已处于不同前缀，不能补救先前的greedy选错。</pre><p>本轮在真实模型上检查这种情况是否足够常见。teacher forcing使用已知答案前缀；greedy结果直接读取原保存生成，不把两者混为同一运行轨迹。</p>'
    summaries='<table><tr><th>模型</th><th>原生成满分</th><th>残错</th><th>持续原负例</th><th>整段胜/局部败且持续错误</th></tr>'
    for role in LABELS:
        s=done['summaries'][role]
        summaries+=f'<tr><td>{LABELS[role]}</td><td>{s["saved_full_score"]}/128</td><td>{s["residual"]}</td><td>{s["persistent_negative_residual"]}</td><td>{s["sequence_win_local_loss_persistent"]}</td></tr>'
    summaries+='</table>'
    cards=[]
    for qid in selected:
        q=dpo[qid];im=data['images'][q['image_id']]['path']
        table='<table><tr><th>模型</th><th>保存生成</th><th>Δseq</th><th>Δfirst</th><th>正确token rank</th></tr>'
        for role in LABELS:
            a=rows[role][qid]
            table+=f'<tr><td>{LABELS[role]}</td><td>{e(a["saved_prediction"])}</td><td>{a["sequence_margin"]:+.4f}</td><td>{a.get("first_margin",float("nan")):+.4f}</td><td>{a.get("first_chosen_rank","无可比分歧")}</td></tr>'
        table+='</table>'
        cards.append(f'<article><h3>QA {qid}</h3><img src="{e(os.path.relpath(im,ROOT))}"><p>{e(q["question"])}</p><p>训练目标：{e(q["target"])}；原负例：{e(q["rejected_target"])}</p>{table}<p>首分歧位置{q["first_divergence_index"]}；正确词元 {e(repr(q.get("first_chosen_text")))} vs原负例词元 {e(repr(q.get("first_rejected_text")))}。这些是token片段，不一定是完整单词。</p></article>')
    full='<table><tr><th>QA</th><th>正确答案</th><th>原负例</th><th>DPO保存生成</th><th>Δseq</th><th>Δfirst</th><th>持续原负例</th></tr>'
    for q in dpo.values():
        full+=f'<tr><td>{e(q["question_id"])}</td><td>{e(q["target"])}</td><td>{e(q["rejected_target"])}</td><td>{e(q["saved_prediction"])}</td><td>{q["sequence_margin"]:+.5f}</td><td>{q.get("first_margin",float("nan")):+.5f}</td><td>{q["original_negative_persists"]}</td></tr>'
    full+='</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DPO首分歧机制</title><style>body{font:16px/1.8 system-ui;max-width:1400px;margin:25px auto;padding:20px;color:#173044;background:#f4f6f8}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{padding:18px;background:white}img{width:100%;height:390px;object-fit:contain}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccd5dd}pre{white-space:pre-wrap;background:#e8eff3;padding:16px}details{overflow:auto}@media(max-width:800px){.cards{grid-template-columns:1fr}}</style>'
    page+=intro+summaries+formula+'<div class="cards">'+''.join(cards)+'</div><p>事后选择的真实训练案例，仅原图等比缩放，不是模型注意力或独立评测；不修改GT/分数，不回填新训练。</p><details><summary>全部128题概率与保存生成</summary>'+full+'</details><p><a href="COMPLETION.json">完成结论</a> · <a href="../rarevlm_monitor/状态看板.html">观察台</a></p></html>'
    (ROOT/'整段偏好与首分歧.html').write_text(page)
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    fontpath='/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'
    fonts={size:ImageFont.truetype(fontpath,size) for size in (18,20,27)}
    canvas=Image.new('RGB',(1500,1120),'#f4f6f8');draw=ImageDraw.Draw(canvas)
    draw.text((24,20),title,fill='#173044',font=fonts[27])
    draw.text((24,67),f'目标错型 {count}/门槛16；三模型共768 TF序列，0训练/0生成；原完整任务仍未达标',fill='#475569',font=fonts[20])
    for column,qid in enumerate(selected):
        q=dpo[qid];x=20+column*745;y=120
        draw.rounded_rectangle((x,y,x+720,1065),radius=12,fill='white')
        draw.text((x+15,y+12),f'QA {qid} · '+('整段胜出但局部败' if qid in mismatch else '整段仍未排对的持续错误'),fill='#173044',font=fonts[20])
        im=ImageOps.contain(Image.open(data['images'][q['image_id']]['path']).convert('RGB'),(685,450))
        canvas.paste(im,(x+(720-im.width)//2,y+50+(450-im.height)//2))
        lines=textwrap.wrap(q['question'],width=60)+['目标：'+q['target'],'原负例：'+q['rejected_target'],'']
        for role in LABELS:
            a=rows[role][qid]
            lines+=textwrap.wrap(LABELS[role]+'生成：'+a['saved_prediction'],width=58)
            lines+=[f'整段差 {a["sequence_margin"]:+.4f}；首分歧差 {a.get("first_margin",float("nan")):+.4f}；正确token rank {a.get("first_chosen_rank","NA")}']
        lines+=['','整段差>0：正确序列概率较高；首分歧差<0：此处偏向错token。']
        ty=y+520
        for line in lines:
            for part in [line[i:i+47] for i in range(0,len(line),47)] or ['']:
                draw.text((x+15,ty),part,fill='#173044',font=fonts[18]);ty+=27
    draw.text((24,1080),'真实训练原图与保存回答；概率为本轮teacher forcing，不是greedy逐步trace或注意力。',fill='#475569',font=fonts[18])
    canvas.save(ROOT/'首分歧机制与原图.png')
    print({'status':'rendered_saved_results','html':str(ROOT/'整段偏好与首分歧.html'),'png':str(ROOT/'首分歧机制与原图.png'),'case_ids':selected})


if __name__=='__main__':main()
