import html
import json
from pathlib import Path
from PIL import Image
from .common import write_json
from vlm_probe.evaluate import now

def build(config_path,result):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);out=root/'learning';out.mkdir(exist_ok=True)
    roster=json.loads((root/'DATA_ROSTER.json').read_text());rows={r['question_id']:r for r in roster['questions']}
    records=[json.loads(line) for role in cfg['roles'] for line in (root/'roles'/f'role{role}'/'predictions.jsonl').read_text().splitlines()]
    eligible=[x for x in records if x['preference'] is not None];zero=[x for x in records if max(x['rewards'])==min(x['rewards'])]
    selected=eligible[:4]+zero[:2];cards=[];pictures=[]
    for x in selected:
        row=rows[x['question_id']];image=roster['images'][row['image_id']]
        name=f'question_{row["question_id"]}.jpg';im=Image.open(Path(roster['image_root'])/image['path']).convert('RGB')
        im.thumbnail((700,480));im.save(out/name,quality=90)
        pictures.append({'path':str(out/name),'image_id':row['image_id'],'source_sha256':image['sha256'],
            'scope':'source thumbnail;not actual1024 preprocessing or attention'})
        table=''.join(f'<tr><td>{i}</td><td>{html.escape(s["prediction"])}</td><td>{r:.3f}</td><td>{a:.3f}</td></tr>'
            for i,(s,r,a) in enumerate(zip(x['samples'],x['rewards'],x['advantages'])))
        cards.append(f'<article><h3>{html.escape(row["question"])}</h3><img src="{name}"><p>greedy：{html.escape(x["greedy"]["prediction"])}；score={x["greedy_score"]:.3f}。实际visual tokens={x["input"]["visual_tokens"]}。</p><p>十参考（只生成后评分）：{html.escape(str(row["answers"]))}</p><table><tr><th>采样编号</th><th>模型实际回答</th><th>共识reward</th><th>组内advantage</th></tr>{table}</table><p>DPO候选索引：{html.escape(str(x["preference"]))}；规则分数差不自动等于真实人类偏好，语义审核待完成。</p></article>')
    intro='<h1>多模态后训练：先检查可学习信号</h1><p>SFT教模型生成给定回答；DPO比较同题优选/劣选回答相对reference的概率；GRPO在同题多个生成间比较reward，通过advantage指导策略更新。</p><p>DPO：L=−logσ[β((logπchosen−logπrejected)−(logπref_chosen−logπref_rejected))]，β=.1。logπ只加总assistant token，不包含图像/问题/padding；ref停止梯度。初始policy=ref时L=log2≈.6931，chosen logπ导数−.05、rejected +.05。</p><p>GRPO信号诊断：A=(r−mean(r))/(std_population(r)+1e−8)。同题四个reward相同则全部A=0；改变输出措辞不保证获得可靠语义信号。这里只计算优势，没有在线GRPO更新。</p><p>512不同训练图，greedy1＋采样4，共2560输出；固定SFT seed42，原BF16/1024。不是dev提分或四训练seed。GPU仅检查真实DPO前后向，不执行optimizer.step。</p>'
    text='<!doctype html><html lang="zh"><meta charset="utf-8"><title>RareVLM 后训练信号</title><style>body{max-width:1150px;margin:24px auto;padding:16px;font:16px/1.65 sans-serif;color:#18222e}img{max-width:700px;width:90%;max-height:480px;object-fit:contain}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}article{padding:18px;margin:20px 0;background:#f3f6fa}pre{white-space:pre-wrap;font-size:13px}</style>'+intro+'<h2>真实统计</h2><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)+'<p>代码：prepare.py数据边界→generate.py实际生成→common.py偏好/优势→loss.py assistant logp/DPO→analysis.py全量重评分。检查chosen/rejected是否只是同义表达、裁剪文本、单位差异或确切错误，然后才讨论完整训练。个人掌握待确认。</p></html>'
    (out/'学习页.html').write_text(text)
    write_json(out/'PROVENANCE.json',{'created_at':now(),'pictures':pictures,'semantic_review':'pending;no label injection',
        'new_optimizer_updates':0,'personal_mastery':'pending'})
