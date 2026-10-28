"""Source photographs and actual run outputs, with uncertainty kept visible."""
import html
import json
import os
from pathlib import Path

def build(cfg, result=None):
    root=Path(cfg['output_root']);data=Path(cfg['public_data_root']);out=root/'learning';out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((data/'semantic_audit/FROZEN_64.json').read_text())
    review=json.loads((data/'semantic_audit/VISUAL_REVIEW.json').read_text());esc=html.escape
    cards=[]
    for row,judgment in zip(audit['samples'],review['records']):
        path=Path(row['image_path']);path=path if path.is_absolute() else data/path
        rel=os.path.relpath(path,out)
        cards.append(f'<article><a href="{rel}"><img loading="lazy" src="{rel}" alt="源图 {row["source_idx"]}"></a><div>'
                     f'<h3>{row["source_idx"]} · {esc(row["question"])}</h3><p><b>chosen</b> {esc(row["chosen"])}</p>'
                     f'<p><b>rejected</b> {esc(row["rejected"])}</p><p><b>{esc(judgment["category"])}</b>：{esc(judgment["note"])}</p></div></article>')
    summary='<p>训练/评测尚未完整完成；本页没有新增提分数字。</p>' if result is None else '<pre>'+esc(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RareVLM 公开局部纠错偏好诊断</title>
<style>body{max-width:1180px;margin:30px auto;padding:0 16px;font:17px/1.7 system-ui;color:#172434;background:#f5f7fa}article{display:grid;grid-template-columns:35% 1fr;gap:22px;background:white;padding:20px;border:1px solid #cfd8e3;border-radius:12px;margin:20px 0}img{max-width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;background:#edf1f6;padding:18px;overflow-wrap:anywhere}code{background:#edf1f6}aside{border-left:5px solid #ba5931;padding:15px;background:#fff5ee}@media(max-width:700px){article{grid-template-columns:1fr}}</style>
<h1>人类局部纠错数据，为什么仍需要真实生成对照？</h1>
<aside>事实质量门槛未通过。这里是单独预注册的噪声偏好诊断，不是可信数据/无噪声RLHF成功，也不自动晋级或修改简历。原749/199失败候选没有训练。</aside>
<p>官方RLHF-V5733对→排121→5612对双completion实际编码→每图一对4007→训练3568、heldout439。SHA/像素去重同时排既有split和POPE500图。上限1024视觉token和2048总token逐样本实际检查，不截断；上限不代表每张图都1024。</p>
<h2>公式、张量与真实代码</h2>
<pre>同图/同问题：SFT batch [1,L_chosen]，DPO pair batch [2,max(L_chosen,L_rejected)]
labels只保留assistant回答和原结束token；prompt、image、padding为-100
sequence_logps: [2,L,V] → assistant gather/sum → [2]
SFT = -logp_chosen / chosen_token_count
DPO = -logsigmoid(beta * ((lp_c-lp_r) - (ref_c-ref_r)))
beta=.1，初始化policy==reference：loss=log(2)，dL/dlp_c=-.05，dL/dlp_r=+.05</pre>
<p>reference必须来自固定初始SFT缓存，不能每次取当前actor.detach。训练LoRA保存FP32以容纳小更新，基座BF16冻结，两组相同初始化/曝光/223更新。真实预检执行临时一步后完整恢复参数/RNG，工程step与正式step分开统计。</p>
<p>真实代码：<code>vlm_posttrain/loss.py::pair_objective</code>、<code>train.py::reference_cache/preflight</code>、<code>vlm_posttrain_pope/scoring.py</code>。CPU/GPU合同不是提分；loss变小也不等于更少幻觉。</p>
<h2>三个观察窗口</h2><p>POPE9000实际yes/no生成测试物体幻觉；TextVQA1617实际短答测试能力保持；heldout439是teacher-forced概率比较，受答案长度影响，不叫生成正确率。三者分开报告，不把seed×question或三category×image当独立重复。</p>
<h2>运行结果</h2>'''+summary+f'<h2>固定64例数据审查</h2><p>明确{review["clear"]}/64，Wilson95% {review["wilson95"]}；助手局部观察，不是独立人审或全回答无错误证明。</p>'+''.join(cards)+'</html>'
    (out/'学习页.html').write_text(doc)

