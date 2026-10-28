"""Actual prefix budgets, calibration data and returned-image comparisons."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from clip_retrieval_rerank.common import write_json
from vlm_probe.evaluate import sha_file,now


def build(config_path,dev,test=None):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);folder=root/'learning';folder.mkdir(exist_ok=True)
    policy=json.loads((root/'POLICY.json').read_text());cal=json.loads((root/'CALIBRATION.json').read_text())
    rows=json.loads((root/'DATA_MANIFEST.json').read_text())['splits']['dev'];index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    photos={};examples=[]
    def picture(image,name):
        handles={}
        try:
            im=read_image(rows[image],index,handles);im.thumbnail((360,360));im.save(folder/name,quality=92)
        finally:
            for f in handles.values():f.close()
        photos[name]={'source_image_id':rows[image]['image_id'],'source_image_sha256':rows[image]['image_sha256'],
            'thumbnail_sha256':sha_file(folder/name),'scope':'display thumbnail,not processor input/attention'}
        return '<img src="'+name+'" alt="实际返回的候选源图缩略展示">'
    if dev['status']!='calibration_rejected_before_gpu':
        with np.load(root/'routes/dev/seed42.npz') as z:budgets={d:z[d+'_budget'].copy() for d in ['T2I','I2T']}
        with np.load(root/'confidence/dev/seed42.npz') as z:margins={d:z[d+'_margin'].copy() for d in ['T2I','I2T']}
        outputs={}
        for condition in ['full','adaptive']:
            with np.load(root/'evaluation'/f'{condition}_seed42'/'dev/per_query.npz') as z:outputs[condition]={k:z[k].copy() for k in z.files}
        confirm=np.zeros(1000,dtype=bool);confirm[json.loads((root/'PARTITION.json').read_text())['confirm_image_indices']]=True
        for direction in ['T2I','I2T']:
            eligible=np.repeat(confirm,5) if direction=='T2I' else confirm
            a=outputs['full'][direction+'_R@1'];b=outputs['adaptive'][direction+'_R@1']
            groups=[('Top20保持',eligible&(budgets[direction]==20)&(a==b)),('Top20退化',eligible&(budgets[direction]==20)&(b<a)),
                    ('Top20改进',eligible&(budgets[direction]==20)&(b>a)),('Top50',eligible&(budgets[direction]==50))]
            for label,mask in groups:
                ids=np.flatnonzero(mask).tolist();ids.sort(key=lambda q:(rows[q//5 if direction=='T2I' else q]['image_sha256'],q))
                if not ids:continue
                q=ids[0];image=q//5 if direction=='T2I' else q;views=[]
                if direction=='T2I':
                    lead='<p>输入caption：'+html.escape(rows[image]['captions'][q%5])+'</p>'
                    for condition in ['full','adaptive']:
                        target=int(outputs[condition]['t2i_top50'][q,0]);views.append('<section><h4>'+('full50' if condition=='full' else '预算分配')+f' Top1图库{target}</h4>'+picture(target,f'{direction}_{q}_{condition}.jpg')+'</section>')
                else:
                    lead=picture(image,f'{direction}_{q}_query.jpg');
                    for condition in ['full','adaptive']:
                        items=[]
                        for k in outputs[condition]['i2t_top50'][q,:5]:
                            i,j=divmod(int(k),5);items.append('<li>'+('标注正描述：' if i==image else '其他图描述：')+html.escape(rows[i]['captions'][j])+'</li>')
                        views.append('<section><h4>'+condition+'</h4><ol>'+''.join(items)+'</ol></section>')
                examples.append('<article><h3>'+direction+' '+label+f' query{q}，seed42</h3>'+lead+
                    f'<p>真实margin={float(margins[direction][q]):.6f}，fast阈值={policy["selection"][direction]["threshold"]:.6f}，κ={policy["budget_selection"][direction]["cutoff"]:.6f}，实际K={int(budgets[direction][q])}。</p><div class="columns">'+''.join(views)+'</div></article>')
    table=''
    if 'summaries' in dev:
        table='<table><tr><th>500确认图</th><th>文搜图R1均值±SD</th><th>图搜文R1均值±SD</th></tr>'
        for condition,s in dev['summaries'].items():table+='<tr><td>'+condition+'</td>'+''.join(f'<td>{s[k]["mean"]:.3f} ± {s[k]["sample_sd"]:.3f}</td>' for k in ['T2I_R@1','I2T_R@1'])+'</tr>'
        table+='</table>'
    state={'gate':dev['gate'],'comparisons':dev.get('comparisons'),'cost':dev['cost'],'test_state':'completed' if test else 'not_run_dev_gate_failed'}
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM：0/20/50候选预算</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto;padding:24px;color:#222}img{width:350px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px}.columns{display:flex;gap:18px;flex-wrap:wrap}.columns section{flex:1;min-width:300px}table{border-collapse:collapse}th,td{border:1px solid #777;padding:8px}article{border-top:1px solid #888;margin-top:24px}</style><h1>需要重排多少个候选？</h1>
<p>对应Embedding/Rerank和检索成本优化JD。0新神经网络训练，固定旧3CLIP＋1BLIP；只cal500图选择κ，confirm500图验证，历史数据已曝光，不称新盲测。</p>
<h2>原理与真实代码</h2><pre>m = original CLIP top1 cosine − top2 cosine
if m &gt; frozen_fast_threshold: K = 0       # unchanged CLIP
elif m &lt; calibration_only_cutoff: K = 50 # harder query
else: K = 20                             # rerank first20 only
pair_keys = unique(original_pool_prefix[:K], selected queries)
output[:K] = ITM reorder; output[K:] = original CLIP tail
</pre><p>route.py的budgets_from_margin、budget_keys、budget_ranking不接收标注。calibrate.py使用旧完整分数仅模拟cal选择；matcher.py复用固定实际评分器，对新选中的keys分别fresh forward。双塔一次生成512维表征，BLIP crossattention每32pair处理文本[32,64]和图像hidden[32,577,768]，ITM输出[32,2]。减少K直接减少联合匹配次数。</p>
<p>保留输出Top50不代表全部50候选都评分：Top20分支不能把尾部正项升到Top1。JSON另外报告effective_prefix_recall；K0的上限是原Top1是否正确，K20/50上限是对应prefix是否含标注正项。覆盖只在评分后计算，不能把参考答案传给路由。</p>
<h2>校准扫描及Top20保留率</h2><pre>'''+html.escape(json.dumps({'policy':policy,'cal_only':cal},ensure_ascii=False,indent=2))+'''</pre>
<h2>真实质量与成本</h2>'''+table+'''<pre>'''+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'''</pre><p>NI每方向.5pp、6000图像簇/单侧alpha=.05/2；实际pair至少减少30%、三轮同card同batch阶段速度中位≥1.2且每轮≥1。视图/模型加载/旧输入缓存/排队与线上SLA分开；三计时重复不是独立质量样本。</p>
<h2>真实Top20／Top50病例</h2><p>按confirm源图SHA取各类首例，预测完成后展示，不用于调κ。源图为展示缩略图，文搜图输入是caption；其他图描述不自动等于所有语义错误。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>手算margin并判断0/20/50。</li><li>解释K20为什么使R50保持但有效Top1上限可能变小。</li><li>核对只有实际budget prefix出现在pair_keys中。</li><li>说明cal选点与confirm NI之间的分工。</li><li>解释完整BLIP分数用于cal模拟与用于伪造加速的区别。</li></ol><p><a href="../POLICY.json">冻结策略</a>；<a href="../CALIBRATION.json">校准诊断</a>；<a href="../analysis/dev/ANALYSIS.json">完整确认统计</a>。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'policy_sha256':sha_file(root/'POLICY.json'),'photos':photos,'scope':'actual budgets/margins/source pictures/saved rankings,no attention',
        'manual_visual_review':'pending'})
