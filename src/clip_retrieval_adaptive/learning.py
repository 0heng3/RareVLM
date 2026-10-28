"""Show actual margin decisions, calibration traces, images and ranking errors."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from clip_retrieval_rerank.common import write_json
from vlm_probe.evaluate import now,sha_file


def build(config_path,dev,test=None):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    d=root/'learning';d.mkdir(exist_ok=True);policy=json.loads((root/'POLICY.json').read_text());cal=json.loads((root/'CALIBRATION.json').read_text())
    rows=json.loads((root/'DATA_MANIFEST.json').read_text())['splits']['dev'];index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    partition=json.loads((root/'PARTITION.json').read_text());mask=np.zeros(1000,dtype=bool);mask[partition['confirm_image_indices']]=True
    photos={}
    def picture(i,name):
        handles={}
        try:
            im=read_image(rows[i],index,handles);im.thumbnail((400,400));im.save(d/name,quality=92)
        finally:
            for f in handles.values():f.close()
        photos[name]={'image_id':rows[i]['image_id'],'source_image_sha256':rows[i]['image_sha256'],'thumbnail_sha256':sha_file(d/name),
            'scope':'source display thumbnail,not224/384processor input or attention'}
        return '<img src="'+name+'" alt="真实源图的展示缩略副本">'
    with np.load(root/'confidence/dev/seed42.npz') as z:confidence={k:z[k].copy() for k in z.files}
    with np.load(root/'routes/dev/seed42.npz') as z:routes={k:z[k].copy() for k in z.files}
    with np.load(old/'candidates/dev/seed42.npz') as z:clip={k:z[k].copy() for k in z.files}
    data={}
    for variant in ['full','adaptive']:
        with np.load(root/'evaluation'/f'{variant}_seed42'/'dev/per_query.npz') as z:data[variant]={k:z[k].copy() for k in z.files}
    examples=[]
    for direction in ['T2I','I2T']:
        eligible=np.repeat(mask,5) if direction=='T2I' else mask;route=routes[direction+'_rerank']
        full=data['full'][direction+'_R@1'];adaptive=data['adaptive'][direction+'_R@1']
        groups=[('fast且保持',eligible&~route&(full==adaptive)),('fast退化',eligible&~route&(adaptive<full)),
                ('fast改进',eligible&~route&(adaptive>full)),('rerank',eligible&route)]
        for label,select in groups:
            ids=np.flatnonzero(select).tolist();ids.sort(key=lambda q:(rows[q//5 if direction=='T2I' else q]['image_sha256'],q))
            if not ids:continue
            q=ids[0];image=q//5 if direction=='T2I' else q;margin=float(confidence[direction+'_margin'][q]);threshold=policy['selection'][direction]['threshold']
            text=('<p>输入caption：'+html.escape(rows[image]['captions'][q%5])+'</p><p>下图是标注配对图片的参照，不是文搜图的query输入。</p>') if direction=='T2I' else '<p>输入是图像；下图为源图展示，不是实际224/384预处理张量。</p>'
            views=[]
            for variant,rank in [('CLIP',clip[direction.lower()]),('full',data['full'][direction.lower()+'_top50']),('adaptive',data['adaptive'][direction.lower()+'_top50'])]:
                items=[]
                for k in rank[q,:5]:
                    if direction=='I2T':
                        i,j=divmod(int(k),5);items.append('<li>'+('标注正描述：' if i==image else '其他图描述：')+html.escape(rows[i]['captions'][j])+'</li>')
                    else:items.append('<li>图库行'+str(int(k))+('（标注配对）' if int(k)==image else '（其他图）')+'</li>')
                views.append('<section><h4>'+variant+'</h4><ol>'+''.join(items)+'</ol></section>')
            examples.append('<article><h3>'+direction+' '+label+f'；query{q}，seed42</h3>'+text+picture(image,f'{direction}_{q}.jpg')+
                f'<p>真实top1 cosine={float(confidence[direction+"_top1_cosine"][q]):.6f}，top2={float(confidence[direction+"_top2_cosine"][q]):.6f}；margin={margin:.6f}，τ={threshold:.6f}；实际路由={"BLIP" if route[q] else "直接CLIP"}。</p><div class="columns">'+''.join(views)+'</div></article>')
    table='<table><tr><th>确认500图</th><th>T2I R1均值±旧checkpoint SD</th><th>I2T R1均值±旧checkpoint SD</th></tr>'
    for variant,s in dev['summaries'].items():table+='<tr><td>'+variant+'</td>'+''.join(f'<td>{s[k]["mean"]:.3f} ± {s[k]["sample_sd"]:.3f}</td>' for k in ['T2I_R@1','I2T_R@1'])+'</tr>'
    table+='</table>'
    calibration_tables=[]
    for direction,scan in cal['scan'].items():
        tab='<h3>'+direction+'：只使用cal标签的固定扫描</h3><table><tr><th>τ</th><th>cal重排查询比例</th><th>cal R1差pp</th><th>cal nDCG差pp</th><th>可行</th></tr>'
        for row in scan:tab+=f'<tr><td>{row["threshold"]:.6f}</td><td>{100*row["cal_rerank_query_fraction"]:.1f}%</td><td>{row["cal_r1_delta_pp"]:.3f}</td><td>{row["cal_ndcg_delta_pp"]:.3f}</td><td>{row["feasible"]}</td></tr>'
        calibration_tables.append(tab+'</table>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM：自适应重排序</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto;padding:24px;color:#222}img{max-width:100%;width:360px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f2f2;padding:12px}table{border-collapse:collapse}td,th{border:1px solid #777;padding:8px}.columns,.flow{display:flex;gap:18px;flex-wrap:wrap}.columns section{flex:1;min-width:270px}.flow div{border:2px solid #4477aa;padding:12px}article{border-top:1px solid #999;margin-top:26px}</style><h1>先判断是否需要重排，再计算图文匹配</h1>
<p>0新训练，三个旧CLIP checkpoint、一固定BLIP。500图cal选择阈值，500图confirm验证；每图五caption同fold，图库保持1000/5000。dev/test已有历史曝光，不是新盲测。</p>
<div class="flow"><div>CLIP完整Top50＋实际margin</div><div>margin&gt;τ → 原CLIP排名</div><div>margin≤τ → 提取pair、全局去重、实际BLIP forward</div></div>
<h2>基本原理与代码</h2><pre>margin = cosine(top1) − cosine(top2)
rerank = margin &lt;= frozen_direction_threshold
needed_pair_keys = unique(image_index * 5000 + caption_index, only selected queries)
adaptive output = BLIP reranking for selected rows; unchanged CLIP for others
</pre><p>margin大表示第一候选领先，但不保证正确；I2T前两caption可能都属于正确图片，margin小也不一定错误。cal选择与运行时路由分开：calibrate.py只用cal标签；route.py的route_mask/selected_keys不接收参考答案。matcher.py对选中keys重新forward，没有把历史完整分数查表冒充加速。</p>
<p>BLIP视觉缓存[1000,577,768]，一batch32pair文本[32,64]，image hidden[32,577,768]，ITM logits[32,2]；分数=logit1−logit0。full/adaptive同role同GPU、同batch32，视觉编码为公共成本，旧权重/候选/处理器保持。</p>
<h2>冻结策略与cal曲线数据</h2><pre>'''+html.escape(json.dumps(policy,ensure_ascii=False,indent=2))+'''</pre>'''+''.join(calibration_tables)+'''
<h2>确认质量与实际成本</h2>'''+table+'''<pre>'''+html.escape(json.dumps({'comparisons':dev['comparisons'],'gate':dev['gate'],'cost':dev['cost'],
        'test_state':'completed' if test else 'not_run_dev_gate_failed'},ensure_ascii=False,indent=2))+'''</pre><p>两方向NI容许损失.5pp，图像整簇6000次bootstrap、单侧alpha=.05/2。三轮计时不当新训练seed或独立质量样本。实际质量unique pair与额外计时forward次数分开，cached matcher＋route速度不等于冷启动、完整服务或线上p95。</p>
<h2>真实路由病例</h2><p>只展示confirm中按图SHA固定排序的若干类别首例；病例选择在策略锁定及预测后进行，不用于调τ。标注相关性不等于所有其他图caption均语义错误，图片区分输入和参照。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>用真实top1/top2数值手算margin和路由。</li><li>解释为什么路由函数不应接收参考答案。</li><li>核对跳过的查询没有请求ITM keys。</li><li>解释逻辑pair、去重pair与计时重复forward的区别。</li><li>根据95%区间与.5pp非劣界限判断质量，不能只看均值。</li></ol><p><a href="../POLICY.json">冻结策略</a>；<a href="../CALIBRATION.json">cal扫描</a>；<a href="../analysis/dev/ANALYSIS.json">确认分析</a>。</p></html>'''
    (d/'学习页.html').write_text(page)
    write_json(d/'PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'policy_sha256':sha_file(root/'POLICY.json'),'analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json'),
        'photos':photos,'scope':'real margins/routes/source thumbnails/saved ranks;no attention','manual_visual_review':'pending'})
