"""Actual training images/pools and saved rankings; no synthetic attention."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from vlm_probe.evaluate import now,sha_file,write_json


def build(config_path,dev=None,test=None):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);folder=root/'learning';folder.mkdir(exist_ok=True)
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    real=json.loads((root/'REAL_CPU_PRECHECK.json').read_text());pictures={}
    def picture(row,name):
        handles={}
        try:
            im=read_image(row,index,handles);im.thumbnail((480,480));im.save(folder/name,quality=92)
        finally:
            for stream in handles.values():stream.close()
        pictures[name]={'row_image_id':row.get('image_id',row.get('id')),'source_image_sha256':row.get('image_sha256'),
            'thumbnail_sha256':sha_file(folder/name),'transformation':'source RGB thumbnail only;not actual224encoder crop or attention'}
        return '<img src="'+name+'" alt="真实源图的展示缩略副本；CLIP编码为固定224裁剪">'
    anchor=real['anchor_ids'][0];hero=picture(manifest['splits']['train'][anchor],'train_anchor.jpg')
    captions='<ol>'+''.join('<li>'+html.escape(c)+'</li>' for c in real['captions'])+'</ol>'
    actual=[]
    for method in cfg['new_training_conditions']:
        d=root/'runs'/method/'seed42';p=d/('FIRST_CAP_EVENT.json' if method=='capped_distill' else 'FIRST_GPU_POOL.json')
        if p.exists():
            event=json.loads(p.read_text());photos=[]
            for j,k in enumerate(event['raw_top32_ids'][:3]):photos.append(picture(manifest['splits']['train'][k],f'{method}_hard{j}.jpg'))
            actual.append('<h3>'+method+'：实际GPU采样事件</h3><div class="images">'+''.join(photos)+'</div><pre>'+html.escape(json.dumps(event,ensure_ascii=False,indent=2))+'</pre>')
    table='';examples=[]
    if dev:
        table='<table><tr><th>组</th><th>文搜图R1均值±样本SD</th><th>图搜文R1均值±样本SD</th></tr>'
        for method,s in dev['summaries'].items():
            table+='<tr><td>'+method+'</td>'+''.join(f'<td>{s[k]["mean"]:.3f} ± {s[k]["sample_sd"]:.3f}</td>' for k in ['T2I_R@1','I2T_R@1'])+'</tr>'
        table+='</table><pre>'+html.escape(json.dumps(dev['comparisons'],ensure_ascii=False,indent=2))+'</pre>'
        table+='<h3>真实曝光计数</h3><p>以下Top1%为总负例曝光占比；最大hard为单epoch计数。二者分母和意义不同。</p><table><tr><th>seed</th><th>组</th><th>Top1%总曝光占比</th><th>最大hard/epoch</th><th>hard不足更新</th></tr>'
        for record in dev['training_contracts']:
            for method,v in record['conditions'].items():table+=f'<tr><td>{record["seed"]}</td><td>{method}</td><td>{100*v["top1pct_total_negative_exposure_share"]:.3f}%</td><td>{v["max_hard_image_exposure_in_one_epoch"]}</td><td>{v["hard_shortage_updates"]}</td></tr>'
        table+='</table>'
        rankings={}
        for method in cfg['conditions']:
            with np.load(root/'evaluation'/f'{method}_seed42'/'dev/per_query.npz') as a:rankings[method]={k:a[k].copy() for k in ['I2T_R@1','i2t_top10']}
        delta=rankings['capped_distill']['I2T_R@1'].astype(int)-rankings['distill']['I2T_R@1'].astype(int)
        rows=manifest['splits']['dev']
        chosen=[]
        for label,mask in [('改进',delta>0),('退化',delta<0)]:
            ids=np.flatnonzero(mask).tolist();ids.sort(key=lambda k:rows[k].get('image_sha256',str(k)))
            chosen.extend((k,label) for k in ids[:3])
        for i,label in chosen:
            views=[]
            for method,a in rankings.items():
                items=[]
                for k in a['i2t_top10'][i,:5]:
                    image,j=divmod(int(k),5);items.append('<li>'+('标注正描述：' if image==i else '其他图描述：')+html.escape(rows[image]['captions'][j])+'</li>')
                views.append('<section><h4>'+method+'</h4><ol>'+''.join(items)+'</ol></section>')
            examples.append('<article><h3>'+label+f'；dev行{i}，seed42</h3>'+picture(rows[i],f'dev_{i}.jpg')+'<div class="columns">'+''.join(views)+'</div></article>')
    state={'dev_gate':dev['gate'] if dev else 'pending','test_state':'completed' if test else ('not_run_dev_gate_failed' if dev else 'pending')}
    test_table=''
    if test:test_table='<h2>固定配置test确认</h2><pre>'+html.escape(json.dumps({'summaries':test['summaries'],'comparisons':test['comparisons'],'gate':test['gate']},ensure_ascii=False,indent=2))+'</pre>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM：hard曝光上限与蒸馏</title><style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}img{max-width:100%;width:350px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px}.columns,.images{display:flex;gap:20px;flex-wrap:wrap}.columns section{flex:1;min-width:270px}.images img{width:280px}table{border-collapse:collapse}td,th{border:1px solid #888;padding:8px}article{border-top:1px solid #aaa;margin-top:24px}</style>
<h1>困难负例：更难是否真的更好？</h1><p>完整28000训练图/140000描述，固定两侧132480参数adapter。复用3份随机负例蒸馏基线；B/C各新增3训练，从相同初始化开始，均10epoch/8750更新。历史dev/test已曝光、预训练重叠未知。</p>
<h2>原理与真实代码</h2><p>原batch-max挖掘会反复挑选对很多anchor都相似的图。C只为每张训练图的hard入池设置8次/epoch上限，达到后从hard候选中排除；随机部分仍可选该图。上限不等于8次所有负例，也不直接检测语义误负。</p><pre>score[i] = max over32anchors cosine(mean5caption, frozen_image[i])
eligible = train images excluding anchor / same SHA / known identical caption association
B: hard = stable_top32(eligible); random = uniform32(eligible excluding hard)
C: hard = stable_top32(eligible with epoch_hard_count[i] &lt; 8)
   random = uniform(64 - len(hard), eligible excluding hard)
L = unchanged symmetric mean-positive CE + 1 × KL(frozen teacher || student)
I2T logits [32,480]; T2I logits [160,96]
</pre><p>代码：clip_retrieval_capped/sampler.py 的sample_pool，train.py保存每epoch hard/random计数；直接调用clip_retrieval_distill/loss.py。改变负池也会改变teacher分布，不能将效果全部归因单独曝光次数。</p>
<h2>真实训练图与五条描述</h2>'''+hero+captions+'''<p>下面为真实训练缓存上的CPU预检。人为将16项计数设为8，仅检查容量规则，不是实际训练曝光；实际GPU事件另列。</p><pre>'''+html.escape(json.dumps(real,ensure_ascii=False,indent=2))+'''</pre>'''+''.join(actual)+'''
<h2>完整dev指标与曝光</h2>'''+table+'''<pre>'''+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'''</pre><p>C/A文搜图至少+1pp且校正单侧下界&gt;0；图搜文非劣下界&gt;−.5pp。两端点alpha=.05/2、6000次图像簇bootstrap；三个训练seed样本SD单列。B/A与C/B探索区间不用于换主候选。失败停止，不调cap或追加test。</p>'''+test_table+'''
<h2>真实改进与退化病例</h2><p>按seed42保存排名选择图搜文变化样本，各类按源图SHA排序取最多3例。结果展示不参与策略选择；其他图caption不自动等于语义错误。照片是展示缩略图，不是attention。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>解释为什么hard次数、随机次数和梯度贡献不同。</li><li>用真实GPU事件找到被排除的图，核对计数达到8及64完整预算。</li><li>解释候选池变化如何影响CE与KL的softmax分母。</li><li>比较曝光下降与Recall变化，说明哪些结论仍缺因果证据。</li><li>解释三个seed SD与1000图像簇区间的区别。</li></ol><p><a href="../analysis/dev/ANALYSIS.json">完整dev分析</a>；<a href="../REUSED_BASELINES.json">旧基线绑定</a>。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'PROVENANCE.json',{'created_at':now(),'learning_source_sha256':sha_file(Path(__file__)),
        'real_cpu_precheck_sha256':sha_file(root/'REAL_CPU_PRECHECK.json'),'pictures':pictures,
        'dev_analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json') if dev else None,
        'scope':'source thumbnails/actual sampling/actual saved rankings,no attention or personal mastery claim','manual_visual_review':'pending'})
