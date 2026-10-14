"""Pinned source reading, real loss/gradient plots, exposure and actual rankings."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np

from clip_retrieval.features import read_image
from vlm_probe.evaluate import now, sha_file, write_json


def build(config_path,dev=None,test=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    folder=root/'learning';folder.mkdir(exist_ok=True)
    real=json.loads((root/'CPU_LOSS_PRECHECK.json').read_text())
    positive=np.flatnonzero(real['first_anchor_positive'])
    negative=np.flatnonzero(~np.asarray(real['first_anchor_positive']))
    probability=np.asarray(real['first_anchor_probability'])
    hardest=negative[np.argsort(-probability[negative],kind='stable')[:3]]
    selected=np.r_[positive,hardest];labels=[f'P{i+1}' for i in range(len(positive))]+['N1','N2','N3']
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text())
    index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets'];handles={}
    anchor=manifest['splits']['train'][real['first_anchor_id']]
    try:
        photo=read_image(anchor,index,handles);photo.thumbnail((600,600));photo.save(folder/'real_train_anchor.jpg',quality=92)
    finally:
        for handle in handles.values():handle.close()
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(len(selected));width=.24
        for offset,key,label,color,hatch in [(-width,'first_anchor_probability','Current probability','#4477AA','///'),
                (0,'mean_positive_targets','Mean-positive target','#228833','xx'),
                (width,'set_positive_effective_targets','Set-positive effective target','#AA3377','..')]:
            axes[0].bar(x+offset,np.asarray(real[key])[selected],width,label=label,color=color,hatch=hatch,edgecolor='black')
        axes[0].set_xticks(x,labels);axes[0].set_ylim(0,1);axes[0].set_ylabel('Probability / effective target')
        axes[0].set_title(f'Real seed42 step0 frozen-CLIP query\n{len(positive)} labeled positives and 3 highest-probability negatives')
        axes[0].legend(fontsize=8)
        for key,label,color,marker in [('mean_gradient','Mean-positive gradient','#228833','o'),('set_gradient','Set-positive gradient','#AA3377','s')]:
            axes[1].plot(x,np.asarray(real[key])[selected],label=label,color=color,marker=marker)
        axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_xticks(x,labels);axes[1].set_ylabel('d(single-query loss) / d(logit)')
        axes[1].set_title(f'Exact autograd vs analytical gradient verified\nselected entries; other {480-len(selected)} entries retained in source JSON')
        axes[1].legend(fontsize=8);fig.savefig(folder/'real_loss_gradient.png',dpi=160,facecolor='white');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        curves={}
        for condition,label,color,style in [('uniform_negatives','Uniform, seed42','#228833','-'),('mixed_hard_negatives','Mixed hard, seed42','#AA3377','--')]:
            exposure=np.sort(np.load(old/'runs'/condition/'seed42/negative_occurrences.npy').astype(float))
            xx=np.arange(len(exposure)+1)/len(exposure);yy=np.r_[0,np.cumsum(exposure)/exposure.sum()]
            ax.plot(xx,yy,label=label,color=color,linestyle=style);curves[condition]={'x':xx.tolist(),'y':yy.tolist()}
        ax.plot([0,1],[0,1],color='black',linestyle=':',label='Equal exposure reference')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Cumulative fraction of training images (least exposed first)',
            ylabel='Cumulative fraction of negative exposures',title='Saved v1 negative sampling concentration\nNot proof of semantic false negatives')
        ax.legend(fontsize=8);fig.savefig(folder/'negative_exposure.png',dpi=160,facecolor='white');plt.close(fig)
        write_json(folder/'negative_lorenz_source.json',curves)
        if dev is not None:
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for ax,metric in zip(axes,['T2I_R@1','I2T_R@1']):
                for i,condition in enumerate(['frozen','uniform_mean','uniform_set']):
                    value=dev['summaries'][condition][metric]
                    ax.bar(i,value['mean'],color=['#4477AA','#228833','#AA3377'][i],hatch=['///','xx','..'][i],edgecolor='black')
                    if value['sample_sd'] is not None:
                        ax.errorbar(i,value['mean'],yerr=value['sample_sd'],color='black',fmt='none',capsize=4)
                        ax.scatter(i+np.array([-.07,0,.07]),list(value['per_seed'].values()),color='black',s=15,zorder=4)
                ax.set_xticks(range(3),['Frozen','Mean positive','Positive set']);ax.set_ylim(0,100);ax.set_ylabel('Recall@1 (%)')
                ax.set_title(metric+' complete dev\npoints:3 seeds; errorbars:sample SD')
            fig.savefig(folder/'dev_recall.png',dpi=160,facecolor='white');plt.close(fig)
    sources=json.loads((root/'upstream/SOURCES.json').read_text())
    source_links=''.join('<li>'+html.escape(j['repository'])+' @ '+html.escape(j['commit'])+'<ul>'+''.join(
        f'<li><a href="{html.escape(item["url"])}">{html.escape(item["path"])}</a>；SHA256 {item["sha256"]}</li>'
        for item in j['files'] if 'path' in item)+'</ul></li>' for j in sources)
    audit=json.loads((root/'reading/TEXTVQA_SCORING_AUDIT.json').read_text()) if (root/'reading/TEXTVQA_SCORING_AUDIT.json').exists() else None
    exposure=json.loads((root/'NEGATIVE_EXPOSURE_DIAGNOSTIC.json').read_text())
    captions='<ol>'+''.join('<li>'+html.escape(c)+'</li>' for c in real['first_anchor_captions'])+'</ol>'
    snippets=[]
    selections=[('OpenCLIP：一对一CLIP损失','mlfoundations__open_clip/src/open_clip/loss.py',78,141),
        ('BLIP：同ID多正与动量软目标','salesforce__BLIP/models/blip_retrieval.py',86,118),
        ('BLIP：困难负例进入匹配损失','salesforce__BLIP/models/blip_retrieval.py',174,225),
        ('BLIP：召回候选与匹配重排','salesforce__BLIP/train_retrieval.py',113,155),
        ('lmms-eval：十参考留一共识评分','EvolvingLMMs-Lab__lmms-eval/lmms_eval/tasks/textvqa/utils.py',15,40)]
    for title,rel,begin,end in selections:
        lines=(root/'upstream'/rel).read_text().splitlines()
        code='\n'.join(f'{i+1}: {lines[i]}' for i in range(begin-1,min(end,len(lines))))
        snippets.append('<details><summary>'+html.escape(title)+'</summary><pre>'+html.escape(code)+'</pre></details>')
    def ranking_examples():
        if dev is None:return '<p>六次完整训练与dev排名尚未完成，以下不预填性能结果。</p>'
        rows=manifest['splits']['dev'];rankings={}
        for condition in ['uniform_mean','uniform_set']:
            with np.load(root/'evaluation'/f'{condition}_seed42'/'dev/per_query.npz') as values:
                rankings[condition]=values['i2t_top10'].copy()
        gallery=[]
        try:
            for i,row in enumerate(rows[:8]):
                filename=f'dev_query_{i}.jpg';photo=read_image(row,index,handles);photo.thumbnail((480,480));photo.save(folder/filename,quality=92)
                views=[]
                for condition,order in rankings.items():
                    candidates=[]
                    for rank,caption_idx in enumerate(order[i,:5]):
                        image_idx,within=divmod(int(caption_idx),5)
                        text=rows[image_idx]['captions'][within]
                        candidates.append(f'<li>Top{rank+1}；{"标注正描述" if image_idx==i else "其他图描述"}：{html.escape(text)}</li>')
                    views.append('<h4>'+condition+'</h4><ol>'+''.join(candidates)+'</ol>')
                gallery.append(f'<article><h3>固定dev行号{i}：{html.escape(row["image_id"])}</h3><img src="{filename}" alt="WebDataset源图展示缩略副本，非模型attention"><div class="columns">'+''.join('<section>'+v+'</section>' for v in views)+'</div></article>')
        finally:
            for handle in handles.values():handle.close()
        return ''.join(gallery)
    table=''
    if dev is not None:
        lines=['<table><tr><th>方法</th><th>文搜图R1</th><th>图搜文R1</th><th>图搜文nDCG10</th></tr>']
        for condition,values in dev['summaries'].items():
            cells=[]
            for metric in ['T2I_R@1','I2T_R@1','I2T_nDCG@10']:
                v=values[metric];cells.append(f'{v["mean"]:.3f}'+(f' ± {v["sample_sd"]:.3f}' if v['sample_sd'] is not None else '（无seed SD）'))
            lines.append('<tr><td>'+condition+'</td>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
        table=''.join(lines)+'</table><img src="dev_recall.png" alt="完整dev两个检索方向，误差条是3训练seed样本SD，三个点为逐seed指标">'
    result_text='尚未完成训练与dev评测。' if dev is None else json.dumps({'dev_gate':dev['gate'],'comparisons':dev['comparisons'],
        'test_state':'completed' if test is not None else 'not_run_dev_gate_failed'},ensure_ascii=False,indent=2)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM：源码精读与多正检索损失</title><style>body{font:17px/1.8 sans-serif;max-width:1150px;margin:auto;padding:24px;color:#222}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:14px;font-size:14px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #777;padding:8px}.columns{display:flex;gap:24px;flex-wrap:wrap}.columns section{flex:1;min-width:300px}article{border-top:1px solid #999;margin-top:24px}article>img{max-width:480px}</style><h1>从真实CLIP检索退化，读懂损失、负例和评测</h1>
<p>本轮完整28000训练图；冻结FP32 CLIP-B/32的512维表征；两侧132480参数adapter；uniform_mean/uniform_set各seeds0/42/200、10轮Final@10。只改图搜文正项聚合，文搜图loss与uniform64负图保持不变。它是已有多正学习目标的受控对照，不是BLIP/SigLIP复现或新算法原创。</p>
<h2>1. OpenCLIP：相似度矩阵、温度与对称CE</h2><p>OpenCLIP归一后的图/文向量相乘得到logits，logit_scale对应1/τ；一对一数据用arange标签。跨卡gather扩大候选库，gather_with_grad/local_loss控制梯度和查询范围。我们的真实训练每步32锚图＋64负图、每图5caption，图搜文[32,480]、文搜图[160,96]，正项以源SHA组判断，所以不能直接套用对角线标签。当前每个训练独占一张卡，两个任务并行；没有把本轮描述成CLIP编码器DDP预训练。</p>
<h2>2. 从真实图像看两种正项目标</h2><img src="real_train_anchor.jpg" alt="真实训练锚图的展示缩略副本，CLIP特征来自原协议224中心裁剪"><p>下列五条描述来自该图的实际标注：</p>'''+captions+'''
<pre>s_ij = normalize(adapter_image(x_i)) · normalize(adapter_text(t_j)) / 0.07
L_mean(i) = logsumexp_j(s_ij) − mean_{j∈P_i}(s_ij)
L_set(i)  = logsumexp_j(s_ij) − logsumexp_{j∈P_i}(s_ij)
L_total = (mean_i L_image(i) + unchanged mean_caption CE_text) / 2
∂L_mean/∂s_j = softmax_all(s)_j − 1[j∈P]/|P|
∂L_set/∂s_j = softmax_all(s)_j − softmax_positive(s)_j
</pre><p>mean目标鼓励每条标注描述；set目标提高标注描述集合的概率质量。两者都使用所有标注正项，但set可能更依赖已有高相似度描述。它与hit-any Recall接近，不保证全量召回、nDCG或泛化更好。梯度大于0时，梯度下降会降低该logit；小于0则提高。两种loss数值尺度不同，不能比较训练loss大小来判断哪组检索更好。</p>
<img src="real_loss_gradient.png" alt="真实训练query的五个正项与三个高概率负项，比较概率、有效目标与真实梯度"><p>数据来自seed42第0步的完整真实缓存批次，所有480项保存在CPU_LOSS_PRECHECK.json；图中展示8项，未展示项并未从训练排除。已验证autograd与解析梯度。这是工程/教学检查，不是新benchmark。</p>
<h2>3. BLIP：对比、匹配、动量目标和重排</h2><p>BLIP的同ID多正分布与动量模型softmax分布混合，构造对比目标；匹配分支则让文本encoder通过cross-attention读取图像patch，输出二分类logits。困难负例按每个查询softmax相似度抽样，同ID正例先屏蔽，再分别抽负图/负文本，形成B正＋2B负匹配对。推理先做向量召回，再对TopK执行匹配重排；不能找回不在候选中的正项。它与我们按整批最大相似度构造共享负池的目标和范围不同，因此本轮不直接移植。</p>
<img src="negative_exposure.png" alt="上一轮seed42随机与混合困难负例的完整曝光Lorenz曲线，等曝光参考线，非误负例因果证明"><pre>'''+html.escape(json.dumps(exposure,ensure_ascii=False,indent=2))+'''</pre>
<h2>4. lmms-eval：评分相同，输入不一定相同</h2><p>TextVQA保留十参考，逐个留出参考后计算min(匹配数/3,1)，再平均。因此1/2/3/4个匹配得到0.3/0.6/0.9/1.0。当前上游函数会原位归一化answers，本次只传入副本，保留原始数据。评分审计实际执行固定上游类/函数，不安装整个评测框架、不新增GPU预测。</p><pre>'''+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'''</pre>
<p>Qwen当前上游默认像素上限1605632、下限200704；RareVLM为802816/50176。上限不是实际token保证。问题大小写、OCR条件、system/后缀、dtype和生成长度也应逐项核对。相同计分不能证明输入或整个框架等价。</p>
<h2>5. 原源码对应位置</h2>'''+''.join(snippets)+'''<h2>6. 实验结果与停止规则</h2>'''+table+'''<pre>'''+html.escape(result_text)+'''</pre><p>主门槛：图搜文比mean至少+1pp且配对下界&gt;0；文搜图对mean及图搜文对冻结底座各容许−0.5pp、必须通过条件非劣；三项单侧alpha=.05/3。另有nDCG点值保护和至少2种子正方向。完整dev失败则停止，不追加新test；历史test已曝光、pretrain重叠未知。单checkpoint平均后的图像簇区间与三seed SD分开，caption不当独立图像。</p>
<h2>7. 完整dev真实图搜文Top5</h2><p>预先固定dev前8行，seed42，两方法真实保存排序；展示副本来自上游WebDataset，编码输入仍是原224中心裁剪。标注正描述按图ID判断，不把其他图描述直接认定为语义错误。</p>'''+ranking_examples()+'''
<h2>独立讲述检查</h2><ol><li>画出[32,480]和[160,96]，说明5个正项及源SHA重复组。</li><li>手算本页真实query两个loss与一项梯度，解释为何loss更小不等于Recall更高。</li><li>解释BLIP匹配困难负例与共享对比负池的区别。</li><li>解释TextVQA留一评分、像素上限与真实token的区别。</li><li>报告正负结果、种子SD、图像簇区间及dev门槛，不宣称已证明的退化机制。</li></ol>
<h2>固定来源与复核入口</h2><ul>'''+source_links+'''</ul><p><a href="../CPU_LOSS_PRECHECK.json">实际logits与梯度</a>；<a href="negative_exposure.csv">全部曝光计数</a>；<a href="../reading/TEXTVQA_SCORING_AUDIT.json">评分审计</a>；<a href="../analysis/dev/ANALYSIS.json">dev完整结果</a>。Scientific Agent Skills图表工作流参考：<a href="https://doi.org/10.48550/arXiv.2609.00065">Kassis等，2026</a>；本页为私有教学网页，无期刊投稿目标。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'FIGURE_PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'real_loss_source_sha256':sha_file(root/'CPU_LOSS_PRECHECK.json'),'negative_exposure_source_sha256':sha_file(root/'NEGATIVE_EXPOSURE_DIAGNOSTIC.json'),
        'dev_analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json') if dev is not None else None,
        'transformations':['real training-query select all positives+top3negative probabilities for display; original480 logits retained',
            'seed42 full negative counts Lorenz; all6 complete counts inCSV','source-image display thumbnails,encoder original224 crop unchanged',
            'complete dev 3seed mean/sampleSD and rawpoints;fixedfirst8dev rows'],
        'manual_visual_review':'pending','scope':'private HTML learning page; no attention weights or journal target'})
