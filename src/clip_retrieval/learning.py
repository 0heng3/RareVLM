"""Real retrieval examples and seed uncertainty;similarity is not attention."""
from __future__ import annotations
import html
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from vlm_probe.evaluate import write_json,sha_file
from .features import read_image


def build(root,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    learning=root/'learning';learning.mkdir(exist_ok=True)
    conditions=['frozen','uniform_negatives','mixed_hard_negatives']
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,direction in zip(axes,['T2I','I2T']):
            values=[result['summaries'][c]['test'][direction+'_R@1'] for c in conditions]
            bars=ax.bar(range(3),[v['mean'] for v in values],color=['#4477AA','#228833','#AA3377'],edgecolor='black')
            for i,(bar,v) in enumerate(zip(bars,values)):
                bar.set_hatch(['///','xx','..'][i])
                if v['sample_sd'] is not None:ax.errorbar(i,v['mean'],yerr=v['sample_sd'],fmt='none',color='black',capsize=5)
            ax.set_xticks(range(3),['Frozen','Uniform','Mixed hard']);ax.set_ylim(0,100);ax.set_ylabel('Recall@1 (%)')
            ax.set_title(direction+' full official test\nerror bars:3 training-seed sample SD')
        fig.savefig(learning/'recall_seed_sd.png',dpi=180,facecolor='white');fig.savefig(learning/'recall_seed_sd.svg',facecolor='white');plt.close(fig)
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());rows=manifest['splits']['dev']
    index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets'];handles={};examples=[]
    keys=['frozen','uniform_negatives_seed42','mixed_hard_negatives_seed42'];rankings={}
    for key in keys:
        with np.load(root/'evaluation'/key/'dev/per_query.npz') as x:rankings[key]=x['t2i_top10'].copy()
    def picture(i):
        filename=f'dev_{i}.jpg';path=learning/filename
        if not path.exists():
            image=read_image(rows[i],index,handles);image.thumbnail((420,420));image.save(path,quality=92)
        return filename
    try:
        for i,row in enumerate(rows[:8]):
            caption=row['captions'][0];views=[]
            for key in keys:
                ids=rankings[key][i*5,:5].tolist()
                thumbnails=''.join(f'<figure><img src="{picture(j)}" alt="上游webdataset图像的展示缩略副本，encoder使用标准224中心裁剪"><figcaption>Top{k+1}；{"标注正图" if j==i else "其他图"}</figcaption></figure>' for k,j in enumerate(ids))
                views.append(f'<h4>{key}</h4><div class="gallery">{thumbnails}</div>')
            examples.append(f'<article><h3>预先按dev行号选取的真实query {i}</h3><p>{html.escape(caption)}</p><p>标注正图ID：{html.escape(row["image_id"])}</p>{"".join(views)}</article>')
    finally:
        for h in handles.values():h.close()
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CLIP真实图文检索：对比学习与困难负样本</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto;padding:24px;color:#222}.gallery{display:flex;flex-wrap:wrap}figure{width:180px;margin:8px}img{max-width:100%}article{border-top:1px solid #777}pre{white-space:pre-wrap;background:#eee;padding:14px}</style><h1>图文检索：从已有CLIP分类走向真实跨模态对齐</h1><p>完整公开Flickr30k WebDataset、固定预训练CLIP-B/32表征、两侧132480参数残差adapter，uniform与32hard+32uniform对照，seeds0/42/200各Final@10。正式test不回调策略。预训练重叠未知，不声称盲测或工业商品检索。</p><h2>1. 原理对应源码</h2><p>归一后的图像/文本向量均为512维，sᵢⱼ=z_imgᵢ·z_txtⱼ/τ，τ=.07用于训练。冻结模型的排序不因正的标量温度而改变，温度会影响softmax概率及梯度。</p><pre>L_image = −mean_i sum_j P_ij log softmax_j(s_ij)
P_ij = positive(i,j) / number_of_positive_captions(i)
L_text uses the matching image targets; L = (L_image + L_text)/2
∂L/∂s = p − P; before scaling the cosine gradient is (p−P)/τ
</pre><p>同图的五条描述都属于正样本，不能把其他四条当负例。SourceSHA重复图也共享训练正组。困难负例在训练图库中按冻结CLIP相似度查找，排除锚点ID、相同源图及已知相同caption关联图，再与随机负例混合。已知排除规则仍不能消除所有语义误负例。</p><p>代码：clip_retrieval/train.py中的multi_positive_ce、contrastive_loss、choose_negatives；两方法匹配初始化/参数/轮数/锚点顺序/64负图槽位，但负例曝光分布不同，日志记录这些成本。参数量=2×(32768+64+128+32768+512)=132480。</p><h2>2. 真正的检索指标</h2><p>T2I查询是caption，图库是图片；I2T查询是图片，库是caption。CLIP_benchmark的Recall@K为“至少命中一个标注正项”的比例；nDCG@10按二值相关性及log2位置折扣另算，不把VQA共识分当召回率。</p><h2>3. 全量结果</h2><img src="recall_seed_sd.png" alt="完整官方检索test的两个方向Recall@1，学习方法误差条为3训练种子样本SD，frozen只一模型无SD"><p>图像簇配对CI另列JSON；不把seed×query当独立样本。全组同一个固定test一次，未依据test改epoch/学习率/负采样。不同方向/指标分别报告。</p><pre>'''+html.escape(json.dumps(result['gates'],ensure_ascii=False,indent=2))+'''</pre><h2>4. 真实caption与Top5图片</h2><p>案例为dev最前8行的第一caption，与上游标注对应。图片是WebDataset表示的缩略副本；encoder用224中心裁剪/标准归一化。Top排序来自真实保存NPZ，不是人工挑选或模型attention图。</p>'''+''.join(examples)+'''<h2>学习检查</h2><ol><li>手算一个真实query的softmax、P和对比损失。</li><li>解释多正caption为什么不能使用单纯对角线标签。</li><li>用p−P解释hardnegative为何更有梯度、为什么也可能有误负例。</li><li>独立写hit-any Recall@K与nDCG，区分query数、image簇和seed。</li></ol><p><a href="../analysis/ANALYSIS.json">完整指标、三种子、配对区间与门槛</a>。图表工作流引用：Kassis等(2026)，<a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>；数据/Recall依据为已锁定CLIP_benchmark，不是新算法原创声明。</p></html>'''
    (learning/'学习页.html').write_text(page)
    write_json(learning/'FIGURE_PROVENANCE.json',{'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'source_sha256':sha_file(Path(__file__)),
        'transformations':['testmean/sampleSD,zero-based percentaxes,no smoothing','first8dev caption queries,true savedtop5','display thumbnails only'],
        'uncertainty':'3seed sampleSD infigure;imageclusterCI inJSON;frozenSDnotavailable','manual_visual_review':'pending','medium':'privatelearningwebpage,no journal target'})
