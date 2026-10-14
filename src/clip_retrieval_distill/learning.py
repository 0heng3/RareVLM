"""Real teacher/student probabilities and guarded three-seed retrieval results."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from vlm_probe.evaluate import now,sha_file,write_json


def build(config_path,dev=None,test=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    folder=root/'learning';folder.mkdir(exist_ok=True)
    real=json.loads((root/'REAL_DISTILL_PRECHECK.json').read_text())
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    def picture(row,name):
        handles={}
        try:
            image=read_image(row,index,handles);image.thumbnail((480,480));image.save(folder/name,quality=92)
        finally:
            for stream in handles.values():stream.close()
    picture(manifest['splits']['train'][real['first_anchor_id']],'real_train_query.jpg')
    positives=np.flatnonzero(real['positive']);negatives=np.flatnonzero(~np.asarray(real['positive']))
    teacher=np.asarray(real['teacher_probability']);student=np.asarray(real['baseline_probability'])
    selected=np.r_[positives,negatives[np.argsort(-teacher[negatives],kind='stable')[:3]]]
    labels=[f'P{i+1}' for i in range(len(positives))]+['N1','N2','N3'];x=np.arange(len(selected))
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].bar(x-.18,teacher[selected],.36,label='Frozen teacher',color='#4477AA',hatch='///',edgecolor='black')
        axes[0].bar(x+.18,student[selected],.36,label='Trained mean-CE baseline',color='#228833',hatch='xx',edgecolor='black')
        axes[0].set_xticks(x,labels);axes[0].set_ylim(0,1);axes[0].set_ylabel('Softmax probability')
        axes[0].set_title('Same real training query/candidate pool\nHistorical baseline, not new student outcome');axes[0].legend(fontsize=8)
        axes[1].plot(x,np.asarray(real['kl_gradient'])[selected],marker='o',color='#AA3377',label='dKL/d(student logit)')
        axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_xticks(x,labels);axes[1].set_ylabel('Gradient: student probability - teacher probability')
        axes[1].set_title('Exact analytical/autograd check\nPositive gradient lowers that student logit');axes[1].legend(fontsize=8)
        fig.savefig(folder/'teacher_student_gradient.png',dpi=160,facecolor='white');plt.close(fig)
        if dev is not None:
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for ax,metric in zip(axes,['T2I_R@1','I2T_R@1']):
                for i,condition in enumerate(['frozen','uniform_mean','distill']):
                    v=dev['summaries'][condition][metric]
                    ax.bar(i,v['mean'],color=['#4477AA','#228833','#AA3377'][i],hatch=['///','xx','..'][i],edgecolor='black')
                    if v['sample_sd'] is not None:
                        ax.errorbar(i,v['mean'],yerr=v['sample_sd'],fmt='none',color='black',capsize=4)
                        ax.scatter(i+np.array([-.07,0,.07]),list(v['per_seed'].values()),s=15,color='black',zorder=4)
                ax.set_xticks(range(3),['Frozen','Mean baseline','Distill']);ax.set_ylim(0,100);ax.set_ylabel('Recall@1 (%)')
                ax.set_title(metric+' complete dev\n3-seed sampleSD; controls reused')
            fig.savefig(folder/'dev_recall.png',dpi=160,facecolor='white');plt.close(fig)
    examples=[];table=''
    if dev is not None:
        rows=manifest['splits']['dev'];rankings={}
        for condition,key in [('frozen','frozen'),('uniform_mean','uniform_mean_seed42'),('distill','distill_seed42')]:
            with np.load(root/'evaluation'/key/'dev/per_query.npz') as a:rankings[condition]=a['i2t_top10'].copy()
        for i,row in enumerate(rows[:8]):
            picture(row,f'dev_{i}.jpg');views=[]
            for condition,order in rankings.items():
                captions=[]
                for rank,k in enumerate(order[i,:5]):
                    image,within=divmod(int(k),5)
                    captions.append(f'<li>Top{rank+1}；{"标注正描述" if image==i else "其他图描述"}：{html.escape(rows[image]["captions"][within])}</li>')
                views.append('<section><h4>'+condition+'</h4><ol>'+''.join(captions)+'</ol></section>')
            examples.append(f'<article><h3>固定dev行号{i}</h3><img src="dev_{i}.jpg" alt="WebDataset图像展示缩略副本，encoder使用原224中心裁剪"><div class="columns">'+''.join(views)+'</div></article>')
        table='<table><tr><th>方法</th><th>文搜图R1</th><th>图搜文R1</th><th>图搜文nDCG10</th></tr>'
        for condition,values in dev['summaries'].items():
            cells=[]
            for metric in ['T2I_R@1','I2T_R@1','I2T_nDCG@10']:
                v=values[metric];cells.append(f'{v["mean"]:.3f}'+(f' ± {v["sample_sd"]:.3f}' if v['sample_sd'] is not None else '（无seed SD）'))
            table+='<tr><td>'+condition+'</td>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>'
        table+='</table><img src="dev_recall.png" alt="完整dev两方向，误差条为3种子样本SD；冻结与原目标为绑定历史对照">'
    state='训练/dev尚未完成，无新增指标。' if dev is None else json.dumps({'dev_gate':dev['gate'],'comparisons':dev['comparisons'],
        'test_state':'completed' if test is not None else 'not_run_dev_gate_failed'},ensure_ascii=False,indent=2)
    source='https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/loss.py'
    captions='<ol>'+''.join('<li>'+html.escape(c)+'</li>' for c in real['captions'])+'</ol>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM：冻结CLIP关系蒸馏</title><style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:14px}.columns{display:flex;gap:20px;flex-wrap:wrap}.columns section{flex:1;min-width:270px}article{border-top:1px solid #777;padding-top:18px;margin-top:25px}article>img{max-width:480px}td,th{border:1px solid #777;padding:8px}table{border-collapse:collapse}</style><h1>冻结CLIP教师：保留关系与监督适配</h1>
<p>完整28k训练图/140k描述、两侧132480参数adapter；新增seeds0/42/200各10epoch，复用已核验的3个原目标checkpoint及4个dev对照格。固定lambda=1，不做权重扫描。encoder不训练，teacher为同候选池的冻结CLIP缓存特征。不是新VLM蒸馏、EMA教师、Reranker或工业检索部署。</p>
<h2>1. 为什么这轮研究关系保持</h2><p>历史dev冻结CLIP的文搜图/图搜文为60.580/81.200%，原多正适配为64.153/78.233%。上一轮正集合目标同时降低两个方向。本轮保留原多正CE，增加教师分布约束，检验能否缓解适配取舍；此前结果尚未证明“关系漂移”就是退化原因。</p>
<h2>2. 真实源图、五caption与候选矩阵</h2><img src="real_train_query.jpg" alt="真实训练query展示副本，原编码输入是224中心裁剪">'''+captions+'''
<p>每步32锚图和64uniform负图，均带5caption。图搜文logits[32,480]、文搜图[160,96]。teacher/student使用完全相同候选顺序；源SHA组决定监督正项。除KL项外，初始化、采样/正暴露/轮数/参数预算均保持一致。KL增加算术成本，不称等总计算。</p>
<h2>3. 原理、梯度和真实数值</h2><pre>P_T = softmax(normalize(frozen_image) · normalize(frozen_text) / 0.07)
P_S = softmax(normalize(adapter_image) · normalize(adapter_text) / 0.07)
KL(T || S) = sum_j P_T,j (log P_T,j − log P_S,j)
L = unchanged multi-positive mean CE + 1 × (mean KL_I2T + mean KL_T2I)/2
For one query: dKL/d(student logit_j) = P_S,j − P_T,j
</pre><p>当学生比教师赋予某候选更多概率时，KL梯度为正，梯度下降会降低相应logit；反之提高。监督CE仍拉近标注正项。教师也可能不够准确或带入语义误负，低KL不能证明质量保持。teacher在no_grad中计算，log目标再次detach，学生有梯度；lambda=0可恢复原损失与全部参数梯度，三项CPU数学合同另核验。</p>
<img src="teacher_student_gradient.png" alt="同一真实训练query的冻结教师与历史训练后基线概率及KL梯度，不是新候选的结果"><p>这里的学生是上一轮原目标seed42最终模型。CPU复现FP32缓存/checkpoint的logits，再以float64计算概率/梯度；图中仅展示全部正项和teacher最高概率3个负项，完整480项保留。这属于工程教学诊断。</p><pre>'''+html.escape(json.dumps({k:real[k] for k in ['identity_supervised_loss','identity_relation_loss','baseline_query_kl','upstream_soft_target_ce','teacher_entropy','upstream_minus_entropy_equals_kl']},ensure_ascii=False,indent=2))+'''</pre>
<h2>4. 对应OpenCLIP源码</h2><p><a href="'''+source+'''">固定DistillClipLoss源码</a>中的dist_loss计算teacher softmax与student log_softmax的交叉熵；减去固定教师熵后得到KL，学生梯度相同。已实际执行该未修改方法核对CE−H=KL。本项目监督项仍是同图多正目标，没有替换成其一对一标签；只借鉴分布匹配机制，不声称整仓库复现。</p><p>实际代码：clip_retrieval_distill/loss.py的teacher_kl、distill_terms；训练循环记录完整采样流与负例计数SHA，先校验三个原checkpoint及原dev数据。各250步保存optimizer/model/step，支持来源锁与原GPU UUID恢复。</p>
<h2>5. 完整dev结果与门槛</h2>'''+table+'''<pre>'''+html.escape(state)+'''</pre><p>图搜文对原目标均值至少+1pp且单侧下界&gt;0；文搜图对原目标、图搜文对冻结CLIP各容许−.5pp，均需条件非劣。三主统计端点alpha=.05/3、6000次配对图像簇bootstrap；另有nDCG点值/至少2seed保护。dev失败不追加test、不调lambda。历史dev/test已曝光，pretrain重叠未知；种子SD与固定checkpoint平均后的条件图像CI分开，不当盲测或全历史FWER保证。</p>
<h2>6. 三种配置的真实图搜文Top5</h2><p>固定dev前8行；原目标与蒸馏为seed42，冻结底座一份。排序来自真实NPZ，标注相关性按源图ID，不将其他图描述自动判为语义错误。展示缩略图不是attention或224编码输入。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>解释监督标签与teacher软分布的不同信息。</li><li>用本页实际query手算KL与一项P_S−P_T。</li><li>解释为什么先验证lambda=0的参数梯度及teacher无梯度。</li><li>解释低KL、相似度保持与Recall保持之间为什么不能直接画等号。</li><li>核对历史对照、新训练/评分数量、种子SD和图像簇区间。</li></ol>
<p><a href="../REAL_DISTILL_PRECHECK.json">完整真实分布与梯度</a>；<a href="../REUSED_BASELINES.json">历史对照与来源绑定</a>；<a href="../analysis/dev/ANALYSIS.json">完整dev指标/区间</a>。图表工作流参考<a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>，私有教学网页，无期刊目标。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'FIGURE_PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'real_query_sha256':sha_file(root/'REAL_DISTILL_PRECHECK.json'),'dev_analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json') if dev else None,
        'transformations':['FP32 cache/checkpoint CPU logits,float64 softmax/gradient diagnostic','all positives+3teacher-highestnegative display,all480 retained',
            'three seed mean/sampleSD+rawpoints;historical controls explicit','source-image thumbnails;fixedfirst8dev actualsavedrankings'],
        'manual_visual_review':'pending','scope':'private learning,no attention or publisher compliance claim'})
