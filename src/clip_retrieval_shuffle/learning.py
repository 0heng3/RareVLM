"""Actual CUDA target correspondence, real images and full-gallery results."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from vlm_probe.evaluate import now, sha_file, write_json


def build(config_path, dev, test=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=json.loads(config_path.read_text()); root=Path(cfg['output_root'])
    folder=root/'learning'; folder.mkdir(exist_ok=True)
    source=root/'runs/negative_shuffle/seed42/FIRST_UPDATE_DIAGNOSTIC.json'
    real=json.loads(source.read_text())
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text())
    offsets=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    def picture(row, name):
        handles={}
        try:
            image=read_image(row,offsets,handles); image.thumbnail((480,480)); image.save(folder/name,quality=92)
        finally:
            for stream in handles.values():stream.close()
    picture(manifest['splits']['train'][real['first_anchor_id']], 'real_train_query.jpg')
    original=np.exp(np.asarray(real['teacher_original_logp'],dtype=np.float64))
    shuffled=np.exp(np.asarray(real['teacher_shuffled_logp'],dtype=np.float64))
    student=np.exp(np.asarray(real['student_logp'],dtype=np.float64))
    positive=np.asarray(real['positive']); neg=np.flatnonzero(~positive)
    selected=list(np.flatnonzero(positive))
    for v in [original,shuffled]:
        for i in neg[np.argsort(-v[neg],kind='stable')[:3]]:
            if int(i) not in selected:selected.append(int(i))
    labels=[('P' if positive[k] else 'N')+str(k) for k in selected]; x=np.arange(len(selected))
    gradients=student-shuffled
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        axes[0].bar(x-.18,original[selected],.36,color='#4477AA',hatch='///',edgecolor='black',label='Original teacher')
        axes[0].bar(x+.18,shuffled[selected],.36,color='#AA3377',hatch='..',edgecolor='black',label='Negative-shuffled teacher')
        axes[0].set_xticks(x,labels,rotation=45); axes[0].set_ylim(0,1); axes[0].set_ylabel('Actual CUDA target probability')
        axes[0].set_title('Same positive targets and negative-value multiset\nDifferent negative candidate associations'); axes[0].legend(fontsize=8)
        axes[1].plot(x,gradients[selected],marker='o',color='#228833')
        axes[1].axhline(0,color='black',linewidth=.8); axes[1].set_xticks(x,labels,rotation=45)
        axes[1].set_ylabel('Unreduced KL gradient: P_student - P_shuffled')
        axes[1].set_title('Actual seed42 first update, before optimizer\nIdeal normalized-target identity; FP32 rounding remains')
        fig.savefig(folder/'negative_relation_gradient.png',dpi=160,facecolor='white'); plt.close(fig)
        for split,analysis in [('dev',dev),('test',test)]:
            if analysis is None:continue
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
            for ax,metric in zip(axes,['T2I_R@1','I2T_R@1']):
                for i,condition in enumerate(['frozen','uniform_mean','distill','negative_shuffle']):
                    s=analysis['summaries'][condition][metric]
                    ax.bar(i,s['mean'],color=['#4477AA','#228833','#AA3377','#666666'][i],hatch=['///','xx','..','--'][i],edgecolor='black')
                    if s['sample_sd'] is not None:
                        ax.errorbar(i,s['mean'],yerr=s['sample_sd'],fmt='none',color='black',capsize=4)
                        ax.scatter(i+np.array([-.07,0,.07]),list(s['per_seed'].values()),color='black',s=16,zorder=4)
                ax.set_xticks(range(4),['Frozen','Mean CE','Teacher','Shuffled'],rotation=15)
                ax.set_ylim(0,100); ax.set_ylabel('Recall@1 (%)')
                ax.set_title(f'{metric}, complete {split}\n3 training seeds: mean, sample SD, raw points')
            fig.savefig(folder/(split+'_recall.png'),dpi=160,facecolor='white'); plt.close(fig)
    def entropy(p):return float(-(p*np.log(p.clip(1e-300))).sum())
    gt=positive/positive.sum()
    diagnosis={'source_first_update_sha256':sha_file(source),'selected_columns':selected,
        'teacher_probability_sum':float(original.sum()),'shuffled_probability_sum':float(shuffled.sum()),
        'teacher_entropy':entropy(original),'shuffled_entropy':entropy(shuffled),
        'original_mixed_entropy':entropy((gt+original)/2),'shuffled_mixed_entropy':entropy((gt+shuffled)/2),
        'positive_max_probability_error':float(np.max(np.abs(original[positive]-shuffled[positive]))),
        'full_teacher_probability':original.tolist(),'full_shuffled_probability':shuffled.tolist(),
        'full_student_probability':student.tolist(),'ideal_unreduced_kl_gradient':gradients.tolist(),
        'scope':'Actual archived CUDA first update probabilities; ideal KL gradient identity shown within FP32 rounding,not total CE+KL parameter gradient'}
    write_json(folder/'TEACHING_DATA.json',diagnosis)
    candidates=[]; train_rows=manifest['splits']['train']
    for k in selected:
        cap_id=real['caption_ids'][k]; image_id,within=divmod(cap_id,5)
        candidates.append('<tr><td>'+labels[selected.index(k)]+'</td><td>'+html.escape(train_rows[image_id]['captions'][within])+
            f'</td><td>{original[k]:.6f}</td><td>{shuffled[k]:.6f}</td><td>{gradients[k]:+.6f}</td></tr>')
    def result_block(split,analysis):
        if analysis is None:return '<p>开发集未通过机制门槛，因此没有新增测试结果。</p>'
        rows=[]
        for condition,s in analysis['summaries'].items():
            cells=[]
            for metric in ['T2I_R@1','I2T_R@1','I2T_nDCG@10']:
                value=s[metric];cells.append(f'{value["mean"]:.3f}'+(f' ± {value["sample_sd"]:.3f}' if value['sample_sd'] is not None else '（无种子SD）'))
            rows.append('<tr><td>'+condition+'</td>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
        return '<h3>'+split+'</h3><table><tr><th>方法</th><th>文搜图R@1</th><th>图搜文R@1</th><th>图搜文nDCG@10</th></tr>'+''.join(rows)+\
            '</table><img src="'+split+'_recall.png" alt="完整图库Recall，三种子样本SD和原始点；冻结底座无种子SD">'+\
            '<pre>'+html.escape(json.dumps({'gate':analysis['gate'],'primary_comparisons':analysis['comparisons'],
                'seed_directions':analysis['positive_seed_directions']},ensure_ascii=False,indent=2))+'</pre>'
    rankings={}
    for condition in ['uniform_mean','distill','negative_shuffle']:
        with np.load(root/'evaluation'/f'{condition}_seed42/dev/per_query.npz') as z:
            rankings[condition]=z['i2t_top10'].copy()
    examples=[]; rows=manifest['splits']['dev']; selected_rows=list(range(4))
    a=rankings['distill'][:,0]//5 == np.arange(1000)
    b=rankings['negative_shuffle'][:,0]//5 == np.arange(1000)
    for mask in [a&~b,b&~a]:
        ids=np.flatnonzero(mask)
        if len(ids) and int(ids[0]) not in selected_rows:selected_rows.append(int(ids[0]))
    for i in selected_rows:
        picture(rows[i],f'dev_{i}.jpg'); views=[]
        for condition,rank in rankings.items():
            captions=[]
            for n,k in enumerate(rank[i,:5]):
                image,within=divmod(int(k),5)
                captions.append(f'<li>Top{n+1}；'+('标注正描述' if image==i else '其他图描述')+'：'+html.escape(rows[image]['captions'][within])+'</li>')
            views.append('<section><h4>'+condition+'</h4><ol>'+''.join(captions)+'</ol></section>')
        examples.append(f'<article><h3>dev行号{i}</h3><img src="dev_{i}.jpg" alt="来源核验的真实dev图展示缩略副本，非attention"><div class="columns">'+''.join(views)+'</div></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM教师负关系机制对照</title>
<style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:14px}table{border-collapse:collapse}td,th{border:1px solid #777;padding:8px}.columns{display:flex;gap:20px;flex-wrap:wrap}.columns section{flex:1;min-width:270px}article{border-top:1px solid #777;padding-top:16px;margin-top:25px}article>img{max-width:480px}</style>
<h1>教师给多少概率，以及给哪个负样本</h1>
<p>本轮是算法机制对照，置乱不是提分方案。新增三个seed各完整28k图、10epoch、8750更新、132480参数；复用六个历史checkpoint，不把对照复用写成新增训练。</p>
<h2>1. 为什么要做这个对照</h2><p>原蒸馏相对CE测试文搜图/图搜文提高6.180/5.300pp，但改善可能包含正样本置信度、平滑和负关系信息。我们保持正样本教师概率，只把负项概率转交给不同负候选。正项固定、负项概率集合不变、教师熵和混合目标熵不变，损失权重也不变。</p>
<pre>每方向 L = CE(P_gt,P_S) + KL(P_T||P_S)
             = 2 CE((P_gt+P_T)/2,P_S) - H(P_T)
负样本列的P_gt=0：只置乱这些列，同时保持教师熵与混合目标熵。
理想归一化目标：dKL/d(student_logit_j) = P_S,j - P_T,j
若学生比置乱目标概率更高，该KL项倾向降低该候选logit。</pre>
<p>原教师优于置乱对照只支持负候选对应关系的价值，不解释全部收益；正项教师置信度仍相同。当前数据公开且历史已曝光，不能称盲测或电商线上效果。</p>
<h2>2. 第一更新的真实图、五描述与概率</h2><img src="real_train_query.jpg" alt="seed42第一CUDA更新使用的真实源图缩略副本；CLIP特征原编码使用224中心裁剪"><ol>'''+''.join('<li>'+html.escape(c)+'</li>' for c in real['first_anchor_captions'])+'''</ol>
<p>图搜文[32,480]、文搜图[160,96]；下面是seed42第一CUDA更新、optimizer.step之前的真实目标。全480项保存在链接JSON中。图只展示所有正项及原/置乱目标各最高3个负项的并集，不是所有候选或attention。FP32舍入保留，梯度图为单query理想归一化KL恒等式，不是整个CE+KL参数梯度。</p>
<img src="negative_relation_gradient.png" alt="同一query的原教师和负概率置乱目标，正项保持；第二图为单query KL理想梯度">
<table><tr><th>候选列</th><th>真实描述</th><th>原教师概率</th><th>置乱目标概率</th><th>理想KL梯度</th></tr>'''+''.join(candidates)+'''</table>
<pre>'''+html.escape(json.dumps({k:v for k,v in diagnosis.items() if not isinstance(v,list)},ensure_ascii=False,indent=2))+'''</pre>
<h2>3. 代码与工程检查</h2><p>src/clip_retrieval_shuffle/loss.py：negative_permutation按query构造负项双射；teacher_targets先归一化再gather；target_kl停教师梯度；shuffle_terms保留原多正CE。独立Generator避免污染采样流，seed和step确定同设备置乱；各250步checkpoint保存模型、optimizer与step。CPU/CUDA随机流不宣称相同。</p>
<p>四项CPU数学契约验证双射/正项不变/两种熵、独立随机流、关闭置乱恢复原蒸馏全部参数梯度、lambda=0恢复原CE梯度、教师停梯度。完整真实缓存批次另验证两方向；最终三seed采样流和负暴露SHA须与两组历史方法匹配，Top10独立重评分，旧文件与正式简历只读保护。</p>
<h2>4. 完整结果与预设统计</h2><p>每方向原蒸馏减置乱均需至少+.5pp、单侧97.5%下界&gt;0、至少2seed同向，两方向均通过才新增测试确认。6000次图像整簇bootstrap在固定三checkpoint平均上计算；T2I同图五caption先平均。种子SD与条件图像CI分开，不将seed×image当独立。其他比较为探索。</p>'''+result_block('dev',dev)+result_block('test',test)+'''
<h2>5. 真实dev图与三配置Top5</h2><p>固定前4行，另按行号选择第一条原教师正确/置乱错误以及相反情况（若存在），用于解释而非估计发生率。相关性按源图ID，其他图描述不自动等于语义错误；展示缩略副本不是模型原224输入。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>解释保持熵为什么比直接改成均匀教师更接近单因素对照。</li><li>用真实负候选手算P_S-P_T，说明这一项梯度方向。</li><li>解释为什么不能改变正样本，也不能使用采样的全局RNG。</li><li>解释机制对照为何允许对照性能下降，却不代表新提分。</li><li>独立说明哪个结论由三seed、哪个由图像簇CI支持。</li></ol>
<p><a href="TEACHING_DATA.json">完整实际概率及图表数据</a>；<a href="../REAL_MECHANISM_PRECHECK.json">真实CPU合同</a>；<a href="../analysis/ANALYSIS.json">完整实验结果</a>；<a href="../analysis/VERIFICATION.json">来源/训练/评分核验</a>。</p>
<p>图表工作流参考Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner（2026），<a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>，官方当前v2已核验；本页为私有教学资料，无期刊合规声明。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'FIGURE_PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'first_cuda_update_sha256':sha_file(source),'teaching_data_sha256':sha_file(folder/'TEACHING_DATA.json'),
        'dev_analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json'),
        'test_analysis_sha256':sha_file(root/'analysis/test/ANALYSIS.json') if test else None,
        'selected_dev_rows':selected_rows,'transformations':['FP32 actual CUDA log-targets exponentiated in float64 for display; no target renormalization',
        'allpositives+union of each target top3 negatives; all480 values retained','actualsourceimage thumbnail,no attention/no image enhancement',
        'zero baseline Recall bars,three seed sampleSD and rawpoints','fixed first4dev+first discordant row each direction if any'],
        'manual_visual_review':'pending','scope':'private learning,not publisher certification'})
