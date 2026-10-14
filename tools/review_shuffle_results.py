"""Presentation-only review of completed shuffle study; keep locked science."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import now,sha_file,write_json


def main():
    cfg=json.loads((PROJECT_ROOT/'configs/clip_retrieval_shuffle_v1.json').read_text())
    root=Path(cfg['output_root']);folder=root/'learning'
    result=json.loads((root/'analysis/ANALYSIS.json').read_text())
    done=json.loads((root/'COMPLETION.json').read_text())
    if done['analysis_sha256']!=sha_file(root/'analysis/ANALYSIS.json'):raise ValueError('Analysis changed')
    data=json.loads((folder/'TEACHING_DATA.json').read_text());selected=data['selected_columns']
    original=np.asarray(data['full_teacher_probability']);shuffled=np.asarray(data['full_shuffled_probability'])
    gradient=np.asarray(data['ideal_unreduced_kl_gradient'])
    real=json.loads((root/'runs/negative_shuffle/seed42/FIRST_UPDATE_DIAGNOSTIC.json').read_text())
    labels=[('P' if real['positive'][k] else 'N')+str(k) for k in selected];x=np.arange(len(selected))
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        axes[0].bar(x-.18,original[selected],.36,color='#4477AA',hatch='///',edgecolor='black',label='Original teacher')
        axes[0].bar(x+.18,shuffled[selected],.36,color='#AA3377',hatch='..',edgecolor='black',label='Negative-shuffled target')
        axes[0].set_xticks(x,labels,rotation=45)
        axes[0].set_ylim(0,float(max(original[selected].max(),shuffled[selected].max()))*1.25)
        axes[0].set_ylabel('Actual CUDA target probability (not renormalized)')
        axes[0].set_title(f'All positives + target top negatives ({len(selected)} / 480)\nSame positive values; negative associations changed')
        axes[0].legend(fontsize=8)
        axes[1].scatter(x,gradient[selected],marker='o',color='#228833',s=40)
        axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_xticks(x,labels,rotation=45)
        axes[1].set_ylabel('Ideal single-query KL gradient: P_student - P_target')
        axes[1].set_title('Actual seed42 update0, before optimizer\nDifferent candidate columns; no continuous interpolation')
        fig.savefig(folder/'negative_relation_gradient.png',dpi=160,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,metric in zip(axes,['T2I_R@1','I2T_R@1']):
            for i,condition in enumerate(['frozen','uniform_mean','distill','negative_shuffle']):
                s=result['dev']['summaries'][condition][metric]
                ax.bar(i,s['mean'],color=['#4477AA','#228833','#AA3377','#666666'][i],hatch=['///','xx','..','--'][i],edgecolor='black')
                if s['sample_sd'] is not None:
                    ax.errorbar(i,s['mean'],yerr=s['sample_sd'],fmt='none',color='black',capsize=4)
                    ax.scatter(i+np.array([-.07,0,.07]),list(s['per_seed'].values()),color='black',s=18,zorder=4)
                ax.text(i,s['mean']+2.2,f'{s["mean"]:.2f}',ha='center',fontsize=10)
            ax.set_xticks(range(4),['Frozen','Mean CE','Teacher','Shuffled'],rotation=15)
            ax.set_ylim(0,100);ax.set_ylabel('Recall@1 (%)')
            ax.set_title(metric+', complete internal dev\nAdapted models:3seed SD/rawpoints; frozen:1model')
        fig.savefig(folder/'dev_recall.png',dpi=160,facecolor='white');plt.close(fig)
        provenance=json.loads((folder/'FIGURE_PROVENANCE.json').read_text())
        photos=['real_train_query.jpg',*[f'dev_{i}.jpg' for i in provenance['selected_dev_rows']]]
        fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
        for ax in axes.flat:ax.axis('off')
        for ax,name in zip(axes.flat,photos):
            with Image.open(folder/name) as im:ax.imshow(im)
            ax.set_title(name)
        fig.suptitle('Source-image display thumbnails; not encoder input or attention')
        fig.savefig(folder/'thumbnail_review.png',dpi=130,facecolor='white');plt.close(fig)
    page=folder/'学习页.html';text=page.read_text()
    note='<p>概率图纵轴从0起，终点覆盖全部展示概率；只展示11/480项，未重归一化。梯度图使用独立候选点，不用连线暗示连续性。</p>'
    marker='<img src="negative_relation_gradient.png"'
    if note not in text:text=text.replace(marker,note+marker)
    boundary='<p>机制边界：按query独立置乱同时破坏负候选与概率的对应及跨query/双向关系一致性，不能进一步单独归因纯语义相似度。原教师对文搜图有支持，对图搜文未通过；置乱仍超过CE是探索证据，不证明正样本置信度就是唯一原因，也不证明两个蒸馏目标等价。</p>'
    text=text.replace('<h2>5. 真实dev图与三配置Top5</h2>',boundary+'<h2>5. 真实dev图与三配置Top5</h2>')
    page.write_text(text)
    provenance.update(review_source_sha256=sha_file(Path(__file__)),reviewed_render_at=now(),
        presentation_changes=['probability bars start0/endcover actual selected maximum,no renormalization',
        'categorical gradients shown as points without connecting lines','dev raw means annotated,seed structure named',
        'source-image contact sheet for manual inspection'],
        plot_sha256={name:sha_file(folder/name) for name in ['negative_relation_gradient.png','dev_recall.png','thumbnail_review.png']},
        manual_visual_review='pending')
    write_json(folder/'FIGURE_PROVENANCE.json',provenance)
    print('Presentation exported; analysis and scientific source bytes untouched.')


if __name__=='__main__':main()
