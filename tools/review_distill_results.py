"""Export the completed study and independently check metrics from saved top10."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vlm_probe.evaluate import sha_file,write_json,now
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT


def main():
    cfg=json.loads((PROJECT_ROOT/'configs/clip_retrieval_distill_v1.json').read_text());root=Path(cfg['output_root'])
    result=json.loads((root/'analysis/ANALYSIS.json').read_text());done=json.loads((root/'COMPLETION.json').read_text())
    if done['analysis_sha256']!=sha_file(root/'analysis/ANALYSIS.json') or done['status']!='completed':raise ValueError('Incomplete report')
    lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    if any(sha_file(SOURCE_ROOT/name)!=digest for name,digest in lock['sources'].items()):raise ValueError('Frozen study sources changed')
    if sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md')!=lock['protocol_sha256']:raise ValueError('Protocol changed')
    guard=json.loads((root/'FROZEN_GUARD.json').read_text())
    if any(sha_file(Path(path))!=digest for path,digest in guard['protected_sha256'].items()):raise ValueError('Protected legacy/resume files changed')
    checks=[];old=root.parent/'clip_retrieval_v1'
    for condition,seeds in [('frozen',[0]),('uniform_mean',[0,42,200]),('distill',[0,42,200])]:
        for seed in seeds:
            key='frozen' if condition=='frozen' else f'{condition}_seed{seed}'
            path=root/'evaluation'/key/'test';cell=json.loads((path/'EVALUATION.json').read_text())
            with np.load(path/'per_query.npz') as a:
                positive_t=a['t2i_top10']==np.arange(5000)[:,None]//5
                positive_i=a['i2t_top10']//5==np.arange(1000)[:,None]
                for direction,positive in [('T2I',positive_t),('I2T',positive_i)]:
                    for k in [1,5,10]:
                        if not np.array_equal(positive[:,:k].any(1),a[f'{direction}_R@{k}']):raise ValueError('Top10/Recall inconsistent')
                    discount=1/np.log2(np.arange(10)+2)
                    ideal=discount[:5 if direction=='I2T' else 1].sum()
                    score=(positive*discount).sum(1)/ideal
                    if not np.allclose(score,a[f'{direction}_nDCG@10'],atol=1e-7):raise ValueError('Top10/nDCG inconsistent')
                baseline_equal=None
                if condition=='uniform_mean':
                    with np.load(old/'evaluation'/f'uniform_negatives_seed{seed}'/'test/per_query.npz') as b:
                        baseline_equal=all(np.array_equal(a[name],b[name]) for name in b.files)
                    if not baseline_equal:raise ValueError('Reused baseline test differs from v1')
            checks.append({'key':key,'test_rankings_sha256':cell['per_query_sha256'],'independent_top10_rescore':'passed','baseline_v1_exact':baseline_equal})
    public=dict(result);public['source_analysis_sha256']=sha_file(root/'analysis/ANALYSIS.json');public['source_study']='clip_retrieval_distill_v1'
    write_json(PROJECT_ROOT/'results/clip_retrieval_distill_v1_metrics.json',public)
    test=result['test'];folder=root/'learning'
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,metric in zip(axes,['T2I_R@1','I2T_R@1']):
            for i,condition in enumerate(['frozen','uniform_mean','distill']):
                v=test['summaries'][condition][metric]
                ax.bar(i,v['mean'],color=['#4477AA','#228833','#AA3377'][i],hatch=['///','xx','..'][i],edgecolor='black')
                if v['sample_sd'] is not None:
                    ax.errorbar(i,v['mean'],yerr=v['sample_sd'],fmt='none',color='black',capsize=4)
                    ax.scatter(i+np.array([-.07,0,.07]),list(v['per_seed'].values()),color='black',s=15,zorder=4)
                ax.text(i,v['mean']+2,f'{v["mean"]:.2f}',ha='center',fontsize=10)
            ax.set_xticks(range(3),['Frozen','Mean baseline','Distill']);ax.set_ylim(0,100);ax.set_ylabel('Recall@1 (%)')
            ax.set_title(metric+' supplied test\n3-seed sample SD')
        fig.suptitle('Fixed configuration; known historical test exposure',fontsize=12)
        fig.savefig(folder/'test_recall.png',dpi=160,facecolor='white');plt.close(fig)
    table='<h2>固定配置的完整test确认</h2><p>1000图/5000caption；dev通过后固定原设置，全test一次。三种子均值±样本SD；图像簇区间另列。公开test已曝光，不能称盲测。</p><table><tr><th>方法</th><th>文搜图R1</th><th>图搜文R1</th><th>图搜文nDCG10</th></tr>'
    for condition,values in test['summaries'].items():
        table+='<tr><td>'+condition+'</td>'
        for metric in ['T2I_R@1','I2T_R@1','I2T_nDCG@10']:
            v=values[metric];table+=f'<td>{v["mean"]:.3f}'+(f' ± {v["sample_sd"]:.3f}' if v['sample_sd'] is not None else '（无seed SD）')+'</td>'
        table+='</tr>'
    table+='</table><img src="test_recall.png" alt="完整历史test三配置两个检索方向，三个点与误差条为训练种子指标与样本SD"><p>相对mean，T2I差+6.180pp，95%图像区间[5.060,7.320]；I2T差+5.300pp，[3.467,7.133]。相对冻结I2T点值+2.233pp，区间[−.067,4.533]跨0，预设非劣通过，不称显著超过冻结。新3训练、dev新增3格/复用4格、test新7格，共10个新评分格，不当独立样本。</p>'
    page=folder/'学习页.html';text=page.read_text();marker='<h2>6. 三种配置的真实图搜文Top5</h2>'
    if '<h2>固定配置的完整test确认</h2>' not in text:
        if marker not in text:raise ValueError('Learning page insertion point absent')
        page.write_text(text.replace(marker,table+marker))
    text=page.read_text()
    if '监督标签与教师分布的平均' not in text:
        lesson='<p>单方向lambda=1时，L=CE(P_GT,P_S)+KL(P_T||P_S)=2CE((P_GT+P_T)/2,P_S)−H(P_T)。因此相当于用监督标签与教师分布的平均训练学生，教师熵是固定项。本轮没有均匀软目标或相似熵的对照，不能将增益全部归因于教师语义关系。</p>'
        page.write_text(text.replace('<h2>5. 完整dev结果与门槛</h2>',lesson+'<h2>5. 完整dev结果与门槛</h2>'))
    write_json(root/'analysis/VERIFICATION.json',{'status':'passed','checked_at':now(),'analysis_sha256':done['analysis_sha256'],
        'review_source_sha256':sha_file(Path(__file__)),'legacy_resume_guard':'unchanged','frozen_sources_and_protocol':'unchanged',
        'test_top10_rescore':checks,'learning_page_sha256':sha_file(page)})
    p=folder/'FIGURE_PROVENANCE.json';provenance=json.loads(p.read_text());provenance.update(
        test_analysis_sha256=sha_file(root/'analysis/test/ANALYSIS.json'),test_plot_review_source_sha256=sha_file(Path(__file__)),
        test_plot_transformations='complete test3seed mean/sampleSD/rawpoints,zero-to-100 axes,no smoothing;frozen one model no SD')
    write_json(p,provenance)
    print('Completed study verified; all seven test cells independently rescored; baseline test arrays reproduce v1; sources/resume unchanged.')


if __name__=='__main__':main()
