"""Rescore complete QA, pair by images, and separate attention change from batch size."""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
import statistics
from vlm_probe.evaluate import write_json,now,sha_file
from vlm_weighted.evaluate import dev_data
from vlm_ocr.analysis import prepare_slices
from vlm_batching.analysis import read_cell,cluster_interval
from .prepare import DEFAULT_CONFIG,read_context


def analyze(config_path=DEFAULT_CONFIG):
    cfg,root,plan,executor=read_context(config_path)
    rows,images,image_root,metadata=dev_data(Path(cfg['asset_root']));by_id={r['question_id']:r for r in rows}
    expected={r['question_id']:r for r in json.loads(Path(cfg['upstream_input_contract']).read_text())['records'] if r['condition']=='baseline'}
    slices=prepare_slices(Path(cfg['asset_root']),root);features={r['question_id']:r for r in slices['features']}
    cells={};quality={};timing={};comparisons={};gates={};status=json.loads((root/'STATUS.json').read_text())
    for group,policies in cfg['groups'].items():
        cells[group]={};quality[group]={};timing[group]={};locks={}
        for policy in policies:
            cell,records,lock=read_cell(root/'runs'/group/policy/'dev',plan['schedules'][policy],by_id,expected,True)
            if lock['segmented_executor_sha256']!=sha_file(root/'SEGMENTED_EXECUTOR_LOCK.json') or lock['config_sha256']!=sha_file(config_path):raise ValueError('executor/config mismatch')
            cells[group][policy]=cell;quality[group][policy]=records;locks[policy]=lock;timing[group][policy]=[]
            for repeat in range(3):
                timed,_,tl=read_cell(root/'timing'/group/f'repeat{repeat}'/policy,plan['timing_schedules'][policy],by_id,expected,False)
                if tl['hardware']!=lock['hardware']:raise ValueError('timing device changed')
                timing[group][policy].append(timed)
        baseline,candidate=policies
        if locks[baseline]['hardware']!=locks[candidate]['hardware']:raise ValueError('same-batch contrast requires same GPU')
        base=quality[group][baseline];adapted=quality[group][candidate]
        delta=[adapted[r['question_id']]['score']-base[r['question_id']]['score'] for r in rows]
        interval=cluster_interval(rows,delta,cfg['bootstrap_replicates'],cfg['bootstrap_seed'],cfg['gate']['one_sided_alpha_per_candidate'])
        sliced={}
        for label in slices['slice_counts']:
            chosen=[r for r in rows if features[r['question_id']]['sft_target_answer_frequency']==label]
            sliced[label]={'n':len(chosen),'delta_pp':100*statistics.mean(adapted[r['question_id']]['score']-base[r['question_id']]['score'] for r in chosen),'scope':'point only,not slice noninferiority'}
        ratios=[c['resource']['preprocess_h2d_generation_qa_per_second']/b['resource']['preprocess_h2d_generation_qa_per_second'] for c,b in zip(timing[group][candidate],timing[group][baseline])]
        comparisons[group]={'baseline':baseline,'candidate':candidate,'quality':interval,'slices':sliced,
            'phase_speedup_repeats':ratios,'median_phase_speedup':statistics.median(ratios),
            'raw_answers_changed':sum(adapted[r['question_id']]['prediction']!=base[r['question_id']]['prediction'] for r in rows),
            'score_changed':sum(abs(d)>1e-12 for d in delta)}
        gate=cfg['gate'];checks={'conditional_overall_noninferiority':interval['one_sided_lower_bound_pp']>-gate['overall_noninferiority_margin_pp'],
            'rare_point_protection':sliced['rare_1_5']['delta_pp']>=gate['rare_point_delta_min_pp'],
            'frequent_point_protection':sliced['frequent_gt20']['delta_pp']>=gate['frequent_point_delta_min_pp'],
            'median_phase_speedup':statistics.median(ratios)>=gate['median_paired_phase_speedup_min'],
            'repeat_consistency':min(ratios)>=gate['each_repeat_phase_speedup_min'],
            'memory_cap':cells[group][candidate]['resource']['peak_allocated_gib']<=65,
            'observed_resources_clean':status['groups'][group]['contention_observations']==0 and status['resource_query_failures']==0}
        gates[group]={'passed':all(checks.values()),'checks':checks}
    result={'schema_version':'rarevlm.segmented.analysis.v1','status':'completed','completed_at':now(),
        'config_sha256':sha_file(config_path),'executor_sha256':sha_file(root/'SEGMENTED_EXECUTOR_LOCK.json'),
        'cells':cells,'timing':timing,'comparisons':comparisons,'gates':gates,
        'scored_predictions':6468,'unscored_timing_observations':3072,'unique_qa':1617,'unique_images':1024,
        'statistics':'6000paired image-cluster draws;two specified same-batch contrasts one-sided alpha .025;historically explored dev;no training-seed replication',
        'limits':cfg['limits']}
    (root/'analysis').mkdir(exist_ok=True);write_json(root/'analysis/ANALYSIS.json',result)
    with (root/'analysis/cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['group','policy','score_percent','phase_qa_per_second','generation_qa_per_second','peak_allocated_gib'])
        writer.writeheader()
        for group,entries in cells.items():
            for policy,cell in entries.items():
                r=cell['resource'];writer.writerow({'group':group,'policy':policy,'score_percent':cell['score_percent'],
                    'phase_qa_per_second':r['preprocess_h2d_generation_qa_per_second'],
                    'generation_qa_per_second':r['pure_generation_qa_per_second'],'peak_allocated_gib':r['peak_allocated_gib']})
    build_reader(root,cfg,result,rows,images,image_root,quality)
    return result


def build_reader(root,cfg,result,rows,images,image_root,quality):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    learning=root/'learning';learning.mkdir(exist_ok=True)
    labels=[p for ps in cfg['groups'].values() for p in ps];cells=[c for entries in result['cells'].values() for c in entries.values()]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,key,title in zip(axes,['preprocess_h2d_generation_qa_per_second','peak_allocated_gib'],['Offline phases: QA/s','Peak allocated: GiB']):
            bars=ax.barh(labels,[c['resource'][key] for c in cells],color=['#4477AA','#228833','#4477AA','#228833'],edgecolor='black')
            for i,b in enumerate(bars):b.set_hatch('///' if i%2==0 else 'xx')
            ax.set_xlim(left=0);ax.invert_yaxis();ax.set_title(title)
        fig.savefig(learning/'measured_cost.png',dpi=180,facecolor='white');plt.close(fig)
        feature=json.loads((root/'workers/batch4/FEATURE_CHECK.json').read_text())['records'][1]
        bounds=np.array(feature['cu_seqlens']);n=int(bounds[-1]);sample=(np.arange(256)+.5)*n/256
        group=np.searchsorted(bounds,sample,side='right')-1;mask=group[:,None]==group[None,:]
        fig,ax=plt.subplots(figsize=(5,4.5),layout='constrained')
        ax.imshow(mask,interpolation='nearest',cmap=matplotlib.colors.ListedColormap(['#DDDDDD','#4477AA']))
        ax.set_title('Allowed blocks from real cu_seqlens\n256x256 sampled shape; NOT attention weights');ax.set_xlabel('Key position (sampled)');ax.set_ylabel('Query position (sampled)')
        fig.savefig(learning/'allowed_blocks.png',dpi=160,facecolor='white');plt.close(fig)
    table=''.join(f'<tr><td>{p}</td><td>{c["score_percent"]:.3f}</td><td>{c["resource"]["preprocess_h2d_generation_qa_per_second"]:.3f}</td><td>{c["resource"]["peak_allocated_gib"]:.3f}</td></tr>' for p,c in zip(labels,cells))
    cases=[]
    for row in sorted(rows,key=lambda r:int(r['question_id']))[:8]:
        q=row['question_id'];name=q+'.jpg'
        with Image.open(image_root/images[row['image_id']]['path']) as image:image=image.convert('RGB');image.thumbnail((640,640));image.save(learning/name,quality=92)
        answers=''.join(f'<li>{p}: {html.escape(quality[g][p][q]["prediction"])}；score={quality[g][p][q]["score"]:.2f}</li>' for g,ps in cfg['groups'].items() for p in ps)
        cases.append(f'<article><h3>QA {q}</h3><img src="{name}" alt="原图等比缩小副本，模型输入仍是原图"><p>{html.escape(row["question"])}</p><ul>{answers}</ul></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视觉SDPA：从稠密mask到分段同长度合批</title><style>body{font:17px/1.8 sans-serif;max-width:1100px;margin:auto;padding:24px;color:#222}img{max-width:100%}td,th{border:1px solid #777;padding:8px}table{border-collapse:collapse}pre{background:#eee;padding:14px;white-space:pre-wrap}article{border-top:1px solid #777}</style><h1>从真实显存瓶颈到注意力执行优化</h1><p>同checkpoint、原图、问题、BF16/1024与生成规则不变；分别在batch4和batch16比较原始稠密mask与分段合批。两组各在同GPU重跑对照，4格共6468评分预测，另3072不评分计时观察，共享1617QA不是独立数据重复。</p><h2>基本原理</h2><p>Attention=softmax(QKᵀ/√d+M)V。原视觉SDPA创建[1,N,N]布尔mask，只允许cu_seqlens划分的块内注意。块之间不交互，因此可以按块长度将Q/K/V放入批量维度，每组不带mask计算SDPA，再按原索引写回。</p><pre># same qkv, same RoPE, same projections
for length, count, indices in layouts(cu_seqlens):
    q_group = q[indices].reshape(count,length,heads,dim).transpose(1,2)
    out_group = SDPA(q_group,k_group,v_group,mask=None,dropout=0)
    out.index_copy_(0,indices,out_group.transpose(1,2).reshape(-1,heads,dim))</pre><p>原mask存储为N²个布尔元素，块内必要关系为ΣLᵢ²；实际运行还包含投影、RoPE、索引分组、64项layout缓存和SDPA后端选择，不能把全部收益单独归因于mask字节。数学变换不保证BF16逐bit一致，需真实局部特征误差和完整QA质量检验。</p><h2>真实允许注意范围</h2><img src="allowed_blocks.png" alt="从真实视觉层cu_seqlens重建并采样的允许注意块，不是模型attention权重"><h2>工程实测</h2><img src="measured_cost.png" alt="同batch的dense与分段SDPA离线阶段吞吐和allocated显存，分组同卡比较"><table><tr><th>条件</th><th>共识分%</th><th>离线阶段QA/s</th><th>allocated GiB</th></tr>'''+table+'''</table><p>额外预热后测全部QA。三轮同256QA计时显示每轮比值与中位数；没有网络、队列、TTFT或在线服务SLA。整体0.25pp非劣界限、图像整簇6000次bootstrap、两比较单侧alpha=.025；rare/frequent仅点值保护。内部dev已探索，结果不证明盲测、无损或种子稳定性。</p><pre>'''+html.escape(json.dumps(result['gates'],indent=2,ensure_ascii=False))+'''</pre><h2>真实图片和实际回答</h2>'''+''.join(cases)+'''<h2>学习检查</h2><ol><li>为什么block-diagonal允许范围能改为批量维度？</li><li>RoPE为什么应先按原位置计算再分组？</li><li>为什么只比较dense16与segmented16才能隔离执行变化？</li><li>数学关系相同为何仍检查BF16回答质量？</li></ol><p><a href="../analysis/cells.csv">指标CSV</a> · <a href="../analysis/ANALYSIS.json">全量对照与区间JSON</a>。</p><p>图表规范引用：Kassis等(2026)，<a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>；算法依据为锁定Transformers4.51.1官方源码，图表工作流文献不作为性能证据。</p></html>'''
    (learning/'学习页.html').write_text(page)
    write_json(learning/'FIGURE_PROVENANCE.json',{'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),
        'source_sha256':sha_file(Path(__file__)),'real_cu_seqlens':feature['cu_seqlens'],
        'transformations':['no smoothing;all4 complete cells','allowed mask sampled at256positions,not attention weights','original image thumbnails for display only'],
        'medium':'private learning webpage,no target journal','manual_visual_review':'pending'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    print(analyze(parser.parse_args().config)['gates'])
