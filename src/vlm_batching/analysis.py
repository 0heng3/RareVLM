"""Independent rescoring, image-cluster noninferiority and repeated same-card costs."""
from __future__ import annotations
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import now,sha_file,write_json,content_sha
from vlm_deep.evaluate import score_prediction
from vlm_weighted.evaluate import dev_data
from vlm_ocr.analysis import prepare_slices
from .prepare import DEFAULT_CONFIG,read_plan
from .evaluate import cached_batches,summarize


def cluster_interval(rows,delta,replicates,seed,alpha):
    import numpy as np
    ids=sorted({r['image_id'] for r in rows});index={iid:i for i,iid in enumerate(ids)}
    sums=np.zeros(len(ids));counts=np.zeros(len(ids))
    for row,d in zip(rows,delta):
        i=index[row['image_id']];sums[i]+=d;counts[i]+=1
    rng=np.random.default_rng(seed);samples=[]
    for begin in range(0,replicates,128):
        draws=rng.integers(0,len(ids),size=(min(128,replicates-begin),len(ids)))
        samples.extend((100*sums[draws].sum(axis=1)/counts[draws].sum(axis=1)).tolist())
    return {'delta_pp':100*statistics.mean(delta),'exploratory_two_sided_95_ci_pp':np.quantile(samples,[.025,.975]).tolist(),
            'one_sided_lower_bound_pp':float(np.quantile(samples,alpha)),
            'one_sided_confidence':1-alpha,'image_clusters':len(ids),'replicates':replicates,'seed':seed,
            'unit':'paired whole image, ratio-of-total-QA consensus estimator',
            'scope':'conditional on fixed checkpoint and historically explored dev; no seed or blind generalization claim'}


def read_cell(directory,schedule,by_id,expected,scored):
    lock=json.loads((directory/'INPUT_LOCK.json').read_text())
    result=json.loads((directory/'EVALUATION.json').read_text())
    batches,_=cached_batches(directory/'batches.jsonl',schedule,lock,by_id,expected,scored)
    if len(batches)!=len(schedule) or result['batches_sha256']!=sha_file(directory/'batches.jsonl'):
        raise ValueError('completed cell incomplete/changed')
    if result['input_lock_sha256']!=content_sha(lock) or result['predictions_sha256']!=sha_file(directory/'predictions.jsonl'):
        raise ValueError('completed cell lock/predictions differ')
    recalculated=summarize(batches,lock,result['session'],scored)
    if result['resource']!=recalculated['resource']:
        raise ValueError('timing summary differs')
    if scored and not math.isclose(result['score_percent'],recalculated['score_percent'],abs_tol=1e-10):
        raise ValueError('score summary differs')
    records={r['question_id']:r for b in batches for r in b['records']}
    return result,records,lock


def analyze(config_path=DEFAULT_CONFIG):
    cfg,root,plan=read_plan(config_path)
    rows,images,image_root,metadata=dev_data(Path(cfg['asset_root']));by_id={r['question_id']:r for r in rows}
    expected={r['question_id']:r for r in json.loads(Path(cfg['upstream_input_contract']).read_text())['records'] if r['condition']=='baseline'}
    slices=prepare_slices(Path(cfg['asset_root']),root)
    features={r['question_id']:r for r in slices['features']}
    quality={};results={};locks={};timing={};comparisons={};gates={};manifest={}
    for group,policies in cfg['groups'].items():
        quality[group]={};results[group]={};locks[group]={};timing[group]={}
        for policy in policies:
            directory=root/'runs'/group/policy/'dev'
            result,records,lock=read_cell(directory,plan['schedules'][policy],by_id,expected,True)
            if lock['source_hashes']!=plan['source_hashes'] or lock['config_sha256']!=sha_file(config_path):
                raise ValueError('worker source/config differs')
            quality[group][policy]=records;results[group][policy]=result;locks[group][policy]=lock
            manifest[str(directory/'predictions.jsonl')]=sha_file(directory/'predictions.jsonl')
            timing[group][policy]=[]
            for repeat in range(3):
                d=root/'timing'/group/f'repeat{repeat}'/policy
                timed,_,timelock=read_cell(d,plan['timing_schedules'][policy],by_id,expected,False)
                if timelock['hardware']!=lock['hardware']:raise ValueError('timing and full-dev GPU differ')
                timing[group][policy].append(timed)
        base=quality[group]['fifo4']
        for policy in policies:
            if policy=='fifo4':continue
            if locks[group][policy]['hardware']!=locks[group]['fifo4']['hardware']:
                raise ValueError('candidate/baseline not on same GPU')
            candidate=quality[group][policy]
            delta=[candidate[r['question_id']]['score']-base[r['question_id']]['score'] for r in rows]
            interval=cluster_interval(rows,delta,cfg['bootstrap_replicates'],cfg['bootstrap_seed'],cfg['gate']['one_sided_alpha_per_candidate'])
            sliced={}
            for label in slices['slice_counts']:
                selected=[r for r in rows if features[r['question_id']]['sft_target_answer_frequency']==label]
                sliced[label]={'n':len(selected),'baseline':100*statistics.mean(base[r['question_id']]['score'] for r in selected),
                    'candidate':100*statistics.mean(candidate[r['question_id']]['score'] for r in selected),
                    'delta_pp':100*statistics.mean(candidate[r['question_id']]['score']-base[r['question_id']]['score'] for r in selected),
                    'scope':'point estimate only, not slice noninferiority'}
            ratios=[c['resource']['preprocess_h2d_generation_qa_per_second']/b['resource']['preprocess_h2d_generation_qa_per_second']
                for c,b in zip(timing[group][policy],timing[group]['fifo4'])]
            comparisons[policy]={'group':group,'quality':interval,'slices':sliced,
                'raw_answers_changed':sum(candidate[r['question_id']]['prediction']!=base[r['question_id']]['prediction'] for r in rows),
                'score_changed':sum(abs(d)>1e-12 for d in delta),'paired_phase_speedup_repeats':ratios,
                'median_paired_phase_speedup':statistics.median(ratios),
                'full_dev_phase_speedup':results[group][policy]['resource']['preprocess_h2d_generation_qa_per_second']/results[group]['fifo4']['resource']['preprocess_h2d_generation_qa_per_second']}
    status=json.loads((root/'STATUS.json').read_text())
    for policy,comparison in comparisons.items():
        group=comparison['group'];gate=cfg['gate']
        checks={'conditional_overall_noninferiority':comparison['quality']['one_sided_lower_bound_pp']>-gate['overall_noninferiority_margin_pp'],
            'rare_point_protection':comparison['slices']['rare_1_5']['delta_pp']>=gate['rare_point_delta_min_pp'],
            'frequent_point_protection':comparison['slices']['frequent_gt20']['delta_pp']>=gate['frequent_point_delta_min_pp'],
            'median_phase_speedup':comparison['median_paired_phase_speedup']>=gate['median_paired_phase_speedup_min'],
            'repeat_speedup_consistency':min(comparison['paired_phase_speedup_repeats'])>=gate['each_repeat_phase_speedup_min'],
            'allocated_memory_cap':results[group][policy]['resource']['peak_allocated_gib']<=gate['peak_allocated_gib_max'],
            'resource_observation_clean':status['groups'][group]['contention_observations']==0 and status['resource_query_failures']==0}
        gates[policy]={'passed':all(checks.values()),'checks':checks}
    result={'schema_version':'rarevlm.batching.analysis.v1','status':'completed','completed_at':now(),
        'config_sha256':sha_file(config_path),'plan_sha256':sha_file(root/'PLAN.json'),
        'cells':results,'timing':timing,'comparisons':comparisons,'gates':gates,
        'predictions_sha256':manifest,'scored_predictions':8085,'unique_qa':1617,'unique_images':1024,
        'limits':cfg['limits'],'statistics':'paired image-cluster percentile screen;one-sided alpha .05/3 for three prespecified candidates;conditional exploratory noninferiority only'}
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    write_json(analysis/'ANALYSIS.json',result)
    with (analysis/'cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['group','policy','score_percent','phase_qa_per_second','generation_qa_per_second','peak_allocated_gib','padding_fraction'])
        writer.writeheader()
        for group,cells in results.items():
            for policy,cell in cells.items():
                r=cell['resource'];writer.writerow({'group':group,'policy':policy,'score_percent':cell['score_percent'],
                    'phase_qa_per_second':r['preprocess_h2d_generation_qa_per_second'],
                    'generation_qa_per_second':r['pure_generation_qa_per_second'],'peak_allocated_gib':r['peak_allocated_gib'],
                    'padding_fraction':r['padding_fraction']})
    build_reader(cfg,root,plan,result,rows,images,image_root,quality)
    return result


def build_reader(cfg,root,plan,result,rows,images,image_root,quality):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    learning=root/'learning';learning.mkdir(exist_ok=True)
    labels=[f'{g}/{p}' for g,policies in cfg['groups'].items() for p in policies]
    cells=[result['cells'][g][p] for g,policies in cfg['groups'].items() for p in policies]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
        for ax,key,title in zip(axes,['preprocess_h2d_generation_qa_per_second','peak_allocated_gib','padding_fraction'],['Offline phases: QA/s','Peak allocated: GiB','Input padding: fraction']):
            values=[c['resource'][key] for c in cells]
            bars=ax.barh(labels,values,color=['#4477AA','#228833','#AA3377','#4477AA','#CC6677'],edgecolor='black')
            for i,b in enumerate(bars):b.set_hatch(['///','xx','..','///','++'][i])
            ax.set_xlim(left=0);ax.set_title(title);ax.invert_yaxis()
        fig.savefig(learning/'measured_cost.png',dpi=180,facecolor='white')
        fig.savefig(learning/'measured_cost.svg',facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,policy in zip(axes,['fifo16','bucket16']):
            ids=plan['schedules'][policy][0];source=json.loads(Path(cfg['upstream_input_contract']).read_text())
            lens={r['question_id']:r['input_tokens'] for r in source['records'] if r['condition']=='baseline'}
            width=max(lens[q] for q in ids)
            array=np.zeros((len(ids),width),dtype=np.uint8)
            for i,q in enumerate(ids):array[i,-lens[q]:]=1
            ax.imshow(array,aspect='auto',interpolation='nearest',cmap=matplotlib.colors.ListedColormap(['#DDDDDD','#4477AA']),vmin=0,vmax=1)
            ax.set_title(f'{policy}: actual first batch\ngrey=left padding, blue=valid input')
            ax.set_xlabel('Padded input position');ax.set_ylabel('Batch row');ax.set_yticks(range(len(ids)),ids,fontsize=7)
        fig.savefig(learning/'actual_padding.png',dpi=160,facecolor='white');plt.close(fig)
    table=[]
    for group,policies in cfg['groups'].items():
        for policy in policies:
            c=result['cells'][group][policy];r=c['resource']
            table.append(f'<tr><td>{group}/{policy}</td><td>{c["score_percent"]:.3f}</td><td>{r["preprocess_h2d_generation_qa_per_second"]:.3f}</td><td>{r["peak_allocated_gib"]:.3f}</td><td>{100*r["padding_fraction"]:.2f}%</td></tr>')
    cases=[]
    selected=sorted(rows,key=lambda r:int(r['question_id']))[:8]
    for row in selected:
        qid=row['question_id'];source=image_root/images[row['image_id']]['path'];name=f'{qid}.jpg'
        with Image.open(source) as image:
            image=image.convert('RGB');image.thumbnail((640,640));image.save(learning/name,quality=92)
        answers=[]
        for group,policies in cfg['groups'].items():
            for policy in policies:
                r=quality[group][policy][qid]
                answers.append(f'<li>{group}/{policy}: {html.escape(r["prediction"])}；score={r["score"]:.2f}；input={r["input_tokens"]}；padded={r["padded_input_tokens"]}；visual={r["visual_tokens"]}</li>')
        cases.append(f'<article><h3>真实 QA {qid}</h3><img src="{name}" alt="原图展示副本，仅等比缩小；模型仍使用锁定原图"><p>{html.escape(row["question"])}</p><ul>{"".join(answers)}</ul></article>')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM 批量推理：代码、真实图像与结果</title>
<style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #777;padding:8px}article{border-top:1px solid #777;padding:20px 0}pre{white-space:pre-wrap;background:#eee;padding:16px}</style>
<h1>保持质量的离线批量推理</h1><p>研究不是新增训练：固定普通 CE seed42、原权重 BF16、视觉预算上限1024、贪心最多32新token。5格完整dev各1617QA；两组在各自GPU上重跑FIFO4基线。答案仅生成后评分。</p>
<h2>原理</h2><p>设一批有B个输入，真实长度为Lᵢ。左padding长度为Lmax，总输入槽位=B×Lmax，有效槽位=ΣLᵢ；padding比例=1−ΣLᵢ/(B×Lmax)。长度分桶将相近Lᵢ放在同一批。这个比例是输入形状代理，不是注意力、完整FLOPs或时延预测；实际收益还受视觉编码、KV cache和本批最长输出影响。</p>
<pre>sorted_ids = sorted(question_ids, key=lambda q: (input_lengths[q], int(q)))
batches = [sorted_ids[i:i+B] for i in range(0, len(sorted_ids), B)]
# make_batches只接收ID、长度和策略；不接收answers/target/score。
# collate对input_ids左补pad，对attention_mask左补0。
# generate后才查十个人工答案并计算TextVQA共识分。
</pre><p>真实代码：src/vlm_batching/schedule.py、evaluate.py；复用src/vlm_deep/common.py中的collate，以及锁定BF16加载器。按ID恢复配对关系，不能把排序后行号当原始QA。</p>
<h2>真实输入padding</h2><img src="actual_padding.png" alt="FIFO16与bucket16的真实首批输入长度和左padding；不同首批样本，形状展示不构成质量对照">
<h2>工程实测</h2><p>吞吐含CPU原图解码/校验/processor、H2D与生成；每格先额外预热一批，之后计量全部计划QA。完整wall另存JSON。输入长度元数据的历史准备成本单列。不含网络、排队或线上并发，不宣称服务SLA。</p><img src="measured_cost.png" alt="五格完整dev的离线阶段吞吐、allocated显存和输入padding比例；只在各组同GPU内比较">
<table><tr><th>组/策略</th><th>共识分%</th><th>离线阶段QA/s</th><th>allocated GiB</th><th>padding</th></tr>'''+''.join(table)+'''</table>
<h2>预设验收与边界</h2><p>整体非劣容许损失0.25pp，图像整簇bootstrap6000次，三候选单侧alpha=0.05/3；罕见/高频仅点值保护。3次固定256QA计时用于重复执行，不是3个独立数据集，也不是训练种子。既有dev已探索，非劣结论仅限固定checkpoint与该dev；不证明无损、盲测泛化或跨任务保持。</p><pre>'''+html.escape(json.dumps(result['gates'],ensure_ascii=False,indent=2))+'''</pre>
<h2>8个预先按ID选取的真实图像案例</h2><p>展示图片是原图的等比缩小副本，未替代模型输入。长度图按保存attention-mask形状重建，不是模型注意力。</p>'''+''.join(cases)+'''<h2>自检</h2><ol><li>为什么左补0的attention mask仍可能浪费计算？</li><li>为什么batch16对batch4的变化不能全部归因分桶？</li><li>为何逐题像素与ID相同，BF16答案仍可能变化？</li><li>如何区分非劣、等价、分数相同与无损？</li><li>QA/s与单请求延迟、TTFT及线上并发有什么不同？</li></ol><p>数据下载：<a href="../analysis/cells.csv">完整格指标CSV</a>；<a href="../analysis/ANALYSIS.json">重复计时、配对区间和门槛JSON</a>。</p>
<p>图表核验参考：Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026). <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>（arXiv记录2026-10-13核对，当前v2）；未将其作为算法性能依据。</p></html>'''
    (learning/'学习页.html').write_text(text)
    with Image.open(learning/'measured_cost.png') as im:dimensions=list(im.size)
    write_json(learning/'FIGURE_PROVENANCE.json',{'source':str(root/'analysis/ANALYSIS.json'),
        'source_sha256':sha_file(root/'analysis/ANALYSIS.json'),'figure_source_sha256':sha_file(Path(__file__)),
        'png_dimensions':dimensions,'dpi':180,'medium':'private learning webpage;no target journal',
        'transformations':['no smoothing;complete cell summaries','raw image display thumbnails only','first actual batch valid/padding mask shape reconstruction'],
        'uncertainty':'shown numerically in linked JSON;bars are measured point costs,not uncertainty',
        'metadata_checked':True,'manual_visual_review':'pending'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    print(json.dumps(analyze(parser.parse_args().config)['gates'],indent=2))
