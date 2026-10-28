"""Source images, actual ITM margins, candidate bounds and quality/cost results."""
import html
import json
from pathlib import Path
import numpy as np
import torch
from clip_retrieval.features import read_image
from vlm_probe.evaluate import sha_file,now
from .common import write_json


def build(config_path,dev,test=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import BlipProcessor
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);folder=root/'learning';folder.mkdir(exist_ok=True)
    manifest=json.loads((Path(cfg['feature_root'])/'DATA_MANIFEST.json').read_text())
    offsets=json.loads((Path(cfg['feature_root'])/'IMAGE_INDEX.json').read_text())['offsets'];rows=manifest['splits']['dev']
    with np.load(root/'candidates/dev/seed42.npz') as z:before=z['i2t'].copy()
    with np.load(root/'evaluation/rerank_seed42/dev/per_query.npz') as z:
        after=z['i2t_top50'].copy();margins=z['i2t_scores_original_pool'].copy()
    original_hit=before[:,0]//5==np.arange(1000);new_hit=after[:,0]//5==np.arange(1000)
    selected=list(range(4))
    for mask in [~original_hit&new_hit,original_hit&~new_hit]:
        matches=np.flatnonzero(mask)
        if len(matches) and int(matches[0]) not in selected:selected.append(int(matches[0]))
    handles={};examples=[];teaching_rows=[]
    try:
        for i in selected:
            image=read_image(rows[i],offsets,handles);image.thumbnail((480,480));image.save(folder/f'dev_{i}.jpg',quality=92)
            lookup={int(c):float(s) for c,s in zip(before[i],margins[i])};blocks=[]
            for name,order in [('CLIP原排序',before[i]),('BLIP重排',after[i])]:
                entries=[]
                for rank,c in enumerate(order[:5]):
                    index,within=divmod(int(c),5);related=index==i
                    entries.append(f'<li>Top{rank+1}，'+('标注正描述' if related else '其他图描述')+
                        f'，ITM logit差={lookup[int(c)]:.4f}：'+html.escape(rows[index]['captions'][within])+'</li>')
                blocks.append('<section><h4>'+name+'</h4><ol>'+''.join(entries)+'</ol></section>')
            examples.append(f'<article><h3>内部dev行号{i}</h3><img src="dev_{i}.jpg" alt="来源SHA核验的原图展示缩略副本"><div class="columns">'+''.join(blocks)+'</div></article>')
            teaching_rows.append({'dev_index':i,'source_image_id':rows[i]['image_id'],'source_image_sha256':rows[i]['image_sha256'],
                'baseline_top50_caption_ids':before[i],'rerank_top50_caption_ids':after[i],'itm_margin_original_pool':margins[i]})
    finally:
        for stream in handles.values():stream.close()
    processor=BlipProcessor.from_pretrained(cfg['model_root'],local_files_only=True)
    pixels=torch.load(root/'prepared/dev/pixels.pt',map_location='cpu',weights_only=True)[0]
    array=pixels.permute(1,2,0).numpy()*np.asarray(processor.image_processor.image_std)+np.asarray(processor.image_processor.image_mean)
    from PIL import Image
    Image.fromarray(np.round(np.clip(array,0,1)*255).astype(np.uint8)).save(folder/'blip_actual_input_display.png')
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        for split,analysis in [('dev',dev),('test',test)]:
            if analysis is None:continue
            fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
            for ax,direction in zip(axes,['T2I','I2T']):
                metric=direction+'_R@1'
                for i,condition in enumerate(['baseline','rerank']):
                    s=analysis['summaries'][condition][metric]
                    ax.bar(i,s['mean'],color=['#4477AA','#AA3377'][i],hatch=['///','..'][i],edgecolor='black')
                    ax.errorbar(i,s['mean'],yerr=s['sample_sd'],fmt='none',color='black',capsize=4)
                    ax.scatter(i+np.array([-.06,0,.06]),list(s['per_seed'].values()),color='black',s=18,zorder=4)
                    ax.text(i,s['mean']+2,f'{s["mean"]:.2f}',ha='center')
                bound=analysis['summaries']['baseline'][direction+'_R@50']['mean']
                ax.axhline(bound,color='#333333',linestyle='--',label=f'Top50 hit bound {bound:.2f}%')
                ax.set_xticks([0,1],['CLIP distill','+ BLIP ITM']);ax.set_ylim(0,103);ax.set_ylabel('Recall@1 (%)')
                ax.set_title(direction+', full '+split+'\n3 old CLIP checkpoints; sampleSD/rawpoints');ax.legend(fontsize=8,loc='lower right')
            fig.savefig(folder/(split+'_quality.png'),dpi=160,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        query=selected[0];caps=before[query,:5];new_positions={int(c):rank+1 for rank,c in enumerate(after[query])}
        axes[0].bar(np.arange(5),margins[query,:5],color='#4477AA',hatch='///',edgecolor='black');axes[0].axhline(0,color='black',linewidth=.8)
        axes[0].set_xticks(np.arange(5),[f'Old Top{i+1}' for i in range(5)]);axes[0].set_ylabel('Actual ITM logit1 - logit0')
        axes[0].set_title('Same first query,original top5 candidates')
        axes[1].scatter(np.arange(1,6),[new_positions[int(c)] for c in caps],color='#AA3377',marker='o')
        axes[1].set_xlabel('Original CLIP rank');axes[1].set_ylabel('Rank after ITM');axes[1].set_ylim(51,0)
        axes[1].set_title('Actual positions inside fixed Top50\nSmaller rank is better')
        fig.savefig(folder/'actual_margin_and_rank.png',dpi=160,facecolor='white');plt.close(fig)
    write_json(folder/'TEACHING_DATA.json',{'selected_dev_rows':teaching_rows,'selection_rule':'first4dev+first gain and first loss if present',
        'candidate_source_sha256':sha_file(root/'candidates/dev/seed42.npz'),
        'rerank_source_sha256':sha_file(root/'evaluation/rerank_seed42/dev/per_query.npz'),
        'actual_input_display':'FP32 prepared pixel0 denormalized with recorded means/std,clipped/rounded for PNG display;forwardBF16 tensor differs last bits'})
    def results(split,analysis):
        if analysis is None:return '<p>本轮未通过dev门槛，没有新增test结果。</p>'
        table='<table><tr><th>条件</th><th>文搜图R@1均值±SD</th><th>图搜文R@1均值±SD</th><th>Top50文搜图命中上界</th></tr>'
        for condition,s in analysis['summaries'].items():
            table+='<tr><td>'+condition+'</td>'+''.join(f'<td>{s[m]["mean"]:.3f} ± {s[m]["sample_sd"]:.3f}</td>' for m in ['T2I_R@1','I2T_R@1','T2I_R@50'])+'</tr>'
        return '<h3>'+split+'</h3>'+table+'</table><img src="'+split+'_quality.png" alt="完整图库质量、三已有CLIP模型样本SD和原始点以及不变的Top50命中上界"><pre>'+html.escape(json.dumps({
            'comparisons':analysis['comparisons'],'gate':analysis['gate'],'logical_pairs':analysis['logical_pairs'],
            'unique_pairs':analysis['unique_pairs'],'matching_roles':analysis['matching_roles'],'rerank_sort_seconds':analysis['rerank_sort_seconds']},ensure_ascii=False,indent=2))+'</pre>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM图文两阶段检索</title><style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:14px}td,th{border:1px solid #777;padding:8px}table{border-collapse:collapse}.columns{display:flex;gap:20px;flex-wrap:wrap}.columns section{flex:1;min-width:270px}article{border-top:1px solid #777;padding-top:20px;margin-top:20px}article>img{max-width:480px}</style>
<h1>双塔快速召回，联合图文匹配重排</h1><p>本轮固定三个已训练的CLIP蒸馏checkpoint与一份COCO匹配BLIP；没有新增训练。新增384图像/模型容量/BF16计算，实测两阶段配置质量与额外成本。旧模型的三个seed是评测轴，不是三个新BLIP训练。</p>
<h2>1. 用实际矩阵理解两阶段</h2><pre>CLIP image features [1000,512], caption features [5000,512]
all-gallery cosine [5000,1000] -> same Top50 as frozen contract
T2I:5000*50 + I2T:1000*50 = 300000 logical pairs per old seed
3old seeds:900000 logical pairs -> unique(image_id,caption_id) union
BLIP vision cache [1000,577,768] (384/16 patches + CLS)
text input [32,64] with [ENC]30523 -> cross-modal text states [32,64,768]
ITM head [32,2]; score = float32(logit_match - logit_nonmatch)
sort inside Top50,stable ties preserve CLIP order</pre>
<p>CLIP分别编码图像与文本，向量可预存，适合大图库召回。ITM让文本token与图像patch表征联合处理，更细致地判断图文匹配，代价是每个候选pair都要匹配计算。Top50没有正确结果时，重排无法救回；因此先检查命中上界。</p>
<h2>2. 真实模型图像输入</h2><img src="dev_0.jpg" alt="真实源图展示缩略副本"><img src="blip_actual_input_display.png" alt="本轮实际FP32像素记录反归一化的384展示图，forward使用BF16；不是attention"><p>左为源图缩略副本，右为prepared像素记录按mean/std反归一化并四舍五入的展示。CLIP仍沿用原224中心裁剪特征；BLIP有更多像素和另一模型，不能把改善全部归因注意力或称等信息预算。</p>
<h2>3. 匹配分数如何改变真实排序</h2><img src="actual_margin_and_rank.png" alt="第一个真实dev查询的旧Top5匹配logit差，以及它们重排后的真实位置"><p>匹配概率p=sigmoid(logit1−logit0)，本轮直接按logit差排序，避免概率饱和后的额外同分。图中的负值也保留，不隐藏失败候选。相关性按标注源图ID，其他图描述不自动等于语义错误。</p>
<h2>4. 代码与工程实践</h2><p>common.py/cached_logits保留固定BLIP的text_encoder与itm_head，以缓存图像latent作为encoder_hidden_states；HF原生同输入64pair另做数值检查，有限预检不证明所有输入等价。candidates.py先完整复现旧Top10；matcher.py按唯一pair分两GPU role、每2048pair保存双类logits/margin/sourceSHA；analysis.py核对全部key/候选池/R@50不变与配对统计。独立Generator没有新训练，已有checkpoint保持。</p>
<p>起始[ENC]来自官方BLIP ITM源码，区别于HF默认processor的CLS。COCO匹配权重revision/SHA固定，pretrain与Flickr重叠未知；不声称无泄漏或SOTA。GPU空闲三次采样与共享UUID锁，不抢占；缓存/批量成本是离线测量，不当线上p95/SLA。</p>
<h2>5. 完整结果、上限、质量和成本</h2><p>T2I至少+1pp且单侧97.5%下界>0；I2T下界>−.5pp；两主端点alpha=.05/2、6000图像簇bootstrap。三已有模型平均后的条件CI与三模型样本SD分开；不能把seed×image当独立。另有两方向nDCG点值保护和显存上限。失败停止、不改阈值、不新增test。</p>'''+results('dev',dev)+results('test',test)+'''
<h2>6. 原排序与重排的实际Top5</h2><p>固定前4dev行，另按行号选择第一条原错/重排对以及原对/重排错的图（若存在），用于学习而非发生率估计。源图与实际排序SHA保留，未伪造attention或人工语义标签。</p>'''+''.join(examples)+'''
<h2>学习检查</h2><ol><li>为什么CLIP向量适合召回，而ITM不适合直接匹配亿级图库？</li><li>指出本页一个候选的CLIP旧位置、ITM分数与新位置。</li><li>解释Top50命中上限及图搜文五正描述的Recall/nDCG区别。</li><li>区分900000逻辑pair、实际唯一pair和实际执行重复计时。</li><li>说清增加BLIP和384像素带来的质量/延迟取舍与结论边界。</li></ol>
<p><a href="TEACHING_DATA.json">源图/实际排序/分数</a>；<a href="../analysis/ANALYSIS.json">完整结果</a>；<a href="../DEV_PREFLIGHT_COMPLETE.json">原生与缓存64pair预检</a>。</p>
<p>模型来源：<a href="https://huggingface.co/Salesforce/blip-itm-base-coco">官方模型卡</a>、<a href="https://github.com/salesforce/BLIP">BLIP</a>。图表工作流参考Kassis等（2026）<a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>，本页为私有学习材料，无期刊合规声明。</p></html>'''
    (folder/'学习页.html').write_text(page)
    write_json(folder/'FIGURE_PROVENANCE.json',{'created_at':now(),'source_sha256':sha_file(Path(__file__)),
        'teaching_data_sha256':sha_file(folder/'TEACHING_DATA.json'),'dev_analysis_sha256':sha_file(root/'analysis/dev/ANALYSIS.json'),
        'test_analysis_sha256':sha_file(root/'analysis/test/ANALYSIS.json') if test else None,
        'selected_dev_rows':selected,'transformations':['source thumbnails','actualFP32pixels denormalized/clipped/rounded for display',
        'zero-baseline Recall/actualsampleSD/rawpoints+unchangedR50 bound','actualITM margins and categorical rank points','first4+first gains/losses if present'],
        'manual_visual_review':'pending','scope':'private learning,not attention or publisher certification'})
