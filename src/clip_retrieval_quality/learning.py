"""Real source photographs, actual selection and saved ranking changes."""
import html
import json
from pathlib import Path
import numpy as np
from clip_retrieval.features import read_image
from vlm_probe.evaluate import now,write_json


def build(config_path,dev,test):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);out=root/'learning';out.mkdir(exist_ok=True)
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    handles={};pictures=[];cases=[]
    def picture(row,name):
        path=out/name
        im=read_image(row,index,handles);im.thumbnail((560,380));im.save(path,quality=88)
        pictures.append({'path':str(path),'source_image_id':row['image_id'],'source_image_sha256':row['image_sha256'],
            'purpose':'source photograph thumbnail for teaching;not reconstructed model input or attention'})
        return name
    with np.load(root/'selection/indices.npz') as raw:sel={k:raw[k] for k in raw.files}
    rows=manifest['splits']['train'];captions=[c for row in rows for c in row['captions']]
    fixed=sorted(range(len(rows)),key=lambda i:rows[i]['image_sha256'])[:2]
    flags=sel['noisy_score4']//5!=np.arange(len(rows))[:,None]
    failed=np.flatnonzero(flags.any(1));chosen=fixed+([int(failed[0])] if len(failed) else [])
    cards=[]
    for i in chosen:
        name=picture(rows[i],f'train_{i}.jpg');actual=sel['noisy_actual'][i];score=sel['noisy_scores'][i]
        detail=[]
        for j,idx in enumerate(actual):
            moved=int(idx)//5!=i
            detail.append({'slot':j,'caption':captions[int(idx)],'cosine':float(score[j]),'reassigned_provenance':moved,
                'score_selected':j in sel['noisy_score4_slots'][i],'random_selected':j in sel['noisy_random4_slots'][i]})
        table=''.join('<tr><td>'+str(d['slot'])+'</td><td>'+html.escape(d['caption'])+'</td><td>'+f'{d["cosine"]:.4f}'+'</td><td>'+('来源置换' if d['reassigned_provenance'] else '原来源')+'</td><td>'+('保留' if d['score_selected'] else '删除')+'</td><td>'+('保留' if d['random_selected'] else '删除')+'</td></tr>' for d in detail)
        donor=int(actual[np.flatnonzero(actual//5!=i)[0]])//5
        donor_name=picture(rows[donor],f'train_{i}_donor.jpg')
        cards.append(f'<article><h3>训练图 {i}：先筛选，后做置换诊断</h3><img src="{name}"><img src="{donor_name}"><p>左：接收图原图缩略图；右：被置换描述的来源图，仅用于事后诊断。</p><table><tr><th>槽</th><th>实际描述</th><th>cosine</th><th>来源诊断</th><th>分数筛选</th><th>随机筛选</th></tr>{table}</table></article>')
        cases.append({'train_index':i,'actual_caption_rows':detail,'donor_index_diagnostic_only':donor})
    diagnostics=json.loads((root/'selection/DIAGNOSTICS.json').read_text())
    metrics=[]
    for split,result in [('dev',dev),('test',test)]:
        if result is None:continue
        for condition,s in result['summaries'].items():
            metrics.append(f'<tr><td>{split}</td><td>{condition}</td><td>{s["T2I_R@1"]["mean"]:.3f} ± {s["T2I_R@1"]["sample_sd"]:.3f}</td><td>{s["I2T_R@1"]["mean"]:.3f} ± {s["I2T_R@1"]["sample_sd"]:.3f}</td></tr>')
    ranking=[]
    for branch in ['clean','noisy']:
        paths=[root/'evaluation'/f'{branch}_{m}4_seed42/dev/per_query.npz' for m in ['random','score']]
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            delta=b['T2I_R@1']-a['T2I_R@1']
            for kind,condition in [('改善',delta>0),('退化',delta<0)]:
                ids=np.flatnonzero(condition)
                if not len(ids):continue
                q=int(ids[0]);truth=q//5;left=int(a['t2i_top10'][q,0]);right=int(b['t2i_top10'][q,0])
                row=manifest['splits']['dev'][truth];query=row['captions'][q%5]
                p0=picture(row,f'{branch}_{kind}_target.jpg');p1=picture(manifest['splits']['dev'][left],f'{branch}_{kind}_random.jpg')
                p2=picture(manifest['splits']['dev'][right],f'{branch}_{kind}_score.jpg')
                ranking.append(f'<article><h3>{branch} seed42：标注Recall@1{kind}</h3><p>{html.escape(query)}</p><img src="{p0}"><img src="{p1}"><img src="{p2}"><p>依次为标注配对图、随机筛选Top1、一致性筛选Top1。配对错误不自动等于语义错误。</p></article>')
    for handle in handles.values():handle.close()
    intro='<h1>图文数据质量：代码—原理—真实结果</h1><p>CLIP把图像和文字编码成归一化512维向量，点积就是cosine。每图5条描述按分数取最高4条，与固定随机取4条比较；评分只读取train。分数代表教师一致性，可能删掉真实细节，不是准确性标签。</p><p>训练采用32图锚点＋64额外图，4描述/图：I2T[32,384]，T2I[128,96]。L=平均多正CE＋KL(P_teacher||P_student)，τ=.07；teacher停止梯度，两个筛选策略loss完全相同。</p><p>原始数据和20%来源置换压力测试分开。已知置换标签只在此处诊断，不输入选择或loss。压力测试提分不能当原始数据提分；五caption历史强基线监督量更多，必须同时看差距。</p>'
    body=intro+'<h2>完整三seed结果</h2><table><tr><th>划分</th><th>条件</th><th>文搜图R1</th><th>图搜文R1</th></tr>'+''.join(metrics)+'</table><pre>'+html.escape(json.dumps({'dev_comparisons':dev['comparisons'],'test_comparisons':test['comparisons'] if test else None,'selection_diagnostics':diagnostics},ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)+''.join(ranking)
    tail='<h2>源码与面试追问</h2><p>src/clip_retrieval_quality/selection.py:select_slots → prepare.py:prepare → loss.py:selected_terms → train.py:train → analysis.py:analyze。</p><ol><li>为什么删除最低分不等于删除错误？教师偏差和细粒度细节分别举例。</li><li>为什么需要random4而不能只对full5？监督量和logits宽度不同。</li><li>教师KL为何不使用未选描述？这会使两策略信息预算不同。</li><li>为什么高错配清除率不保证干净集提高？同一教师相关性、训练信息损失、分布改变。</li><li>按图像簇而非5000caption独立bootstrap，是因为同图问题相关；seed SD与固定模型CI有何不同？</li></ol><p>图片为真实原图缩略图，实际冻结CLIP编码来自原224预处理。不是attention。案例由脚本选取，语义审查与本人掌握待确认。</p>'
    (out/'学习页.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>RareVLM 图文数据质量</title><style>body{max-width:1200px;margin:24px auto;font:16px/1.65 sans-serif;padding:18px;color:#18222e}img{max-width:30%;max-height:300px;object-fit:contain;margin:6px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}article{margin:24px 0;padding:16px;background:#f4f7fb}pre{white-space:pre-wrap;font-size:12px;max-height:400px;overflow:auto}</style>'+body+tail+'</html>')
    write_json(out/'PROVENANCE.json',{'created_at':now(),'pictures':pictures,'train_cases':cases,
        'semantic_review':'pending;not human labels or training injection','personal_mastery':'pending'})
