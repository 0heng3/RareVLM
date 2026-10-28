"""Presentation-only Top1 photo comparisons from frozen actual predictions."""
import html
import json
from pathlib import Path
import numpy as np
from rarevlm_paths import PROJECT_ROOT
from clip_retrieval.features import read_image
from clip_retrieval_rerank.common import write_json
from vlm_probe.evaluate import sha_file,now


def main():
    cfg=json.loads((PROJECT_ROOT/'configs/clip_retrieval_adaptive_v1.json').read_text());root=Path(cfg['output_root']);d=root/'learning'
    page=d/'学习页.html';text=page.read_text();before=sha_file(page)
    if '<h2 id="top1-photo-comparison">' in text:print('Already enriched');return
    provenance=json.loads((d/'PROVENANCE.json').read_text());rows=json.loads((root/'DATA_MANIFEST.json').read_text())['splits']['dev']
    index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets'];old=Path(cfg['rerank_root'])
    queries=sorted(int(Path(name).stem.split('_')[1]) for name in provenance['photos'] if name.startswith('T2I_'))
    ranked={}
    with np.load(old/'candidates/dev/seed42.npz') as z:ranked['CLIP']=z['t2i'].copy()
    for variant in ['full','adaptive']:
        with np.load(root/'evaluation'/f'{variant}_seed42'/'dev/per_query.npz') as z:ranked[variant]=z['t2i_top50'].copy()
    articles=[];records=[]
    for q in queries:
        views=[]
        for variant,ranks in ranked.items():
            image=int(ranks[q,0]);name=f'top1_{q}_{variant}.jpg';handles={}
            try:
                photo=read_image(rows[image],index,handles);photo.thumbnail((380,380));photo.save(d/name,quality=92)
            finally:
                for f in handles.values():f.close()
            record={'query':q,'condition':variant,'gallery_image_index':image,'annotated_pair':image==q//5,
                'image_id':rows[image]['image_id'],'source_image_sha256':rows[image]['image_sha256'],'thumbnail_sha256':sha_file(d/name),
                'scope':'actual savedTop1 source-photo thumbnail;not384processor input or attention'}
            records.append(record);provenance['photos'][name]=record
            views.append('<section><h4>'+variant+f' Top1：图库{image}</h4><img src="'+name+'" alt="保存的实际Top1候选源图缩略展示"><p>'+('标注配对图片' if image==q//5 else '另一张图库图片；不能自动等同语义完全错误')+'</p></section>')
        caption=rows[q//5]['captions'][q%5]
        articles.append(f'<article><h3>文搜图query{q}，seed42</h3><p>输入只有caption：'+html.escape(caption)+'</p><div class="columns">'+''.join(views)+'</div></article>')
    block='<h2 id="top1-photo-comparison">图片对照：保存的实际Top1</h2><p>文搜图输入是文字，下面为CLIP/full/adaptive实际返回的图库图片。使用已固定类别首例，预测后展示；不参与τ选择。照片为展示缩略副本，不冒充模型输入或attention。</p>'+''.join(articles)
    marker='<h2>学习检查</h2>'
    if marker not in text:raise ValueError('Learning insertion point missing')
    page.write_text(text.replace(marker,block+marker))
    write_json(d/'TOP1_ENRICHMENT.json',{'status':'completed','created_at':now(),'renderer_source_sha256':sha_file(Path(__file__)),
        'html_before_sha256':before,'html_after_sha256':sha_file(page),'image_records':records,
        'prediction_sha256':{v:sha_file(root/'evaluation'/f'{v}_seed42'/'dev/per_query.npz') for v in ['full','adaptive']},
        'new_gpu_predictions':0,'new_training_runs':0,'scope':'presentation-only,scientific sources/policy/results unchanged'})
    write_json(d/'PROVENANCE.json',provenance);print('LEARNING_ENRICHED',len(records),'actualTop1 photos')


if __name__=='__main__':main()
