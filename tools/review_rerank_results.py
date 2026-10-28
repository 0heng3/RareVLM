"""Independent row-wise metric procedure and presentation review artifacts."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import sha_file,now
from clip_retrieval_rerank.common import write_json


def verify(root):
    checks=[]
    for split in ['dev','test']:
        for seed in [0,42,200]:
            errors={}
            with np.load(root/f'evaluation/rerank_seed{seed}/{split}/per_query.npz') as z:
                for direction,n,positive_count in [('T2I',5000,1),('I2T',1000,5)]:
                    ranking=z[direction.lower()+'_top50'];values={k:[] for k in [1,5,10,50]};ndcg=[]
                    ideal=sum(1/math.log2(rank+2) for rank in range(positive_count))
                    for query,row in enumerate(ranking):
                        target=query//5 if direction=='T2I' else query
                        hits=[int(c)==target if direction=='T2I' else int(c)//5==target for c in row]
                        for k in values:values[k].append(float(any(hits[:k])))
                        ndcg.append(sum(float(hit)/math.log2(rank+2) for rank,hit in enumerate(hits[:10]))/ideal)
                    for k,v in values.items():assert np.array_equal(v,z[f'{direction}_R@{k}'])
                    error=float(np.max(np.abs(np.asarray(ndcg)-z[direction+'_nDCG@10'])))
                    assert error<1e-12;errors[direction]=error
            checks.append({'split':split,'seed':seed,'recall_1_5_10_50_exact':True,'ndcg_float64_max_error':errors})
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    cfg=json.loads((PROJECT_ROOT/'configs/clip_retrieval_rerank_v1.json').read_text());root=Path(cfg['output_root']);folder=root/'learning'
    verification=json.loads((root/'analysis/VERIFICATION.json').read_text())
    if args.verify:verification['independent_python_top50_rescore']=verify(root)
    if len(verification['independent_python_top50_rescore'])!=6:raise ValueError('Independent six-cell rescore absent')
    verification['independent_verification_procedure_source_sha256']=sha_file(Path(__file__))
    write_json(root/'analysis/VERIFICATION.json',verification)
    data=json.loads((folder/'TEACHING_DATA.json').read_text());rows=[v['dev_index'] for v in data['selected_dev_rows']]
    photos=[f'dev_{i}.jpg' for i in rows]+['blip_actual_input_display.png']
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
        for ax in axes.flat:ax.axis('off')
        for ax,name in zip(axes.flat,photos):
            with Image.open(folder/name) as image:ax.imshow(image)
            ax.set_title(name)
        fig.suptitle('Source thumbnails + recorded384 input display; not attention')
        fig.savefig(folder/'thumbnail_review.png',dpi=140,facecolor='white');plt.close(fig)
        first=data['selected_dev_rows'][0];margin=np.asarray(first['itm_margin_original_pool'])[:5]
        new={int(c):rank+1 for rank,c in enumerate(first['rerank_top50_caption_ids'])}
        positions=[new[int(c)] for c in first['baseline_top50_caption_ids'][:5]]
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        axes[0].bar(range(1,6),margin,color='#4477AA',hatch='///',edgecolor='black');axes[0].axhline(0,color='black',linewidth=.8)
        axes[0].set_xticks(range(1,6),[f'Old Top{i}' for i in range(1,6)])
        axes[0].set_ylabel('Actual ITM logit1 - logit0');axes[0].set_title('Same first query,original top5 candidates')
        axes[1].scatter(range(1,6),positions,color='#AA3377',marker='o');axes[1].set_xticks(range(1,6))
        axes[1].set_yticks([1,5,10,20,30,40,50]);axes[1].set_ylim(51,0)
        axes[1].set_xlabel('Original CLIP rank');axes[1].set_ylabel('Rank after ITM')
        axes[1].set_title('Actual positions inside fixed Top50\nSmaller rank is better')
        fig.savefig(folder/'actual_margin_and_rank.png',dpi=160,facecolor='white');plt.close(fig)
    p=folder/'FIGURE_PROVENANCE.json';provenance=json.loads(p.read_text())
    provenance.update(review_source_sha256=sha_file(Path(__file__)),review_export_at=now(),
        additional_presentation=['contactsheet from recordedsource thumbnails','integer original-rank axis; categorical points,no interpolation'],
        plot_sha256={name:sha_file(folder/name) for name in ['dev_quality.png','test_quality.png','actual_margin_and_rank.png','thumbnail_review.png']})
    write_json(p,provenance)
    print('Review images ready; existing independent six-cell check retained.')


if __name__=='__main__':main()
