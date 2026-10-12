"""Draw teaching diagrams from real adapter tensors and the encoded dev case.

These are CPU-only learning assets. No model inference or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np

FONT = Path('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf')
ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_learning_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box(ax, x, y, w, h, title, sub, color, fontsize=19):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                 facecolor=color, edgecolor='#566375', linewidth=1.4))
    ax.text(x+w/2, y+h*.65, title, ha='center', va='center', fontsize=fontsize)
    ax.text(x+w/2, y+h*.25, sub, ha='center', va='center', fontsize=15, color='#46515f')


def arrow(ax, start, end, color='#46515f'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                 mutation_scale=17, color=color, linewidth=2))


def export(fig, root, stem):
    directory=root/'figures';directory.mkdir(parents=True,exist_ok=True)
    files=[]
    for suffix in ['png','svg']:
        p=directory/f'{stem}.{suffix}'
        fig.savefig(p, dpi=100, facecolor='white', bbox_inches=None)
        files.append({'path':str(p.relative_to(root)), 'sha256':digest(p)})
    plt.close(fig)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root
    font_manager.fontManager.addfont(str(FONT))
    family=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({'font.family':family,'font.size':16,'axes.unicode_minus':False,
                         'svg.fonttype':'path','pdf.fonttype':42})
    tensor_path=root/'code/LORA_TENSORS.json';mask_path=root/'code/TOKEN_MASK.json'
    tensors=json.loads(tensor_path.read_text());mask=json.loads(mask_path.read_text())
    q=tensors['projections']['q_proj'];v=tensors['projections']['v_proj']
    assert q['A_shape']==[8,2048] and q['B_shape']==[2048,8]
    assert v['A_shape']==[8,2048] and v['B_shape']==[256,8]
    assert tensors['trainable_parameters_from_safetensors']==1843200
    assert mask['input_length']==277 and mask['prefix_length']==273
    assert mask['visual_token_count']==234 and mask['supervised_token_count']==4
    products=[]

    fig,ax=plt.subplots(figsize=(16,8.4));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.08)
    fig.suptitle('LoRA 实际更新了什么？看第 0 层 q_proj',x=.05,ha='left',fontsize=29)
    ax.text(.015,.95,'真实 q/v rank8、seed0 最终 checkpoint；图中的形状与参数来自保存的 A/B 张量',fontsize=17)
    box(ax,.03,.47,.15,.15,'输入 x','当前 token：2048 维','#e9f0f7',18)
    box(ax,.28,.70,.29,.19,'冻结的 W0：2048 × 2048','NF4 存储；计算为 FP32','#edf0f3')
    box(ax,.27,.23,.20,.20,'A：8 × 2048','2048 维 → 8 维','#fff0d9')
    box(ax,.57,.23,.20,.20,'B：2048 × 8','8 维 → 2048 维','#fff0d9')
    ax.add_patch(Circle((.86,.57),.043,facecolor='white',edgecolor='#566375',linewidth=1.5))
    ax.text(.86,.57,'+',fontsize=29,ha='center',va='center')
    arrow(ax,(.18,.55),(.24,.55));arrow(ax,(.24,.55),(.29,.72))
    arrow(ax,(.24,.55),(.27,.34),color='#a7630a')
    arrow(ax,(.47,.33),(.57,.33),color='#a7630a')
    arrow(ax,(.77,.33),(.84,.53),color='#a7630a')
    arrow(ax,(.57,.79),(.84,.61));arrow(ax,(.905,.57),(.98,.57))
    ax.text(.67,.49,'× (α/r) = 2',fontsize=17,color='#a7630a')
    ax.text(.51,.60,'h = W0x + 2BAx',fontsize=24,ha='center')
    ax.text(.035,.115,'q_proj：8 × (2048 + 2048) = 32,768 参数 / 层',fontsize=19)
    ax.text(.035,.055,'v_proj 输出只有 256 维：8 × (2048 + 256) = 18,432 参数 / 层',fontsize=19)
    ax.text(.035,-.005,'36 × (32,768 + 18,432) = 1,843,200 可训练参数',fontsize=22,color='#8e5405')
    fig.text(.05,.027,'冻结的是 W0 参数；计算图仍可传递梯度。橙色框表示可训练 A/B，不是注意力热图。',fontsize=14,color='#46515f')
    products.append({'name':'mechanism_lora','caption':'同一真实checkpoint的q/v形状与参数计数。输入为语言层单个token的隐藏向量。',
                     'alt':'Frozen q projection plus a rank-eight trainable A-B branch, with real q and v matrix dimensions.',
                     'files':export(fig,root,'mechanism_lora')})

    delta=np.asarray(tensors['delta_crop']['values'],dtype=float)
    singular=np.asarray(tensors['singular_values'],dtype=float)
    assert delta.shape==(64,64) and singular.shape==(8,) and np.isfinite(delta).all()
    fig,axes=plt.subplots(1,2,figsize=(16,7.7),layout='constrained')
    fig.suptitle('真实训练后的低秩更新：权重局部与 8 个奇异值',fontsize=26)
    scale=float(np.abs(delta).max())
    im=axes[0].imshow(delta,cmap='RdBu_r',vmin=-scale,vmax=scale,origin='lower',interpolation='nearest')
    axes[0].set(title='ΔW = 2BA 的前 64 × 64 个权重',xlabel='输入维度索引',ylabel='输出维度索引')
    fig.colorbar(im,ax=axes[0],label='权重增量（无量纲）',shrink=.83)
    axes[1].plot(np.arange(1,9),singular,marker='o',color='#0072b2',linewidth=2.5)
    axes[1].set(title='完整 q_proj 更新的非零奇异值',xlabel='奇异值序号',ylabel='奇异值',xticks=np.arange(1,9),ylim=(0,float(singular.max())*1.16))
    axes[1].grid(alpha=.25)
    axes[1].text(.04,.04,'从小型 QR/SVD 精确计算\n未对 2048 × 2048 大矩阵做完整 SVD',transform=axes[1].transAxes,fontsize=14,color='#46515f')
    fig.supxlabel('这是一个投影层的真实参数更新；局部热图不是图像注意力，奇异值也不代表任务贡献。',fontsize=14)
    products.append({'name':'lora_update','caption':'真实q_proj更新64×64局部；完整低秩更新的八个奇异值，未将其解释为视觉注意力或任务归因。',
                     'alt':'A 64-by-64 crop of a real LoRA weight update and the eight singular values of the full update.',
                     'files':export(fig,root,'lora_update')})

    n=mask['input_length'];prefix=mask['prefix_length'];tokens=mask['tokens']
    supervised=[t for t in tokens if t['supervised']]
    visual=[t['position'] for t in tokens if t['is_visual_token']]
    assert len(supervised)==4 and len(visual)==234
    fig,ax=plt.subplots(figsize=(16,9.2));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.subplots_adjust(left=.045,right=.965,top=.88,bottom=.08)
    fig.suptitle('SFT 到底惩罚哪些 token？沿用“ALBERTVILLE 距离”案例',fontsize=26,x=.045,ha='left')
    ax.text(.015,.94,'问题：how many miles to albertville?    确定性训练目标：40',fontsize=19)
    start=.03;width=.92;y=.69;h=.13
    ax.add_patch(Rectangle((start,y),width*prefix/n,h,facecolor='#d9e7f4',edgecolor='#60748a'))
    ax.add_patch(Rectangle((start+width*prefix/n,y),width*(n-prefix)/n,h,facecolor='#a8d1bd',edgecolor='#34634c'))
    vs=start+width*min(visual)/n;vw=width*len(visual)/n
    ax.add_patch(Rectangle((vs,y),vw,h,facecolor='#bbd4e9',edgecolor='#60748a',hatch='//'))
    ax.text(vs+vw/2,y+h/2,'234 个视觉 token\nlabels = −100',ha='center',va='center',fontsize=19)
    ax.text(start,.64,'位置 0…272：273 个前缀 token，不计 loss',fontsize=18)
    ax.text(.61,.58,'位置 273…276：4 个监督 token',fontsize=18,color='#20523b')
    arrow(ax,(start+width*.99,y-.01),(.82,.49),color='#34634c')
    for i,t in enumerate(supervised):
        x=.08+i*.215
        shown=t['decoded_token'].replace('\n','\\n')
        box(ax,x,.29,.185,.20,f"位置 {t['position']}",f"{shown}\nlabel = {t['label']}",'#e3f0e8',18)
    ax.text(.015,.20,'−100 表示忽略这个位置的损失；图像和问题仍然保留在模型输入中。',fontsize=19)
    ax.text(.015,.115,'训练时采用因果移位：前一个位置的 logits，预测下一个位置的标签。',fontsize=18)
    ax.text(.015,.055,'每题先对自己的 4 个有效 token 平均，再对问题平均；长回答不会仅因长度更大而得到更高权重。',fontsize=17)
    fig.text(.05,.025,'这里是训练编码；生成评测只输入图像和问题，不提供人类答案。图中每个位置与真实 TOKEN_MASK.json 对应。',fontsize=14,color='#46515f')
    products.append({'name':'sft_mask','caption':'真实277-token训练序列：273前缀忽略，234视觉token为条件，4回答/结束token监督。',
                     'alt':'A real 277-token training sequence, showing 273 ignored prefix tokens and four supervised answer/end tokens.',
                     'files':export(fig,root,'sft_mask')})
    record={'schema_version':'rarevlm.learning_mechanism_figures.v1','source_sha256':{
        str(tensor_path):digest(tensor_path),str(mask_path):digest(mask_path),str(Path(__file__).resolve()):digest(__file__),
        str(FONT):digest(FONT)},'products':products,
        'data_integrity':'Real checkpoint values and encoded dev tokens only; layout is a teaching diagram, not a new model result.',
        'font_source':'https://github.com/notofonts/noto-cjk','publisher_requirement':'Not a journal submission; teaching artifacts.'}
    (root/'figures/MECHANISMS_PROVENANCE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('Created',len(products),'mechanism/token figures from real data')


if __name__=='__main__':main()
