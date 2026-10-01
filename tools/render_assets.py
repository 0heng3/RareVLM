"""Render repository overview and a source-backed CE-minus-ZS figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#0072B2'
ORANGE = '#D55E00'
INK = '#172B4D'
SETTINGS = ('imagenet_lt_b32','imagenet_lt_b16','imagenet_lt_b32_online','inat2018_b32')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, destination, stem):
    for extension in ('png','svg'):
        fig.savefig(destination/f'{stem}.{extension}',dpi=200,facecolor='white',transparent=False,
                    metadata={'Creator':'RareVLM source-backed figure renderer'})
    plt.close(fig)


def overview(destination):
    fig,ax=plt.subplots(figsize=(16,7))
    fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.97)
    ax.set(xlim=(0,16),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,label,fill='#F2F5F9',edge='#A6B4C5',size=13):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',
                                  facecolor=fill,edgecolor=edge,linewidth=1.4))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK,linespacing=1.5)
    def arrow(a,b,style='solid'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,linewidth=1.6,
                                    color='#52677F',linestyle=style))
    ax.text(.35,6.7,'RareVLM',fontsize=28,weight='bold',color=INK)
    ax.text(.35,6.25,'Frozen CLIP → residual adaptation → fixed-checkpoint decision interventions',fontsize=16,color=INK)
    ax.text(.45,5.08,'Image',fontsize=14,color=INK,va='center')
    box(1.55,4.55,2.15,1.1,'CLIP image encoder\nFrozen · eval/no_grad')
    box(4.2,4.55,1.55,1.1,'Normalized\nfeature v',size=12)
    box(6.3,4.45,2.15,1.3,'Residual adapter\n512 → 64 → 512\n66,240 parameters',fill='#E5F3FC',edge=BLUE,size=13)
    box(9.65,4.55,2.3,1.1,'Cosine classifier\nFixed CLIP scale s',size=13)
    arrow((1.02,5.1),(1.46,5.1));arrow((3.8,5.1),(4.1,5.1));arrow((5.83,5.1),(6.2,5.1))
    arrow((8.55,5.1),(9.55,5.1));ax.text(9.02,5.43,'L2 norm',ha='center',fontsize=11,color=INK)
    arrow((12.05,5.1),(12.55,5.1));ax.text(12.57,5.1,'z',fontsize=16,color=INK,va='center')
    ax.text(6.34,4.03,'Only trainable model module',fontsize=11,color=BLUE)
    ax.text(.45,3.2,'All class names\nFixed prompt',fontsize=13,color=INK,va='center')
    box(2.35,2.65,2.45,1.1,'CLIP text encoder\nFrozen',size=13)
    box(6.1,2.65,2.35,1.1,'Normalized text\nprototypes {t_c}\nFixed',size=12)
    arrow((1.88,3.2),(2.25,3.2));arrow((4.9,3.2),(6.0,3.2))
    ax.plot([8.55,9.12,9.12],[3.2,3.2,4.82],color='#52677F',linewidth=1.6)
    arrow((9.12,4.82),(9.55,4.82))
    box(.55,.9,6.7,1.03,'Training supervision: CE(z, y) or CE(z + log π_train, y)\nPaired runs: same initialization and views; independent adapter / AdamW',
        fill='#F6F8FB',size=12)
    box(8.05,.9,4.7,1.03,'Fix the selected CE checkpoint\nα=1 · validation-selected α · train-only P2P',
        fill='#FFF3E7',edge=ORANGE,size=12)
    ax.plot([12.6,12.6],[4.85,2.22],color='#52677F',linewidth=1.6)
    arrow((12.6,2.22),(10.4,2.01))
    ax.text(12.33,3.95,'raw logits',rotation=90,fontsize=10,color='#52677F')
    box(13.28,.95,2.35,4.72,'Evidence scope\n\nImageNet-LT\nB/32 · B/16\nB/32 online views\n\niNaturalist 2018\nB/32\n\nSeeds 0 / 42 / 200',
        fill='#F8FAFC',edge='#CFD9E4',size=12)
    ax.text(.55,.36,'Ground-truth labels are used by training losses and evaluation metrics, never by classifier inference.',
            fontsize=11,color=INK)
    save(fig,destination,'overview')


def main_results(results,destination):
    with (results/'seed_results.csv').open() as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(12,6.2),layout='constrained')
    payload={}
    for i,setting in enumerate(SETTINGS):
        baseline=next(r for r in rows if r['setting']==setting and r['method']=='zero_shot')
        ce=sorted([r for r in rows if r['setting']==setting and r['method']=='ce'],key=lambda r:int(r['seed']))
        if {int(r['seed']) for r in ce}!={0,42,200}:raise ValueError('missing or duplicate research seed')
        payload[setting]={}
        for group,offset,color,hatch,marker in [('many',-.19,BLUE,None,'o'),('few',.19,ORANGE,'///','^')]:
            values=[float(r[group])-float(baseline[group]) for r in ce]
            mean,sd=statistics.mean(values),statistics.stdev(values)
            x=i+offset
            ax.bar(x,mean,width=.34,color=color,hatch=hatch,edgecolor='#263646',linewidth=.7,
                   yerr=sd,capsize=4,error_kw={'elinewidth':1.3,'ecolor':INK},
                   label=f'{group.title()} (CE − ZS)' if i==0 else None,zorder=2)
            ax.scatter([x-.055,x,x+.055],values,c='black',s=19,marker=marker,zorder=4)
            ax.text(x,mean+(1.1 if mean>=0 else -1.1),f'{mean:+.2f}',ha='center',
                    va='bottom' if mean>=0 else 'top',fontsize=12,weight='bold',color=INK)
            payload[setting][group]={'seed_order':[int(r['seed']) for r in ce],'delta_pp':values,'mean_pp':mean,'seed_sd_pp':sd}
    ax.axhline(0,color=INK,linewidth=1.2,zorder=1)
    ax.set(ylim=(-28,28),ylabel='Accuracy change from zero-shot (percentage points)',
           xticks=range(4),xticklabels=['ImageNet-LT\nB/32 · fixed view','ImageNet-LT\nB/16 · fixed view',
                                     'ImageNet-LT\nB/32 · online views','iNaturalist 2018\nB/32 · scientific names'])
    ax.set_title('CE adaptation shifts accuracy toward frequent classes',loc='left',fontsize=18,weight='bold',pad=18)
    ax.legend(loc='upper left',frameon=False,ncol=2,fontsize=11)
    ax.grid(axis='y',color='#E6EBF1',linewidth=.6,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',labelsize=10,length=0);ax.tick_params(axis='y',labelsize=10)
    fig.supxlabel('Bars: mean of three training seeds · dots: individual seeds · whiskers: seed SD\n'
                  'Seeds share final images; these are not confidence intervals. Many >100, Few <20 actual train examples/class.',
                  fontsize=10,color=INK)
    save(fig,destination,'main_results')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=ROOT/'results')
    p.add_argument('--assets-dir',type=Path,default=ROOT/'assets')
    args=p.parse_args();args.assets_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none',
                         'figure.facecolor':'white','axes.facecolor':'white','axes.labelcolor':INK}):
        overview(args.assets_dir)
        values=main_results(args.results_dir,args.assets_dir)
    metadata={'audience':'GitHub repository readers and hiring reviewers','medium':'web PNG with editable SVG companion',
        'conclusion':'within-setting CE minus ZS improves Many and reduces Few in the four completed conditions',
        'metric':'percentage-point accuracy difference','estimator':'three-seed arithmetic mean',
        'uncertainty':'sample SD across training seeds, ddof=1; shared evaluation images',
        'missing_values':'none; all seeds required','new_training_runs':0,
        'source_csv_sha256':sha(args.results_dir/'seed_results.csv'),
        'transformations':['select completed CE and matching ZS within each setting','subtract ZS accuracy in pp','mean and SD across seeds'],
        'matplotlib_version':matplotlib.__version__,'palette':{'many':BLUE,'few':ORANGE},'plot_data':values,
        'files':{f.name:sha(f) for f in args.assets_dir.iterdir() if f.suffix in ('.png','.svg')}}
    (args.assets_dir/'figure_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('rendered overview and main_results in PNG/SVG from archived results')


if __name__=='__main__':main()
