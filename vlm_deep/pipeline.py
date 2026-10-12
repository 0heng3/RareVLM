"""Persistent phase execution for the full A matrix and two-stage B experiments."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from .common import content_sha,read_manifest,sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--public-output',type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_deep_v2.json')
    args=p.parse_args();c=json.loads(args.config.read_text());repo=Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True,exist_ok=True)
    sources={str(f.relative_to(repo)):sha_file(f) for f in (repo/'vlm_deep').rglob('*.py')}
    source_lock={'config_sha256':content_sha(c),'source_sha256':sources,
        'pilot_train_result_sha256':sha_file(args.pilot_root/'runs/train/TRAIN_RESULT.json')}
    source_file=args.run_root/'SOURCE_LOCK.json'
    if source_file.exists() and json.loads(source_file.read_text())!=source_lock:raise ValueError('frozen execution source/config changed')
    write_json(source_file,source_lock)
    phase='initialization'
    def execute(name,command,completed=None):
        nonlocal phase
        phase=name
        if completed is not None and completed.exists():
            print('reuse completed',name,flush=True);return
        write_json(args.run_root/'STATUS.json',{'phase':'complete' if name=='final_report' else name,
            'status':'complete' if name=='final_report' else 'running','config_sha256':content_sha(c)})
        print('START',name,flush=True)
        with (args.run_root/(name.replace('/','__')+'.log')).open('a') as log:
            r=subprocess.run(command,cwd=repo,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{name} failed with exit {r.returncode}')
        print('DONE',name,flush=True)
    shared=['--asset-root',str(args.asset_root),'--model-root',str(args.pilot_root/'model'),'--config',str(args.config)]
    def distributed(module,extra):
        return [sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={c["world_size"]}',
                '-m',module,*shared,*extra]
    def evaluate(name,directory,task,adapter=None,split='final',quant='nf4_language_only',batch=4):
        extra=['--output-dir',str(directory),'--task',task,'--split',split,'--quantization',quant,'--batch-size',str(batch)]
        if adapter:extra+=['--adapter-dir',str(adapter)]
        execute(name,distributed('vlm_deep.evaluate',extra),directory/'EVALUATION.json')
    try:
        for method in c['methods']:
            pf=json.loads((args.asset_root/'preflight'/f'{method}.json').read_text())
            if not pf['passed'] or pf['config_sha256']!=content_sha(c):raise ValueError('missing/mismatched method preflight')
        numerical=json.loads((args.asset_root/'preflight/original_fp32.json').read_text())
        if not numerical['passed'] or numerical['config_sha256']!=content_sha(c):raise ValueError('original FP32 path not preflighted')
        def wait_ready(path,label):
            while not path.exists():
                write_json(args.run_root/'STATUS.json',{'phase':'waiting_'+label,'status':'running'})
                print('WAITING',label,flush=True);time.sleep(30)
        wait_ready(args.asset_root/'prepared/manifest.json','full_textvqa')
        m=read_manifest(args.asset_root/'prepared/manifest.json')
        if len(m['splits']['final'])!=5000 or len(m['splits']['train_10k'])!=10000:raise ValueError('not the frozen full-data protocol')
        pilot_adapter=args.pilot_root/'runs/train/adapter'
        for quant in ['original','nf4_language_only']:
            for label,adapter in [('base',None),('pilot',pilot_adapter)]:
                name=f'numerical/{quant}_{label}'
                evaluate(name,args.run_root/name,'textvqa',adapter,quant=quant,batch=1)
        for task in ['textvqa','docvqa','mmstar']:
            if task!='textvqa':wait_ready(args.asset_root/'transfer'/task/'manifest.json',task)
            evaluate('baselines/'+task,args.run_root/'baselines'/task,task)
        def run_stage(stage,methods,split):
            for method in methods:
                for seed in c['training_seeds']:
                    directory=args.run_root/stage/method/f'seed{seed}'
                    train_dir=directory/'train'
                    command=distributed('vlm_deep.train',['--output-dir',str(train_dir),'--method',method,'--seed',str(seed),'--split',split])
                    execute(f'{stage}/{method}/seed{seed}/train',command,train_dir/'TRAIN_RESULT.json')
                    for task in ['dev','textvqa','docvqa','mmstar']:
                        evaluate(f'{stage}/{method}/seed{seed}/eval_{task}',directory/f'eval_{task}',
                            'textvqa' if task=='dev' else task,train_dir/'adapter',split='dev' if task=='dev' else 'final')
                    report=[sys.executable,'-m','vlm_deep.report','--asset-root',str(args.asset_root),'--run-root',str(args.run_root)]
                    execute(f'report_{stage}_{method}_{seed}',report)
        run_stage('stage1',list(c['methods']),'train_10k')
        candidates=[]
        for method in c['methods']:
            scores=[];nll=[];hashes={}
            for seed in c['training_seeds']:
                directory=args.run_root/'stage1'/method/f'seed{seed}'
                e=directory/'eval_dev/EVALUATION.json';t=directory/'train/TRAIN_RESULT.json'
                scores.append(json.loads(e.read_text())['score_percent'])
                nll.append(json.loads(t.read_text())['internal_dev_token_nll'])
                hashes[str(seed)]={'dev':sha_file(e),'train':sha_file(t)}
            candidates.append({'method':method,'mean_dev_score':statistics.mean(scores),'mean_dev_nll':statistics.mean(nll),'input_hashes':hashes})
        candidates.sort(key=lambda r:(-r['mean_dev_score'],r['mean_dev_nll'],r['method']))
        selection={'selected_methods':[r['method'] for r in candidates[:2]],'dev_ranking':candidates,
            'selection_uses_final_metrics':False,'rule':c['stage2_method_selection']}
        write_json(args.run_root/'STAGE2_SELECTION.json',selection)
        run_stage('stage2',selection['selected_methods'],'train_all')
        execute('verification',[sys.executable,'-m','vlm_deep.verify','--asset-root',str(args.asset_root),
            '--run-root',str(args.run_root),'--config',str(args.config)])
        write_json(args.run_root/'STATUS.json',{'phase':'complete','status':'complete'})
        report=[sys.executable,'-m','vlm_deep.report','--asset-root',str(args.asset_root),'--run-root',str(args.run_root)]
        if args.public_output:report+=['--public-output',str(args.public_output)]
        execute('final_report',report)
        write_json(args.run_root/'STATUS.json',{'phase':'complete','status':'complete'})
        print('ALL DEEP EXPERIMENTS COMPLETE',flush=True)
    except Exception as error:
        write_json(args.run_root/'STATUS.json',{'phase':phase,'status':'failed','error':str(error)})
        raise


if __name__=='__main__':main()
