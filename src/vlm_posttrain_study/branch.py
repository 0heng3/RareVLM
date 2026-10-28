"""One admitted GPU branch; checkpoints determine resume, not a new config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from vlm_probe.evaluate import now,write_json

def train_arguments(config,root,condition,seed):
    args=[sys.executable,'-m','vlm_posttrain.train','--config',str(config),'--output-root',str(root),
          '--condition',condition,'--seed',str(seed)]
    if (root/'runs'/condition/f'seed{seed}'/'train/LATEST.json').exists():args.append('--resume')
    return args

def run(args):
    cfg=json.loads(args.config.read_text());root=args.output_root;seed=args.seed;condition=args.condition
    directory=root/'branches'/f'{condition}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    def call(phase,command):
        write_json(directory/'PROGRESS.json',{'status':phase,'updated_at':now(),'condition':condition,'seed':seed})
        print('BRANCH_STAGE',condition,phase,flush=True);subprocess.run(command,check=True)
    if condition!='initial':
        if not (root/'runs'/condition/f'seed{seed}'/'train/TRAIN_RESULT.json').exists():
            call('training',train_arguments(args.config,root,condition,seed))
        adapter=root/'runs'/condition/f'seed{seed}'/'train/adapter'
    else:adapter=Path(cfg['initial_adapter'])
    common=['--config',str(args.config),'--condition',condition,'--seed',str(seed),'--adapter-dir',str(adapter),'--output-root',str(root)]
    call('text_dev',[sys.executable,'-m','vlm_posttrain_eval.evaluate',*common])
    call('pope',[sys.executable,'-m','vlm_posttrain_pope.evaluate',*common,'--data-root',cfg['pope_data_root']])
    call('preference_heldout',[sys.executable,'-m','vlm_posttrain_study.heldout',*common])
    write_json(directory/'COMPLETE.json',{'status':'completed','condition':condition,'seed':seed,'completed_at':now()})
    write_json(directory/'PROGRESS.json',{'status':'completed','condition':condition,'seed':seed,'updated_at':now()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--condition',choices=['initial','sft','dpo'],required=True);p.add_argument('--seed',type=int,default=42);run(p.parse_args())
