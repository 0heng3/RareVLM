"""Execute the fixed generative-VLM pilot; preserve phase logs and failures."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from .common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_sft_textvqa.json')
    p.add_argument('--public-output',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text());root=PROJECT_ROOT
    args.run_root.mkdir(parents=True,exist_ok=True)
    common=['--asset-root',str(args.asset_root),'--config',str(args.config)]
    stages=[]
    if not (args.asset_root/'prepared/manifest.json').exists():stages.append(('prepare',[sys.executable,'-u','-m','vlm_sft.prepare',*common]))
    if not (args.asset_root/'PREFLIGHT.json').exists():stages.append(('preflight',[sys.executable,'-u','-m','vlm_sft.preflight',*common]))
    def distributed(module,extra):return [sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={config["expected_world_size"]}', '-m',module,*common,*extra]
    if not (args.run_root/'base/EVALUATION.json').exists():stages.append(('base',distributed('vlm_sft.evaluate',['--output-dir',str(args.run_root/'base')])))
    if not (args.run_root/'train/TRAIN_RESULT.json').exists():stages.append(('train',distributed('vlm_sft.train',['--output-dir',str(args.run_root/'train')])))
    if not (args.run_root/'adapted/EVALUATION.json').exists():stages.append(('adapted',distributed('vlm_sft.evaluate',['--output-dir',str(args.run_root/'adapted'),'--adapter-dir',str(args.run_root/'train/adapter')])))
    report=[sys.executable,'-u','-m','vlm_sft.report','--asset-root',str(args.asset_root),'--run-root',str(args.run_root)]
    if args.public_output:report+=['--public-output',str(args.public_output)]
    stages.append(('verify',[sys.executable,'-u','-m','vlm_sft.verify',*common,'--run-root',str(args.run_root)]))
    stages.append(('report',report))
    try:
        for phase,command in stages:
            write_json(args.run_root/'STATUS.json',{'phase':phase,'status':'running'})
            print('stage',phase,flush=True)
            with (args.run_root/f'{phase}.log').open('a') as log:
                result=subprocess.run(command,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{phase} failed with exit {result.returncode}; see {phase}.log')
        write_json(args.run_root/'STATUS.json',{'phase':'complete','status':'complete'})
        print('all fixed pilot stages complete',flush=True)
    except Exception as error:
        write_json(args.run_root/'STATUS.json',{'phase':phase,'status':'failed','error':str(error)})
        raise


if __name__=='__main__':main()
