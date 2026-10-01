"""Summarize one manifest's runs and optional calibration controls."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--controls',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();runs=[]
    for directory in [args.input_dir,args.controls]:
        if directory:
            for f in sorted(directory.glob('*.json')):
                row=json.loads(f.read_text())
                if 'test' in row and 'method' in row:runs.append(row)
    if not runs:raise ValueError('no completed result JSON files found')
    if len({r['manifest_sha256'] for r in runs})!=1:raise ValueError('refusing to mix different manifests')
    summary={}
    for method in sorted({r['method'] for r in runs}):
        selected=[r for r in runs if r['method']==method]
        metrics={}
        for key in ['oa','many','medium','few','macro_f1','nll','ece15']:
            values=[r['test']['groups'][key]['accuracy'] if key in ['many','medium','few'] else r['test'][key] for r in selected]
            metrics[key]={'mean':float(np.mean(values)),'seed_std':float(np.std(values,ddof=1)) if len(values)>1 else None}
        summary[method]={'n_runs':len(selected),'metrics':metrics}
        print(method,' '.join(f'{k}={100*metrics[k]["mean"]:.2f}%' for k in ['oa','many','medium','few']))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
