"""Idle-GPU three-task strong-baseline confirmation without modifying archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from research_runtime.queue import run_queue
from vlm_probe.evaluate import sha_file,now,write_json
from .evaluate import DEFAULT_CONFIG,source_hashes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'PID.json',{'pid':__import__('os').getpid(),'started_at':now(),'python':sys.executable})
    jobs=[{'key':task,'command':[sys.executable,'-m','vlm_strong_confirm.evaluate','--task',task,'--config',str(args.config)],
        'completion_marker':str(root/'workers'/task/'STATUS.json')} for task in cfg['tasks']]
    identity={'config_sha256':sha_file(args.config),'scientific_sources':source_hashes(),
        'attention_library_sha256':sha_file(Path(cfg['attention_library_path']))}
    run_queue(root,jobs,identity)
    from .analysis import analyze
    result=analyze(args.config);print('FORMAL_STRONG_COMPLETE',result['gates'],flush=True)
    write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json')})


if __name__=='__main__':main()
