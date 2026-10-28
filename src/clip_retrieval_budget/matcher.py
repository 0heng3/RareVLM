"""Reuse frozen actual key-driven full/selected scorer;no score lookup shortcut."""
import argparse
from pathlib import Path
from clip_retrieval_adaptive.matcher import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--role',type=int,required=True)
    a=p.parse_args();run(a.config,a.split,a.role)
