"""Dev margins are bound;gated test may use the frozen full-score builder."""
import argparse
from pathlib import Path
from clip_retrieval_adaptive.confidence import build

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['test'],required=True)
    a=p.parse_args();build(a.config,a.split)
