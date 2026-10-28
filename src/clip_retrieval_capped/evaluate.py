"""Use the unchanged full-gallery evaluator and this study's own gate."""
import argparse
from pathlib import Path
from clip_retrieval_set.evaluate import evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    args=parser.parse_args();evaluate(args.config,args.condition,args.seed,args.split)
