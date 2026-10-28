#!/usr/bin/env python
"""CPU contracts, common initialization preparation, or full visual merger KL study."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vlm_visual_merger_kl.study import main
if __name__=='__main__':main()
