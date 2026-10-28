#!/usr/bin/env python
"""Versioned stage33 recovery; all original source and completed Object remain immutable."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vlm_lora_coordinate_shrink_recovery_v1.study import main
if __name__=='__main__':main()
