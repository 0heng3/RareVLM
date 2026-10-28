#!/usr/bin/env python
"""Independent replay coverage study; explicit prepare-only/run modes."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
from vlm_object_replay_coverage.study import main

if __name__=='__main__':main()
