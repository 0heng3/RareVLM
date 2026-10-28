#!/usr/bin/env python
"""Behavior replay: explicit static prepare-only or full shared-queue run."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
from vlm_object_behavior_replay.study import main

if __name__=='__main__':main()
