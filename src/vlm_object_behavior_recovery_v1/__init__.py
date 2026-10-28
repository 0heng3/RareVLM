"""Explicit post-training recovery; original scientific modules stay frozen."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'

SCHEMA='rarevlm.object_behavior_replay.recovery.v1'
