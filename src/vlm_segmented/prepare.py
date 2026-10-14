"""Reuse frozen input schedules, add explicit implementation/library locks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import sha_file,lock_file
from vlm_batching.prepare import prepare as base_prepare,read_plan

DEFAULT_CONFIG=PROJECT_ROOT/'configs/vlm_segmented_sdpa_v1.json'
LIBRARY=Path('/opt/data/private/.venvs/rarevlm-vlm/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py')


def bindings():
    paths=sorted((SOURCE_ROOT/'vlm_segmented').glob('*.py'))+[LIBRARY]
    return {str(p):sha_file(p) for p in paths}


def prepare(config_path=DEFAULT_CONFIG):
    base_prepare(config_path)
    cfg,root,plan=read_plan(config_path)
    lock_file(root/'SEGMENTED_EXECUTOR_LOCK.json',{'config_sha256':sha_file(config_path),
        'plan_sha256':sha_file(root/'PLAN.json'),'source_hashes':bindings(),
        'vision_layers':32,'layout_cache_entries':64,'inference_only':True})


def read_context(config_path=DEFAULT_CONFIG):
    cfg,root,plan=read_plan(config_path);lock=json.loads((root/'SEGMENTED_EXECUTOR_LOCK.json').read_text())
    if lock['source_hashes']!=bindings() or lock['config_sha256']!=sha_file(config_path) or lock['plan_sha256']!=sha_file(root/'PLAN.json'):
        raise ValueError('segmented implementation or upstream library changed')
    return cfg,root,plan,lock


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    prepare(parser.parse_args().config)
