"""Freeze source/data/input schedules before new GPU observations."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_probe.evaluate import content_sha, now, sha_file, write_json, lock_file, private_path
from vlm_weighted.evaluate import dev_data
from .schedule import make_batches, padding_summary

DEFAULT_CONFIG = PROJECT_ROOT/'configs/vlm_batching_v1.json'


def source_hashes():
    names = ['vlm_probe/evaluate.py', 'vlm_precision/evaluate.py', 'vlm_joint/evaluate.py',
             'vlm_joint/pipeline.py', 'vlm_precision/pipeline.py', 'vlm_weighted/evaluate.py',
             'vlm_ocr/evaluate.py', 'vlm_ocr/analysis.py', 'vlm_deep/evaluate.py',
             'vlm_deep/model.py', 'vlm_deep/common.py', 'vlm_sft/common.py',
             'vlm_deep/third_party/vqa_eval_metric.py', 'rarevlm_paths.py']
    names += [str(p.relative_to(SOURCE_ROOT)) for p in sorted((SOURCE_ROOT/'vlm_batching').glob('*.py'))]
    return {name: sha_file(SOURCE_ROOT/name) for name in names}


def read_plan(config_path=DEFAULT_CONFIG):
    config_path = private_path(config_path)
    cfg = json.loads(config_path.read_text())
    root = private_path(cfg['output_root'])
    plan = json.loads((root/'PLAN.json').read_text())
    if plan['config_sha256'] != sha_file(config_path) or plan['source_hashes'] != source_hashes():
        raise ValueError('config/source changed; start a new version')
    for key, item in plan['bindings'].items():
        if sha_file(Path(item['path'])) != item['sha256']:
            raise ValueError(f'upstream binding changed:{key}')
    return cfg, root, plan


def prepare(config_path=DEFAULT_CONFIG):
    started = time.monotonic()
    cfg = json.loads(config_path.read_text())
    root = private_path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    rows, images, image_root, metadata = dev_data(Path(cfg['asset_root']))
    upstream = private_path(cfg['upstream_input_contract'])
    contract = json.loads(upstream.read_text())
    if contract['status'] != 'passed' or not contract['complete']:
        raise ValueError('upstream processor contract incomplete')
    if contract['manifest_sha256'] != metadata['manifest_sha256']:
        raise ValueError('input metadata from a different manifest')
    for name, digest in contract['source_hashes'].items():
        if sha_file(SOURCE_ROOT/name) != digest:
            raise ValueError('upstream source changed')
    expected = {r['question_id']: r for r in contract['records'] if r['condition'] == 'baseline'}
    qids = [r['question_id'] for r in rows]
    if set(expected) != set(qids) or len(expected) != 1617:
        raise ValueError('complete baseline processor metadata required')
    lengths = {qid: expected[qid]['input_tokens'] for qid in qids}
    timing_ids = sorted(qids, key=int)[:cfg['timing_questions']]
    schedules = {}; summaries = {}; timing_schedules = {}
    for name, policy in cfg['policies'].items():
        schedules[name] = make_batches(qids, lengths, policy)
        timing_schedules[name] = make_batches(timing_ids, lengths, policy)
        summaries[name] = padding_summary(schedules[name], lengths)
    reference = PROJECT_ROOT/cfg['reference_config']
    plan = {'schema_version':'rarevlm.batching.plan.v1', 'config_sha256':sha_file(config_path),
            'source_hashes':source_hashes(), 'bindings':{
                'input_contract':{'path':str(upstream),'sha256':sha_file(upstream)},
                'manifest':{'path':metadata['manifest_path'],'sha256':metadata['manifest_file_sha256']},
                'reference_config':{'path':str(reference),'sha256':sha_file(reference)}},
            'data':metadata, 'schedules':schedules, 'timing_schedules':timing_schedules,
            'timing_question_ids':timing_ids, 'padding':summaries,
            'length_metadata_used':['question_id','input_tokens'],
            'reference_answers_used_for_scheduling':False,
            'upstream_preparation_seconds':contract['elapsed_seconds'],
            'scope':'immutable upstream input-only metadata; new predictions required; no historical timing reuse'}
    lock_file(root/'PLAN.json',plan)
    from vlm_ocr.analysis import prepare_slices
    slices = prepare_slices(Path(cfg['asset_root']),root)
    write_json(root/'PREPARE_STATUS.json',{'status':'passed','completed_at':now(),
               'elapsed_seconds':time.monotonic()-started,'plan_sha256':sha_file(root/'PLAN.json'),
               'slice_lock_sha256':slices['slice_lock_sha256'],'padding':summaries})
    print(json.dumps({'status':'passed','padding':summaries}, indent=2),flush=True)
    return plan


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    prepare(parser.parse_args().config)
