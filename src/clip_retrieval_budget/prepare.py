"""Use frozen asset binding and verify reused margin/fast-policy lineage."""
import json
from pathlib import Path
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import sha_file,lock_file
from clip_retrieval_adaptive.prepare import bind as bind_assets,link,verify_guard as verify_base_guard


def bind(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['adaptive_root'])
    if cfg['candidate_k']!=50 or cfg['small_candidate_k']!=20:raise ValueError('Fixed budgets required')
    assets=bind_assets(config_path);lock=json.loads((old/'STUDY_INPUT_LOCK.json').read_text())
    done=json.loads((old/'COMPLETION.json').read_text());analysis=json.loads((old/'analysis/ANALYSIS.json').read_text())
    if done['status']!='completed' or done['analysis_sha256']!=sha_file(old/'analysis/ANALYSIS.json'):raise ValueError('Previous adaptive study changed/incomplete')
    if analysis['policy_sha256']!=sha_file(old/'POLICY.json'):raise ValueError('Old fast policy changed')
    if any(sha_file(SOURCE_ROOT/name)!=digest for name,digest in lock['sources'].items()):raise ValueError('Frozen adaptive source changed')
    oldassets=json.loads((old/'REUSED_ASSETS.json').read_text())
    for split in ['dev','test']:
        if oldassets['features'][split]['features_sha256']!=assets['features'][split]['features_sha256']:raise ValueError('Gallery mismatch')
    for seed in cfg['training_seeds']:
        if oldassets['checkpoints'][str(seed)]['checkpoint_sha256']!=assets['checkpoints'][str(seed)]['checkpoint_sha256']:raise ValueError('Model mismatch')
    oldpart=json.loads((old/'PARTITION.json').read_text());newpart=json.loads((root/'PARTITION.json').read_text())
    if any(oldpart[k]!=newpart[k] for k in ['cal_image_indices','confirm_image_indices','salt']):raise ValueError('Calibration groups changed')
    source=old/'confidence/dev';meta=json.loads((source/'CONFIDENCE.json').read_text())
    if any(sha_file(source/name)!=digest for name,digest in meta['files'].items()):raise ValueError('Old margins changed')
    link(root/'confidence/dev',source)
    record={'source_root':str(old),'old_analysis_sha256':done['analysis_sha256'],'old_policy_sha256':analysis['policy_sha256'],
        'old_partition_sha256':sha_file(old/'PARTITION.json'),'confidence_marker_sha256':sha_file(source/'CONFIDENCE.json'),
        'confidence_files':meta['files'],'scope':'bound historical margins and fastthresholds,not new confidence inference or independent baseline'}
    lock_file(root/'REUSED_FAST_POLICY.json',record)
    verify_guard(root)
    return assets


def verify_guard(root):
    result=verify_base_guard(root)
    record=json.loads((root/'REUSED_FAST_POLICY.json').read_text());old=Path(record['source_root'])
    for p,key in [(old/'analysis/ANALYSIS.json','old_analysis_sha256'),(old/'POLICY.json','old_policy_sha256'),
                  (old/'PARTITION.json','old_partition_sha256'),(old/'confidence/dev/CONFIDENCE.json','confidence_marker_sha256')]:
        if sha_file(p)!=record[key]:raise ValueError('Historical margin/fast source changed')
    if any(sha_file(old/'confidence/dev'/name)!=digest for name,digest in record['confidence_files'].items()):raise ValueError('Historical margin array changed')
    result['reused_fast_and_margin_sources']='unchanged';return result
