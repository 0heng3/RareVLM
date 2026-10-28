"""Native-JSON reporting recovery; frozen math/source/predictions unchanged.

Run this wrapper for this study's pipeline. It converts NumPy scalar containers
at the analysis/reporting boundary only; all training and evaluation functions,
checkpoint selection, dev gates and queue identity remain the frozen originals.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import lock_file,now,sha_file,write_json


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock_file(root/'REPORTING_RECOVERY_LOCK.json',{'helper_sha256':sha_file(Path(__file__)),
        'config_sha256':sha_file(a.config),'original_analysis_source_sha256':sha_file(PROJECT_ROOT/'src/clip_retrieval_quality/analysis.py'),
        'original_pipeline_source_sha256':sha_file(PROJECT_ROOT/'src/clip_retrieval_quality/pipeline.py'),
        'scope':'convert NumPy scalar values at JSON/return boundary only;no changed arithmetic,thresholds,models,training,selection or rankings'})
    record=root/'analysis/REPORTING_RECOVERY.json'
    if not record.exists():
        state=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
        write_json(record,{'created_at':now(),'previous_status':state,'reason':'numpy.bool_ in positive_seed gate not JSON serializable',
            'reused_completed_training_runs':12,'reused_completed_dev_cells':12,
            'original_source_edits':0,'recomputed_neural_predictions':0,
            'recovery_lock_sha256':sha_file(root/'REPORTING_RECOVERY_LOCK.json')})
    from clip_retrieval_quality import analysis,pipeline
    original=analysis.analyze
    analysis.write_json=lambda path,value:write_json(path,native(value))
    def analyze_native(*args,**kwargs):return native(original(*args,**kwargs))
    pipeline.analyze=analyze_native
    sys.argv=[sys.argv[0],'--config',str(a.config)]
    print('NATIVE_JSON_REPORTING_RECOVERY; frozen science unchanged',flush=True)
    pipeline.main()


if __name__=='__main__':main()
