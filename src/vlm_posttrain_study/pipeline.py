"""Complete a fixed two-arm noise diagnostic; never auto-promote failed data."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_probe.evaluate import lock_file,now,sha_file,content_sha
from vlm_alignment_signal.common import write_json
from .learning import build

def publication_decision(admission, statistical_criteria):
    """A score cannot change the predeclared diagnostic/failed-data lineage."""
    if admission.get('scope')=='noisy_public_preference_diagnostic':
        return {'automatic_promotion_allowed':False,'factual_quality_gate_passed':admission['factual_quality_gate_passed'],
                'statistical_criteria':statistical_criteria,'reason':'diagnostic_only_scope;failed factual-data gate preserved'}
    raise ValueError('This study only accepts the explicit diagnostic scope')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    config_path=args.config.resolve();cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();admission=json.loads(Path(cfg['data_admission']).read_text())
    if (admission.get('status')!='admitted' or admission.get('scope')!=cfg['study_scope']
        or admission.get('factual_quality_gate_passed') is not False or admission.get('automatic_promotion_allowed') is not False
        or cfg.get('automatic_replication') is not False):raise ValueError('Diagnostic scope/admission/promotion changed')
    protocol=PROJECT_ROOT/'docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md'
    if sha_file(protocol)!=admission['diagnostic_protocol_sha256']:raise ValueError('Diagnostic protocol changed')
    names=[]
    for folder in ['vlm_posttrain','vlm_posttrain_eval','vlm_posttrain_pope','vlm_posttrain_study','vlm_public_preference']:
        names.extend(sorted((SOURCE_ROOT/folder).glob('*.py')))
    for name in ['vlm_deep/common.py','vlm_sft/common.py','vlm_probe/evaluate.py','vlm_precision/evaluate.py',
                 'vlm_joint/evaluate.py','vlm_alignment_signal/loss.py','research_runtime/queue.py']:
        names.append(SOURCE_ROOT/name)
    data_files=[Path(cfg[k]) for k in ['train_manifest','heldout_manifest','data_admission']]
    data_files.extend([Path(cfg['public_data_root'])/'semantic_audit/FROZEN_64.json',
                       Path(cfg['public_data_root'])/'semantic_audit/VISUAL_REVIEW.json',
                       Path(cfg['pope_data_root'])/'manifest.json'])
    identity={'config_sha256':sha_file(config_path),'protocol_sha256':sha_file(protocol),
              'sources':{str(n.relative_to(SOURCE_ROOT)):sha_file(n) for n in sorted(set(names))},
              'data_sha256':{str(f):sha_file(f) for f in data_files},
              'cpu_contracts_sha256':sha_file(root/'CPU_CONTRACTS.json')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    old_guard=root/'INITIALIZATION_CV_GUARD.json'
    paths=list(Path(cfg['initial_adapter']).glob('*'))+[Path(cfg['initial_adapter']).parent/'TRAIN_RESULT.json',
           PROJECT_ROOT/'docs/RESUME.md',Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf')]
    protected={str(f):sha_file(f) for f in paths if f.is_file()};lock_file(old_guard,protected)
    def status(phase,**extra):
        write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
                   'study_scope':cfg['study_scope'],'factual_quality_gate_passed':False,
                   'automatic_promotion_allowed':False,'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
                   'existing_tasks_untouched':True,**extra})
    if (root/'COMPLETION.json').exists():status('completed');return
    try:
        build(cfg);seed=cfg['pilot_seed'];status('engineering_preflight')
        preflight=root/'engineering_runs'
        jobs=[{'key':f'preflight_{c}','command':[sys.executable,'-m','vlm_posttrain.train','--config',str(config_path),
               '--condition',c,'--seed',str(seed),'--output-root',str(preflight),'--preflight'],
               'completion_marker':str(preflight/'runs'/c/f'seed{seed}'/'train/PREFLIGHT.json')} for c in ['sft','dpo']]
        run_queue(root/'preflight_queue',jobs,identity)
        status('training_and_evaluation_branches')
        jobs=[{'key':f'{c}_seed{seed}','command':[sys.executable,'-m','vlm_posttrain_study.branch','--config',str(config_path),
               '--condition',c,'--seed',str(seed),'--output-root',str(root)],
               'completion_marker':str(root/'branches'/f'{c}_seed{seed}'/'COMPLETE.json')} for c in ['sft','dpo','initial']]
        run_queue(root/'branch_queue',jobs,identity);status('paired_analysis')
        trains={c:json.loads((root/'runs'/c/f'seed{seed}'/'train/TRAIN_RESULT.json').read_text()) for c in ['sft','dpo']}
        for key in ['initial_adapter_tensor_sha256','frozen_parameter_sha256_before','global_pair_order_sha256',
                    'actual_chosen_exposures','completed_updates','chosen_input_contract_sha256','training_chosen_answer_tokens']:
            if trains['sft'][key]!=trains['dpo'][key]:raise ValueError('Matched training contract differs:'+key)
        if trains['sft']['completed_updates']!=223 or trains['sft']['actual_chosen_exposures']!=3568:raise ValueError('Full final training not complete')
        subprocess.run([sys.executable,'-m','vlm_posttrain_eval.analysis','--config',str(config_path),'--output-root',str(root),'--seeds',str(seed)],check=True)
        text_path=root/'analysis'/f'pilot_seed{seed}'/'ANALYSIS.json'
        subprocess.run([sys.executable,'-m','vlm_posttrain_pope.analysis','--config',str(config_path),'--output-root',str(root),
                        '--seed',str(seed),'--text-analysis',str(text_path)],check=True)
        pope_path=root/'analysis'/f'pope_pilot_seed{seed}'/'ANALYSIS.json'
        text=json.loads(text_path.read_text());pope=json.loads(pope_path.read_text())
        heldout={c:json.loads((root/'runs'/c/f'seed{seed}'/'heldout/EVALUATION.json').read_text()) for c in ['initial','sft','dpo']}
        for file,digest in protected.items():
            if sha_file(Path(file))!=digest:raise ValueError('Initialization/CV modified:'+file)
        engineering_steps=0
        for base in [preflight,root]:
            for c in ['sft','dpo']:
                history=base/'runs'/c/f'seed{seed}'/'train/PREFLIGHT_HISTORY.jsonl'
                engineering_steps+=sum(json.loads(line)['discarded_engineering_optimizer_updates'] for line in history.read_text().splitlines())
        result={'status':'completed','completed_at':now(),'study_scope':cfg['study_scope'],'pilot_seed':seed,
                'new_trained_adapters':2,'formal_optimizer_updates':sum(t['completed_updates'] for t in trains.values()),
                'discarded_engineering_optimizer_updates':engineering_steps,
                'matched_training_contracts_passed':True,'train_questions_per_arm':3568,'heldout_questions':439,
                'text_dev_model_x_qa':3*1617,'pope_model_x_qa':3*9000,'heldout_model_x_pairs':3*439,
                'textvqa_analysis':text,'pope_analysis':pope,'heldout_likelihood_diagnostics':heldout,
                'training':{c:{k:t[k] for k in ['completed_updates','actual_chosen_exposures','actual_rejected_exposures',
                           'training_input_tokens','training_chosen_answer_tokens','training_rejected_answer_tokens',
                           'training_active_seconds','model_load_seconds','reference_cost','peak_gpu_allocated_gib',
                           'lora_delta_l2','frozen_parameters_unchanged']} for c,t in trains.items()},
                'decision':publication_decision(admission,pope.get('overall_promotion',pope.get('gates',{}))),
                'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'controller_session_seconds':time.monotonic()-start,
                'initialization_cv_guard_unchanged':True,'automatic_replication':False,
                'limits':['single fixed SFT initializer and order-seed42;not3seed stability','human local corrections are imperfect;data factual gate failed',
                          'conditional image-cluster CIs;public benchmark/pretraining overlap unknown','teacher-forced heldout ranking is not generated accuracy']}
        write_json(root/'analysis/ANALYSIS.json',result)
        write_json(PROJECT_ROOT/'results/vlm_public_posttrain_diagnostic_v1_metrics.json',result)
        build(cfg,result);write_json(root/'COMPLETION.json',{'status':'completed','completed_at':now(),
                   'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'automatic_promotion_allowed':False})
        status('completed',formal_optimizer_updates=result['formal_optimizer_updates'])
    except BaseException as exc:
        status('failed',reason=repr(exc));raise
    finally:owner.close()

if __name__=='__main__':main()
