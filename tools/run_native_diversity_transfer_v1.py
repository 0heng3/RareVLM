"""Independent fixed-Final transfer characterization, zero training updates."""
import argparse
import importlib
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import read
from vlm_native_scope_v1.model import load

BASE=PROJECT_ROOT.parents[1]/'outputs'
PARENT=BASE/'vlm_native_diversity_v1'
ROOT=BASE/'vlm_native_diversity_transfer_v1'
PREVIOUS=BASE/'vlm_visual_preference_v1/runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json'
PROTOCOL=PROJECT_ROOT/'docs/VLM_NATIVE_DIVERSITY_TRANSFER_V1_PROTOCOL.md'
ROLES=('repeat96','diverse192')


def prepare():
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('Followup already frozen')
    previous=read(PARENT/'COMPLETION.json')
    if previous['phase']!='screen_negative_no_expansion' or previous['screen_checks']['diverse192']['hard_learning']:
        raise ValueError('Require preserved original diagnostic gate outcome')
    models={r:read(PARENT/'runs'/r/'train/TRAIN_COMPLETE.json') for r in ROLES}
    c={'output_root':str(ROOT),'data_manifest':str(PARENT/'data/MANIFEST.json'),'models':models,'roles':list(ROLES),
       'baseline_predictions':str(PARENT/'baseline/transfer_dev/PREDICTIONS.json'),'previous_dpo128_predictions':str(PREVIOUS),
       'new_optimizer_updates':0,'new_generations':3234,'policy':'imagewise_fp32_language','evaluation_budget':1024,
       'evaluation_batch_size':4,'max_new_tokens':32,'posthoc_followup':True,'original_gate_remains_failed':True,'automatic_more_training':False}
    write_json(ROOT/'CONFIG.json',c)
    paths=[Path(__file__),PROTOCOL,ROOT/'CONFIG.json',Path(c['data_manifest']),Path(c['baseline_predictions']),PREVIOUS,
           PARENT/'STUDY_INPUT_LOCK.json',PARENT/'COMPLETION.json',PARENT/'SCREEN_ANALYSIS.json',
           PROJECT_ROOT/'src/vlm_native_scope_v1/model.py',PROJECT_ROOT/'src/vlm_batch_stable_v1/model.py',
           PROJECT_ROOT/'tools/run_native_correction_sft_dpo_v2.py',PROJECT_ROOT/'src/research_runtime/queue.py']
    for r,m in models.items():paths += [PARENT/'runs'/r/'train/TRAIN_COMPLETE.json',Path(m['adapter'])/'adapter_config.json',Path(m['adapter'])/'adapter_model.safetensors']
    write_json(ROOT/'STUDY_INPUT_LOCK.json',{'status':'frozen','files':{str(p):sha_file(p) for p in paths},'new_optimizer_updates':0,'new_generations':3234,'roles':list(ROLES),'original_study_result_unchanged':True})


def verify():
    for path,digest in read(ROOT/'STUDY_INPUT_LOCK.json')['files'].items():
        if sha_file(path)!=digest:raise ValueError('Frozen followup input changed: '+path)


def worker(role):
    from vlm_human_sft.evaluate import lora_hash
    verify();c=read(ROOT/'CONFIG.json');final=c['models'][role];model,processor,_=load(final['adapter'],'visual',False)
    model.eval();model.requires_grad_(False)
    if lora_hash(model)!=final['final_lora_sha256']:raise ValueError('Wrong fixed Final')
    evaluator=importlib.import_module('run_native_correction_sft_dpo_v2');evaluator.ROOT=ROOT;begun=time.monotonic()
    directory=ROOT/'runs'/role/'transfer';evaluator.evaluate(model,processor,directory,('transfer_dev',),read(c['data_manifest']))
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','role':role,'new_optimizer_updates':0,'new_generations':1617,
        'model_hash_unchanged':lora_hash(model)==final['final_lora_sha256'],'active_seconds':time.monotonic()-begun,
        'predictions_sha256':sha_file(directory/'transfer_dev/PREDICTIONS.json')})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    c=read(ROOT/'CONFIG.json');records={'start':read(c['baseline_predictions']),'previous_dpo128':read(c['previous_dpo128_predictions'])}
    for r in ROLES:records[r]=read(ROOT/'runs'/r/'transfer/transfer_dev/PREDICTIONS.json')
    expected=read(c['data_manifest'])['evaluations']['transfer_dev'];frame=[(q['question_id'],q['inference_input_sha256']) for q in expected]
    if any([(q['question_id'],q['input_sha256']) for q in rows]!=frame for rows in records.values()):raise ValueError('Fixed evaluation frames differ')
    metrics={r:100*sum(q['vqa_score'] for q in rows)/len(rows) for r,rows in records.items()}
    comparisons={r:paired(rows,records['diverse192']) for r,rows in records.items() if r!='diverse192'}
    rare={q['question_id'] for q in expected if 1<=q['answer_frequency']<=5}
    rare_comparisons={r:paired([q for q in records[r] if q['question_id'] in rare],[q for q in records['diverse192'] if q['question_id'] in rare]) for r in ('start','repeat96')}
    checks=read(PARENT/'SCREEN_ANALYSIS.json')['screen_checks']['diverse192']
    signal=all(comparisons[r]['delta_pp']>=.5 and comparisons[r]['ci95'][0]>0 for r in ('start','repeat96')) and comparisons['previous_dpo128']['delta_pp']>=0 and all(q['delta_pp']>=-.5 for q in rare_comparisons.values()) and checks['ocr_point_protection'] and checks['native_point_protections']
    result={'status':'completed','metrics':metrics,'candidate_comparisons':comparisons,'full1617_rare':rare_comparisons,'task_signal':signal,
        'phase':'fixed_model_internal_signal_pending_replication' if signal else 'fixed_model_task_threshold_not_met',
        'new_optimizer_updates':0,'new_generations':3234,'original_study_gate_remains_failed':True,'goal_complete':False,
        'scope':'Posthoc fixed-checkpoint characterization on historically used internal cohort; original failed gate unchanged, no external/seed-stability claim.'}
    write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-run',action='store_true');parser.add_argument('--worker',choices=ROLES);args=parser.parse_args()
    if args.prepare_run and not (ROOT/'STUDY_INPUT_LOCK.json').exists():prepare()
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();run_queue(ROOT/'queue',[{'key':r,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',r],'completion_marker':str(ROOT/'runs'/r/'COMPLETE.json')} for r in ROLES],{'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')});analyze()


if __name__=='__main__':main()
