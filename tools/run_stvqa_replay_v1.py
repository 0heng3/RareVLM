"""Fixed source replacement, reusing frozen vision CE/DPO training."""
import argparse
from collections import Counter
import importlib
from pathlib import Path
import shutil
import sys

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json,content_sha
from vlm_grpo_iteration_v1.common import read

BASE=PROJECT_ROOT.parents[1]/'outputs';PARENT=BASE/'vlm_visual_preference_v1'
ROOT=BASE/'vlm_stvqa_replay_v1';SOURCE=BASE/'vlm_stvqa_headroom_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_STVQA_REPLAY_V1_PROTOCOL.md'
ROLES=('ce','ce_dpo');INTERNAL=('native_hard','literal_validation','seen_validation','retention')


def prepare():
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('New replay study frozen')
    if not read(SOURCE/'COMPLETION.json')['supervision_signal']:raise ValueError('No new source supervision signal')
    original=read(PARENT/'data/MANIFEST.json');cpu=read(PARENT/'data/CPU_INPUTS.json');schedule=read(PARENT/'data/SCHEDULE.json')
    source=read(SOURCE/'DATA.json');selected=[q for q in source['rows'] if q['probe_cohort']=='train_slots']
    if len(selected)!=128 or len({q['image_id'] for q in selected})!=128:raise ValueError('Require fixed128 distinct train scenes')
    old_ocr=[q for q in original['rows'] if q['study_source']=='common' and q['image_id'].startswith('iiit5k:')]
    if len(old_ocr)!=128:raise ValueError('Expected128 old OCR slots')
    mapping={old['question_id']:new['question_id'] for old,new in zip(old_ocr,selected)}
    newrows=[{**q,'study_source':'common','training_domain':'stvqa_scene','study_source_lineage':'original_MMf_subtrain_positive_only'} for q in selected]
    rows=[q for q in original['rows'] if q['question_id'] not in mapping]+newrows
    images={**original['images'],**{q['image_id']:source['images'][q['image_id']] for q in selected}}
    for q in selected:
        cpu[q['question_id']+':chosen']={'input_ids_sha256':q['chosen_input_sha256'],'labels_sha256':q['chosen_labels_sha256']}
    updates=[{'micros':[[mapping.get(q,q) for q in ids] for ids in update['micros']]} for update in schedule['updates']]
    exposures=Counter(q for update in updates for ids in update['micros'] for q in ids)
    if len(rows)!=384 or sum(exposures.values())!=512 or any(exposures[q['question_id']]!=1 for q in selected):raise ValueError('Source exposure budget differs')
    if set(q['image_id'] for q in selected)&set(q['image_id'] for q in source['rows'] if q['probe_cohort']=='source_holdout'):raise ValueError('Source holdout overlap')
    write_json(ROOT/'data/MANIFEST.json',{'rows':rows,'images':images,'evaluations':original['evaluations']})
    write_json(ROOT/'data/CPU_INPUTS.json',cpu);write_json(ROOT/'data/SCHEDULE.json',{'updates':updates})
    config={**read(PARENT/'CONFIG.json'),'output_root':str(ROOT),'source_replacement':'IIIT_crop128_to_fixed_STVQA_scene128','automatic_main_evaluation':True}
    write_json(ROOT/'CONFIG.json',config)
    for file in ['reference/CACHE.json','reference/COMPLETE.json']:
        dest=ROOT/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/file,dest)
    reuse={}
    copied_baselines=[]
    for name in INTERNAL+('transfer_dev',):
        file=PARENT/'baseline'/name/'PREDICTIONS.json';dest=ROOT/'baseline'/name/'PREDICTIONS.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        copied_baselines.append(dest)
        reuse[name]={'start':str(file),'start_sha256':sha_file(file),'old_ce':str(PARENT/'runs/ce'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json'),
                     'old_ce_dpo':str(PARENT/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json')}
        for role in ('ce','ce_dpo'):reuse[name]['old_'+role+'_sha256']=sha_file(reuse[name]['old_'+role])
    write_json(ROOT/'CONTROL_REUSE.json',{'cohorts':reuse,'new_control_generations':0,'new_reference_forwards':0,'not_independent_repeats':True})
    write_json(ROOT/'DATA_REPLACEMENT.json',{'old_ocr_to_new_scene':mapping,'qa':384,'chosen_exposures':512,'new_scenes':128,'unique_new_targets':len({q['target'].casefold() for q in selected}),
        'old_native_and_preference_rows_unchanged':True,'source_holdout_excluded':True,'new_model_calls':0})
    paths=[Path(__file__),PROTOCOL,ROOT/'CONFIG.json',ROOT/'CONTROL_REUSE.json',ROOT/'DATA_REPLACEMENT.json',SOURCE/'COMPLETION.json',SOURCE/'PLAN.json',SOURCE/'SOURCE_INFERENCE_ADAPTER_PLAN.json']
    paths += [ROOT/'data'/n for n in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json')]+[ROOT/'reference'/n for n in ('CACHE.json','COMPLETE.json')]
    paths += copied_baselines
    paths += [Path(q) for meta in reuse.values() for k,q in meta.items() if not k.endswith('sha256')]
    paths += [Path(images[q['image_id']]['path']) for q in selected]
    parent=read(PARENT/'STUDY_INPUT_LOCK.json')
    write_json(ROOT/'STUDY_INPUT_LOCK.json',{'status':'frozen','files':{**parent['files'],**{str(p):sha_file(p) for p in paths}},
        'initial_lora_sha256':parent['initial_lora_sha256'],'roles':list(ROLES),'new_updates':64,'new_generations':5154,'automatic_more_training':False})
    print({'status':'prepared','roles':list(ROLES),'training_qa':384,'updates_each':32},flush=True)


def runner():
    module=importlib.import_module('run_visual_preference_v1');module.ROOT=ROOT;return module


def verify():return runner().verify()


def analyze():
    parent=runner();result=parent.analyze(True)
    from run_native_stable_kl_v1 import paired
    reuse=read(ROOT/'CONTROL_REUSE.json')['cohorts'];comparisons={}
    for role in ROLES:
        predictions=read(ROOT/'runs'/role/'transfer/transfer_dev/PREDICTIONS.json')
        comparisons[role]={old:paired(read(reuse['transfer_dev']['old_'+old]),predictions) for old in ROLES}
    data=read(ROOT/'data/MANIFEST.json');rare_ids={q['question_id'] for q in data['evaluations']['transfer_dev'] if 1<=q['answer_frequency']<=5}
    candidate=read(ROOT/'runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json');rare=[q for q in candidate if q['question_id'] in rare_ids]
    rare_comparisons={old:paired([q for q in read(path) if q['question_id'] in rare_ids],rare) for old,path in
        [('start',str(ROOT/'baseline/transfer_dev/PREDICTIONS.json')),('old_ce_dpo',reuse['transfer_dev']['old_ce_dpo'])]}
    old_strong_guards={}
    for name in INTERNAL:
        fresh=read(ROOT/'runs/ce_dpo/internal'/name/'PREDICTIONS.json');old=read(reuse[name]['old_ce_dpo'])
        if name=='literal_validation':
            fresh=[{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in fresh]
            old=[{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in old]
        old_strong_guards[name]=paired(old,fresh)
    fresh=read(ROOT/'runs/ce_dpo/internal/seen_validation/PREDICTIONS.json');old=read(reuse['seen_validation']['old_ce_dpo'])
    for part in ('singleword','multiword','rare_1_5'):
        ids={q['question_id'] for q in data['evaluations']['seen_validation'] if (1<=q['answer_frequency']<=5 if part=='rare_1_5' else q['stratum']==part)}
        old_strong_guards[part]=paired([q for q in old if q['question_id'] in ids],[q for q in fresh if q['question_id'] in ids])
    strong=comparisons['ce_dpo']['ce_dpo'];protections=all(q['delta_pp']>=-.5 for q in list(rare_comparisons.values())+list(old_strong_guards.values()))
    main=result['native_positive_signal']['ce_dpo'] and strong['delta_pp']>=.5 and strong['ci95'][0]>0 and protections
    result.update(vs_historical_controls_transfer=comparisons,full_rare342_comparisons=rare_comparisons,
                  old_strong_internal_protections=old_strong_guards,extra_protection_gates_passed=protections,primary_data_replacement_signal=main,goal_complete=False)
    write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',{'status':'completed','phase':'internal_task_signal' if main else 'completed_negative',
        'primary_data_replacement_signal':main,'new_updates':64,'new_generations':5154,'goal_complete':False})
    write_json(ROOT/'DECISION.json',{'status':'completed','decision':'preregister_independent_confirmation' if main else 'stop_fixed_source_replacement_no_sweep_or_extra_steps','automatic_training':False})
    print('STVQA_REPLAY_COMPLETE',result,flush=True)


def worker(argument):
    phase,role=argument.split(':');parent=runner();parent.verify()
    if phase=='train':parent.train(role);return
    done=read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json');model,processor,_=parent.load(done['adapter'],'visual',False)
    parent.evaluate(model,processor,role,INTERNAL if phase=='internal' else ('transfer_dev',),phase)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();entry=str(Path(__file__).resolve());identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')}
    run_queue(ROOT/'training_queue',[{'key':role,'command':[sys.executable,'-u',entry,'--worker','train:'+role],
        'completion_marker':str(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json')} for role in ROLES],identity)
    run_queue(ROOT/'evaluation_queue',[{'key':role+'_'+phase,'command':[sys.executable,'-u',entry,'--worker',phase+':'+role],
        'completion_marker':str(ROOT/'runs'/role/(phase+'_COMPLETE.json'))} for role in ROLES for phase in ('internal','transfer')],identity)
    analyze()


if __name__=='__main__':main()
