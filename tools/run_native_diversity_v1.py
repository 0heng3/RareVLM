"""Equal-exposure corrective-data coverage; reuse the frozen vision DPO loop."""
import argparse
from collections import Counter
import copy
import gc
import importlib
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_literal_preference_v1.common import sample, make_batch
from vlm_native_scope_v1.model import load

BASE = PROJECT_ROOT.parents[1]/'outputs'
ROOT = BASE/'vlm_native_diversity_v1'
SOURCE = BASE/'vlm_visual_preference_v1'
MINING = BASE/'vlm_native_diversity_mining_v1'
OLD = BASE/'vlm_native_correction_pool_v2'
PROTOCOL = PROJECT_ROOT/'docs/VLM_NATIVE_DIVERSITY_V1_PROTOCOL.md'
ROLES = ('repeat96','diverse192')
INTERNAL = ('native_hard','literal_validation','seen_validation','retention')


def prepare():
    from vlm_probe.evaluate import processor_for
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Study frozen')
    review = read(MINING/'REVIEWED_ERRORS.json')
    if review['status']!='passed' or review['selected_new_unique_targets']!=64:
        raise ValueError('Need64 reviewed novel singleword targets')
    for path,digest in review['source_files_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Review source changed')
    oldreview=read(OLD/'REVIEWED_ERRORS.json'); oldlookup={q['question_id']:q for q in read(OLD/'ERROR_CANDIDATES.json')['rows']}
    selected=[];used=set()
    for stratum in ('singleword','multiword'):
        group=[]
        for qid in oldreview['accepted_question_ids']:
            q=oldlookup[qid];t=NORMALIZER(q['target'])
            if q['stratum']==stratum and t not in used:
                group.append(copy.deepcopy(q));used.add(t)
                if len(group)==64:break
        if len(group)!=64:raise ValueError('Need64 unique old targets per stratum')
        selected+=group
    lookup={q['question_id']:q for q in read(MINING/'ERROR_CANDIDATES.json')['rows']}
    fresh=[copy.deepcopy(lookup[q]) for q in review['selected_new_question_ids']]
    if len(fresh)!=64 or any(q['stratum']!='singleword' for q in fresh):raise ValueError('Fixed new64 differs')
    hard=selected+fresh
    if len({NORMALIZER(q['target']) for q in hard})!=192 or len({q['image_id'] for q in hard})!=192:raise ValueError('Hard target/image uniqueness')
    for q in hard:
        p=q['initial_prediction']
        if not p['terminated']:raise ValueError('Need observed EOS negative')
        q.update(study_source='preference',rejected_target=p['prediction'],rejected_completion_ids=p['completion_ids'])
    groups=lambda rows:[rows[i:i+4] for i in range(0,len(rows),4)]
    oldgroups=groups(hard[:128]);newgroups=groups(hard[128:]);anchors=oldgroups[::2]
    base=read(SOURCE/'data/MANIFEST.json');common=[q for q in base['rows'] if q['study_source']=='common']
    if len(common)!=256:raise ValueError('Original replay changed')
    images={**base['images'],**read(OLD/'ERROR_CANDIDATES.json')['images'],**read(MINING/'DATA.json')['images']}
    union={'rows':common+hard,'images':images,'evaluations':base['evaluations']}
    def pixels(q):return q['raw_pixel_sha256'] if 'raw_pixel_sha256' in q else images[q['image_id']]['pixel_sha256']
    held={pixels(q) for n,rows in base['evaluations'].items() if n!='native_hard' for q in rows}
    if {pixels(q) for q in union['rows']} & held:raise ValueError('Train/evaluation pixel overlap')
    c=read(SOURCE/'CONFIG.json');modelbase=read(c['baseline_config']);processor=processor_for(Path(modelbase['model_root']),read(modelbase['reference_config']),1024)
    previous={q['question_id']:q for q in base['rows']};cpu={};oldcpu=read(SOURCE/'data/CPU_INPUTS.json');encoded=0;reused=0
    for q in union['rows']:
        old=previous.get(q['question_id']);same=old and old['target']==q['target'] and old.get('rejected_completion_ids')==q.get('rejected_completion_ids')
        for kind in (('chosen','rejected') if q['study_source']=='preference' else ('chosen',)):
            key=q['question_id']+':'+kind
            if same and key in oldcpu:cpu[key]=oldcpu[key];reused+=1;continue
            item=sample(processor,union,q,kind=='rejected')
            if item['input_ids'].shape[1]>2048:raise ValueError('Sequence budget')
            cpu[key]={k+'_sha256':content_sha(item[k].tolist()) for k in ('input_ids','labels')};encoded+=1
    write_json(ROOT/'data/MANIFEST.json',union);write_json(ROOT/'data/CPU_INPUTS.json',cpu)
    original_updates=read(SOURCE/'data/SCHEDULE.json')['updates'];role_schedules={};paths=[];receipt={}
    for role in ROLES:
        og=oldgroups[::2] if role=='repeat96' else oldgroups;ng=newgroups[::2] if role=='repeat96' else newgroups
        variable=[q for group in og+ng for q in group];rolehard=variable
        data={**union,'rows':common+rolehard};updates=copy.deepcopy(original_updates)
        for b in range(16):
            hard_groups=[og[(2*b)%len(og)],og[(2*b+1)%len(og)],ng[b%len(ng)],anchors[b]]
            for k,group in enumerate(hard_groups):updates[2*b+k//2]['micros'][1+2*(k%2)]=[q['question_id'] for q in group]
        counts=Counter(q for u in updates for batch in u['micros'] for q in batch)
        pref={q['question_id'] for q in rolehard};exposures={n:sum(counts[q['question_id']] for q in rolehard if q['stratum']==n) for n in ('singleword','multiword')}
        if len(pref)!=(96 if role=='repeat96' else 192) or sum(counts[q] for q in pref)!=256 or sum(counts.values())!=512 or exposures!={'singleword':160,'multiword':96}:raise ValueError('Exposure schedule differs')
        unit=ROOT/'units'/role
        for name,value in [('MANIFEST.json',data),('CPU_INPUTS.json',cpu),('SCHEDULE.json',{'updates':updates})]:write_json(unit/'data'/name,value);paths.append(unit/'data'/name)
        config={**c,'output_root':str(unit),'roles':[role],'dpo_weight':{role:1.},'algorithm':'fixed_visual_DPO_corrective_coverage','unique_hard_qa':len(pref)};write_json(unit/'CONFIG.json',config);paths.append(unit/'CONFIG.json')
        (unit/'runs'/role).mkdir(parents=True,exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True);(ROOT/'runs'/role).symlink_to(unit/'runs'/role,target_is_directory=True)
        role_schedules[role]={'unit':str(unit),'reference_unique_batches':24 if role=='repeat96' else 48}
        receipt[role]={'unique_hard':len(pref),'unique_training_qa':len(data['rows']),'hard_exposures':256,'replay_native':128,'replay_ocr':128,'single_multi_exposures':exposures,'hard_multiplicity_histogram':dict(Counter(counts[q] for q in pref)),'original_hard128_training_members':sum(q['question_id'] in pref for q in base['evaluations']['native_hard'])}
    for part,names in [('a',('native_hard','literal_validation','retention')),('b',('seen_validation',))]:
        for n in names:
            src=SOURCE/'baseline'/n/'PREDICTIONS.json';dst=ROOT/('baseline_'+part)/n/'PREDICTIONS.json';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
        write_json(ROOT/('baseline_'+part)/'COMPLETE.json',{'status':'completed','cohorts':list(names),'reused':True,'new_generations':0})
    reuse={}
    for n in INTERNAL+('transfer_dev',):
        src=SOURCE/'baseline'/n/'PREDICTIONS.json';dst=ROOT/'baseline'/n/'PREDICTIONS.json';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());paths += [src,dst];reuse[n]={'source':str(src),'sha256':sha_file(src),'qa':len(read(src))}
    write_json(ROOT/'baseline_TRANSFER_COMPLETE.json',{'status':'completed','reused':True,'new_generations':0})
    write_json(ROOT/'BASELINE_REUSE.json',{'status':'completed','cohorts':reuse,'new_generations':0})
    write_json(ROOT/'DATA_RECEIPT.json',{**read(SOURCE/'DATA_RECEIPT.json'),'role_training':receipt,'training_candidate_unique_targets':192,'selected_new_reviewed_single':64,'original_hard128_is_diagnostic_not_full_training_accuracy':True,'new_cpu_sequences_encoded':encoded,'old_cpu_sequences_reused':reused,'train_eval_pixel_overlap':0})
    write_json(ROOT/'CONFIG.json',{**c,'output_root':str(ROOT),'roles':list(ROLES),'dpo_weight':{r:1. for r in ROLES},'algorithm':'fixed_visual_DPO_corrective_coverage','units':role_schedules})
    paths += [Path(__file__),PROTOCOL,ROOT/'data/MANIFEST.json',ROOT/'data/CPU_INPUTS.json',ROOT/'DATA_RECEIPT.json',ROOT/'CONFIG.json',ROOT/'BASELINE_REUSE.json',MINING/'REVIEWED_ERRORS.json',OLD/'REVIEWED_ERRORS.json',OLD/'ERROR_CANDIDATES.json',SOURCE/'STUDY_INPUT_LOCK.json',SOURCE/'CONFIG.json',SOURCE/'COMPLETION.json',SOURCE/'runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json',PROJECT_ROOT/'tools/run_visual_preference_v1.py',PROJECT_ROOT/'src/vlm_native_scope_v1/model.py',PROJECT_ROOT/'src/vlm_literal_preference_v1/common.py']
    adapter=Path(c['initial_adapter']);paths += [adapter/n for n in ('adapter_config.json','adapter_model.safetensors')]
    initial=read(SOURCE/'STUDY_INPUT_LOCK.json')['initial_lora_sha256'];write_json(ROOT/'STUDY_INPUT_LOCK.json',{'status':'frozen','files':{str(p):sha_file(p) for p in paths},'initial_lora_sha256':initial,'new_updates':64,'new_generations_phase1':1920,'conditional_generations':3234,'roles':list(ROLES),'automatic_more_training':False})
    bindings={}
    for role in ROLES:
        unit=ROOT/'units'/role;write_json(unit/'STUDY_INPUT_LOCK.json',{'status':'frozen','initial_lora_sha256':initial,'files':{str(ROOT/'STUDY_INPUT_LOCK.json'):sha_file(ROOT/'STUDY_INPUT_LOCK.json'),**{str(p):sha_file(p) for p in [unit/'CONFIG.json',unit/'data/MANIFEST.json',unit/'data/CPU_INPUTS.json',unit/'data/SCHEDULE.json']}}});bindings[role]=sha_file(unit/'STUDY_INPUT_LOCK.json')
    write_json(ROOT/'UNIT_BINDINGS.json',bindings);print('DIVERSITY_PREPARED',receipt,flush=True)


def verify():
    lock=read(ROOT/'STUDY_INPUT_LOCK.json')
    for path,digest in lock['files'].items():
        if sha_file(path)!=digest:raise ValueError('Frozen diversity input changed: '+path)
    for role,digest in read(ROOT/'UNIT_BINDINGS.json').items():
        if sha_file(ROOT/'units'/role/'STUDY_INPUT_LOCK.json')!=digest:raise ValueError('Unit binding changed')
    return lock


def reference(role):
    import torch
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sft.common import to_device
    unit=ROOT/'units'/role;closed=unit/'reference/COMPLETE.json'
    if closed.exists():return
    lock=verify();c=read(unit/'CONFIG.json');data=read(unit/'data/MANIFEST.json');cpu=read(unit/'data/CPU_INPUTS.json');rows={q['question_id']:q for q in data['rows']}
    model,processor,_=load(c['initial_adapter'],'visual',False);model.eval();model.requires_grad_(False);model.config.use_cache=False;before=lora_hash(model)
    if before!=lock['initial_lora_sha256']:raise ValueError('Wrong reference initializer')
    memo={};batches={};started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for step,update in enumerate(read(unit/'data/SCHEDULE.json')['updates']):
            for micro,ids in enumerate(update['micros']):
                chunk=[rows[q] for q in ids]
                if chunk[0]['study_source']!='preference':continue
                batch,binding,_,_=make_batch(processor,data,chunk,cpu);key=content_sha(binding)
                if key not in memo:
                    logp,counts=sequence_logps(model,to_device(batch,next(model.parameters()).device))
                    if not bool(torch.isfinite(logp).all()):raise FloatingPointError('Nonfinite reference')
                    memo[key]={'binding':binding,'chosen_logps':logp[:4].cpu().tolist(),'rejected_logps':logp[4:].cpu().tolist(),'counts':counts.cpu().tolist()}
                batches[f'{step+1}:{micro}']=memo[key];write_json(unit/'reference/PROGRESS.json',{'status':'reference','completed_batches':len(batches),'expected':64,'unique_forward_batches':len(memo),'active_seconds':time.monotonic()-started})
                if time.monotonic()-started>600 or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Reference budget')
    if len(batches)!=64 or len(memo)!=(24 if role=='repeat96' else 48) or lora_hash(model)!=before:raise ValueError('Reference coverage/identity')
    write_json(unit/'reference/CACHE.json',{'reference_lora_sha256':before,'batches':batches});write_json(closed,{'status':'completed','forward_sequences':8*len(memo),'unique_forward_batches':len(memo),'optimizer_updates':0,'active_seconds':time.monotonic()-started,'cache_sha256':sha_file(unit/'reference/CACHE.json')})
    del model,processor,logp,counts,batch;gc.collect();torch.cuda.empty_cache()


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    data=read(ROOT/'data/MANIFEST.json');names=list(INTERNAL)+(['transfer_dev'] if transfer else []);rec={'start':{n:read(ROOT/'baseline'/n/'PREDICTIONS.json') for n in names}}
    for role in ROLES:rec[role]={n:read(ROOT/'runs'/role/('transfer' if n=='transfer_dev' else 'internal')/n/'PREDICTIONS.json') for n in names}
    for records in rec.values():
        for group in ('singleword','multiword','rare_1_5'):
            ids={q['question_id'] for q in data['evaluations']['seen_validation'] if (q['stratum']==group if group!='rare_1_5' else 1<=q['answer_frequency']<=5)};records[group]=[q for q in records['seen_validation'] if q['question_id'] in ids]
    def norm(rows,n):return [{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in rows] if n=='literal_validation' else rows
    metrics={r:{n:100*sum(q['vqa_score'] for q in norm(rows,n))/len(rows) for n,rows in records.items()} for r,records in rec.items()};vs={r:{n:paired(norm(rec['start'][n],n),norm(rows,n)) for n,rows in rec[r].items()} for r in ROLES};adv={n:paired(norm(rec['repeat96'][n],n),norm(rows,n)) for n,rows in rec['diverse192'].items()}
    checks={r:{'hard_learning':sum(q['vqa_score']>=1-1e-9 for q in rec[r]['native_hard'])>=32 and vs[r]['native_hard']['delta_pp']>=20,'ocr_point_protection':vs[r]['literal_validation']['delta_pp']>=-.5,'native_point_protections':all(vs[r][n]['delta_pp']>=-.5 for n in ('seen_validation','singleword','multiword','rare_1_5','retention'))} for r in ROLES}
    expand=all(checks['diverse192'].values()) and all(adv[n]['delta_pp']>=-.5 for n in ('literal_validation','seen_validation','rare_1_5'))
    result={'status':'completed','metrics':metrics,'vs_start':vs,'screen_checks':checks,'screen_passed':{r:all(q.values()) for r,q in checks.items()},'candidate_vs_repeat96':adv,'expand_transfer':expand,'goal_complete':False,'scope':'Single-seed internal corrective-coverage intervention, matched exposure/source/stratum counts; old128 diagnosis is not complete training accuracy.'}
    if transfer:
        previous=read(SOURCE/'runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json');result['vs_previous_dpo128']=paired(previous,rec['diverse192']['transfer_dev']);rare={q['question_id'] for q in data['evaluations']['transfer_dev'] if 1<=q['answer_frequency']<=5}
        rarity={r:paired([q for q in rec[r]['transfer_dev'] if q['question_id'] in rare],[q for q in rec['diverse192']['transfer_dev'] if q['question_id'] in rare]) for r in ('start','repeat96')};result['full1617_rare']=rarity
        result['native_positive_signal']={'diverse192':expand and all(q['delta_pp']>=.5 and q['ci95'][0]>0 for q in (vs['diverse192']['transfer_dev'],adv['transfer_dev'])) and result['vs_previous_dpo128']['delta_pp']>=0 and all(q['delta_pp']>=-.5 for q in rarity.values())}
    write_json(ROOT/('TRANSFER_ANALYSIS.json' if transfer else 'SCREEN_ANALYSIS.json'),result);return result


def worker(arg):
    verify();kind,role=arg.split(':');unit=ROOT/'units'/role;parent=importlib.import_module('run_visual_preference_v1');parent.ROOT=unit
    if kind=='pilot':
        reference(role);model,processor,_=parent.train(role);parent.evaluate(model,processor,role,INTERNAL,'internal')
    else:
        final=read(unit/'runs'/role/'train/TRAIN_COMPLETE.json');model,processor,_=load(final['adapter'],'visual',False);parent.evaluate(model,processor,role,('transfer_dev',),'transfer')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-run',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare or (args.prepare_run and not (ROOT/'STUDY_INPUT_LOCK.json').exists()):
        prepare()
        if args.prepare:return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json'),'unit_bindings_sha256':sha_file(ROOT/'UNIT_BINDINGS.json')};entry=str(Path(__file__).resolve())
    def job(kind,r):return {'key':r,'command':[sys.executable,'-u',entry,'--worker',kind+':'+r],'completion_marker':str(ROOT/'runs'/r/('internal_COMPLETE.json' if kind=='pilot' else 'transfer_COMPLETE.json'))}
    run_queue(ROOT/'training_queue',[job('pilot',r) for r in ROLES],identity);write_json(ROOT/'reference/COMPLETE.json',{'status':'completed','schedule_entries':128,'unique_forward_batches':72,'forward_sequences':576});result=analyze()
    if not result['expand_transfer']:
        write_json(ROOT/'COMPLETION.json',{**result,'phase':'screen_negative_no_expansion','new_updates':64,'new_generations':1920,'automatic_more_training':False});return
    run_queue(ROOT/'transfer_queue',[job('transfer',r) for r in ROLES],identity);result=analyze(True);write_json(ROOT/'COMPLETION.json',{**result,'phase':'pilot_passed_pending_replication' if any(result['native_positive_signal'].values()) else 'transfer_negative_no_expansion','new_updates':64,'new_generations':5154,'automatic_more_training':False})


if __name__=='__main__':main()
