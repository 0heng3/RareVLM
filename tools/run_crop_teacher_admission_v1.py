"""Bounded train-only crop-teacher admission; no training or benchmark gain."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, encode, sha_file, write_json, collate
from vlm_grpo_iteration_v1.common import read, chosen_sample
from inventory_native_smalltext_v1 import literal

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_crop_teacher_admission_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_CROP_TEACHER_ADMISSION_V1_PROTOCOL.md'
PARENT = BASE / 'vlm_crop_teacher_signal_v1'


def prepare():
    import torch
    from PIL import Image
    from vlm_probe.evaluate import processor_for, inference_row
    from vlm_region_ocr_sft_v1.data import bounds, excludes_other_word_centers
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Admission plan already frozen')
    started = time.monotonic(); torch.set_num_threads(2)
    paths = [Path(__file__), PROTOCOL,
             BASE/'vlm_native_ocr_correction_v1/revision2/DATA.json',
             BASE/'vlm_native_smalltext_v1/CANDIDATE_ROWS.json',
             BASE/'vlm_native_ocr_correction_v1/ERROR_CANDIDATES.json',
             BASE/'vlm_native_smalltext_v1/ERROR_CANDIDATES.json',
             BASE/'vlm_late_rank_matched_dpo_v1/data/MANIFEST.json',
             BASE/'vlm_deep_v2/prepared/manifest.json', PARENT/'DATA.json', PARENT/'PLAN.json']
    sources = [read(paths[2]), read(paths[3])]
    errors = {q['question_id'] for p in paths[4:6] for q in read(p)['rows']}
    current = read(paths[6]); manifest = read(paths[7]); prior = read(paths[8])
    blocked_rows = current['rows'] + [q for rows in current['evaluations'].values() for q in rows]
    blocked_rows += [q for name in ('dev','final') for q in manifest['splits'][name]]
    meta = {**manifest['images'], **current['images']}
    pixel = lambda iid: meta[iid].get('pixel_sha256', meta[iid].get('sha256'))
    blocked = {q['image_id'] for q in blocked_rows}
    blocked.update(p['source_image_id'] for p in prior['pairs'])
    blocked_pixels = {pixel(i) for i in blocked if i in meta}
    cfg = read(PROJECT_ROOT/'configs/vlm_region_ocr_sft_v1.json')
    annotations = Path(cfg['annotations']); raw = read(annotations); paths += [annotations]
    candidates = []
    for origin, data in zip(('native_ocr_revision2','small_text'), sources):
        for q in data['rows']:
            candidates.append({**q, 'admission_origin': origin,
                'historical_error_priority': q['question_id'] in errors})
    def ordering(q):
        return (0 if q['historical_error_priority'] else 1 if q['admission_origin']=='small_text' else 2,
                q['ocr_evidence']['effective_region_pixels'][1],
                content_sha(['crop_teacher_admission_v1',q['question_id']]))
    candidates.sort(key=ordering)
    c = read(BASE/'vlm_visual_preference_v1/CONFIG.json'); base = read(c['baseline_config'])
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    pairs, images, seen, targets, rejects = [], {}, set(), Counter(), Counter()
    for q in candidates:
        iid = q['image_id']; im = manifest['images'][iid]; pix = pixel(iid)
        if iid in blocked or pix in blocked_pixels:
            rejects['current_training_or_evaluation_or_prior7'] += 1; continue
        if pix in seen or targets[literal(q['target'])] >= 2:
            rejects['duplicate_pixel_or_target_cap'] += 1; continue
        ann = raw['anns'][q['ocr_annotation_id']]
        if literal(ann['utf8_string']) != literal(q['target']) or len(q['target'].split()) != 1:
            raise ValueError('Independent literal evidence differs')
        if sum(literal(a)==literal(q['target']) for a in q['answers']) < 8:
            rejects['literal_consensus_below8'] += 1; continue
        x0,y0,x1,y1 = q['ocr_evidence']['box_xyxy_original_pixels']
        box = [max(0,math.floor(x0)-2),max(0,math.floor(y0)-2),
               min(im['width'],math.ceil(x1)+2),min(im['height'],math.ceil(y1)+2)]
        # This processor requires both native input dimensions >= its factor28.
        # Expand the source region, never synthesize/resample missing pixels.
        for lo,hi,limit in [(0,2,im['width']),(1,3,im['height'])]:
            if box[hi]-box[lo]<28:
                begin=max(0,min(limit-28,math.floor((box[lo]+box[hi]-28)/2)))
                box[lo],box[hi]=begin,begin+28
        others=[]
        for aid in raw['imgToAnns'][iid]:
            if aid == q['ocr_annotation_id']: continue
            try: others.append(bounds(raw['anns'][aid]['points'],im['width'],im['height']))
            except (ValueError,KeyError,TypeError): continue
        if not excludes_other_word_centers(box, others):
            rejects['other_text_center_in_padded_crop'] += 1; continue
        source = next(d['images'][iid] for d in sources if iid in d['images'])
        path = Path(source['path'])
        if sha_file(path) != source['sha256']:
            raise ValueError('Source image bytes changed')
        image = Image.open(path).convert('RGB')
        if image.size != (im['width'],im['height']): raise ValueError('Image size differs')
        crop_path = ROOT/'teacher_inputs'/f"{q['question_id']}.png"
        crop_path.parent.mkdir(parents=True,exist_ok=True); image.crop(box).save(crop_path)
        pair = {'source_question_id':q['question_id'],'source_image_id':iid,'pixel_sha256':pix,
                'teacher_box_xyxy':box,'origin':q['admission_origin'],
                'historical_error_priority':q['historical_error_priority'],'conditions':{}}
        for condition, image_path in [('full',path),('crop',crop_path)]:
            image_id = iid+':'+condition
            row = {k:v for k,v in q.items() if k not in ('initial_prediction','enlarged_prediction')}
            row.update(question_id=q['question_id']+':'+condition,image_id=image_id)
            images[image_id]={'path':str(image_path),'sha256':sha_file(image_path)}
            prefix = encode(processor,inference_row(row),image_path,False)
            sample = chosen_sample(processor,row,images)
            if sample['input_ids'].shape[1]>2048: raise ValueError('Prefix exceeds budget')
            row.update(expected_prefix_tokens=int(prefix['input_ids'].shape[1]),
                expected_grid=prefix['image_grid_thw'][0].tolist(),
                inference_input_sha256=content_sha(prefix['input_ids'].tolist()),
                chosen_input_sha256=content_sha(sample['input_ids'].tolist()),
                chosen_labels_sha256=content_sha(sample['labels'].tolist()))
            pair['conditions'][condition]=row; paths.append(image_path)
        pairs.append(pair); seen.add(pix); targets[literal(q['target'])]+=1
        if len(pairs)%32 == 0:
            write_json(ROOT/'CPU_PROGRESS.json',{'status':'preparing','pairs':len(pairs),'cap':512,'seconds':time.monotonic()-started})
        if len(pairs)==512: break
    write_json(ROOT/'DATA.json',{'pairs':pairs,'images':images,'original_sft_seen':True,'train_only_privileged_crop':True})
    write_json(ROOT/'INVENTORY.json',{'status':'completed','source_qa':len(candidates),'selected_images':len(pairs),
        'unique_targets':len(targets),'historical_error_candidates':sum(p['historical_error_priority'] for p in pairs),
        'origins':dict(Counter(p['origin'] for p in pairs)),'rejections_before_cap':dict(rejects),
        'excluded_image_ids':len(blocked),'pixel_isolation_passed':not bool(set(seen)&blocked_pixels),
        'maximum_candidate_images':512,'new_model_calls':0,'optimizer_updates':0,'seconds':time.monotonic()-started})
    parent = read(PARENT/'PLAN.json')
    paths += [ROOT/'DATA.json',ROOT/'INVENTORY.json',PROJECT_ROOT/'tools/inventory_native_smalltext_v1.py',
              PROJECT_ROOT/'src/vlm_region_ocr_sft_v1/data.py',PROJECT_ROOT/'src/vlm_deep/common.py']
    files={**parent['files'],**{str(p):sha_file(p) for p in paths}}
    write_json(ROOT/'PLAN.json',{'status':'frozen','pairs':len(pairs),'models':{'start':parent['models']['start']},
        'primary_teacher':'start','shards':2,'max_seconds_per_shard':1200,'optimizer_updates':0,
        'new_generations':2*len(pairs),'teacher_forced_sequences':2*len(pairs),
        'automatic_training':False,'files':files,'scope':'Train-only teacher admission, not a model ability test'})
    print(read(ROOT/'INVENTORY.json'),flush=True)


def verify():
    for f,digest in read(ROOT/'PLAN.json')['files'].items():
        if sha_file(f)!=digest: raise ValueError('Frozen input changed: '+f)


def distribution_metrics(full, crop, ids):
    """Output-logit CE/KD gradients, not parameter gradients."""
    import torch
    p=full.exp(); t=crop.exp(); onehot=torch.zeros_like(p).scatter_(1,ids[:,None],1)
    a=(p-onehot)[:-1]; b=(p-t)[:-1]
    def norm(x): return x.square().sum().sqrt()
    cosine=float((a*b).sum()/(norm(a)*norm(b)).clamp_min(1e-20))
    return {'body_kl_teacher_to_full':float((t[:-1]*(crop[:-1]-full[:-1])).sum(-1).mean()),
        'body_teacher_non_gt_mass':float((1-t[:-1].gather(1,ids[:-1,None]).squeeze(1)).mean()),
        'eos_teacher_non_gt_mass':float(1-t[-1,ids[-1]]),
        'output_gradient_cosine_ce_kd':cosine,
        'output_gradient_relative_difference':float(norm(b-a)/norm(a).clamp_min(1e-20))}


def worker(shard):
    import torch
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_fit_positive_control_v1.study import generated,token_statistics
    from vlm_sft.common import to_device
    verify(); plan=read(ROOT/'PLAN.json'); data=read(ROOT/'DATA.json'); started=time.monotonic()
    model,processor,_=load(plan['models']['start']['adapter'],'visual',False)
    model.eval(); model.requires_grad_(False); before=lora_hash(model)
    if before!=plan['models']['start']['lora_sha256']: raise ValueError('Wrong teacher')
    pairs=[p for i,p in enumerate(data['pairs']) if i%2==shard]
    directory=ROOT/'runs'/f'shard_{shard}'; results=[]; torch.cuda.reset_peak_memory_stats()
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    with torch.inference_mode():
        for index,pair in enumerate(pairs):
            path=directory/'pairs'/f'{index:03d}.json'
            if path.exists():
                record=read(path)
                if record['source_question_id']!=pair['source_question_id'] or record['lora_sha256']!=before:
                    raise ValueError('Resume identity differs')
                results.append(record); continue
            records={}; logps={}; ids=None
            for condition in (('full','crop') if index%2==0 else ('crop','full')):
                q=pair['conditions'][condition]; prediction=generated(model,processor,[q],data)[0]
                if prediction['input_sha256']!=q['inference_input_sha256']: raise ValueError('Prompt changed')
                sample=chosen_sample(processor,q,data['images'])
                if content_sha(sample['input_ids'].tolist())!=q['chosen_input_sha256'] or content_sha(sample['labels'].tolist())!=q['chosen_labels_sha256']:
                    raise ValueError('GT prefix changed')
                batch=to_device(collate([sample],processor.tokenizer.pad_token_id),next(model.parameters()).device)
                keep=int(batch['labels'].ne(-100).sum(1).max())+1
                hook=model.get_output_embeddings().register_forward_pre_hook(lambda m,args:(args[0][:,-keep:,:],))
                try: output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
                finally: hook.remove()
                stats=token_statistics(output.logits,batch['labels'],eos)[0]
                targets=batch['labels'][:,-output.logits.shape[1]:][:,1:]
                valid=targets[0].ne(-100); actual_ids=targets[0,valid]
                if ids is not None and not torch.equal(ids,actual_ids): raise ValueError('Unaligned target actions')
                ids=actual_ids.clone(); logps[condition]=output.logits[0,:-1][valid].float().log_softmax(-1)
                records[condition]={'prediction':prediction,'teacher_prefix':stats,
                    'strict_target_match':literal(prediction['prediction'])==literal(q['target'])}
                del output,batch,sample
            record={'source_question_id':pair['source_question_id'],'lora_sha256':before,'conditions':records,
                'distribution':distribution_metrics(logps['full'],logps['crop'],ids)}
            del logps,ids
            write_json(path,record); results.append(record)
            elapsed=time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'teacher_data_admission','completed_pairs':len(results),
                'expected_pairs':len(pairs),'new_generations':2*len(results),'teacher_forced_sequences':2*len(results),
                'optimizer_updates':0,'active_seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
            if elapsed>1200 or torch.cuda.max_memory_allocated()/2**30>65: raise RuntimeError('Admission budget exceeded')
    if len(results)!=len(pairs) or lora_hash(model)!=before: raise ValueError('Count/weights changed')
    write_json(directory/'ROWS.json',results)
    write_json(directory/'COMPLETE.json',{'status':'completed','pairs':len(results),'optimizer_updates':0,
        'model_hash_unchanged':True,'active_seconds':time.monotonic()-started})


def analyze():
    import statistics
    verify(); data=read(ROOT/'DATA.json'); lookup={r['source_question_id']:r for s in range(2) for r in read(ROOT/f'runs/shard_{s}/ROWS.json')}
    eligible=[]; selected=[]; targets=Counter(); reasons=Counter(); rows=[]
    for pair in data['pairs']:
        r=lookup[pair['source_question_id']]; full=r['conditions']['full']; crop=r['conditions']['crop']
        stats=crop['teacher_prefix']; drop=full['teacher_prefix']['content_ce']-stats['content_ce']; why=[]
        for passed,reason in [(crop['strict_target_match'],'crop_literal_wrong'),
                (crop['prediction']['terminated'],'crop_not_natural_eos'),
                (stats['content_target_argmax'],'crop_body_not_all_argmax'),
                (stats['content_ce']<=.2,'crop_body_nll_above_point2'),
                (stats['eos_ce']<=.2,'crop_eos_nll_above_point2'),(drop>=.05,'body_nll_drop_below_point05')]:
            if not passed: why.append(reason)
        reasons.update(why); target=literal(pair['conditions']['full']['target'])
        item={'question_id':pair['source_question_id'],'image_id':pair['source_image_id'],'target':target,
            'eligible':not why,'reject_reasons':why,'body_nll_drop':drop,
            'full_wrong_to_crop_correct':not full['strict_target_match'] and crop['strict_target_match'],
            'distribution':r['distribution']}
        rows.append(item)
        if not why:
            eligible.append(item)
            if len(selected)<128 and targets[target]<2:
                selected.append(item); targets[target]+=1
    def mean(xs):
        values=list(xs)
        return statistics.mean(values) if values else None
    drop=mean(q['body_nll_drop'] for q in selected); corrections=sum(q['full_wrong_to_crop_correct'] for q in selected)
    gates={'images_at_least128':len(selected)>=128,'targets_at_least96':len(targets)>=96,
        'corrections_at_least32':corrections>=32,'mean_body_nll_drop_at_least_point2':drop is not None and drop>=.2}
    result={'status':'completed','phase':'sufficient_teacher_data' if all(gates.values()) else 'insufficient_teacher_data',
        'candidate_images':len(rows),'eligible_images':len(eligible),'selected_images':len(selected),
        'selected_unique_targets':len(targets),'selected_full_wrong_to_crop_correct':corrections,
        'selected_correct_but_underconfident':len(selected)-corrections,'selected_mean_body_nll_drop':drop,
        'all_candidate_full_strict_correct':sum(r['conditions']['full']['strict_target_match'] for r in lookup.values()),
        'all_candidate_crop_strict_correct':sum(r['conditions']['crop']['strict_target_match'] for r in lookup.values()),
        'gates':gates,'rejection_counts_nonexclusive':dict(reasons),
        'selected_mean_teacher_non_gt_mass':mean(q['distribution']['body_teacher_non_gt_mass'] for q in selected),
        'selected_median_ce_kd_output_gradient_cosine':statistics.median(q['distribution']['output_gradient_cosine_ce_kd'] for q in selected) if selected else None,
        'selected_mean_ce_kd_relative_difference':mean(q['distribution']['output_gradient_relative_difference'] for q in selected),
        'new_generations':2*len(rows),'teacher_forced_sequences':2*len(rows),'optimizer_updates':0,
        'new_training_admitted':False,'goal_complete':False,'scope':'Train-only crop admission, no model ability gain'}
    write_json(ROOT/'ADMISSION_ROWS.json',{'rows':rows}); write_json(ROOT/'SELECTED.json',{'rows':selected})
    write_json(ROOT/'COMPLETION.json',result)
    write_json(ROOT/'DECISION.json',{'status':'completed','new_training_admitted':False,
        'decision':'prepare_separate_weight_matched_sft_kd_protocol' if all(gates.values()) else 'stop_no_pool_or_threshold_expansion'})
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--worker',type=int,choices=(0,1)); args=parser.parse_args()
    if args.prepare: prepare(); return
    if args.worker is not None: worker(args.worker); return
    from research_runtime.queue import run_queue
    verify(); entry=str(Path(__file__).resolve())
    run_queue(ROOT/'queue',[{'key':f'shard_{s}','command':[sys.executable,'-u',entry,'--worker',str(s)],
        'completion_marker':str(ROOT/f'runs/shard_{s}/COMPLETE.json')} for s in range(2)],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    analyze()


if __name__=='__main__': main()
