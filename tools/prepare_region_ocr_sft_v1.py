"""Prepare matched TextVQA/TextOCR training views using no model predictions."""

from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]/'outputs/vlm_region_ocr_sft_v1'


def main():
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from vlm_deep.common import NORMALIZER,content_sha,encode,read_manifest,sha_file,write_json
    from vlm_probe.evaluate import processor_for
    from vlm_region_ocr_sft_v1.data import bounds,excludes_other_word_centers,region,region_question,schedule
    from prepare_semantic_admission_v1 import write_once

    torch.set_num_threads(2)
    config_path=PACKAGE/'configs/vlm_region_ocr_sft_v1.json';config=json.loads(config_path.read_text())
    if (ROOT/'DATA_COMPLETE.json').exists():
        print('DATA_ALREADY_PREPARED',flush=True);return
    ROOT.mkdir(parents=True,exist_ok=True)
    write_once(ROOT/'DATA_POLICY.json',{'seed':config['seed'],'region_filter':config['region_filter'],
        'minimum_original_answer_votes':config['minimum_original_answer_votes'],
        'images':config['images'],'selection':'Image/QA/region salted SHA order; no prediction-based mining',
        'coordinate_convention':'xyxy normalized1000 from original polygon, full-image input',
        'no_label_synthesis':'One official word transcription, not ten fabricated answers'})
    source=read_manifest(config['source_manifest']);ref=json.loads(Path(config['baseline_config']).read_text())
    text_root=Path(config['source_manifest']).parent
    human=json.loads(Path(ref['human_manifest']).read_text())
    blocked=[source['images'][q['image_id']] for split in ('dev','final') for q in source['splits'][split]]
    blocked+=list(human['images'].values())
    denied={key:{x[key] for x in blocked if key in x} for key in ('sha256','pixel_sha256')}
    by_image=defaultdict(list)
    for q in source['splits']['train_all']:
        votes=sum(NORMALIZER(a)==NORMALIZER(q['target']) for a in q['answers'])
        if votes>=config['minimum_original_answer_votes']:by_image[q['image_id']].append(q)
    whitelist={iid for iid,qs in by_image.items() if len({q['question'] for q in qs})>=2 and
               not any(source['images'][iid].get(k) in denied[k] for k in denied)}
    annotation_path=Path(config['annotations']);raw=json.loads(annotation_path.read_text())
    counters=Counter();records=[];images={}
    candidates=sorted(whitelist & set(raw['imgs']),key=lambda iid:content_sha(['region_ocr_image_v1',config['seed'],iid]))
    reference=json.loads(Path(ref['reference_config']).read_text())
    for iid in candidates:
        item=source['images'][iid];metadata=raw['imgs'][iid]
        if (metadata['width'],metadata['height'])!=(item['width'],item['height']):
            counters['image_dimensions_differ']+=1;continue
        path=(text_root/item['path']).resolve(strict=True)
        with Image.open(path) as original:
            if original.size!=(item['width'],item['height']):
                counters['actual_image_dimensions_differ']+=1;continue
        if sha_file(path)!=item['sha256']:raise ValueError('Approved source image bytes changed')
        rh,rw=smart_resize(item['height'],item['width'],factor=28,min_pixels=reference['min_pixels'],
                           max_pixels=config['train_budget']*28*28)
        ids=raw['imgToAnns'][iid]
        annotations=[raw['anns'][aid] for aid in ids]
        boxes={}
        for ann in annotations:
            try:boxes[ann['id']]=bounds(ann['points'],item['width'],item['height'])
            except (ValueError,KeyError,TypeError):pass
        chosen=None
        for ann in sorted(annotations,key=lambda x:content_sha(['region_ocr_word_v1',config['seed'],x['id']])):
            value,reason=region(ann,item['width'],item['height'],rw,rh,config['region_filter'])
            if reason:counters[reason]+=1;continue
            if not excludes_other_word_centers(value['box_xyxy_original_pixels'],
                    [b for key,b in boxes.items() if key!=ann['id']]):
                counters['another_word_center_inside_region']+=1;continue
            chosen={**value,'annotation_id':ann['id']};break
        if chosen is None:
            counters['image_without_eligible_region']+=1;continue
        original_qa=[]
        for q in sorted(by_image[iid],key=lambda q:content_sha(['region_ocr_qa_v1',config['seed'],q['question_id']])):
            if q['question'] not in {x['question'] for x in original_qa}:original_qa.append(q)
            if len(original_qa)==2:break
        common={**original_qa[0],'study_source':'common'}
        control={**original_qa[1],'study_source':'control_aux'}
        roi={'question_id':'textocr_v01:'+chosen['annotation_id'],'image_id':iid,
            'question':region_question(chosen['box_xyxy_normalized_1000']),
            'prompt_suffix':'\nReturn only the text inside the rectangle, preserving spelling and case.',
            'target':chosen['target'],'source_split':'train','study_source':'region_aux',
            'label_source':'official_TextOCR_v0.1_single_word_transcription',**chosen}
        records.append({'image_id':iid,'common':common,'control':control,'region':roi,
                        'resized_image':[rw,rh]})
        images[iid]={**item,'path':str(path)}
        if len(records)==config['images']:break
    if len(records)!=1600:raise ValueError(f'Insufficient matched reliable training images: {len(records)}')
    catalogs={role:[q for rec in records for q in (rec['common'],rec[role])] for role in ('control','region')}
    schedules=schedule([x['common']['question_id'] for x in records],
                       [x['control']['question_id'] for x in records],
                       [x['region']['question_id'] for x in records])
    for role,rows in catalogs.items():
        manifest={'schema_version':'rarevlm.vlm_data.v2','training_allowed':True,
            'purpose':f'Matched same-image {role} supervision, no predictions used in selection',
            'images':images,'splits':{'train_all':rows,'train_10k':[],'dev':[],'final':[]}}
        manifest['manifest_sha256']=content_sha(manifest)
        write_once(ROOT/f'data/{role.upper()}_TRAIN.json',manifest)
        read_manifest(ROOT/f'data/{role.upper()}_TRAIN.json')
    write_once(ROOT/'data/SCHEDULE.json',{'roles':schedules,'seed':config['seed'],'updates_per_role':200,
        'questions_per_role':3200,'same_image_order_by_microbatch':True})
    write_once(ROOT/'data/TRAINING_RECORDS.json',records)
    # Screens are copied unchanged; confirmation content is not exported here.
    old=PACKAGE.parents[1]/'outputs/vlm_attention_scope_v1/data'
    for task in ('HUMAN','TEXT'):
        write_once(ROOT/f'data/SCREEN_{task}.json',json.loads((old/f'SCREEN_{task}.json').read_text()))
    write_once(ROOT/'data/SCREEN_PAIRS.json',json.loads((old/'SCREEN_PAIRS.json').read_text()))
    probe=sorted([r['region'] for r in records],key=lambda q:content_sha(['region_ocr_probe_v1',q['question_id']]))[:64]
    write_once(ROOT/'data/REGION_TRAIN_PROBE.json',{'qa':probe,'images':images})
    preview=sorted(records,key=lambda q:content_sha(['region_ocr_visual_preview_v1',q['image_id']]))[:8]
    write_once(ROOT/'data/VISUAL_PREVIEW.json',preview)
    # Decode and assistant-mask only the unique 4800 actual training inputs.
    processor=processor_for(Path(ref['model_root']),reference,config['train_budget'])
    unique={q['question_id']:q for rec in records for q in (rec['common'],rec['control'],rec['region'])}
    resize_by_image={rec['image_id']:rec['resized_image'] for rec in records}
    def one(q):
        encoded=encode(processor,q,Path(images[q['image_id']]['path']),True)
        tokens=int(encoded['input_ids'].shape[1]);visual=int(encoded['image_grid_thw'].prod())//4
        actual_size=[int(encoded['image_grid_thw'][0,2])*14,int(encoded['image_grid_thw'][0,1])*14]
        if actual_size!=resize_by_image[q['image_id']]:raise ValueError('Geometry filter resize differs from actual processor')
        if tokens>config['max_sequence_length'] or visual>config['train_budget']:raise ValueError('No truncation permitted')
        targets=encoded['labels'][encoded['labels'].ne(-100)]
        return {'question_id':q['question_id'],'image_id':q['image_id'],'source':q['study_source'],
            'input_tokens':tokens,'visual_tokens':visual,'image_grid_thw':encoded['image_grid_thw'].tolist(),
            'assistant_target_ids':targets.tolist(),'input_ids_sha256':content_sha(encoded['input_ids'].tolist()),
            'labels_sha256':content_sha(encoded['labels'].tolist()),'scope':'Actual CPU processor input, not GPU trace'}
    started=time.monotonic();path=ROOT/'data/CPU_INPUTS.jsonl';totals=Counter()
    with path.open('w') as stream,ThreadPoolExecutor(max_workers=8) as executor:
        for i,result in enumerate(executor.map(one,sorted(unique.values(),key=lambda q:q['question_id'])),1):
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            totals[result['source']]+=1
            if i%400==0:stream.flush();print('CPU_ENCODED',i,len(unique),flush=True)
    summary={'status':'completed','source_images':len(raw['imgs']),'source_region_records':len(raw['anns']),
        'source_unreadable_dot_records':sum(str(x.get('utf8_string','')).strip()=='.' for x in raw['anns'].values()),
        'approved_images_with_two_high_consensus_qa':len(whitelist),'matched_images':len(records),
        'questions_per_role':3200,'unique_questions_per_role':{k:len({q['question_id'] for q in v}) for k,v in catalogs.items()},
        'cpu_unique_qa':len(unique),'cpu_source_counts':dict(totals),'filter_events_on_visited_candidates':dict(counters),
        'data_labels_unchanged':True,'added_human_vote_claims':False,'cpu_encoding_seconds':time.monotonic()-started,
        'annotation_source':config['annotation_url'],'annotation_sha256':sha_file(annotation_path),
        'annotation_bytes':annotation_path.stat().st_size,'cpu_trace_sha256':sha_file(path),
        'source_manifest_sha256':sha_file(config['source_manifest']),'new_model_forwards':0,'training_updates':0}
    write_once(ROOT/'DATA_COMPLETE.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
