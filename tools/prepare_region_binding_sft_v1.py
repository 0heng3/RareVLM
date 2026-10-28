"""Matched same-image region coverage: repeat one region versus cover two."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha,encode,sha_file,write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_binding_sft_v1'
PIXEL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'
ORIGINAL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
POOL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_binding_data_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    from vlm_region_pixel_sft_v1.study import verify
    from vlm_probe.evaluate import processor_for
    verify()
    if (ROOT/'DATA_COMPLETE.json').exists():raise FileExistsError('Already prepared')
    pool=read(POOL/'PAIR_CANDIDATES.json');pairs=pool['pairs'][:800]
    if len(pairs)!=800:raise ValueError('Need the fixed first800 hash-ordered eligible images')
    common_source=read(ORIGINAL/'data/CONTROL_TRAIN.json')
    by_image={}
    for q in common_source['splits']['train_all']:by_image.setdefault(q['image_id'],[]).append(q)
    old_cpu={q['question_id']:q for q in (json.loads(line) for line in (ORIGINAL/'data/CPU_INPUTS.jsonl').read_text().splitlines())}
    old_cpu.update({q['question_id']:q for q in (json.loads(line) for line in (PIXEL/'data/CPU_INPUTS.jsonl').read_text().splitlines())})
    images={p['source_image_id']:pool['images'][p['source_image_id']] for p in pairs}
    common=[];aux=[];new_b=[];shared=[];schedules={role:[] for role in ('repeat','dual')};lookup={}
    for index,pair in enumerate(pairs):
        original=by_image[pair['source_image_id']]
        if len(original)!=2:raise ValueError('Need both admitted original VQA questions per image')
        originals=[{**q,'study_source':'common'} for q in original]
        a={**pair['a'],'study_source':'region_aux'}
        b={**pair['b'],'study_source':'region_aux'}
        common+=originals;aux += [a,b];new_b.append(b)
        anchor,other=(a,b) if index%2==0 else (b,a)
        shared.append({'image_id':pair['source_image_id'],'common':[q['question_id'] for q in originals],
            'anchor':anchor['question_id'],'other':other['question_id'],'anchor_side':'a' if index%2==0 else 'b'})
    rows=common+aux;lookup={q['question_id']:q for q in rows}
    if len(lookup)!=3200:raise ValueError('Expected 1600 original QA and 1600 distinct region questions in shared catalog')
    for start in range(0,800,4):
        group=shared[start:start+4]
        for role in schedules:
            schedules[role].append({'micros':[[v['common'][0] for v in group],[v['anchor'] for v in group],
                [v['common'][1] for v in group],[v['anchor'] if role=='repeat' else v['other'] for v in group]]})
    unique={}
    for role,updates in schedules.items():
        ids=[qid for u in updates for micro in u['micros'] for qid in micro]
        unique[role]=len(set(ids))
        if len(ids)!=3200 or sum(lookup[q]['study_source']=='common' for q in ids)!=1600:raise ValueError('Exposure counts differ')
    if unique!={'repeat':2400,'dual':3200}:raise ValueError('Intended coverage manipulation differs')
    orders={role:[lookup[q]['image_id'] for u in values for m in u['micros'] for q in m] for role,values in schedules.items()}
    if orders['repeat']!=orders['dual']:raise ValueError('Matched image/micro order differs')
    for role in schedules:
        manifest={'schema_version':common_source['schema_version'],'training_allowed':True,
            'purpose':'Shared3200-question training catalog; actual exposure follows role schedule, repeat exposes only2400 unique',
            'images':images,'splits':{'train_all':rows,'train_10k':[],'dev':[],'final':[]}}
        manifest['manifest_sha256']=content_sha(manifest)
        write_json(ROOT/f'data/{role.upper()}_TRAIN.json',manifest)
    write_json(ROOT/'data/SCHEDULE.json',{'roles':schedules,'seed':42,'updates_per_role':200,
        'unique_exposed_qa':unique,'anchor_sides':{'a':400,'b':400},'shared_catalog_items':3200})
    config=read(PROJECT_ROOT/'configs/vlm_region_binding_sft_v1.json');base=read(config['baseline_config'])
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    def record(q):
        item=encode(processor,q,Path(images[q['image_id']]['path']),True);grid=item['image_grid_thw'][0].tolist()
        if [grid[2]*14,grid[1]*14]!=q['actual_processed_dimensions']:raise ValueError('Actual frame differs')
        if int(item['input_ids'].shape[1])>2048 or int(item['image_grid_thw'].prod())//4>1024:raise ValueError('Budget exceeded')
        return {'question_id':q['question_id'],'image_id':q['image_id'],'source':q['study_source'],
            'input_tokens':int(item['input_ids'].shape[1]),'visual_tokens':int(item['image_grid_thw'].prod())//4,
            'image_grid_thw':item['image_grid_thw'].tolist(),'assistant_target_ids':item['labels'][item['labels'].ne(-100)].tolist(),
            'input_ids_sha256':content_sha(item['input_ids'].tolist()),'labels_sha256':content_sha(item['labels'].tolist()),
            'scope':'New actual CPU encoding for second official region'}
    started=time.monotonic()
    with (ROOT/'data/CPU_INPUTS.jsonl').open('w') as stream:
        for q in common+[p['a'] for p in pairs]:
            old=old_cpu[q['question_id']]
            stream.write(json.dumps({**old,'source':q['study_source'],'scope':'Reused frozen identical input/target, source-role tag updated if second VQA'},ensure_ascii=False)+'\n')
        with ThreadPoolExecutor(max_workers=2) as executor:
            for count,value in enumerate(executor.map(record,new_b),1):
                stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                if count%32==0:
                    stream.flush();write_json(ROOT/'DATA_PROGRESS.json',{'status':'encoding_second_regions','completed_qa':count,
                        'expected_qa':800,'reused_identical_cpu_qa':2400,'seconds':time.monotonic()-started})
    result={'status':'completed','matched_images':800,'questions_per_catalog':3200,'exposures_per_role':3200,
        'unique_exposed_qa':unique,'reused_identical_cpu_qa':2400,'new_actual_cpu_encoded_qa':800,
        'cpu_registered_unique_qa':3200,'cpu_trace_sha256':sha_file(ROOT/'data/CPU_INPUTS.jsonl'),
        'cpu_encoding_seconds':time.monotonic()-started,'training_updates':0,
        'single_change':'Second regional exposure repeats anchor vs covers the other official region',
        'information_limit':'Candidate has800 more independently exposed region labels; equal QA does not mean equal information'}
    write_json(ROOT/'DATA_COMPLETE.json',result);write_json(ROOT/'DATA_PROGRESS.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
