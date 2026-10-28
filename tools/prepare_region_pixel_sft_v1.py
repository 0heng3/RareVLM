"""Reuse admitted images/labels; change only the declared region coordinate frame."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, encode, sha_file, write_json
from vlm_region_coordinates_v1.data import pixel_box, question

PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    from vlm_region_ocr_sft_v1.study import actor, verify
    from vlm_probe.evaluate import processor_for
    verify()
    evidence=read(ROOT.parent/'vlm_region_coordinates_v1/COMPLETION.json')
    if not evidence['input_evidence_gate_passed']:raise ValueError('Native-pixel input evidence gate failed')
    if (ROOT/'DATA_COMPLETE.json').exists():raise FileExistsError('Data already prepared; inspect registered study')
    config=read(PROJECT_ROOT/'configs/vlm_region_pixel_sft_v1.json')
    reference=read(config['baseline_config'])
    processor=processor_for(Path(reference['model_root']),read(reference['reference_config']),1024)
    original=read(PARENT/'data/REGION_TRAIN.json');images=original['images']
    common=[q for q in original['splits']['train_all'] if q['study_source']=='common']
    regions=[q for q in original['splits']['train_all'] if q['study_source']=='region_aux']
    old_cpu={q['question_id']:q for q in (json.loads(line) for line in (PARENT/'data/CPU_INPUTS.jsonl').read_text().splitlines())}
    catalogs={};renamings={};new_rows=[]
    for role,mode in [('normalized','normalized_dimensions'),('pixel','pixel_dimensions')]:
        auxiliary=[];renamings[role]={}
        for q in regions:
            grid=old_cpu[q['question_id']]['image_grid_thw'][0];width,height=grid[2]*14,grid[1]*14
            image=images[q['image_id']]
            box=pixel_box(q['box_xyxy_original_pixels'],image['width'],image['height'],width,height)
            coordinates=q['box_xyxy_normalized_1000'] if role=='normalized' else box
            row={**q,'question_id':q['question_id']+':'+role,'source_question_id':q['question_id'],
                 'question':question(coordinates,width,height,mode),'coordinate_frame':mode,
                 'actual_processed_dimensions':[width,height],'box_xyxy_processed_pixels':box}
            renamings[role][q['question_id']]=row['question_id'];auxiliary.append(row)
        catalogs[role]=common+auxiliary;new_rows+=auxiliary
        manifest={**{k:v for k,v in original.items() if k not in ('manifest_sha256','splits')},
                  'purpose':f'Matched {role} coordinate-frame auxiliary SFT; unchanged images and official labels',
                  'splits':{'train_all':catalogs[role],'train_10k':[],'dev':[],'final':[]}}
        manifest['manifest_sha256']=content_sha(manifest)
        write_json(ROOT/f'data/{role.upper()}_TRAIN.json',manifest)
    source_schedule=read(PARENT/'data/SCHEDULE.json')['roles']['region']
    schedules={role:[{'micros':[[renamings[role].get(q,q) for q in micro] for micro in update['micros']]}
                     for update in source_schedule] for role in catalogs}
    image_orders={}
    for role in catalogs:
        mapping={q['question_id']:q for q in catalogs[role]}
        image_orders[role]=[mapping[qid]['image_id'] for u in schedules[role] for micro in u['micros'] for qid in micro]
    if image_orders['normalized']!=image_orders['pixel']:raise ValueError('Matched image order differs')
    if len(common)!=1600 or len(regions)!=1600 or len({q['question_id'] for q in common+new_rows})!=4800:
        raise ValueError('Expected 1600 common plus 3200 new coordinate questions')
    write_json(ROOT/'data/SCHEDULE.json',{'roles':schedules,'seed':42,'updates_per_role':200,
        'same_images_and_assistant_targets':True,'source_schedule_sha256':sha_file(PARENT/'data/SCHEDULE.json')})
    begun=time.monotonic()
    def encoded_record(row):
        item=encode(processor,row,Path(images[row['image_id']]['path']),True)
        tokens=int(item['input_ids'].shape[1]);visual=int(item['image_grid_thw'].prod())//4
        actual=[int(item['image_grid_thw'][0,2])*14,int(item['image_grid_thw'][0,1])*14]
        if tokens>2048 or visual>1024 or actual!=row['actual_processed_dimensions']:
            raise ValueError('New actual encoding differs from declared pixel frame or exceeds budget')
        original_target=old_cpu[row['source_question_id']]['assistant_target_ids']
        target=item['labels'][item['labels'].ne(-100)].tolist()
        if target!=original_target:raise ValueError('Coordinate change modified assistant target tokens')
        return {'question_id':row['question_id'],'image_id':row['image_id'],'source':'region_aux',
            'input_tokens':tokens,'visual_tokens':visual,'image_grid_thw':item['image_grid_thw'].tolist(),
            'assistant_target_ids':target,'input_ids_sha256':content_sha(item['input_ids'].tolist()),
            'labels_sha256':content_sha(item['labels'].tolist()),'scope':'New actual CPU encoding; unchanged full original image',
            'coordinate_frame':row['coordinate_frame']}
    path=ROOT/'data/CPU_INPUTS.jsonl'
    with path.open('w') as stream:
        for row in common:stream.write(json.dumps(old_cpu[row['question_id']],ensure_ascii=False)+'\n')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for count,record in enumerate(pool.map(encoded_record,new_rows),1):
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if count%64==0:
                    stream.flush();elapsed=time.monotonic()-begun
                    write_json(ROOT/'DATA_PROGRESS.json',{'status':'encoding','new_completed_qa':count,'new_expected_qa':3200,
                        'reused_original_common_qa':1600,'seconds':elapsed,'remaining_seconds_estimate':elapsed/count*(3200-count)})
    summary={'status':'completed','matched_images':1600,'questions_per_role':3200,'cpu_registered_unique_qa':4800,
        'new_actual_cpu_encoded_qa':3200,'reused_frozen_common_cpu_qa':1600,'cpu_encoding_seconds':time.monotonic()-begun,
        'cpu_trace_sha256':sha_file(path),'same_original_images_labels_order_and_loss':True,'training_updates':0,
        'scope':'Only coordinate convention and corresponding declared prompt units vary between arms'}
    write_json(ROOT/'DATA_COMPLETE.json',summary);write_json(ROOT/'DATA_PROGRESS.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
