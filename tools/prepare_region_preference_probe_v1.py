"""Prepare reviewed reciprocal hard-region preferences and a fixed32-step schedule."""
from pathlib import Path
import hashlib,json,random,time
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha,encode,sha_file,write_json
from vlm_region_coordinates_v1.data import pixel_box,question

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_preference_probe_v1'
ORIGINAL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    from vlm_probe.evaluate import processor_for
    if (ROOT/'DATA_COMPLETE.json').exists():raise FileExistsError('Already prepared')
    candidates=read(ROOT/'CANDIDATES_WITH_REVIEW.json');review=read(ROOT/'DATA_REVIEW.json')
    admitted={q['candidate_index'] for q in review['rows'] if q['supported']}
    prefs=[];pairs=[];images={}
    for index,item in enumerate(candidates['rows']):
        if index not in admitted:continue
        a=item['a'];b=item['b'];iid=a['image_id'];im=candidates['images'][iid];images[iid]=im
        width,height=a['actual_processed_dimensions'];bbox=pixel_box(b['box_xyxy_original_pixels'],im['width'],im['height'],width,height)
        rows=[]
        for side,chosen,rejected,box in [('a',a['target'],b['target'],a['box_xyxy_processed_pixels']),
                                         ('b',b['target'],a['target'],bbox)]:
            row={'question_id':f'region-pref:{index}:{side}','pair_id':iid,'pair_side':side,'image_id':iid,
                'question':question(box,width,height,'pixel_dimensions'),'prompt_suffix':a['prompt_suffix'],
                'target':chosen,'rejected_target':rejected,'study_source':'preference',
                'actual_processed_dimensions':[width,height],'label_source':'official word transcription, assistant source-crop review',
                'original_candidate_index':index}
            rows.append(row);prefs.append(row)
        pairs.append({'pair_id':iid,'question_ids':[q['question_id'] for q in rows]})
    if len(pairs)!=review['admitted_pairs'] or len(prefs)!=2*len(pairs):raise ValueError('Registered reviewed cohort differs')
    source=read(ORIGINAL/'data/CONTROL_TRAIN.json');hard_images=set(images)
    order=sorted(source['splits']['train_all'],key=lambda q:hashlib.sha256(('region-pref-replay:20261020:'+q['question_id']).encode()).hexdigest())
    replay=[{**q,'study_source':'common'} for q in order if q['image_id'] not in hard_images][:256]
    used=hard_images|{q['image_id'] for q in replay};anchors=[]
    for q in order:
        if q['image_id'] in used:continue
        anchors.append(q);used.add(q['image_id'])
        if len(anchors)==64:break
    if len(replay)!=256 or len(anchors)!=64:raise ValueError('Need256 replay and64 image-separated retention QA')
    for q in replay+anchors:images[q['image_id']]=source['images'][q['image_id']]
    rng=random.Random(42);pair_slots=[]
    while len(pair_slots)<128:
        cycle=pairs.copy();rng.shuffle(cycle);pair_slots+=cycle
    pair_slots=pair_slots[:128];updates=[]
    for step in range(32):
        group=pair_slots[step*4:step*4+4]
        updates.append({'micros':[[q['question_id'] for q in replay[step*8:step*8+4]],
            [qid for pair in group[:2] for qid in pair['question_ids']],
            [q['question_id'] for q in replay[step*8+4:step*8+8]],
            [qid for pair in group[2:] for qid in pair['question_ids']]]})
    rows=replay+prefs
    if len({q['question_id'] for q in rows})!=256+len(prefs):raise ValueError('Unique training QA mismatch')
    write_json(ROOT/'data/MANIFEST.json',{'images':images,'rows':rows,'hard_eval':prefs,'retention_eval':anchors,'pairs':pairs})
    write_json(ROOT/'data/SCHEDULE.json',{'updates':updates,'chosen_qa_exposures':512,'preference_qa_exposures':256,'replay_qa_exposures':256,
        'rejected_forward_sequences_per_role':256,'note':'Both roles run matched chosen+rejected forward batches; control ignores rejected loss'})
    c=read(PROJECT_ROOT/'configs/vlm_region_preference_probe_v1.json');base=read(c['baseline_config'])
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    begun=time.monotonic();records={}
    for row in rows:
        for kind in ('chosen','rejected') if row['study_source']=='preference' else ('chosen',):
            q={**row,'target':row['target'] if kind=='chosen' else row['rejected_target']}
            item=encode(processor,q,Path(images[q['image_id']]['path']),True)
            grid=item['image_grid_thw'][0].tolist();tokens=int(item['input_ids'].shape[1])
            if row['study_source']=='preference' and [grid[2]*14,grid[1]*14]!=q['actual_processed_dimensions']:raise ValueError('Actual region frame mismatch')
            if tokens>2048 or int(item['image_grid_thw'].prod())//4>1024:raise ValueError('Budget exceeded')
            records[row['question_id']+':'+kind]={'input_ids_sha256':content_sha(item['input_ids'].tolist()),
                'labels_sha256':content_sha(item['labels'].tolist()),'assistant_target_ids':item['labels'][item['labels'].ne(-100)].tolist(),
                'image_grid_thw':grid,'input_tokens':tokens,'image_sha256':images[q['image_id']]['sha256']}
    write_json(ROOT/'data/CPU_INPUTS.json',records)
    result={'status':'completed','reciprocal_pairs':len(pairs),'preference_qa':len(prefs),'replay_unique_qa':256,'retention_qa':64,
        'actual_supervised_cpu_encodings':len(records),'chosen_exposures_per_role':512,'new_training_updates':0,
        'cpu_seconds':time.monotonic()-begun,'scope':'Small training-only learning positive control, not CV or benchmark capability'}
    write_json(ROOT/'DATA_COMPLETE.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
