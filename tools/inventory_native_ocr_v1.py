"""Prediction-free inventory of native QA with independent official OCR evidence."""
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import time
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vlm_deep.common import NORMALIZER,content_sha,read_manifest,write_json,sha_file
from vlm_region_ocr_sft_v1.data import region


def main():
    b=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs');out=b/'vlm_native_ocr_correction_v1';out.mkdir(exist_ok=True)
    if (out/'INVENTORY.json').exists():raise FileExistsError('Inventory already exists')
    start=time.monotonic();m_path=b/'vlm_deep_v2/prepared/manifest.json';source=read_manifest(m_path);old=json.loads((b/'vlm_reference_replay_v1/DATA.json').read_text())
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_region_ocr_sft_v1.json').read_text());ann_path=Path(cfg['annotations']);raw=json.loads(ann_path.read_text())
    ref=json.loads(Path(json.loads((b/'vlm_reference_replay_v1/PLAN.json').read_text())['reference_config']).read_text())
    norm=lru_cache(maxsize=300000)(lambda x:NORMALIZER(x));freq=Counter(norm(q['target']) for q in source['splits']['train_all'])
    blocked={q['image_id'] for q in old['training_rows']}
    for rows in old['evaluations'].values():blocked.update(q['image_id'] for q in rows)
    for name in ('dev','final'):blocked.update(q['image_id'] for q in source['splits'][name])
    image_meta={**source['images'],**old['images']};pixel=lambda iid:image_meta[iid].get('pixel_sha256',image_meta[iid]['sha256'])
    blocked_pixels={pixel(i) for i in blocked if i in image_meta};by_image=defaultdict(list);counts=Counter()
    policy={**cfg['region_filter'],'min_resized_width':24,'min_resized_height':16,'max_characters':24}
    for q in source['splits']['train_all']:
        counts['source_train_all']+=1;iid=q['image_id'];t=norm(q['target']);question=q['question'].lower()
        if iid in blocked or pixel(iid) in blocked_pixels:continue
        counts['not_current_or_held_images']+=1
        if sum(norm(a)==t for a in q['answers'])<8:continue
        if not re.fullmatch(r'[a-z0-9]{3,24}',t) or not re.search(r'[a-z]',t):continue
        if not re.search(r'\b(brand|name|say|says|read|reads|written|word|words|text|logo|spell|letters|letter)\b',question):continue
        if re.search(r'\b(color|colour|time|many|much|price|language)\b',question):continue
        by_image[iid].append(q);counts['native_literal_high_consensus_qa']+=1
    rows=[];used=set();targets=Counter()
    for iid in sorted(by_image,key=lambda x:content_sha(['native_ocr_correction_v1:image',x])):
        if iid not in raw['imgs']:continue
        im=source['images'][iid];meta=raw['imgs'][iid]
        if (im['width'],im['height'])!=(meta['width'],meta['height']):continue
        rh,rw=smart_resize(im['height'],im['width'],factor=28,min_pixels=ref['min_pixels'],max_pixels=1024*28*28)
        annotations=defaultdict(list)
        for aid in raw['imgToAnns'][iid]:
            a=raw['anns'][aid];annotations[norm(str(a.get('utf8_string','')).strip())].append(a)
        for q in sorted(by_image[iid],key=lambda q:content_sha(['native_ocr_correction_v1:qa',q['question_id']])):
            t=norm(q['target']);matches=annotations.get(t,[])
            if len(matches)!=1:continue
            value,reason=region(matches[0],im['width'],im['height'],rw,rh,policy)
            if reason or targets[t]>=8 or pixel(iid) in used:continue
            targets[t]+=1;used.add(pixel(iid));f=freq[t]
            rows.append({**q,'fold':'training','stratum':'singleword','answer_frequency':f,
                         'frequency_slice':'rare_1_5' if f<=5 else 'freq_6_20' if f<=20 else 'frequent_gt20',
                         'ocr_evidence':value,'ocr_annotation_id':matches[0]['id'],'original_target_unchanged':True})
            break
    rows.sort(key=lambda q:content_sha(['native_ocr_correction_v1:execution',q['question_id']]))
    images={q['image_id']:{**source['images'][q['image_id']],'path':str((m_path.parent/source['images'][q['image_id']]['path']).resolve())} for q in rows}
    write_json(out/'CANDIDATE_ROWS.json',{'rows':rows,'images':images})
    result={'status':'completed_cpu_inventory','candidate_qa':len(rows),'unique_images':len(images),
            'frequency_counts':dict(Counter(q['frequency_slice'] for q in rows)),'unique_normalized_targets':len(targets),
            'max_per_target':max(targets.values(),default=0),'filter_counts':dict(counts),'policy':policy,
            'native_questions_and_labels_unchanged':True,'model_image_is_full_original':True,'model_forwards':0,'training_updates':0,
            'active_seconds':time.monotonic()-start,'manifest_sha256':sha_file(m_path),'annotation_sha256':sha_file(ann_path),
            'candidate_rows_sha256':sha_file(out/'CANDIDATE_ROWS.json')}
    write_json(out/'INVENTORY.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
