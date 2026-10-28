"""Prepare same-image distinct-region candidates from the unchanged training whitelist."""
from pathlib import Path
import hashlib
import json
import math
import time

from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_region_ocr_sft_v1.data import region,bounds,excludes_other_word_centers
from vlm_region_coordinates_v1.data import pixel_box,question

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_binding_data_v1'
PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'
SOURCE=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'


def read(path):return json.loads(Path(path).read_text())


def disjoint(a,b):return a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1]


def main():
    if (ROOT/'PAIR_CANDIDATES.json').exists():raise FileExistsError('Paired candidates already prepared')
    started=time.monotonic()
    manifest=read(PARENT/'data/PIXEL_TRAIN.json');policy=read(PROJECT_ROOT/'configs/vlm_region_ocr_sft_v1.json')['region_filter']
    source_path=SOURCE/'source_data/TextOCR_0.1_train.json'
    source_sha=sha_file(source_path)
    if source_sha!=read(SOURCE/'DATA_COMPLETE.json')['annotation_sha256']:raise ValueError('Official source changed')
    raw=read(source_path);pairs=[];rejected={}
    for a in manifest['splits']['train_all']:
        if a['study_source']!='region_aux':continue
        iid=a['image_id'];im=manifest['images'][iid];width,height=a['actual_processed_dimensions']
        aids=raw['imgToAnns'][iid];boxes={}
        for aid in aids:
            try:boxes[aid]=bounds(raw['anns'][aid]['points'],im['width'],im['height'])
            except (ValueError,KeyError,TypeError):pass
        choices=[]
        for aid in aids:
            if aid==a['annotation_id']:continue
            b,error=region(raw['anns'][aid],im['width'],im['height'],width,height,policy)
            if b is None:continue
            left,right=a['target'].casefold(),b['target'].casefold()
            if left==right or left in right or right in left:continue
            if not disjoint(a['box_xyxy_original_pixels'],b['box_xyxy_original_pixels']):continue
            others=[box for key,box in boxes.items() if key!=aid]
            if not excludes_other_word_centers(b['box_xyxy_original_pixels'],others):continue
            ac=a['box_xyxy_original_pixels'];bc=b['box_xyxy_original_pixels']
            distance=math.hypot(((ac[0]+ac[2])-(bc[0]+bc[2]))/(2*im['width']),
                ((ac[1]+ac[3])-(bc[1]+bc[3]))/(2*im['height']))
            choices.append((distance,str(aid),b))
        if not choices:
            rejected[iid]='No second region meeting existing geometry, disjointness, and distinct-text rules';continue
        distance,aid,b=min(choices,key=lambda x:(x[0],x[1]))
        box=pixel_box(b['box_xyxy_original_pixels'],im['width'],im['height'],width,height)
        b={**b,'annotation_id':aid,'image_id':iid,'question_id':'textocr_v01:'+aid+':paired_pixel',
            'question':question(box,width,height,'pixel_dimensions'),'prompt_suffix':a['prompt_suffix'],
            'coordinate_frame':'pixel_dimensions','actual_processed_dimensions':[width,height],
            'box_xyxy_processed_pixels':box,'source_split':'train',
            'label_source':'unchanged official TextOCR transcription'}
        assert a['image_id']==b['image_id'] and disjoint(a['box_xyxy_original_pixels'],b['box_xyxy_original_pixels'])
        assert a['target'].casefold()!=b['target'].casefold()
        pairs.append({'pair_id':iid,'source_image_id':iid,'a':a,'b':b,'normalized_center_distance':distance})
    pairs.sort(key=lambda q:hashlib.sha256(('region-binding-v1:20261020:'+q['pair_id']).encode()).hexdigest())
    result={'status':'prepared_for_short_coordinate_switch_probe','training_updates':0,
        'candidate_pairs':len(pairs),'source_images':1600,'rejected_images':len(rejected),
        'selection':'All original admitted1600 training images; closest valid distinct nonoverlapping official word, no prediction inputs',
        'scope':'Training-only candidates; no expert full-pool audit, new model performance, or training admission claim',
        'expected_next':'Same-image two-coordinate zero-update probe before registering any new optimizer study',
        'images':{q['source_image_id']:manifest['images'][q['source_image_id']] for q in pairs},
        'pairs':pairs,'rejected':rejected,'annotation_sha256':source_sha,
        'source_manifest_sha256':sha_file(PARENT/'data/PIXEL_TRAIN.json'),
        'source_geometry_sha256':sha_file(SOURCE_ROOT/'vlm_region_ocr_sft_v1/data.py'),
        'source_code_sha256':sha_file(Path(__file__)),'seconds':time.monotonic()-started}
    write_json(ROOT/'PAIR_CANDIDATES.json',result)
    write_json(ROOT/'SUMMARY.json',{k:result[k] for k in ('status','training_updates','candidate_pairs','source_images','rejected_images','selection','scope','expected_next','seconds')})
    print(json.dumps(read(ROOT/'SUMMARY.json'),ensure_ascii=False))


if __name__=='__main__':main()
