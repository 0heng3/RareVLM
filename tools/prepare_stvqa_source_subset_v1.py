"""Train-only first-shard corpus, mapped to original subtrain identities."""
from collections import Counter,defaultdict
import hashlib
import io
import re
import unicodedata
from pathlib import Path
import time
import pyarrow.parquet as pq
from PIL import Image
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import read

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_stvqa_source_v1'
BASE=ROOT.parent


def literal(value):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(value)).casefold()).strip(' .?!"')


def main():
    if (ROOT/'DATA.json').exists():raise FileExistsError('Source subset already prepared')
    started=time.monotonic();upstream=read(ROOT/'UPSTREAM_SUBTRAIN.json')['rows'];lookup=defaultdict(list)
    source=ROOT/'source/st_vqa/train-00000-of-00002-8f04db983951cc0f.parquet'
    if sha_file(source)!='a7a8dbbe2d4238ec3cb2419189b2ad49dd954d1c725e8acc6d30a05bd681957c':raise ValueError('Pinned first shard differs')
    for q in upstream:
        for answer in q['reference_answers']:lookup[(literal(q['question']),literal(answer))].append(q)
    native_path=BASE/'vlm_deep_v2/prepared/manifest.json';current_path=BASE/'vlm_late_rank_matched_dpo_v1/data/MANIFEST.json'
    known={**read(native_path)['images'],**read(current_path)['images']}
    blocked_pixels={m['pixel_sha256'] for m in known.values() if 'pixel_sha256' in m}
    blocked_bytes={m['sha256'] for m in known.values() if 'sha256' in m}
    records=[];counts=Counter();families=Counter();seen_paths=set();seen_pixels=set();dimension_examples=[]
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for index,item in enumerate(batch.to_pylist()):
            counts['source_images']+=1
            if len(item['images'])!=1:counts['not_single_image']+=1;continue
            matches=[]
            for text in item['texts']:
                if text.get('source')!='ST-VQA':continue
                key=(literal(text['user'].splitlines()[0]),literal(text['assistant']))
                found=lookup.get(key,[])
                if found:matches.append(found)
            if not matches:counts['no_original_train_match']+=1;continue
            identities=set(q['image_path'] for q in matches[0])
            for values in matches[1:]:identities.intersection_update(q['image_path'] for q in values)
            if len(identities)!=1:counts['ambiguous_original_image']+=1;continue
            image_path=next(iter(identities));family=image_path.split('/')[0];families[family]+=1
            if family.casefold() in {'coco-text','visualgenome','iiit_text'}:
                counts['excluded_coco_vg_iiit_source']+=1;continue
            if image_path in seen_paths:counts['same_original_scene']+=1;continue
            qs={q['question_id']:q for values in matches for q in values if q['image_path']==image_path}
            permitted=[]
            for q in qs.values():
                target=q['reference_answers'][0].strip();question=q['question']
                if not re.search(r'\b(say|says|read|reads|written|word|words|text|letter|letters|number|numbers|date|year|spell|printed|title|name)\b',question,re.I):continue
                if re.search(r'\b(brand|company|language|color|colour|many)\b',question,re.I):continue
                if not 1<=len(target.split())<=5 or not 1<=len(target)<=64 or not any(c.isalnum() for c in target):continue
                if literal(target) in {'unanswerable','unknown','none','not available','answer not provided'}:continue
                permitted.append(q)
            if not permitted:counts['no_fixed_reading_question']+=1;continue
            q=min(permitted,key=lambda v:content_sha(['stvqa_source_v1:question',v['question_id']]))
            raw=item['images'][0]['bytes'];image=Image.open(io.BytesIO(raw));fmt=image.format;image=image.convert('RGB')
            if image.size!=(q['image_width'],q['image_height']):
                counts['mirror_dimensions_differ_from_upstream']+=1
                if len(dimension_examples)<8:dimension_examples.append({'image_path':image_path,'mirror':list(image.size),'upstream':[q['image_width'],q['image_height']]})
                continue
            raw_sha=hashlib.sha256(raw).hexdigest();pixel_sha=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            if raw_sha in blocked_bytes or pixel_sha in blocked_pixels:counts['known_image_or_pixel_overlap']+=1;continue
            if pixel_sha in seen_pixels:counts['duplicate_source_pixels']+=1;continue
            seen_paths.add(image_path);seen_pixels.add(pixel_sha)
            row={'question_id':'stvqa_subtrain:'+q['question_id'],'image_id':'stvqa_scene:'+image_path,
                 'question':q['question'],'target':q['reference_answers'][0].strip(),'answers':q['reference_answers'],
                 'source_split':'MMF_subtrain','source_image_path':image_path,'source_family':family,
                 'upstream_question_id':q['question_id'],'original_reference_answers':q['reference_answers'],
                 'stratum':'singleword' if len(q['reference_answers'][0].split())==1 else 'multiword'}
            records.append((row,raw,fmt,raw_sha,pixel_sha,image.size))
    records.sort(key=lambda v:content_sha(['stvqa_source_v1:scene',v[0]['source_image_path']]))
    held=records[:256];pool=records[256:256+1536];selected=held+pool;rows=[];images={}
    for index,(row,raw,fmt,raw_sha,pixel_sha,size) in enumerate(selected):
        row={**row,'cohort':'source_holdout' if index<len(held) else 'source_candidate'}
        suffix='.jpg' if fmt=='JPEG' else '.png' if fmt=='PNG' else '.img'
        dest=ROOT/'images'/(raw_sha+suffix);dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
        images[row['image_id']]={'path':str(dest),'sha256':raw_sha,'pixel_sha256':pixel_sha,'width':size[0],'height':size[1]};rows.append(row)
    candidate=[q for q in rows if q['cohort']=='source_candidate'];targets=Counter();training=[]
    for q in candidate:
        key=literal(q['target'])
        if len(training)<128 and targets[key]<2:training.append(q['question_id']);targets[key]+=1
    inventory={'status':'completed_source_subset_inventory','source_images':counts['source_images'],'eligible_images':len(records),
        'candidate_images':len(candidate),'source_holdout_images':len(held),'future_train_slots':len(training),
        'future_train_unique_targets':len(targets),'filter_counts':dict(counts),'matched_source_families':dict(families),
        'dimension_mismatch_examples':dimension_examples,'protected_exact_pixels':len(blocked_pixels),
        'known_source_identity_and_exact_pixel_isolation_passed':True,'near_duplicates_and_pretraining_overlap_unknown':True,
        'selected_families':dict(Counter(q['source_family'] for q in rows)),'source_scope':'Pinned first shard only; not representative of full benchmark',
        'original_train_only':True,'test_or_val_answers_read':0,'fake_ten_votes_added':False,'model_calls':0,'optimizer_updates':0,
        'source_data_count_gate':len(training)==128 and len(targets)>=96 and len(held)==256,'active_seconds':time.monotonic()-started}
    write_json(ROOT/'DATA.json',{'rows':rows,'images':images,'training_slot_ids':training,'references_are_individual_answers_not_votes':True})
    write_json(ROOT/'SOURCE_INVENTORY.json',inventory)
    write_json(ROOT/'SOURCE_SUBSET_LOCK.json',{'files':{str(p):sha_file(p) for p in [Path(__file__),source,ROOT/'UPSTREAM_SUBTRAIN.json',ROOT/'DATA.json',native_path,current_path]},
        'model_calls':0,'training_admitted':False})
    print(inventory,flush=True)


if __name__=='__main__':main()
