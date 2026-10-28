"""Official training images only; grouped split and matched supervision slots."""
from collections import Counter,defaultdict
import hashlib
import math
from pathlib import Path
import re

from vlm_deep.common import content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import read
from .initialization import ROOT

SOURCE=ROOT.parent/'iiit5k_source_v1/raw/IIIT5K'
QUESTION='Read the main text in this image. Return only English letters and digits, without spaces or punctuation.'


def canonical(text):
    import unicodedata
    return unicodedata.normalize('NFKC',str(text)).casefold().strip()


def cell(q):
    f=q['sft_exact_answer_frequency']
    return str((len(q['target'])>5,any(c.isdigit() for c in q['target']),0 if f==0 else 1 if f<=5 else 2))


def inventory():
    from PIL import Image,ImageDraw,ImageOps,ImageFont
    from scipy.io import loadmat
    if (ROOT/'INVENTORY.json').exists():raise FileExistsError('Inventory already frozen')
    words=loadmat(SOURCE/'traindata.mat',simplify_cells=True)['traindata']
    chars=loadmat(SOURCE/'trainCharBound.mat',simplify_cells=True)['trainCharBound']
    annotations={str(q['ImgName']):q for q in chars}
    original=read(ROOT.parent/'vlm_deep_v2/prepared/manifest.json')
    frequencies=Counter(canonical(q['target']) for q in original['splits']['train_all'])
    records=[];counts=Counter();pixels=set()
    for q in sorted(words,key=lambda q:content_sha(['iiit_scope_v1:deduplicate',str(q['ImgName'])])):
        name=str(q['ImgName']);target=str(q['GroundTruth']);a=annotations.get(name)
        if not re.fullmatch('[a-zA-Z0-9]{2,20}',target):counts['outside_declared_charset_length']+=1;continue
        if a is None or canonical(a['chars'])!=canonical(target):counts['word_character_text_disagreement']+=1;continue
        path=SOURCE/name
        with Image.open(path) as raw:image=raw.convert('RGB')
        boxes=a['charBB'].reshape(-1,4)
        if len(boxes)!=len(str(a['chars'])) or any(x<0 or y<0 or w<=0 or h<=0 or x+w>image.width+1 or y+h>image.height+1 for x,y,w,h in boxes):
            counts['invalid_character_boxes']+=1;continue
        if max(image.size)/min(image.size)>30:counts['aspect_ratio_over30']+=1;continue
        pixel=hashlib.sha256(image.tobytes()).hexdigest()
        if pixel in pixels:counts['duplicate_decoded_pixel']+=1;continue
        pixels.add(pixel);family=Path(name).stem.rsplit('_',1)[0]
        records.append({'question_id':'iiit5k:'+name,'image_id':'iiit5k:'+name,'filename_family':'iiit5k:family:'+family,
            'official_relative_path':name,'raw_path':str(path),'raw_sha256':sha_file(path),'raw_pixel_sha256':pixel,'raw_size':list(image.size),
            'target':target,'character_text':str(a['chars']),'character_boxes':boxes.tolist(),'question':QUESTION,
            'sft_exact_answer_frequency':frequencies[canonical(target)],'frequency_slice':'zero' if frequencies[canonical(target)]==0 else 'rare_1_5' if frequencies[canonical(target)]<=5 else 'more_than5',
            'heldout_family':int(content_sha('iiit_scope_v1:family:'+family)[:8],16)%5==0,'training_domain':'ocr'})
    def unique(rows,salt):
        seen=set();out=[]
        for q in sorted(rows,key=lambda q:content_sha([salt,q['question_id']])):
            t=canonical(q['target'])
            if t not in seen:out.append(q);seen.add(t)
        return out
    holdout=[q for q in records if q['heldout_family']];blocked_words={canonical(q['target']) for q in holdout}
    validation=unique(holdout,'iiit_scope_v1:validation')[:256]
    pool=unique([q for q in records if not q['heldout_family'] and canonical(q['target']) not in blocked_words],'iiit_scope_v1:train')
    if len(validation)!=256:raise ValueError('Need256 fixed official validation words')
    cells=defaultdict(list)
    for q in pool:cells[cell(q)].append(q)
    capacity={k:len(v)//4 for k,v in cells.items() if len(v)>=4};total=sum(capacity.values())
    if total<128:raise ValueError('Need512 stratified different training words')
    ideal={k:128*v/total for k,v in capacity.items()};units={k:int(v) for k,v in ideal.items()}
    for k in sorted(units,key=lambda k:(-(ideal[k]-units[k]),k))[:128-sum(units.values())]:units[k]+=1
    dense=[];sparse=[]
    for k,n in sorted(units.items()):
        selected=cells[k][:4*n];dense.extend(selected);sparse.extend(selected[:n])
    dense.sort(key=lambda q:content_sha(['iiit_scope_v1:order',q['question_id']]))
    if len(dense)!=512 or len(sparse)!=128:raise ValueError('Expected512/128 words')
    replay=read(ROOT.parent/'vlm_replay_diversity_v1/DATA.json');hardids={q['question_id'] for q in replay['hard_rows']}
    def flatten(x):
        if isinstance(x,dict):
            if 'question_id' in x and 'target' in x:yield x
            else:
                for v in x.values():yield from flatten(v)
        elif isinstance(x,list):
            for v in x:yield from flatten(v)
    native={q['question_id']:q for q in flatten(replay['training_rows']) if q['question_id'] not in hardids}
    valwords={canonical(q['target']) for q in validation};native=[q for q in native.values() if not valwords.intersection(re.findall('[a-z0-9]+',canonical(q['target'])))]
    native=sorted(native,key=lambda q:content_sha(['iiit_scope_v1:native',q['question_id']]))[:256]
    if len(native)!=256:raise ValueError('Insufficient nonleaking native replay')
    native=[{**q,'training_domain':'native'} for q in native]
    inventory={'status':'inventory_ready','official_train_rows':len(words),'eligible':len(records),'filter_counts':dict(counts),
        'dense':dense,'sparse_question_ids':[q['question_id'] for q in sparse],'validation':validation,'native':native,
        'source':read(ROOT.parent/'iiit5k_source_v1/SOURCE.json'),'lexicons_used':False,'official_test_labels_or_images_read':False,
        'matched_cells':{k:{'dense':4*n,'sparse':n} for k,n in units.items()},'native_replay_available_after_word_filter':len(native)}
    write_json(ROOT/'INVENTORY.json',inventory)
    try:font=ImageFont.load_default(size=17)
    except TypeError:font=ImageFont.load_default()
    display=[('validation',validation),('training',sorted(dense,key=lambda q:content_sha(['iiit:preview',q['question_id']]))[:32])]
    boards=[]
    for kind,rows in display:
        for number,begin in enumerate(range(0,len(rows),16)):
            canvas=Image.new('RGB',(1040,960),'#eef2f6');draw=ImageDraw.Draw(canvas)
            for j,q in enumerate(rows[begin:begin+16]):
                x=(j%2)*520;y=(j//2)*120;draw.text((x+8,y+4),f'{kind}#{begin+j}: {q["target"]}',font=font,fill='black')
                image=ImageOps.contain(Image.open(q['raw_path']).convert('RGB'),(500,80),Image.Resampling.BICUBIC);canvas.paste(image,(x+8+(500-image.width)//2,y+32+(80-image.height)//2))
            path=ROOT/'audit'/f'{kind}_{number:02d}.png';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);boards.append(str(path))
    write_json(ROOT/'AUDIT_BOARDS.json',{'boards':boards,'validation_rows':256,'training_previews':32,'model_predictions_used':False})
    print('INVENTORY_READY',{'eligible':len(records),'filters':dict(counts),'dense':len(dense),'sparse':len(sparse),'validation':len(validation),'native':len(native),'boards':len(boards)},flush=True)


def prepare():
    import torch
    from PIL import Image
    from vlm_probe.evaluate import processor_for,inference_row
    from vlm_deep.common import encode
    from vlm_grpo_iteration_v1.common import chosen_sample
    from . import initialization
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Study already frozen')
    torch.set_num_threads(2);inventory=read(ROOT/'INVENTORY.json');audit=read(ROOT/'AUDIT_REVIEW.json')
    if audit['model_predictions_used'] or len(audit['supported_validation_ordinals'])<192:raise ValueError('Need pre-prediction audit')
    reference=read(ROOT.parent/'vlm_reference_replay_v1/PLAN.json');proc=processor_for(Path(reference['model_root']),read(reference['reference_config']),1024)
    previous=read(ROOT.parent/'vlm_replay_diversity_v1/DATA.json');old=read(ROOT.parent/'vlm_literal_case_sft_v1/DATA.json');images=dict(previous['images']);dense=[];validation=[]
    for kind,rows in [('training',inventory['dense']),('validation',inventory['validation'])]:
        for source in rows:
            q=dict(source)
            if sha_file(q['raw_path'])!=q['raw_sha256']:raise ValueError('Official crop changed')
            with Image.open(q['raw_path']) as raw:image=raw.convert('RGB')
            raw_size=list(image.size)
            if min(image.size)<28:
                factor=28/min(image.size);image=image.resize((math.ceil(image.width*factor),math.ceil(image.height*factor)),Image.Resampling.BICUBIC)
            path=ROOT/'data/images'/Path(q['official_relative_path']).name;path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            images[q['image_id']]={'path':str(path),'sha256':sha_file(path),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'width':image.width,'height':image.height}
            item=encode(proc,inference_row(q),path,False)
            q.update(fold='iiit_'+kind,original_crop_size=raw_size,actual_input_size=list(image.size),expected_prefix_tokens=int(item['input_ids'].shape[1]),
                     expected_grid=item['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(item['input_ids'].tolist()))
            (dense if kind=='training' else validation).append(q)
    native=inventory['native'];sparseids=set(inventory['sparse_question_ids']);sparse=[q for q in dense if q['question_id'] in sparseids]
    for field in ['filename_family','raw_pixel_sha256']:
        if {q[field] for q in dense}&{q[field] for q in validation}:raise ValueError('OCR train/validation overlap: '+field)
    valwords={canonical(q['target']) for q in validation}
    if valwords&{canonical(q['target']) for q in dense}:raise ValueError('OCR target leakage')
    if any(valwords.intersection(re.findall('[a-z0-9]+',canonical(q['target']))) for q in native):raise ValueError('Native answer word leakage')
    evaluation={n:old['evaluations'][n] for n in ['seen_validation','retention']};evaluation.update(hard=sparse,literal_validation=validation)
    native_hold={q['image_id'] for n in ['seen_validation','retention'] for q in evaluation[n]}
    if {q['image_id'] for q in native}&native_hold:raise ValueError('Native replay overlaps native evaluation image')
    training=dense+native;cpu={}
    for q in training:
        target={**q,'target':canonical(q['target'])} if q['training_domain']=='ocr' else q
        item=chosen_sample(proc,target,images)
        if item['input_ids'].shape[1]>2048:raise ValueError('Training input exceeds cap')
        if content_sha(item['input_ids'][:,:q['expected_prefix_tokens']].tolist())!=q['inference_input_sha256']:raise ValueError('Prefix changed')
        cpu[q['question_id']]={k:content_sha(item[k].tolist()) for k in ['input_ids','labels']};cpu[q['question_id']]['assistant_ids']=item['labels'][item['labels'].ne(-100)].tolist()
    bycell=defaultdict(list)
    for q in sparse:bycell[cell(q)].append(q['question_id'])
    cursors=Counter();dense_order=dense*2;native_order=native*2;dense_updates=[];sparse_updates=[]
    for step in range(64):
        oq=dense_order[step*16:step*16+16];nq=native_order[step*8:step*8+8];dense_micro=[];sparse_micro=[];oi=ni=0
        for kind in ['native','ocr','ocr','native','ocr','ocr']:
            if kind=='native':ids=[q['question_id'] for q in nq[ni:ni+4]];ni+=4;dense_micro.append(ids);sparse_micro.append(ids)
            else:
                chunk=oq[oi:oi+4];oi+=4;dense_micro.append([q['question_id'] for q in chunk]);ids=[]
                for q in chunk:
                    key=cell(q);ids.append(bycell[key][cursors[key]%len(bycell[key])]);cursors[key]+=1
                sparse_micro.append(ids)
        dense_updates.append({'micros':dense_micro});sparse_updates.append({'micros':sparse_micro})
    schedules={'dense_language':dense_updates,'dense_visual':dense_updates,'sparse_language':sparse_updates}
    for role,updates in schedules.items():
        count=Counter(qid for u in updates for m in u['micros'] for qid in m)
        actual=sparse if role=='sparse_language' else dense
        if any(count[q['question_id']]!=(8 if role=='sparse_language' else 2) for q in actual) or any(count[q['question_id']]!=2 for q in native):raise ValueError('Exposure contract differs')
    data={'training_rows':training,'images':images,'evaluations':evaluation,'sparse_question_ids':sorted(sparseids),
          'supported_validation_qids':[validation[i]['question_id'] for i in audit['supported_validation_ordinals']]}
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'CPU_INPUTS.json',cpu);write_json(ROOT/'SCHEDULE.json',schedules)
    from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
    old_plan=read(ROOT.parent/'vlm_literal_case_sft_v1/PLAN.json');files=dict(old_plan['files'])
    protocol=PROJECT_ROOT/'docs/VLM_IIIT_SCOPE_V1_PROTOCOL.md'
    sources=[Path(__file__),Path(initialization.__file__),Path(__file__).parent/'__init__.py',Path(__file__).parent/'train.py',PROJECT_ROOT/'tools/run_iiit_scope_v1.py',protocol,
             ROOT/'INITIALIZATION.json',ROOT/'INVENTORY.json',ROOT/'AUDIT_REVIEW.json',ROOT/'DATA.json',ROOT/'CPU_INPUTS.json',ROOT/'SCHEDULE.json',
             SOURCE/'traindata.mat',SOURCE/'trainCharBound.mat',ROOT.parent/'iiit5k_source_v1/SOURCE.json',ROOT.parent/'vlm_replay_diversity_v1/DATA.json']
    sources+=list((ROOT/'initial_adapter').glob('*'))
    for path in sources:files[str(path)]=sha_file(path)
    used={q['image_id'] for q in training+[q for xs in evaluation.values() for q in xs]}
    plan={'status':'prepared','files':files,'new_image_files':{images[i]['path']:images[i]['sha256'] for i in used},'packages':old_plan['packages'],
          'base_weight_files':old_plan['base_weight_files'],'initial_lora_sha256':read(ROOT/'INITIALIZATION.json')['composite_lora_sha256'],
          'roles':['dense_language','dense_visual','sparse_language'],'updates':64,'lr':2e-4,'warmup_updates':8,'lr_schedule':'linear_decay_64_horizon',
          'optimized_parameters_per_arm':1843200,'stored_adapter_parameters':3686400,'dense_unique_ocr':512,'sparse_unique_ocr':128,'native_unique':256,
          'qa_exposures_per_arm':1536,'source_loss_weights':{'ocr':.5,'native':.5},'evaluation_per_arm':960,'new_baseline_generations':960,
          'primary_ocr_metric':'official_test3000_lexicon_free_ascii36_exact','old_literal_metric_unchanged':True,'official_test_opened':False,
          'official_test_models':4,'official_test_generations':12000,'defer_test_until_all_final64':True,'new_generations_total':15840,
          'preview_qids':[q['question_id'] for q in sparse[:8]]}
    write_json(ROOT/'PLAN.json',plan);print('PREPARED',{'training_catalog':768,'validation':256,'supported_validation':len(data['supported_validation_qids']),'roles':plan['roles'],'exposures_per_role':1536},flush=True)


def prepare_test():
    """Only open official test after all three fixed Final64 adapters exist."""
    import tarfile
    from PIL import Image
    from scipy.io import loadmat
    from vlm_deep.common import encode
    from vlm_probe.evaluate import processor_for,inference_row
    if (ROOT/'TEST_PLAN.json').exists():return
    plan=read(ROOT/'PLAN.json')
    final={r:read(ROOT/'runs'/r/'train/TRAIN_COMPLETE.json') for r in plan['roles']}
    if any(q['updates']!=64 for q in final.values()):raise ValueError('All Final64 required before opening test')
    archive=ROOT.parent/'iiit5k_source_v1/IIIT5K-Word_V3.0.tar.gz';source=read(ROOT.parent/'iiit5k_source_v1/SOURCE.json')
    if sha_file(archive)!=source['sha256']:raise ValueError('Official archive changed')
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and ('/test/' in member.name or member.name.lower().endswith('/testdata.mat')):
                path=(SOURCE.parent/member.name).resolve();path.relative_to(SOURCE.parent.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(member).read())
    annotations=loadmat(SOURCE/'testdata.mat',simplify_cells=True)['testdata']
    if len(annotations)!=3000:raise ValueError('Expected all3000 official test images')
    reference=read(ROOT.parent/'vlm_reference_replay_v1/PLAN.json');processor=processor_for(Path(reference['model_root']),read(reference['reference_config']),1024)
    original=read(ROOT.parent/'vlm_deep_v2/prepared/manifest.json');frequency=Counter(canonical(q['target']) for q in original['splits']['train_all'])
    training=read(ROOT/'DATA.json')['training_rows'];train_targets={canonical(q['target']) for q in training};rows=[];images={}
    for annotation in annotations:
        name=str(annotation['ImgName']);target=str(annotation['GroundTruth']);path=SOURCE/name
        with Image.open(path) as raw:image=raw.convert('RGB')
        size=list(image.size)
        if min(image.size)<28:
            factor=28/min(image.size);image=image.resize((math.ceil(image.width*factor),math.ceil(image.height*factor)),Image.Resampling.BICUBIC)
        destination=ROOT/'test_images'/Path(name).name;destination.parent.mkdir(exist_ok=True);image.save(destination)
        iid='iiit5k:'+name;images[iid]={'path':str(destination),'sha256':sha_file(destination),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'width':image.width,'height':image.height}
        q={'question_id':iid,'image_id':iid,'filename_family':'iiit5k:family:'+Path(name).stem.rsplit('_',1)[0],'question':QUESTION,'target':target,'fold':'official_test',
           'raw_size':size,'actual_input_size':list(image.size),'training_domain':'ocr','sft_exact_answer_frequency':frequency[canonical(target)],'extra_train_exact_target_seen':canonical(target) in train_targets}
        item=encode(processor,inference_row(q),destination,False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]),expected_grid=item['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(item['input_ids'].tolist()));rows.append(q)
    write_json(ROOT/'TEST_DATA.json',{'images':images,'evaluations':{'official_test':rows}})
    write_json(ROOT/'TEST_PLAN.json',{'status':'prepared_after_all_final64','data_sha256':sha_file(ROOT/'TEST_DATA.json'),'archive_sha256':source['sha256'],
        'parent_plan_sha256':sha_file(ROOT/'PLAN.json'),'rows':3000,'all_models_frozen_before_test_open':{r:q['final_lora_sha256'] for r,q in final.items()},
        'test_images_sha256':{v['path']:v['sha256'] for v in images.values()},'test_labels_used_for_configuration':False})


if __name__=='__main__':inventory()
