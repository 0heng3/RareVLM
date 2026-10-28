import hashlib
import json
from collections import Counter
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import read_manifest,NORMALIZER
from vlm_probe.evaluate import content_sha,lock_file,now,sha_file
from .common import write_json

def verify_guard(root):
    saved=json.loads((root/'FROZEN_GUARD.json').read_text())['protected_sha256']
    for p,digest in saved.items():
        if sha_file(Path(p))!=digest:raise ValueError('Historical source/input/weight changed:'+p)
    return len(saved)

def prepare(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(cfg['manifest_path']);m=read_manifest(manifest_path)
    if len(m['splits']['train_all'])!=32985:raise ValueError('Require full frozen training pool')
    images=m['images'];excluded={r['image_id'] for split in ['dev','final'] for r in m['splits'][split]}
    excluded_pixels={images[i]['pixel_sha256'] for i in excluded};excluded_bytes={images[i]['sha256'] for i in excluded}
    def key(row):return hashlib.sha256(f'rarevlm-alignment-v1:{cfg["selection_seed"]}:{row["question_id"]}:{images[row["image_id"]]["pixel_sha256"]}'.encode()).digest()
    rows=[];seen=set();frequency=Counter(NORMALIZER(r['target']) for r in m['splits']['train_all'])
    for r in sorted(m['splits']['train_all'],key=key):
        im=images[r['image_id']];pixel=im['pixel_sha256']
        if r['source_split']!='train' or r['image_id'] in excluded or pixel in excluded_pixels or im['sha256'] in excluded_bytes:continue
        if pixel in seen:continue
        seen.add(pixel);rows.append({**r,'train_target_frequency':frequency[NORMALIZER(r['target'])]})
        if len(rows)==cfg['audit_questions']:break
    if len(rows)!=512:raise ValueError('Insufficient image-disjoint train questions')
    for r in rows:
        im=images[r['image_id']]
        if sha_file(manifest_path.parent/im['path'])!=im['sha256']:raise ValueError('Training source image changed')
    roster={'created_at':now(),'selection':'SHA256 order;one question per distinct pixel hash;no score/answer-based selection',
        'source_manifest_sha256':m['manifest_sha256'],'source_manifest_file_sha256':sha_file(manifest_path),
        'source_train_questions':32985,'questions':rows,'images':{r['image_id']:images[r['image_id']] for r in rows},
        'image_root':str(manifest_path.parent),'excluded_dev_final_image_ids':len(excluded),
        'distinct_selected_pixel_hashes':len(seen),'uses_eval_answers':False}
    path=root/'DATA_ROSTER.json'
    if path.exists():
        old=json.loads(path.read_text());roster['created_at']=old['created_at']
    lock_file(path,roster)
    adapter=Path(cfg['adapter_dir']);model=Path(cfg['model_root'])
    records={'manifest_sha256':sha_file(manifest_path),'roster_sha256':sha_file(path),
        'adapter_files_sha256':{str(p):sha_file(p) for p in sorted(adapter.glob('*')) if p.is_file()},
        'model_provenance_sha256':sha_file(model.parent/'model_provenance.json'),
        'reference_config_sha256':sha_file(PROJECT_ROOT/cfg['reference_config']),
        'zero_new_training_updates':True}
    lock_file(root/'DATA_BINDINGS.json',records)
    guard=root/'FROZEN_GUARD.json'
    if not guard.exists():
        paths=[p for folder in ['src','configs','results'] for p in (PROJECT_ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ['.py','.json'] and 'vlm_alignment_signal' not in str(p)]
        paths += [p for p in (PROJECT_ROOT/'docs').glob('*PROTOCOL*.md') if 'ALIGNMENT_SIGNAL' not in p.name]
        paths += [p for p in model.iterdir() if p.is_file() and p.suffix in ['.json','.safetensors']]
        paths += [p for p in adapter.iterdir() if p.is_file()]
        paths += [manifest_path,model.parent/'model_provenance.json']
        paths += [p for p in [PROJECT_ROOT/'docs/RESUME.md',Path('/opt/data/private/longtail_vlm/private_notes/resume.pdf')] if p.exists()]
        write_json(guard,{'created_at':now(),'protected_sha256':{str(p):sha_file(p) for p in sorted(set(paths))}})
    verify_guard(root);return records
