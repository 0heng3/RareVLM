"""Prepare the whole registered official-native confirmation pool on CPU only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from vlm_object_presence.data import PROMPT_SUFFIX, assert_disjoint, content_sha, sha_file, write_json
from vlm_object_presence.prepare import render_audit
from vlm_object_presence_v2.data import fresh_cpu_encode
from .data import (AUDIT_ORDER_SALT, AUDIT_SALT, CATEGORIES, CLASS_BALANCE_SALT,
    DEDUP_SALT, NEGATIVE_SALT, build_rows, class_balanced_subset, corpus_counts,
    freeze_audit, select_verified_images)
from .download import download_one

PROTOCOL_SHA = '691012e3f040856f73943013b374357fc468f075ea30608d16a1d976c897d1cd'
INVENTORY_SHA = 'ee71c874d4c141ccfe4886f1562a5e0017264b18b76ba00cbec5cebb78bbe7c3'
ANNOTATION_SHA = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'


def freeze_or_match(path, value):
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing external-data definition differs: {path}')
    else:
        write_json(path, value)


def scientific_protection(study_path):
    """Read the existing scientific set without adding this new package to it."""
    from vlm_object_study.common import source_hashes
    study = json.loads(study_path.read_text())
    actual = source_hashes()
    if actual != study['source_sha256']:
        raise ValueError('Existing v2 scientific source set or bytes changed')
    files = {**study['source_sha256'], **study['protocol_files_sha256'],
             study['config_path']: study['config_file_sha256']}
    for name, digest in files.items():
        if sha_file(name) != digest:
            raise ValueError(f'Existing scientific file changed: {name}')
    return {'study_path': str(study_path), 'study_sha256': sha_file(study_path),
            'source_set_exactly_matches': True, 'files_sha256': files}


def collect_boundary_hashes(source_files):
    files, pixels, counts = set(), set(), {}
    for filename, expected in source_files.items():
        path = Path(filename)
        if sha_file(path) != expected:
            raise ValueError(f'Immutable metadata boundary input changed: {path}')
        # Annotation geometry is not an old-image identity manifest.
        if path.name.startswith('instances_'):
            continue
        document = json.loads(path.read_text())
        local_files, local_pixels = set(), set()
        def visit(value):
            if isinstance(value, dict):
                pixel = value.get('pixel_sha256')
                if isinstance(pixel, str) and len(pixel) == 64:
                    local_pixels.add(pixel)
                    for key in ('sha256', 'file_sha256', 'image_sha256'):
                        digest = value.get(key)
                        if isinstance(digest, str) and len(digest) == 64:
                            local_files.add(digest)
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, (dict, list)):
                        visit(child)
        visit(document)
        files.update(local_files)
        pixels.update(local_pixels)
        counts[filename] = {'file_shas': len(local_files), 'pixel_shas': len(local_pixels)}
    return files, pixels, counts


def validate_metadata(inventory, annotation):
    categories = {int(c['id']): c['name'] for c in annotation['categories'] if c['name'] in CATEGORIES}
    if len(categories) != 18:
        raise ValueError('Official 18-category mapping differs')
    excluded = set(inventory['boundary_policy']['excluded_val2017_image_ids'])
    annotations = defaultdict(list)
    for item in annotation['annotations']:
        annotations[item['image_id']].append(item)
    expected = {}
    for image in annotation['images']:
        iid = f'{image["id"]:012d}'
        if iid in excluded:
            continue
        anns = annotations[image['id']]
        positive = defaultdict(list)
        for a in anns:
            if (a['category_id'] in categories and not a['iscrowd'] and min(a['bbox'][2:]) >= 24
                and a['area']/(image['width']*image['height']) >= .005):
                positive[a['category_id']].append(a)
        absent = set(categories) - {a['category_id'] for a in anns}
        if positive and len(absent) >= len(positive):
            expected[iid] = (image, positive, anns)
    rows = inventory['eligible_image_inventory']
    if len(rows) != 1791 or {r['coco_image_id'] for r in rows} != set(expected):
        raise ValueError('Complete metadata candidate population differs')
    qa = 0
    for row in rows:
        image, positive, anns = expected[row['coco_image_id']]
        if (row['width'], row['height'], row['file_name'], row['source_coco_url']) != (
                image['width'], image['height'], image['file_name'], image['coco_url']):
            raise ValueError('Native metadata/source URL differs')
        actual = {p['category_id']: p['qualifying_instances'] for p in row['positive_categories']}
        if set(actual) != set(positive) or set(row['all_annotated_category_ids_including_crowd']) != {a['category_id'] for a in anns}:
            raise ValueError('Positive or absent-all category metadata differs')
        for category, boxes in actual.items():
            by_id = {a['id']: a for a in positive[category]}
            if {b['annotation_id'] for b in boxes} != set(by_id):
                raise ValueError('Qualified instance inventory differs')
            for box in boxes:
                original = by_id[box['annotation_id']]
                if (box['bbox_native_metadata_xywh'] != original['bbox']
                    or box['segmentation_area_fraction'] != original['area']/(image['width']*image['height'])):
                    raise ValueError('Qualified same-instance geometry differs')
        qa += 2*len(positive)
    if qa != 4372:
        raise ValueError('Pre-exact-exclusion QA inventory differs')
    return rows, categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--old-study-lock', type=Path, required=True)
    parser.add_argument('--download-workers', type=int, default=12)
    parser.add_argument('--encoding-workers', type=int, default=8)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('All outputs must remain under the persistent boundary')
    if (root/'COMPLETION.json').exists():
        raise ValueError('External dataset is already frozen and complete')
    if sha_file(args.inventory) != INVENTORY_SHA or sha_file(args.protocol) != PROTOCOL_SHA:
        raise ValueError('Predeclared immutable inventory or protocol differs')
    old_protection = scientific_protection(args.old_study_lock)
    inventory = json.loads(args.inventory.read_text())
    annotation_path = Path(inventory['source_annotation']['path'])
    if sha_file(annotation_path) != ANNOTATION_SHA:
        raise ValueError('Official annotation JSON differs')
    candidates, category_map = validate_metadata(inventory, json.loads(annotation_path.read_text()))
    old_files, old_pixels, boundary_counts = collect_boundary_hashes(inventory['source_files_sha256'])
    v2_source = args.inventory.parent.parent/'vlm_object_presence_v2/SOURCE_LOCK.json'
    prior = json.loads(v2_source.read_text())
    model_root = Path(prior['model_root'])
    reference_path = Path(prior['reference_config']['path']).resolve()
    if sha_file(reference_path) != prior['reference_config']['sha256']:
        raise ValueError('Frozen processor reference config differs')
    for name, digest in prior['processor_files'].items():
        if sha_file(model_root/name) != digest:
            raise ValueError('Frozen processor file differs')
    if sha_file(prior['audit_font']['path']) != prior['audit_font']['sha256']:
        raise ValueError('Persistent audit font differs')
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    policy = {'schema_version': 'rarevlm.object_external.policy.v1', 'protocol_sha256': PROTOCOL_SHA,
        'inventory_sha256': INVENTORY_SHA, 'category_scope': list(CATEGORIES),
        'same_noncrowd_instance_short_side_px_at_least': 24, 'segmentation_area_fraction_at_least': .005,
        'native_jpeg_storage': True, 'stored_images_resized_or_cropped': False,
        'negative_salt': NEGATIVE_SALT,
        'negative_hash_input': '[salt,42,int(coco_image_id),actual_pixel_sha256,category_id]',
        'negative_policy': 'all annotations including crowd and small instances must lack category; same unique count as every qualified positive category',
        'dedup_salt': DEDUP_SALT, 'dedup_hash_input': '[salt,42,coco_image_id_12digit,pixel_sha256]',
        'dedup_policy': 'exclude any old file/pixel overlap; retain one new file/pixel duplicate by fixed SHA order; never replace excluded images',
        'audit_salt': AUDIT_SALT, 'audit_order_salt': AUDIT_ORDER_SALT,
        'audit_hash_input': '[salt,42,question_id,pixel_sha256]',
        'audit_rule': 'yes32 then no32, shared seen pixel set, independent [audit_order_salt,42,qid] final order',
        'class_balance_salt': CLASS_BALANCE_SALT,
        'question_template': 'Is there a {category} in the image?', 'prompt_suffix': PROMPT_SUFFIX,
        'download_source': 'https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}',
        'download_first_validated_sample': 'the six pre-image-inspection official HEAD sample IDs from immutable inventory',
        'download_workers': args.download_workers, 'attempts_per_image_per_invocation': 3,
        'request_timeout_seconds_connect_read': [10,30], 'source_redirects_allowed': False,
        'download_resume': 'complete byte/pixel/dimension verified per-image receipts only; never partial images',
        'failed_download_policy': 'status incomplete; no full manifest/audit or image replacement',
        'expected_pre_exact_candidates': 1791, 'expected_pre_exact_qa': 4372,
        'metadata_class_balance_budget_only': 3342, 'actual_post_download_balance_may_differ': True,
        'fresh_cpu_encoding': True, 'old_encoding_reused': False,
        'visual_token_budget': 1024, 'max_sequence_length': 2048, 'input_truncation': False,
        'model_predictions_used': False, 'model_forward_or_gpu_allowed': False,
        'evaluation_requires_external_admission': True, 'training_allowed': False,
        'semantic_gate': {'clear_fraction_at_least': .90, 'wilson_two_sided_95_lower_at_least': .80},
        'scope': 'independent future confirmation data only; not a frozen model study or v2-gate repair'}
    freeze_or_match(root/'PREPARATION_POLICY.json', policy)
    repo = Path(__file__).parents[2]
    code = [*sorted(Path(__file__).parent.glob('*.py')),
        repo/'tools/prepare_object_external_data.py', repo/'tests/test_object_external_data.py',
        repo/'src/vlm_object_presence/data.py', repo/'src/vlm_object_presence/prepare.py',
        repo/'src/vlm_object_presence_v2/data.py', repo/'src/vlm_public_preference/data.py',
        repo/'src/vlm_deep/common.py', repo/'src/vlm_sft/common.py', repo/'src/vlm_probe/evaluate.py']
    packages = {name: importlib.metadata.version(name) for name in ('requests','Pillow','numpy','torch','transformers')}
    input_lock = {'schema_version': 'rarevlm.object_external.preparation_input.v1',
        'protocol': {'path': str(args.protocol.resolve()), 'sha256': PROTOCOL_SHA},
        'immutable_inventory': {'path': str(args.inventory.resolve()), 'sha256': INVENTORY_SHA},
        'head_network_receipt': {'path': str(args.inventory.with_name('HEAD_RECEIPT.json').resolve()),
            'sha256': sha_file(args.inventory.with_name('HEAD_RECEIPT.json'))},
        'metadata_inputs_sha256': inventory['source_files_sha256'],
        'annotation_path': str(annotation_path), 'annotation_sha256': ANNOTATION_SHA,
        'policy_sha256': sha_file(root/'PREPARATION_POLICY.json'),
        'code_files_sha256': {str(p): sha_file(p) for p in code},
        'model_root': str(model_root), 'reference_config': {'path': str(reference_path), 'sha256': sha_file(reference_path)},
        'processor_files': prior['processor_files'], 'audit_font': prior['audit_font'],
        'old_scientific_protection': old_protection, 'old_image_boundary_counts': boundary_counts,
        'old_unique_file_shas': len(old_files), 'old_unique_pixel_shas': len(old_pixels),
        'category_map': {str(key):value for key,value in category_map.items()},
        'packages': packages, 'python': platform.python_version()}
    freeze_or_match(root/'PREPARATION_INPUT_LOCK.json', input_lock)
    print('definitions_locked_before_any_GET', sha_file(root/'PREPARATION_INPUT_LOCK.json'), flush=True)
    receipts, resumed, failures = {}, 0, {}
    def record_result(meta, result):
        nonlocal resumed
        receipt, reused = result
        if receipt['status'] == 'verified':
            receipts[meta['coco_image_id']] = receipt
            resumed += int(reused)
        else:
            failures[meta['coco_image_id']] = receipt
        if len(receipts)+len(failures) <= 6 or (len(receipts)+len(failures)) % 16 == 0:
            write_json(root/'STATUS.json', {'phase':'official_download', 'verified_images':len(receipts),
                'expected_images':len(candidates), 'failed_images':len(failures), 'verified_cache_resumed':resumed,
                'elapsed_seconds':time.monotonic()-start, 'evaluation_allowed':False})
            print('download_verified',len(receipts),'/',len(candidates),'failed',len(failures),flush=True)
    sample_ids = [row['coco_image_id'] for row in inventory['official_head_sample']['sample']]
    index = {row['coco_image_id']: row for row in candidates}
    for iid in sample_ids:
        record_result(index[iid], download_one(index[iid], root))
    if not failures:
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {pool.submit(download_one, meta, root): meta for meta in candidates if meta['coco_image_id'] not in sample_ids}
            for future in as_completed(futures):
                record_result(futures[future], future.result())
    download_summary = {'complete':not failures and len(receipts)==len(candidates),
        'expected_images':len(candidates),'verified_images':len(receipts),'failed_images':sorted(failures),
        'verified_cache_resumed':resumed, 'sample_first_image_ids':sample_ids,
        'actual_verified_image_bytes':sum(r['identity']['http_content_length'] for r in receipts.values()),
        'singlepart_md5_verified_images':sum(r['identity']['etag_md5_verified'] for r in receipts.values()),
        'non_md5_etag_images':sum(not r['identity']['etag_is_md5'] for r in receipts.values()),
        'elapsed_seconds_since_preparation_start':time.monotonic()-start}
    download_path = root/'DOWNLOAD_COMPLETION.json'
    if download_path.exists() and json.loads(download_path.read_text()).get('complete'):
        previous = json.loads(download_path.read_text())
        for key in ('complete','expected_images','verified_images','failed_images',
                    'actual_verified_image_bytes','singlepart_md5_verified_images','non_md5_etag_images'):
            if previous[key] != download_summary[key]:
                raise ValueError(f'Previously completed download inventory differs: {key}')
        download_summary = previous
    else:
        write_json(download_path, download_summary)
    if not download_summary['complete']:
        write_json(root/'STATUS.json', {'phase':'incomplete_download','detail':download_summary,
            'evaluation_allowed':False,'model_forward_calls':0})
        raise SystemExit(2)
    verified = {iid: {**r['identity'],'path':r['path'],'category_map':category_map} for iid,r in receipts.items()}
    retained, exclusions = select_verified_images(candidates, verified, old_files, old_pixels)
    excluded_index = {row['coco_image_id']: row for row in exclusions}
    freeze_or_match(root/'EXCLUSION_LOG.json', {'candidate_images':len(candidates),'retained_images':len(retained),
        'rows':[excluded_index.get(row['coco_image_id'],{'coco_image_id':row['coco_image_id'],'reasons':[]}) for row in candidates],
        'exclusion_reasons':dict(Counter(reason for row in exclusions for reason in row['reasons']))})
    rows, images = [], {}
    for meta in retained:
        created, image = build_rows(meta, verified[meta['coco_image_id']])
        rows += created
        images[created[0]['image_id']] = image
    rows.sort(key=lambda row:row['question_id'])
    assert_disjoint(rows, [])
    if len(images)<64:
        raise ValueError('Too few exact-disjoint images for the predeclared audit')
    source_lock = {'schema_version':'rarevlm.object_external.source_lock.v1',
        'preparation_input_lock':input_lock, 'preparation_input_lock_file_sha256':sha_file(root/'PREPARATION_INPUT_LOCK.json'),
        'download_completion_sha256':sha_file(root/'DOWNLOAD_COMPLETION.json'),
        'exclusion_log_sha256':sha_file(root/'EXCLUSION_LOG.json'),
        'download_receipts_sha256':{str((root/'downloads'/f'{iid}.json').resolve()):sha_file(root/'downloads'/f'{iid}.json') for iid in sorted(receipts)},
        'native_image_file_sha256':{r['path']:r['identity']['sha256'] for r in receipts.values()},
        'native_original_dimensions_verified_for_all_downloads':True,
        'near_duplicate_and_pretraining_overlap':'unknown'}
    source_lock['source_lock_sha256'] = content_sha(source_lock)
    freeze_or_match(root/'SOURCE_LOCK.json', source_lock)
    def manifest(split, selected, selected_images):
        value={'schema_version':'rarevlm.object_external.manifest.v1','split':split,
            'protocol_sha256':PROTOCOL_SHA,'source_lock_sha256':source_lock['source_lock_sha256'],
            'prompt_suffix':PROMPT_SUFFIX,'category_scope':list(CATEGORIES),
            'evaluation_requires_external_admission':True,'training_allowed':False,
            'rows':selected,'images':selected_images}
        value['manifest_sha256']=content_sha(value)
        return value
    freeze_or_match(root/'MANIFEST.json',manifest('external_confirmation',rows,images))
    balanced, balance_counts = class_balanced_subset(rows)
    balanced_ids={row['image_id'] for row in balanced}
    balanced_images={key:value for key,value in images.items() if key in balanced_ids}
    balanced_manifest=manifest('external_class_balanced',balanced,balanced_images)
    balanced_manifest.pop('manifest_sha256')
    balanced_manifest.update(diagnostic_only=True,category_balance=balance_counts,
        selection='exact full-manifest subset; per-category min(yes,no), fixed SHA per label')
    balanced_manifest['manifest_sha256']=content_sha(balanced_manifest)
    freeze_or_match(root/'CLASS_BALANCED_MANIFEST.json',balanced_manifest)
    counts={'full':corpus_counts(rows,images),'class_balanced':corpus_counts(balanced,balanced_images),
        'metadata_pre_exact_images':1791,'metadata_pre_exact_qa':4372,'metadata_class_balance_budget':3342,
        'exact_excluded_images':len(exclusions),'known_source_id_and_old_file_pixel_overlap_zero':True,
        'native_images_stored_without_resizing':True,'semantic_review':'pending_external_review',
        'evaluation_requires_external_admission':True,'evaluation_allowed':False}
    freeze_or_match(root/'SUMMARY.json',counts)
    audit=freeze_audit(rows)
    frozen={'schema_version':'rarevlm.object_external.frozen_audit.v1',
        'source_full_manifest_sha256':sha_file(root/'MANIFEST.json'),'protocol_sha256':PROTOCOL_SHA,
        'audit_salt':AUDIT_SALT,'audit_order_salt':AUDIT_ORDER_SALT,'seed':42,
        'labels':{'yes':32,'no':32},'unique_images_by_pixel_sha':64,'old_image_pixel_overlap':0,
        'review_state':'pending_external_review','rows':audit}
    frozen['audit_sha256']=content_sha(frozen)
    freeze_or_match(root/'semantic_audit/FROZEN_64.json',frozen)
    render_lock_path=root/'semantic_audit/RENDER_LOCK.json'
    if render_lock_path.exists():
        for name,digest in json.loads(render_lock_path.read_text())['files_sha256'].items():
            if sha_file(root/name)!=digest:
                raise ValueError('Fixed audit rendering changed')
    else:
        render_audit(root,audit,images)
        render_files=[p for p in (root/'semantic_audit').iterdir() if p.is_file() and p.name!='RENDER_LOCK.json']
        write_json(render_lock_path,{'native_images_unmodified':True,'overlays_for_review_only':True,
            'files_sha256':{str(p.relative_to(root)):sha_file(p) for p in sorted(render_files)}})
    write_json(root/'AUDIT_READY.json',{'ready':True,'full_manifest_sha256':sha_file(root/'MANIFEST.json'),
        'audit_sha256':sha_file(root/'semantic_audit/FROZEN_64.json'),'counts':counts,
        'cpu_encoding':'next_or_in_progress','evaluation_allowed':False})
    print('AUDIT_READY',json.dumps(counts),flush=True)
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(1)
    processor=processor_for(model_root,json.loads(reference_path.read_text()),1024)
    journal=root/'ENCODING_CHECKS.jsonl';checks={}
    if journal.exists():
        blob=journal.read_bytes();end=blob.rfind(b'\n')+1
        if end<len(blob):
            write_json(root/'CPU_JOURNAL_TAIL_REPAIR.json',{'original_sha256':sha_file(journal),
                'complete_bytes_retained':end,'incomplete_tail_bytes':len(blob)-end})
            journal.write_bytes(blob[:end])
        for line in journal.read_text().splitlines():
            item=json.loads(line)
            if item['encoding_origin']!='fresh_external_v1_cpu':
                raise ValueError('External CPU journal contains a foreign encoding record')
            checks[item['question_id']]=item
    def encode_one(row):
        value=fresh_cpu_encode(row,processor)
        value.update(encoding_origin='fresh_external_v1_cpu',v1_encoding_reused=False,v2_encoding_reused=False)
        return value
    pending=[row for row in rows if row['question_id'] not in checks]
    with journal.open('a') as handle,ThreadPoolExecutor(max_workers=args.encoding_workers) as pool:
        for number,item in enumerate(pool.map(encode_one,pending),1):
            checks[item['question_id']]=item
            handle.write(json.dumps(item,separators=(',',':'))+'\n')
            if number%128==0 or number==len(pending):
                handle.flush()
                write_json(root/'STATUS.json',{'phase':'fresh_cpu_encoding','complete':len(checks),
                    'total':len(rows),'elapsed_seconds':time.monotonic()-start,'evaluation_allowed':False})
                print('fresh_cpu_encoded',len(checks),'/',len(rows),flush=True)
    if set(checks)!={row['question_id'] for row in rows}:
        raise ValueError('Fresh CPU encoding coverage differs from the frozen full manifest')
    encoding={'questions':len(checks),'fresh_actual_cpu_encodes':len(checks),'old_encoding_records_reused':0,
        'input_ids_labels_and_pixel_hash_saved':True,'sequence_length_min':min(r['sequence_length'] for r in checks.values()),
        'sequence_length_max':max(r['sequence_length'] for r in checks.values()),
        'visual_tokens_min':min(r['visual_tokens'] for r in checks.values()),
        'visual_tokens_max':max(r['visual_tokens'] for r in checks.values()),
        'assistant_tokens':dict(Counter(r['assistant_tokens'] for r in checks.values())),
        'max_sequence_length':2048,'visual_token_budget':1024,'truncation':False,
        'model_forward_or_gpu_used':False,'journal_sha256':sha_file(journal)}
    write_json(root/'ENCODING_SUMMARY.json',encoding)
    after=scientific_protection(args.old_study_lock)
    if after!=old_protection:
        raise ValueError('Old scientific source/config/protocol changed during data preparation')
    write_json(root/'OLD_SCIENTIFIC_PROTECTION.json',{'before':old_protection,'after':after,
        'unchanged':True,'new_external_package_outside_old_registered_scientific_set':True})
    artifacts=['PREPARATION_POLICY.json','PREPARATION_INPUT_LOCK.json','SOURCE_LOCK.json',
        'DOWNLOAD_COMPLETION.json','EXCLUSION_LOG.json','MANIFEST.json','CLASS_BALANCED_MANIFEST.json',
        'SUMMARY.json','semantic_audit/FROZEN_64.json','semantic_audit/RENDER_LOCK.json',
        'ENCODING_SUMMARY.json','OLD_SCIENTIFIC_PROTECTION.json']
    completion={'complete':True,'data_only':True,'optimizer_updates':0,
        'elapsed_seconds':time.monotonic()-start,'download':download_summary,'summary':counts,'encoding':encoding,
        'protocol_sha256':PROTOCOL_SHA,'evaluation_requires_external_admission':True,'evaluation_allowed':False,
        'training_allowed':False,'old_scientific_source_config_protocol_unchanged':True,
        'artifacts_sha256':{name:sha_file(root/name) for name in artifacts}}
    write_json(root/'COMPLETION.json',completion)
    write_json(root/'STATUS.json',{'phase':'complete','questions':len(rows),'images':len(images),
        'elapsed_seconds':time.monotonic()-start,'evaluation_allowed':False,'model_forward_calls':0})
    print('COMPLETE',json.dumps(completion),flush=True)


if __name__=='__main__':
    main()
