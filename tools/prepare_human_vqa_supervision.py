"""Join official human annotations to the existing split, replacing pseudo no labels."""
import argparse
import json
from pathlib import Path
import zipfile

from vlm_deep.common import content_sha, read_manifest, sha_file, write_json
from vlm_human_vqa.data import counts, human_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--image-pool', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to('/opt/data/private'):
        raise ValueError('Use persistent output path')
    documents, sources = {}, {}
    for kind in ('Annotations', 'Questions'):
        path = args.downloads / f'v2_{kind}_Train_mscoco.zip'
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        if sha_file(path) != receipt['sha256']:
            raise ValueError('Official source archive differs from receipt')
        with zipfile.ZipFile(path) as archive:
            member, = archive.namelist()
            document = json.loads(archive.read(member))
        if document['data_subtype'] != 'train2014' or document['data_type'] != 'mscoco':
            raise ValueError('Only official COCO train2014 is admitted')
        documents[kind] = document
        sources[kind] = {'url': receipt['url'], 'sha256': receipt['sha256'], 'member': member}
    questions = {row['question_id']: row for row in documents['Questions']['questions']}
    annotations = documents['Annotations']['annotations']
    if len(questions) != len(documents['Questions']['questions']) or len({row['question_id'] for row in annotations}) != len(annotations):
        raise ValueError('Duplicate source question IDs')
    if set(questions) != {row['question_id'] for row in annotations}:
        raise ValueError('Human question and annotation ID sets differ')
    blocked = {int(value) for value in json.loads(args.source_lock.read_text())['blocked_coco_ids']}
    maps, split_rows, image_map = {}, {}, {}
    for split in ('train', 'val'):
        path = args.image_pool / f'{split.upper()}_CANDIDATES.json'
        pool = json.loads(path.read_text())
        sources[f'image_pool_{split}'] = {'path': str(path.resolve()), 'sha256': sha_file(path)}
        maps[split] = {int(image['coco_image_id']): (image_id, image) for image_id, image in pool['images'].items()}
        if len(maps[split]) != len(pool['images']) or set(maps[split]) & blocked:
            raise ValueError('Duplicate COCO identity or benchmark overlap')
        split_rows[split] = []
    if set(maps['train']) & set(maps['val']):
        raise ValueError('Training and validation images overlap')
    for annotation in annotations:
        question = questions[annotation['question_id']]
        if question['image_id'] != annotation['image_id']:
            raise ValueError('Source question has a different image ID')
        for split, mapping in maps.items():
            pair = mapping.get(int(annotation['image_id']))
            if pair is None:
                continue
            image_id, image = pair
            row = human_row(question, annotation, image_id, image, split)
            if row is not None:
                split_rows[split].append(row)
                image_map[image_id] = image
    for split in split_rows:
        split_rows[split].sort(key=lambda row: row['question_id'])
        if {row['target'] for row in split_rows[split]} != {'yes', 'no'}:
            raise ValueError('Actual human yes/no labels required in each split')
    for identity in ('pixel_sha256', 'sha256'):
        if {image_map[row['image_id']][identity] for row in split_rows['train']} & {
                image_map[row['image_id']][identity] for row in split_rows['val']}:
            raise ValueError('Cross-split image content overlap')
    manifest = {'schema_version': 'rarevlm.vlm_data.v2', 'training_allowed': True,
                'purpose': 'human visual yes/no SFT data; not a gain/evaluation result',
                'task_scope': 'original human questions, including attributes/relations; not 80-class object-presence coverage',
                'sources': sources, 'builder_sha256': sha_file(Path(__file__)),
                'label_module_sha256': sha_file(Path(__file__).resolve().parents[1] / 'src/vlm_human_vqa/data.py'),
                'source_lock_sha256': sha_file(args.source_lock),
                'images': image_map, 'splits': {'train_all': split_rows['train'],
                    'train_10k': split_rows['train'][:10000], 'dev': split_rows['val'], 'final': []},
                'notes': ['Official VQA v2 train2014 annotations only.',
                          'No inference from missing COCO categories; no relabeling of the original question.',
                          'Image-disjoint inherited split; these validation images have prior Object-study exposure.',
                          'No official VQA test result or blind final set is claimed.',
                          'Third-party model pretraining overlap is unknown.']}
    manifest['manifest_sha256'] = content_sha(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'MANIFEST.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to overwrite a different human corpus')
    write_json(path, manifest)
    read_manifest(path, verify_images=True)
    result = {'status': 'completed', 'training_data_prepared': True, 'training_launched': False,
              'source_questions': len(questions), 'manifest_sha256': sha_file(path),
              'splits': {split: counts(rows) for split, rows in split_rows.items()},
              'raw_image_hashes_verified': True, 'benchmark_images_excluded': True,
              'no_gain_claim': True}
    write_json(output / 'SUMMARY.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
