"""Prepare repaired candidates only. Does not admit or launch model training."""
import argparse
import json
from pathlib import Path

from vlm_deep.common import content_sha, sha_file, write_json
from vlm_object_presence_v3.data import repair_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--exclude-audit', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to('/opt/data/private') or root == args.source.resolve():
        raise ValueError('Use a distinct persistent output directory')
    excluded = [row for path in args.exclude_audit for row in json.loads(path.read_text())['rows']]
    source_lock = json.loads((args.source / 'SOURCE_LOCK.json').read_text())
    blocked_ids = {str(value).zfill(12) for value in source_lock['blocked_coco_ids']}
    result, identities = {}, {}
    for split, filename in [('train', 'TRAIN_MANIFEST.json'), ('val', 'VAL_MANIFEST.json')]:
        path = args.source / filename
        original = json.loads(path.read_text())
        if content_sha({key: value for key, value in original.items() if key != 'manifest_sha256'}) != original['manifest_sha256']:
            raise ValueError('Ancestor manifest content changed')
        candidates, unknown, images = repair_rows(original, excluded)
        if any(str(image['coco_image_id']).zfill(12) in blocked_ids for image in images.values()):
            raise ValueError('Inherited benchmark image boundary violated')
        identities[split] = {key: {str(row[key]) for row in candidates + unknown}
                             for key in ('image_id', 'pixel_sha256', 'file_sha256', 'coco_image_id')}
        result[split] = {'schema_version': 'rarevlm.object_presence.candidates.v3', 'split': split,
                         'training_allowed': False, 'source_manifest_sha256': sha_file(path),
                         'excluded_audits': {str(path): sha_file(path) for path in args.exclude_audit},
                         'candidates': candidates, 'quarantined_unknown': unknown, 'images': images,
                         'summary': summary(candidates, unknown, images)}
    for key in identities['train']:
        if identities['train'][key] & identities['val'][key]:
            raise ValueError(f'Inherited split overlap: {key}')
    root.mkdir(parents=True, exist_ok=True)
    for split, document in result.items():
        path = root / f'{split.upper()}_CANDIDATES.json'
        if path.exists() and json.loads(path.read_text()) != document:
            raise ValueError('Refusing to overwrite a different candidate population')
        write_json(path, document)
    output = {'status': 'completed', 'training_allowed': False,
              'splits': {split: document['summary'] for split, document in result.items()},
              'benchmark_images_excluded': True, 'train_val_disjoint': True}
    write_json(root / 'SUMMARY.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
