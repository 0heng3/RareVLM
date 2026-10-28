import io
import json

from PIL import Image
import pytest

from vlm_public_preference.data import (coco_source_id, content_sha, exclusion_reasons, freeze_audit,
    image_identity, load_boundaries, one_pair_per_image, parse_pair, split_pairs)


def image_bytes(fmt):
    handle = io.BytesIO()
    Image.new('RGB', (7, 9), (15, 20, 30)).save(handle, format=fmt)
    return handle.getvalue()


def test_same_pixels_different_file_encoding_are_excluded():
    first, second = image_identity(image_bytes('PNG')), image_identity(image_bytes('BMP'))
    assert first['sha256'] != second['sha256']
    assert first['pixel_sha256'] == second['pixel_sha256']
    boundary = {'files': {first['sha256']: []}, 'pixels': {first['pixel_sha256']: []}}
    pair = {'origin_dataset': 'coco', 'source_image_path': 'coco.jpg'}
    assert exclusion_reasons(pair, second, boundary) == ['existing_train_dev_final_or_formal_pixel_sha']


def test_human_and_ai_metadata_and_original_strings_remain_distinct():
    row = {'text': json.dumps({'question': 'Q?', 'chosen': ' A. ', 'rejected': 'B.'}),
           'question': 'Q?', 'chosen': ' A. ', 'rejected': 'B.', 'idx': 1,
           'origin_split': '{"model":"m","type":"question_answering","feedback_model":"judge"}',
           'origin_dataset': 'coco', 'image_path': 'coco.jpg'}
    human = parse_pair(row, 'openbmb/RLHF-V-Dataset', 'r')
    ai = parse_pair(row, 'openbmb/RLAIF-V-Dataset', 'r')
    assert human['chosen'] == ai['chosen'] == ' A. '
    assert human['prompt_suffix'] == ''
    assert human['label_source'] != ai['label_source']
    assert ai['origin_split']['feedback_model'] == 'judge'
    assert human['question_id'] != '1'


def test_identical_official_responses_are_not_a_preference():
    with pytest.raises(ValueError, match='Identical chosen/rejected'):
        parse_pair({'text': json.dumps({'question':'Q?', 'chosen':'A.', 'rejected':'A.'})},
                   'openbmb/RLHF-V-Dataset', 'r')


def test_coco_source_identity_survives_2014_2017_layout():
    assert coco_source_id('coco2017/train2017/000000179545.jpg') == '000000179545'
    assert coco_source_id('coco/COCO_val2014_000000179545.jpg') == '000000179545'
    assert coco_source_id('other/000000179545.jpg') is None


def test_image_unique_split_and_audit_are_order_independent():
    pairs = [{'pair_id': str(i), 'image_sha256': str(i), 'pixel_sha256': str(i)} for i in range(200)]
    pairs.append({'pair_id': 'reencoded', 'image_sha256': 'other', 'pixel_sha256': '0'})
    unique, excluded = one_pair_per_image(pairs)
    assert len(unique) == 200 and len(excluded) == 1
    assert one_pair_per_image(list(reversed(pairs))) == (unique, excluded)
    splits = split_pairs(unique)
    train, heldout = splits.values()
    assert not {p['pixel_sha256'] for p in train} & {p['pixel_sha256'] for p in heldout}
    assert not {p['image_sha256'] for p in train} & {p['image_sha256'] for p in heldout}
    assert freeze_audit(train) == freeze_audit(list(reversed(train)))
    assert len(freeze_audit(train)) == 64


def test_every_existing_split_and_formal_record_is_indexed_and_locked(tmp_path):
    paths = []
    for task in ('text', 'formal'):
        rows = [{'image_id': key, 'question_id': key, 'question': 'Q ' + key} for key in ('a', 'b', 'c')]
        manifest = {'images': {key: {'sha256': 'file'+key, 'pixel_sha256': 'pixel'+key} for key in ('a','b','c')}}
        manifest.update({'splits': dict(zip(('train_all','dev','final'), ([r] for r in rows)))}
                        if task == 'text' else {'task':'docvqa','records':rows})
        manifest['manifest_sha256'] = content_sha(manifest)
        path = tmp_path / (task+'.json')
        path.write_text(json.dumps(manifest));paths.append(path)
    boundary = load_boundaries(paths)
    assert len(boundary['files']) == len(boundary['pixels']) == 3
    assert boundary['lock'][0]['question_counts'] == {'train_all':1,'dev':1,'final':1}
    damaged = json.loads(paths[0].read_text());damaged['images']['a']['sha256'] = 'changed'
    paths[0].write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='Manifest content changed'):
        load_boundaries(paths)
