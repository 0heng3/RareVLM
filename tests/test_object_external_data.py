"""CPU contracts for official JPEG integrity and new confirmation-data boundaries."""
import hashlib
import io
import json
import socket
from copy import deepcopy
from types import SimpleNamespace

from PIL import Image, ImageFile
import pytest

from vlm_object_presence.data import content_sha
from vlm_object_external_data import download
from vlm_object_external_data.data import (
    build_rows, class_balanced_subset, freeze_audit,
    select_verified_images, verify_image_bytes,
)


CATEGORIES = {
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError('External-data contracts must not use the network')
    monkeypatch.setattr(socket, 'create_connection', reject_connection)
    monkeypatch.setattr(socket.socket, 'connect', reject_connection)


def image_bytes(color=(30, 90, 170), *, image_format='JPEG', comment=None):
    output = io.BytesIO()
    Image.new('RGB', (64, 48), color).save(output, format=image_format)
    data = output.getvalue()
    if comment is not None:
        assert image_format == 'JPEG'
        segment = b'\xff\xfe' + (len(comment) + 2).to_bytes(2, 'big') + comment
        data = data[:2] + segment + data[2:]
    return data


def metadata(number=1):
    return {'id': number, 'coco_image_id': f'{number:012d}',
            'file_name': f'{number:012d}.jpg', 'width': 64, 'height': 48}


def headers(data):
    return {'Content-Length': str(len(data)),
            'ETag': '"' + hashlib.md5(data).hexdigest() + '"'}


def test_verified_native_jpeg_uses_the_existing_file_and_rgb_identity_definition():
    data = image_bytes()
    identity = verify_image_bytes(data, metadata(), headers(data))
    with Image.open(io.BytesIO(data)) as raw:
        rgb = raw.convert('RGB')
        expected_pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    assert identity['sha256'] == hashlib.sha256(data).hexdigest()
    assert identity['file_sha256'] == identity['sha256']
    assert identity['pixel_sha256'] == expected_pixels
    assert (identity['width'], identity['height']) == (64, 48)
    assert identity['md5'] == hashlib.md5(data).hexdigest()
    assert identity['etag_is_md5'] is identity['etag_md5_verified'] is True
    assert identity['official_published_sha256_claimed'] is False


@pytest.mark.parametrize('fault', [
    'truncated_jpeg', 'non_jpeg', 'wrong_width', 'wrong_height',
    'content_length_too_short', 'content_length_too_long', 'missing_content_length', 'wrong_md5',
])
def test_corrupt_bytes_or_transport_or_official_dimensions_cannot_be_verified(fault):
    data, meta = image_bytes(), metadata()
    if fault == 'truncated_jpeg':
        data = data[:-2]
    elif fault == 'non_jpeg':
        data = image_bytes(image_format='PNG')
    elif fault == 'wrong_width':
        meta['width'] += 1
    elif fault == 'wrong_height':
        meta['height'] += 1
    response_headers = headers(data)
    if fault == 'content_length_too_short':
        response_headers['Content-Length'] = str(len(data) - 1)
    elif fault == 'content_length_too_long':
        response_headers['Content-Length'] = str(len(data) + 1)
    elif fault == 'missing_content_length':
        del response_headers['Content-Length']
    elif fault == 'wrong_md5':
        response_headers['ETag'] = '"' + '0' * 32 + '"'
    with pytest.raises((ValueError, OSError)):
        verify_image_bytes(data, meta, response_headers)


@pytest.mark.parametrize('etag', ['"0123456789abcdef0123456789abcdef-2"',
                                 'W/"0123456789abcdef0123456789abcdef"',
                                 '"opaque-etag"', None])
def test_non_md5_or_missing_etag_is_explicitly_unverified_without_claiming_official_sha(etag):
    data = image_bytes()
    response_headers = headers(data)
    if etag is None:
        del response_headers['ETag']
    else:
        response_headers['ETag'] = etag
    identity = verify_image_bytes(data, metadata(), response_headers)
    assert identity['etag_is_md5'] is False
    assert identity['etag_md5_verified'] is False
    assert identity['etag_not_md5_reason']
    assert identity['official_published_sha256_claimed'] is False
    assert identity['sha256'] == hashlib.sha256(data).hexdigest()


def test_http_header_names_and_hex_md5_are_case_insensitive():
    data = image_bytes()
    identity = verify_image_bytes(data, metadata(), {
        'content-length': str(len(data)), 'etag': '"' + hashlib.md5(data).hexdigest().upper() + '"'})
    assert identity['etag_md5_verified'] is True


def test_permissive_truncated_image_decoder_cannot_silently_relax_integrity(monkeypatch):
    data = image_bytes()
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', True)
    with pytest.raises(ValueError):
        verify_image_bytes(data, metadata(), headers(data))


def test_verified_selection_excludes_old_file_or_pixel_and_deduplicates_actual_new_jpegs():
    blue = image_bytes((0, 0, 220))
    yellow = image_bytes((220, 220, 0))
    yellow_variant = image_bytes((220, 220, 0), comment=b'different file, same pixels')
    data = {
        1: image_bytes((220, 0, 0)),
        2: image_bytes((0, 220, 0), comment=b'new bytes of an old RGB image'),
        3: blue, 4: blue, 5: yellow, 6: yellow_variant,
        7: image_bytes((0, 220, 220)),
    }
    rows = [metadata(number) for number in data]
    verified = {row['coco_image_id']: verify_image_bytes(data[row['id']], row, headers(data[row['id']]))
                for row in rows}
    old_green = image_bytes((0, 220, 0))
    old_green_identity = verify_image_bytes(old_green, metadata(2), headers(old_green))
    assert verified['000000000002']['sha256'] != old_green_identity['sha256']
    assert verified['000000000002']['pixel_sha256'] == old_green_identity['pixel_sha256']
    assert verified['000000000005']['sha256'] != verified['000000000006']['sha256']
    assert verified['000000000005']['pixel_sha256'] == verified['000000000006']['pixel_sha256']
    old_files = {verified['000000000001']['sha256']}
    old_pixels = {old_green_identity['pixel_sha256']}
    before = deepcopy((rows, verified, old_files, old_pixels))
    retained, excluded = select_verified_images(rows, verified, old_files, old_pixels, seed=42)
    assert (rows, verified, old_files, old_pixels) == before
    def winner(numbers):
        return min((f'{number:012d}' for number in numbers), key=lambda iid:
            content_sha(['object_external_v1_image_dedup', 42, iid, verified[iid]['pixel_sha256']]))
    expected = {winner((3, 4)), winner((5, 6)), '000000000007'}
    assert {row['coco_image_id'] for row in retained} == expected
    assert all(row in rows for row in retained)
    assert len(excluded) == 4
    assert {row['coco_image_id'] for row in excluded} == set(verified) - expected
    records = {row['coco_image_id']: row for row in excluded}
    assert any('file_sha' in reason for reason in records['000000000001']['reasons'])
    assert any('pixel_sha' in reason for reason in records['000000000002']['reasons'])
    assert (retained, excluded) == select_verified_images(
        list(reversed(rows)), dict(reversed(list(verified.items()))), old_files, old_pixels, seed=42)


def test_incomplete_download_mapping_or_duplicate_candidate_id_is_rejected():
    data = image_bytes()
    row = metadata()
    identity = verify_image_bytes(data, row, headers(data))
    with pytest.raises(ValueError):
        select_verified_images([row], {}, set(), set())
    with pytest.raises(ValueError):
        select_verified_images([row, deepcopy(row)], {row['coco_image_id']: identity}, set(), set())


def qualifying_box(annotation_id=1, width=24, height=24, area_fraction=.005, crowd=0):
    return {'annotation_id': annotation_id, 'bbox_native_metadata_xywh': [2, 3, width, height],
            'segmentation_area_fraction': area_fraction, 'iscrowd': crowd}


def eligible_image():
    row = {**metadata(),
        'source_coco_url': 'http://images.cocodataset.org/val2017/000000000001.jpg',
        'official_https_url': 'https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000001.jpg',
        'positive_categories': [
            {'category_id': 16, 'category_name': 'bird',
             'qualifying_instances': [qualifying_box(), qualifying_box(2, 25, 26, .01)]},
            {'category_id': 17, 'category_name': 'cat',
             'qualifying_instances': [qualifying_box(3, 30, 25, .02)]}],
        # Dog and horse represent a small nonqualifying instance and a crowd
        # annotation. Both remain present and must never become negatives.
        'all_annotated_category_ids_including_crowd': [16, 17, 18, 19]}
    data = image_bytes()
    identity = {**verify_image_bytes(data, row, headers(data)), 'category_map': CATEGORIES,
                'path': '/opt/data/private/synthetic/000000000001.jpg'}
    return row, identity


def test_questions_keep_all_qualified_positives_native_boxes_and_fixed_unique_absent_negatives():
    metadata_row, identity = eligible_image()
    before = deepcopy((metadata_row, identity))
    rows, image = build_rows(metadata_row, identity)
    assert (metadata_row, identity) == before
    assert len(rows) == 4
    assert {row['category_id'] for row in rows if row['target'] == 'yes'} == {16, 17}
    absent = set(CATEGORIES) - {16, 17, 18, 19}
    expected_negatives = set(sorted(absent, key=lambda category: content_sha([
        'object_external_v1_negative', 42, 1, identity['pixel_sha256'], category]))[:2])
    assert {row['category_id'] for row in rows if row['target'] == 'no'} == expected_negatives
    assert len({row['category_id'] for row in rows}) == 4
    assert (image['width'], image['height']) == (64, 48)
    assert image['path'] == identity['path']
    original_boxes = {item['category_id']: item['qualifying_instances']
                      for item in metadata_row['positive_categories']}
    for row in rows:
        assert row['question_id'] == f'object_external_v1:000000000001:category{row["category_id"]:02d}'
        assert row['question'] == f'Is there a {CATEGORIES[row["category_id"]]} in the image?'
        assert row['prompt_suffix'] == '\nAnswer yes or no.'
        assert row['training_allowed'] is False
        assert row['evaluation_requires_external_admission'] is True
        assert row['human_preference'] is False
        assert row['model_prediction_used_for_label_or_sampling'] is False
        if row['target'] == 'no':
            assert row['bbox_evidence'] == []
        else:
            assert len(row['bbox_evidence']) == len(original_boxes[row['category_id']])
            for actual, original in zip(row['bbox_evidence'], original_boxes[row['category_id']]):
                assert actual['bbox_current_image_xywh'] == original['bbox_native_metadata_xywh']
    reversed_metadata = {**metadata_row, 'positive_categories': list(reversed(metadata_row['positive_categories']))}
    assert (rows, image) == build_rows(reversed_metadata, identity)


@pytest.mark.parametrize('fault', ['crowd', 'short_side', 'segmentation_area', 'different_instances',
                                 'wrong_dimensions', 'no_positive', 'insufficient_absent'])
def test_question_generation_rejects_invalid_positive_evidence_or_balance_instead_of_fabricating_labels(fault):
    row, identity = eligible_image()
    instances = row['positive_categories'][0]['qualifying_instances']
    if fault == 'crowd':
        instances[0]['iscrowd'] = 1
    elif fault == 'short_side':
        instances[0]['bbox_native_metadata_xywh'][2] = 23.99
    elif fault == 'segmentation_area':
        instances[0]['segmentation_area_fraction'] = .004999
    elif fault == 'different_instances':
        row['positive_categories'][0]['qualifying_instances'] = [
            qualifying_box(width=23, area_fraction=.1), qualifying_box(width=30, area_fraction=.004)]
    elif fault == 'wrong_dimensions':
        identity['width'] += 1
    elif fault == 'no_positive':
        row['positive_categories'] = []
    elif fault == 'insufficient_absent':
        row['all_annotated_category_ids_including_crowd'] = sorted(set(CATEGORIES) - {9})
    with pytest.raises(ValueError):
        build_rows(row, identity)


def audit_candidates(image_count=96):
    return [{
        'question_id': f'object_external_v1:{number:012d}:category{category:02d}',
        'image_id': f'coco2017:{number:012d}',
        'pixel_sha256': f'pixel:{number}', 'target': label,
    } for number in range(image_count)
       for category, label in ((16, 'yes'), (17, 'yes'), (18, 'no'), (19, 'no'))]


def test_audit_selects_yes_then_no_with_one_shared_pixel_boundary_and_fresh_salts():
    rows = audit_candidates()
    before = deepcopy(rows)
    selected = freeze_audit(rows, count=64, seed=42)
    assert rows == before
    assert len(selected) == 64
    assert sum(row['target'] == 'yes' for row in selected) == 32
    assert len({row['pixel_sha256'] for row in selected}) == 64
    assert selected == freeze_audit(list(reversed(rows)), count=64, seed=42)
    seen, expected = set(), []
    for label in ('yes', 'no'):
        ranked = sorted((row for row in rows if row['target'] == label), key=lambda row:
            content_sha(['object_external_v1_audit', 42, row['question_id'], row['pixel_sha256']]))
        label_selection = []
        for row in ranked:
            if row['pixel_sha256'] not in seen:
                label_selection.append(row)
                seen.add(row['pixel_sha256'])
                if len(label_selection) == 32:
                    break
        expected.extend(label_selection)
    expected.sort(key=lambda row: content_sha([
        'object_external_v1_audit_order', 42, row['question_id']]))
    assert selected == expected


def test_audit_rejects_odd_size_and_too_few_distinct_images_even_when_each_label_has_enough_rows():
    with pytest.raises(ValueError):
        freeze_audit(audit_candidates(), count=63)
    with pytest.raises(ValueError):
        freeze_audit(audit_candidates(image_count=63), count=64)


def test_category_balancing_is_an_exact_subset_and_retains_both_labels_only_when_available():
    specifications = [(16, 'yes'), (16, 'yes'), (16, 'no'),
                      (18, 'yes'), (18, 'no'), (18, 'no'), (18, 'no'),
                      (9, 'yes')]
    rows = [{'question_id': f'object_external_v1:{number:012d}:category{category:02d}',
             'category_id': category, 'category_name': CATEGORIES[category],
             'pixel_sha256': f'pixel:{number}', 'target': label,
             'evidence_sentinel': {'source': number}}
            for number, (category, label) in enumerate(specifications)]
    before = deepcopy(rows)
    selected, counts = class_balanced_subset(rows, seed=42)
    assert rows == before
    assert len(selected) == 4
    assert all(row in rows for row in selected)
    assert {row['category_id'] for row in selected} == {16, 18}
    for category in (16, 18):
        assert sorted(row['target'] for row in selected if row['category_id'] == category) == ['no', 'yes']
    assert counts['9']['selected_yes'] == counts['9']['selected_no'] == 0
    assert (selected, counts) == class_balanced_subset(list(reversed(rows)), seed=42)


def mocked_download(monkeypatch, data, *, status=200, response_headers=None):
    meta, _ = eligible_image()
    calls = []
    headers_for_data = headers(data)
    class Response:
        status_code = status
        url = meta['official_https_url']
        headers = response_headers if response_headers is not None else headers_for_data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def iter_content(self, block_size):
            yield data[:len(data)//2]
            yield data[len(data)//2:]
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(download._local, 'session', SimpleNamespace(get=get), raising=False)
    return meta, calls


def test_download_success_verifies_before_atomic_final_jpeg_and_receipt(monkeypatch, tmp_path):
    assert tmp_path.is_relative_to('/opt/data/private')
    data = image_bytes()
    meta, calls = mocked_download(monkeypatch, data)
    destination = tmp_path / 'images' / meta['file_name']
    receipt_path = tmp_path / 'downloads' / f'{meta["coco_image_id"]}.json'
    def checked_verify(body, official, response_headers):
        assert not destination.exists()
        return verify_image_bytes(body, official, response_headers)
    monkeypatch.setattr(download, 'verify_image_bytes', checked_verify)
    receipt, reused = download.download_one(meta, tmp_path, max_attempts=1)
    assert reused is False
    assert len(calls) == 1
    assert calls[0][0] == meta['official_https_url']
    assert calls[0][1]['allow_redirects'] is False
    assert receipt['status'] == 'verified'
    assert destination.read_bytes() == data
    assert json.loads(receipt_path.read_text()) == receipt
    assert receipt['identity']['sha256'] == hashlib.sha256(data).hexdigest()
    assert not (tmp_path / 'download_tmp' / f'{meta["coco_image_id"]}.part').exists()
    assert not receipt_path.with_suffix('.json.tmp').exists()


def test_download_verified_cache_is_revalidated_without_get_or_receipt_rewrite(monkeypatch, tmp_path):
    assert tmp_path.is_relative_to('/opt/data/private')
    data = image_bytes()
    meta, calls = mocked_download(monkeypatch, data)
    first, _ = download.download_one(meta, tmp_path, max_attempts=1)
    receipt_path = tmp_path / 'downloads' / f'{meta["coco_image_id"]}.json'
    original_receipt = receipt_path.read_bytes()
    validations = []
    def checked_verify(body, official, response_headers):
        validations.append(body)
        return verify_image_bytes(body, official, response_headers)
    def reject_receipt_write(*args, **kwargs):
        raise AssertionError('Complete verified cache must not rewrite its receipt')
    monkeypatch.setattr(download, 'verify_image_bytes', checked_verify)
    monkeypatch.setattr(download, 'write_json', reject_receipt_write)
    receipt, reused = download.download_one(meta, tmp_path, max_attempts=1)
    assert reused is True
    assert receipt == first
    assert len(calls) == 1
    assert validations == [data]
    assert receipt_path.read_bytes() == original_receipt


@pytest.mark.parametrize('fault', ['http_error', 'short_body', 'bad_md5'])
def test_download_failures_never_create_a_verified_receipt_or_final_jpeg(monkeypatch, tmp_path, fault):
    assert tmp_path.is_relative_to('/opt/data/private')
    data = image_bytes()
    response_headers = headers(data)
    status = 503 if fault == 'http_error' else 200
    if fault == 'short_body':
        response_headers['Content-Length'] = str(len(data) + 1)
    elif fault == 'bad_md5':
        response_headers['ETag'] = '"' + '0' * 32 + '"'
    meta, calls = mocked_download(monkeypatch, data, status=status, response_headers=response_headers)
    def reject_sleep(*args, **kwargs):
        raise AssertionError('Single-attempt mock contracts must not wait')
    monkeypatch.setattr(download.time, 'sleep', reject_sleep)
    receipt, reused = download.download_one(meta, tmp_path, max_attempts=1)
    assert reused is False
    assert len(calls) == 1
    assert receipt['status'] == 'failed'
    assert len(receipt['attempts']) == 1
    assert receipt['attempts'][0]['outcome'] == 'failed'
    assert not (tmp_path / 'images' / meta['file_name']).exists()
    saved = json.loads((tmp_path / 'downloads' / f'{meta["coco_image_id"]}.json').read_text())
    assert saved['status'] == 'failed'
    assert 'identity' not in saved
