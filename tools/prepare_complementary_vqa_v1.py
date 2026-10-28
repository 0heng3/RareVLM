"""Inventory official VQA-v2 complementary supervision without downloading images.

This stage prepares candidates and graph feasibility only. It does not admit labels,
select training examples, download COCO photographs, or launch a model. The complete
official image graph is deliberately distinguished from the filtered candidate graph.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import textwrap
import time
from urllib.request import Request, urlopen
import zipfile


PRIVATE = Path('/opt/data/private')
OUTPUTS = PRIVATE / 'longtail_vlm/rarevlm_pilot/outputs'
PAIR_URL = ('https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/'
            'v2_Complementary_Pairs_Train_mscoco.zip')
SOURCE_PAGE = 'https://visualqa.org/download.html'
SALT = 'rarevlm-complementary-vqa-v1-official-full-component'
DEFAULT_EXCLUSIONS = (
    'vlm_human_vqa_v1/MANIFEST.json',
    'vlm_object_presence_v1/TRAIN_MANIFEST.json',
    'vlm_object_presence_v1/VAL_MANIFEST.json',
    'vlm_object_presence_v1/VAL_CLASS_BALANCED_MANIFEST.json',
    'vlm_object_presence_v1/SOURCE_LOCK.json',
    'vlm_object_presence_v2/TRAIN_MANIFEST.json',
    'vlm_object_presence_v2/VAL_MANIFEST.json',
    'vlm_object_presence_v2/VAL_CLASS_BALANCED_MANIFEST.json',
    'vlm_object_presence_v2/SOURCE_LOCK.json',
    'vlm_object_presence_v3/TRAIN_CANDIDATES.json',
    'vlm_object_presence_v3/VAL_CANDIDATES.json',
    'vlm_object_external_data_v1/MANIFEST.json',
    'vlm_object_external_data_v1/CLASS_BALANCED_MANIFEST.json',
    'vlm_object_external_data_v1/SOURCE_LOCK.json',
    'vlm_posttrain_pope_v1/data/manifest.json',
    'vlm_public_preference_v1/CANDIDATE_MANIFEST.json',
    'vlm_public_preference_v1/TRAIN_MANIFEST.json',
    'vlm_public_preference_v1/HELDOUT_MANIFEST.json',
    'vlm_deep_v2/prepared/manifest.json',
    'vlm_sft_textvqa_v1/prepared/manifest.json',
    'vlm_deep_v2/transfer/mmstar/manifest.json',
    'vlm_deep_v2/transfer/docvqa/manifest.json',
)
COCO_PATTERNS = (
    re.compile(r'coco20(?:14|17):(\d{1,12})(?:$|\D)', re.I),
    re.compile(r'COCO_(?:train2014|val2014|test2015)_(\d{12})(?:\D|$)'),
    re.compile(r'(?:^|/)(?:train2017|val2017)/(\d{12})\.(?:jpg|png)(?:$|\?)'),
)
SKIP_CONTENT = {'answers', 'answer', 'target', 'question', 'prediction', 'generated_text',
                'prompt', 'prompt_suffix', 'rejected', 'chosen', 'messages'}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def content_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def normalized(text: str) -> str:
    return ' '.join(text.lower().split())


def write_json(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != payload:
        raise ValueError(f'Refusing to overwrite a different prepared artifact: {path}')
    if not path.exists():
        path.write_text(payload)


def write_jsonl(path: Path, rows) -> None:
    payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                      for row in rows)
    if path.exists() and path.read_text() != payload:
        raise ValueError(f'Refusing to overwrite a different prepared artifact: {path}')
    if not path.exists():
        path.write_text(payload)


def pair_archive(destination: Path) -> tuple[Path, dict]:
    path = destination / 'v2_Complementary_Pairs_Train_mscoco.zip'
    receipt_path = path.with_suffix('.receipt.json')
    if path.exists():
        if not receipt_path.exists():
            raise ValueError('Existing official pair ZIP has no source receipt')
        receipt = json.loads(receipt_path.read_text())
        if receipt['url'] != PAIR_URL or receipt['sha256'] != sha_file(path):
            raise ValueError('Official complementary pair source changed')
        return path, receipt
    partial = path.with_suffix('.partial')
    for attempt in range(3):
        try:
            request = Request(PAIR_URL, headers={'User-Agent': 'RareVLM-data-preparation/1.0'})
            with urlopen(request, timeout=60) as response, partial.open('wb') as stream:
                for block in iter(lambda: response.read(2 ** 20), b''):
                    stream.write(block)
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt official complementary pair ZIP')
            partial.replace(path)
            receipt = {'url': PAIR_URL, 'source_page': SOURCE_PAGE,
                       'sha256': sha_file(path), 'bytes': path.stat().st_size,
                       'downloaded_utc': datetime.now(timezone.utc).isoformat()}
            write_json(receipt_path, receipt)
            return path, receipt
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError('Unreachable')


def existing_official_archive(directory: Path, kind: str) -> tuple[dict, dict]:
    path = directory / f'v2_{kind}_Train_mscoco.zip'
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if sha_file(path) != receipt['sha256']:
        raise ValueError(f'Existing {kind} source differs from its original receipt')
    with zipfile.ZipFile(path) as archive:
        member, = archive.namelist()
        document = json.loads(archive.read(member))
    if document.get('data_type') != 'mscoco' or document.get('data_subtype') != 'train2014':
        raise ValueError('Only official mscoco train2014 inputs are allowed')
    return document, {'path': str(path), 'sha256': receipt['sha256'],
                      'url': receipt['url'], 'member': member}


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, item: int) -> int:
        if item not in self.parent:
            self.parent[item], self.size[item] = item, 1
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

    def components(self) -> tuple[dict[int, str], list[dict]]:
        groups = defaultdict(list)
        for image_id in self.parent:
            groups[self.find(image_id)].append(image_id)
        identities, records = {}, []
        for members in groups.values():
            members.sort()
            identity = content_sha(members)
            fold = int(content_sha([SALT, identity])[:16], 16) % 100
            split = 'train' if fold < 70 else 'screen' if fold < 85 else 'confirm'
            records.append({'component_id': identity, 'images': len(members),
                            'image_ids': members, 'hash_fold': fold, 'split': split})
            identities.update((image_id, identity) for image_id in members)
        records.sort(key=lambda row: (-row['images'], row['component_id']))
        return identities, records


def boundary_identities(document) -> tuple[set[int], set[str], set[str]]:
    coco_ids, pixels, files = set(), set(), set()

    def visit(value, key=''):
        if key in SKIP_CONTENT:
            return
        if key == 'blocked_coco_ids' and isinstance(value, list):
            coco_ids.update(int(item) for item in value)
            return
        if key == 'coco_image_id' and value is not None:
            if isinstance(value, list):
                for item in value:
                    if item is not None:
                        coco_ids.add(int(item))
            else:
                coco_ids.add(int(value))
            return
        if isinstance(value, dict):
            is_image = ('path' in value or 'image_path' in value) and (
                'width' in value or 'pixel_sha256' in value or 'coco_image_id' in value)
            if is_image:
                if value.get('pixel_sha256'):
                    pixels.add(value['pixel_sha256'])
                for field in ('sha256', 'file_sha256'):
                    if value.get(field):
                        files.add(value[field])
            for subkey, subvalue in value.items():
                if isinstance(subkey, str):
                    for pattern in COCO_PATTERNS:
                        coco_ids.update(int(match) for match in pattern.findall(subkey))
                visit(subvalue, subkey)
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str):
            for pattern in COCO_PATTERNS:
                coco_ids.update(int(match) for match in pattern.findall(value))

    visit(document)
    return coco_ids, pixels, files


def collect_boundaries(paths: list[Path]) -> dict:
    ids, pixels, files, sources = set(), set(), set(), []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Explicit historical exclusion input missing: {path}')
        document = json.loads(path.read_text())
        source_ids, source_pixels, source_files = boundary_identities(document)
        ids.update(source_ids)
        pixels.update(source_pixels)
        files.update(source_files)
        sources.append({'path': str(path), 'sha256': sha_file(path),
                        'coco_ids': len(source_ids), 'pixel_sha256': len(source_pixels),
                        'file_sha256': len(source_files)})
    return {'sources': sources, 'blocked_coco_ids': sorted(ids),
            'blocked_pixel_sha256': sorted(pixels), 'blocked_file_sha256': sorted(files),
            'scope': 'All explicitly listed historical image catalogs; original label text ignored. '
                     'All known TextVQA train/dev/final image hashes are excluded, not only dev/final. '
                     'Near-duplicate and model-pretraining overlap remain unknown.'}


def consensus(annotation: dict) -> tuple[str | None, int]:
    answers = annotation['answers']
    if len(answers) != 10 or {row['answer_id'] for row in answers} != set(range(1, 11)):
        raise ValueError('Expected ten distinct official human answer IDs')
    if annotation['answer_type'] != 'yes/no':
        return None, 0
    votes = Counter(normalized(row['answer']) for row in answers)
    target = next((target for target in ('yes', 'no') if votes[target] >= 9), None)
    if target is not None and normalized(annotation['multiple_choice_answer']) != target:
        raise ValueError('Official majority and released multiple-choice answer disagree')
    return target, votes[target] if target else 0


def question_tags(text: str) -> list[str]:
    """Heuristics prioritize image review; tags are never semantic admission."""
    q = normalized(text)
    inferred = re.search(r'\b(?:raining|sunny|nighttime|daytime|night time|day time|'
                         r'hot|cold|professional|talking|thinking|happy|sad|thirsty|hungry|'
                         r'asian|american|japanese|chinese|expensive|cheap|delicious|'
                         r'taste|healthy|dangerous|safe|birthday|public|private)\b', q)
    tags = []
    if inferred:
        tags.append('inference_or_context_review_needed')
    if q.startswith(('is there ', 'are there ')):
        tags.append('existence_prefix_review_candidate')
    if re.search(r'\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|'
                 r'grey|color|colour|colored|coloured)\b', q):
        tags.append('color_word_review_candidate')
    if re.search(r'\b(?:under|above|below|behind|between|beside|inside|left|right)\b|'
                 r'\b(?:on top of|in front of|next to)\b', q):
        tags.append('spatial_word_review_candidate')
    if re.search(r'\b(?:wearing|open|closed|empty|full|striped|spotted|frosted|bruised)\b|'
                 r'\b(?:lamp|light|lights|tv|television)\b.*\bon\b', q):
        tags.append('visible_state_word_review_candidate')
    if not tags:
        tags.append('other_question_review_needed')
    return tags


def pair_counts(rows: list[dict]) -> dict:
    images = {side['coco_image_id'] for row in rows for side in row['sides']}
    qids = {side['question_id'] for row in rows for side in row['sides']}
    return {'pairs': len(rows), 'unique_questions': len(qids), 'unique_images': len(images),
            'normalized_question_texts': len({row['normalized_question'] for row in rows})}


FINAL_SALT = 'rarevlm-complementary-vqa-v1-eligible-component-20261018'
FINAL_ORDER_SALT = 'rarevlm-complementary-vqa-v1-pair-order-20261018'
FINAL_REVIEW_SALT = 'rarevlm-complementary-vqa-v1-blind-review-20261018'
FINAL_TARGETS = {'train': 600, 'screen': 256, 'confirm': 256}
FINAL_BANS = ('full-grown', 'full grown', 'grown', 'adult', 'elderly', 'young', 'old',
              'real', 'fake', 'toy', 'alive', 'dead', 'look', 'looks', 'seem', 'seems',
              'feel', 'feels', 'think', 'thinks', 'want', 'wants', 'need', 'needs',
              'like', 'likes', 'enjoy', 'enjoying', 'interested', 'about to', 'going to',
              'will', 'would', 'could', 'should', 'comfortable', 'friendly', 'bored',
              'tired', 'scared', 'healthy', 'danger', 'beautiful', 'pretty', 'nice',
              'summer', 'winter', 'morning', 'afternoon', 'night', 'daytime', 'daylight',
              'sunny', 'raining', 'bruised', 'toasted')
VISIBLE_TAGS = {'existence_prefix_review_candidate', 'color_word_review_candidate',
                'spatial_word_review_candidate', 'visible_state_word_review_candidate'}
REVISION2_BANS = ('modern', 'art', 'artistic', 'stadium', 'plenty', 'enough', 'fruit',
                  'meat', 'vegetables', 'percent', 'many', 'lots', 'a lot', 'a lot of')
REVISION2_PHRASES = ('apple product', 'product from apple', 'made by', 'brand')
REVISION2_REVIEW_SALT = 'rarevlm-complementary-vqa-v1-blind-review-r2-20261018'
REVISION2_HUMAN_EXISTENCE = (r'^(?:is|are) there (?:a |an |any |some )?'
                            r'(?:person|people|man|men|woman|women|human|child|children|'
                            r'boy|boys|girl|girls|kid|kids)\b')


def atomic_progress(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def final_candidate_pool(output: Path, revision: int = 1) -> tuple[dict, dict[str, list[dict]]]:
    candidates = [json.loads(line) for line in (output / 'CANDIDATE_PAIRS.jsonl').read_text().splitlines()]
    components = json.loads((output / 'FILTERED_GRAPH_DIAGNOSTIC_COMPONENTS.json').read_text())
    image_components = {f'{image_id:012d}': component for component in components
                        for image_id in component['image_ids']}
    forbidden = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(word) for word in FINAL_BANS) + r')(?!\w)')
    race = re.compile(r'\b(?:(?:person|man|woman|people)\s+(?:white|black|brown)|'
                      r'(?:white|black|brown)\s+(?:person|man|woman|people))\b')
    r2_words = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(word) for word in REVISION2_BANS) + r')(?!\w)')
    r2_phrases = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(word) for word in REVISION2_PHRASES) + r')(?!\w)')
    r2_human = re.compile(REVISION2_HUMAN_EXISTENCE)
    rejected = Counter()
    folds = {split: [] for split in FINAL_TARGETS}
    for pair in candidates:
        sides = pair['sides']
        component = image_components[sides[0]['coco_image_id']]
        if image_components[sides[1]['coco_image_id']]['component_id'] != component['component_id']:
            raise ValueError('Eligible official pair crosses its source component')
        if component['images'] > 200:
            rejected['eligible_component_exceeds_200_images'] += 1
            continue
        if sides[0]['question'] != sides[1]['question']:
            rejected['original_question_not_exact_equal'] += 1
            continue
        tags = set(pair['question_tags'])
        if not tags & VISIBLE_TAGS:
            rejected['no_allowed_visible_question_tag'] += 1
            continue
        if 'inference_or_context_review_needed' in tags:
            rejected['original_inference_or_context_tag'] += 1
            continue
        if forbidden.search(pair['normalized_question']):
            rejected['fixed_whole_word_or_phrase_ban'] += 1
            continue
        if race.search(pair['normalized_question']):
            rejected['fixed_race_color_expression_ban'] += 1
            continue
        if revision == 2:
            question = pair['normalized_question']
            if r2_words.search(question):
                rejected['revision2_global_word_or_phrase_ban'] += 1
                continue
            if '%' in question or r2_phrases.search(question):
                rejected['revision2_percent_brand_or_apple_phrase'] += 1
                continue
            if r2_human.search(question):
                rejected['revision2_existence_human_identity_question'] += 1
                continue
        fold = int(content_sha([FINAL_SALT, component['component_id']])[:16], 16) % 100
        split = 'train' if fold < 70 else 'screen' if fold < 85 else 'confirm'
        pair = {**pair, 'final_component_id': component['component_id'],
                'final_component_images': component['images'], 'final_hash_fold': fold,
                'final_split': split}
        folds[split].append(pair)
    priority, capacity = {}, {}
    for split, rows in folds.items():
        ordered = sorted(rows, key=lambda pair: (content_sha([FINAL_ORDER_SALT, split, pair['pair_id']]),
                                                  pair['pair_id']))
        used_images, used_questions, eligible = set(), set(), []
        for pair in ordered:
            images = {side['coco_image_id'] for side in pair['sides']}
            questions = {side['question_id'] for side in pair['sides']}
            if images & used_images or questions & used_questions:
                continue
            used_images.update(images)
            used_questions.update(questions)
            eligible.append(pair)
        priority[split] = eligible
        capacity[split] = {'before_greedy_pairs': len(rows), 'image_and_qid_disjoint_pairs': len(eligible),
                           'target': FINAL_TARGETS[split],
                           'question_types_per_pair': dict(Counter(pair['sides'][0]['question_type']
                                                                  for pair in rows).most_common())}
    plan = {
        'status': 'fixed_before_image_download_and_any_new_model_prediction',
        'candidate_file_sha256': sha_file(output / 'CANDIDATE_PAIRS.jsonl'),
        'eligible_component_file_sha256': sha_file(output / 'FILTERED_GRAPH_DIAGNOSTIC_COMPONENTS.json'),
        'historical_boundary_sha256': sha_file(output / 'HISTORICAL_IMAGE_BOUNDARY.json'),
        'full_eligible_graph': 'All 32192 official pairs after two-sided >=9/10 yes/no, '
                               'opposite answers and historical COCO-ID exclusion; before final question filters.',
        'component_identity': 'Inventory SHA256 of canonical JSON sorted integer image IDs',
        'component_maximum_images': 200, 'fold_salt': FINAL_SALT,
        'fold_rule': 'sha256([salt,component_id]) first16hex modulo100; 0..69 train, '
                     '70..84 screen, 85..99 confirm',
        'pair_order_salt': FINAL_ORDER_SALT,
        'pair_order_rule': 'ascending sha256([pair_order_salt,split,pair_id]); '
                           'greedy skip repeated image or question ID within each fold',
        'original_question_rule': 'Both original strings exactly equal; no rewrite',
        'allowed_tags_at_least_one': sorted(VISIBLE_TAGS),
        'forbidden_original_tag': 'inference_or_context_review_needed',
        'whole_word_or_phrase_bans': list(FINAL_BANS),
        'race_color_regex': race.pattern,
        'sequential_rejection_counts': dict(rejected), 'capacity': capacity,
        'targets': FINAL_TARGETS, 'maximum_downloaded_new_images': 2500,
        'fixed_priority_pair_ids': {split: [pair['pair_id'] for pair in rows]
                                    for split, rows in priority.items()},
        'post_download_skips': 'Historical file/pixel SHA overlap, selected-image file/pixel '
                               'duplication, or failed original-image download; fixed next priority pair only.',
        'model_prediction_selection': False,
        'semantic_admission': 'Pending assistant blind review: >=14/16 training pairs and '
                              '>=7/8 screen pairs have both sides clear and unchanged original labels matching.'}
    if revision == 2:
        plan['data_revision'] = 2
        plan['revision2_global_filters'] = {
            'additional_whole_word_or_phrases': list(REVISION2_BANS),
            'additional_phrases': list(REVISION2_PHRASES), 'any_percent_symbol': True,
            'human_existence_regex': REVISION2_HUMAN_EXISTENCE,
            'no_individual_question_or_image_blacklist_for_training_selection': True}
        plan['review_salt'] = REVISION2_REVIEW_SALT
        plan['review_only_exclusion'] = ('Every image ID/question ID in the original24-pair blind cohort; '
                                        'ordinary train/screen/confirm selection is not affected by these IDs.')
        plan['shared_cache'] = ('Preserve all revision1 images/receipts; current revision original-image '
                               'attempt bound2500 applies to this revision, not to historical cache occupancy.')
    return plan, priority


def download_original_image(side: dict, output: Path, preferred_host: str | None) -> dict:
    from PIL import Image
    coco_id = side['coco_image_id']
    image_path = output / 'images' / f'COCO_train2014_{coco_id}.jpg'
    receipt_path = output / 'image_receipts' / f'{coco_id}.json'
    if image_path.exists():
        if not receipt_path.exists():
            raise ValueError(f'Original image {coco_id} has no receipt; refusing to replace it')
        receipt = json.loads(receipt_path.read_text())
        if receipt['coco_image_id'] != coco_id or sha_file(image_path) != receipt['sha256']:
            raise ValueError(f'Original image receipt mismatch for {coco_id}')
        return receipt
    first_url = side['image_url']
    backup_url = ('https://s3.amazonaws.com/images.cocodataset.org/train2014/'
                  f'COCO_train2014_{coco_id}.jpg')
    urls = [backup_url, first_url] if preferred_host == 's3' else [first_url, backup_url]
    attempts = []
    for url in urls:
        for attempt in range(2):
            try:
                request = Request(url, headers={'User-Agent': 'RareVLM-data-preparation/1.0'})
                with urlopen(request, timeout=25) as response:
                    payload = response.read()
                with Image.open(io.BytesIO(payload)) as raw:
                    original_format, original_mode = raw.format, raw.mode
                    rgb = raw.convert('RGB')
                    rgb.load()
                    if min(rgb.size) < 1 or rgb.width * rgb.height > 50_000_000:
                        raise ValueError('Invalid original COCO image dimensions')
                    if original_format != 'JPEG':
                        raise ValueError('Official COCO photograph is not an original JPEG')
                    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                receipt = {'coco_image_id': coco_id, 'path': str(image_path.resolve()),
                           'sha256': hashlib.sha256(payload).hexdigest(),
                           'file_sha256': hashlib.sha256(payload).hexdigest(),
                           'pixel_sha256': pixels, 'width': rgb.width, 'height': rgb.height,
                           'source_format': original_format, 'source_mode': original_mode,
                           'decoded_mode': 'RGB', 'requested_original_url': first_url,
                           'actual_download_url': url, 'bytes': len(payload),
                           'prior_url_attempts': attempts, 'downloaded_original_bytes': True,
                           'resized_or_cropped': False}
                partial = image_path.with_suffix('.partial')
                partial.write_bytes(payload)
                partial.replace(image_path)
                write_json(receipt_path, receipt)
                return receipt
            except Exception as exc:
                attempts.append({'url': url, 'attempt': attempt + 1,
                                 'error': type(exc).__name__ + ': ' + str(exc)[:240]})
                if attempt == 0:
                    time.sleep(0.5)
    return {'coco_image_id': coco_id, 'download_failed': True, 'url_attempts': attempts}


def blind_review_sheets(manifest: dict, output: Path, review_salt: str = FINAL_REVIEW_SALT,
                        excluded_images=frozenset(), excluded_qids=frozenset()) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    review_dir = output / 'blind_review'
    review_dir.mkdir(exist_ok=True)
    font_paths = list((PRIVATE / '.venvs').glob('*/lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'))
    if not font_paths:
        font_paths = list((PRIVATE / '.tools').glob('**/mpl-data/fonts/ttf/DejaVuSans.ttf'))
    if not font_paths:
        for packages in ('site-packages', 'dist-packages'):
            font_paths.extend(Path('/usr/local/lib').glob(
                f'python*/{packages}/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'))
    font = ImageFont.truetype(str(font_paths[0]) if font_paths else 'DejaVuSans.ttf', 22)
    title_font = ImageFont.truetype(str(font_paths[0]) if font_paths else 'DejaVuSans.ttf', 25)
    keys, sheets = [], []
    rows_by_id = {row['question_id']: row for split in ('train_all', 'dev')
                  for row in manifest['splits'][split]}
    for split, count, prefix in [('train', 16, 'T'), ('screen', 8, 'S')]:
        review_pool = [pair for pair in manifest['complementary_pairs'][split]
                       if not {pair['positive_question_id'], pair['negative_question_id']} & excluded_qids
                       and not {rows_by_id[pair['positive_question_id']]['image_id'],
                                rows_by_id[pair['negative_question_id']]['image_id']} & excluded_images]
        pairs = sorted(review_pool,
                       key=lambda pair: content_sha([review_salt, split, pair['pair_id']]))[:count]
        if len(pairs) != count:
            raise ValueError('Insufficient fresh review pairs; no old review example reused')
        for sheet_index in range(0, count, 4):
            canvas = Image.new('RGB', (1120, 1800), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 12), f'{split.upper()} - blind complementary image review',
                      fill='black', font=title_font)
            draw.text((15, 47), 'Original frames resized only for display; labels hidden.',
                      fill=(65, 65, 65), font=font)
            for local_index, pair in enumerate(pairs[sheet_index:sheet_index + 4]):
                index = sheet_index + local_index
                review_id = f'{prefix}{index:02d}'
                top = 90 + local_index * 425
                lines = textwrap.wrap(f'{review_id}: {pair["question"]}', width=86)
                if len(lines) > 4:
                    raise ValueError('Review question exceeds reserved display area')
                for line_index, text in enumerate(lines):
                    draw.text((15, top + 27 * line_index), text, fill='black', font=font)
                qids = [pair['negative_question_id'], pair['positive_question_id']]
                if int(content_sha([review_salt, 'ab', split, pair['pair_id']])[-1], 16) % 2:
                    qids.reverse()
                mappings = {}
                image_top = top + 110
                for side_index, (letter, qid) in enumerate(zip(('A', 'B'), qids)):
                    row = rows_by_id[qid]
                    item = manifest['images'][row['image_id']]
                    with Image.open(item['path']) as raw:
                        display = raw.convert('RGB')
                        display.thumbnail((535, 285), Image.Resampling.LANCZOS)
                        x = 15 + side_index * 555
                        canvas.paste(display, (x + (535 - display.width) // 2, image_top))
                    draw.text((x, top + 80), f'{review_id} {letter}', fill='black', font=font)
                    mappings[letter] = {'question_id': qid, 'image_id': row['image_id'],
                                        'target': row['target'], 'path': item['path']}
                keys.append({'review_id': review_id, 'split': split, 'pair_id': pair['pair_id'],
                             'question': pair['question'], 'mapping': mappings,
                             'semantic_review_status': 'pending_blind_assistant_review'})
            sheet = review_dir / f'{split}_{sheet_index // 4:02d}.jpg'
            canvas.save(sheet, quality=94)
            sheets.append({'split': split, 'path': str(sheet.resolve()), 'sha256': sha_file(sheet)})
    write_json(review_dir / 'PRIVATE_REVIEW_KEY.json', {'review_salt': review_salt,
                                                      'rows': keys, 'scope': 'No confirm images displayed.'})
    write_json(output / 'BLIND_REVIEW_COHORT.json', {
        'status': 'pending_blind_assistant_review', 'review_salt': review_salt,
        'selection': 'Fixed hash sample: train16 pairs and screen8 pairs; no model predictions.',
        'rows': [{key: value for key, value in row.items() if key != 'mapping'} |
                 {'mapping': {letter: {key: value for key, value in values.items() if key != 'target'}
                              for letter, values in row['mapping'].items()}} for row in keys],
        'label_key': str((review_dir / 'PRIVATE_REVIEW_KEY.json').resolve()),
        'labels_on_sheets': False, 'confirm_displayed': False})
    write_json(review_dir / 'SHEETS.json', {'sheets': sheets, 'labels_on_sheets': False,
                                          'review_status': 'pending', 'confirm_displayed': False})
    return {'training_pairs': 16, 'screen_pairs': 8, 'sheets': len(sheets),
            'key_path': str((review_dir / 'PRIVATE_REVIEW_KEY.json').resolve()),
            'status': 'pending_blind_assistant_review', 'confirm_displayed': False}


def early_review_manifest(selected_seed: list[dict], receipts: dict, review_salt: str = FINAL_REVIEW_SALT,
                          excluded_images=frozenset(), excluded_qids=frozenset()) -> dict:
    """Display-only structure for the fixed sample; no model or evaluation input."""
    manifest = {'images': {}, 'splits': {'train_all': [], 'dev': []},
                'complementary_pairs': {'train': [], 'screen': []}}
    for split, count, row_split in [('train', 16, 'train_all'), ('screen', 8, 'dev')]:
        all_pairs = [pair for pair in selected_seed if pair['final_split'] == split]
        review_pool = [pair for pair in all_pairs if
                       not {side['question_id'] for side in pair['sides']} & excluded_qids and
                       not {side['image_id'] for side in pair['sides']} & excluded_images]
        chosen = sorted(review_pool, key=lambda pair: content_sha(
            [review_salt, split, pair['pair_id']]))[:count]
        if len(chosen) != count:
            raise ValueError('Insufficient fresh review examples')
        for pair in all_pairs:
            ids = {side['target']: side['question_id'] for side in pair['sides']}
            manifest['complementary_pairs'][split].append({
                'pair_id': pair['pair_id'], 'positive_question_id': ids['yes'],
                'negative_question_id': ids['no'], 'question': pair['sides'][0]['question'],
                'component_id': pair['final_component_id'], 'question_tags': pair['question_tags']})
            manifest['splits'][row_split].extend(pair['sides'])
        for pair in chosen:
            for side in pair['sides']:
                manifest['images'][side['image_id']] = receipts[side['coco_image_id']]
    return manifest


def select_and_download(args) -> None:
    from vlm_deep.common import content_sha as manifest_content_sha, read_manifest
    base_output = args.output.resolve()
    if base_output != (OUTPUTS / 'vlm_complementary_rank_v1/data_preparation').resolve():
        raise ValueError('Final selection is scoped to its registered persistent output')
    revision = 2 if args.stage == 'revision2' else 1
    output = base_output / 'revision2' if revision == 2 else base_output
    output.mkdir(exist_ok=True)
    cache_root = base_output
    plan, priority = final_candidate_pool(base_output, revision=revision)
    excluded_images, excluded_qids = set(), set()
    review_salt = REVISION2_REVIEW_SALT if revision == 2 else FINAL_REVIEW_SALT
    if revision == 2:
        old_review_path = base_output / 'BLIND_REVIEW_COHORT.json'
        old_review = json.loads(old_review_path.read_text())
        for row in old_review['rows']:
            for side in row['mapping'].values():
                excluded_images.add(side['image_id'])
                excluded_qids.add(side['question_id'])
        plan['original_review_cohort_sha256'] = sha_file(old_review_path)
        plan['review_only_excluded_image_ids'] = sorted(excluded_images)
        plan['review_only_excluded_question_ids'] = sorted(excluded_qids)
        write_json(output / 'SOURCE_LOCK.json', {
            'source_page': SOURCE_PAGE, 'parent_source_lock': str(base_output / 'SOURCE_LOCK.json'),
            'parent_source_lock_sha256': sha_file(base_output / 'SOURCE_LOCK.json'),
            'parent_sources': json.loads((base_output / 'SOURCE_LOCK.json').read_text()),
            'candidate_file_sha256': sha_file(base_output / 'CANDIDATE_PAIRS.jsonl'),
            'historical_boundary_sha256': sha_file(base_output / 'HISTORICAL_IMAGE_BOUNDARY.json'),
            'revision': 2, 'shared_original_image_cache': str(cache_root / 'images')})
    write_json(output / 'FINAL_SELECTION_PLAN.json', plan)
    if any(len(priority[split]) < target for split, target in FINAL_TARGETS.items()):
        raise ValueError('Fixed data rules leave an undersized fold; stop without relaxing the rules')
    for name in ('images', 'image_receipts'):
        (cache_root / name).mkdir(exist_ok=True)
    selected_seed = [pair for split, target in FINAL_TARGETS.items() for pair in priority[split][:target]]
    review_pairs = [pair for split, count in [('train', 16), ('screen', 8)]
                    for pair in sorted([pair for pair in selected_seed if pair['final_split'] == split and
                                        not {side['question_id'] for side in pair['sides']} & excluded_qids and
                                        not {side['image_id'] for side in pair['sides']} & excluded_images],
                                       key=lambda pair: content_sha(
                                           [review_salt, split, pair['pair_id']]))[:count]]
    if len(review_pairs) != 24:
        raise ValueError('Insufficient fresh blind review pairs')
    review_image_ids = {side['coco_image_id'] for pair in review_pairs for side in pair['sides']}
    side_by_id = {side['coco_image_id']: side for pair in selected_seed for side in pair['sides']}
    if len(side_by_id) != 2 * sum(FINAL_TARGETS.values()):
        raise ValueError('Final component folds or greedy pair identities overlap')
    existing_cache_ids = {path.stem for path in (cache_root / 'image_receipts').glob('*.json')}
    cache_reuse_plan = {
        'current_revision_target_images': len(side_by_id),
        'reusable_original_images': len(set(side_by_id) & existing_cache_ids),
        'new_original_images_needed': len(set(side_by_id) - existing_cache_ids),
        'shared_cache_images_before_revision': len(existing_cache_ids),
        'new_blind_review_image_count': len(review_image_ids),
        'new_blind_images_already_cached': len(review_image_ids & existing_cache_ids),
        'old_review_image_overlap': len({side['image_id'] for pair in review_pairs for side in pair['sides']} & excluded_images),
        'old_review_question_overlap': len({side['question_id'] for pair in review_pairs for side in pair['sides']} & excluded_qids)}
    if not (output / 'CACHE_REUSE_PLAN.json').exists():
        write_json(output / 'CACHE_REUSE_PLAN.json', cache_reuse_plan)
    if len(existing_cache_ids) + len(set(side_by_id) - existing_cache_ids) > 2500:
        raise ValueError('Cumulative shared original-image cache would exceed2500 images')
    print(json.dumps({'event': 'final_selection_capacity', 'revision': revision,
                      'capacity': {split: len(rows) for split, rows in priority.items()},
                      'cached': len(set(side_by_id) & existing_cache_ids),
                      'new_images': len(set(side_by_id) - existing_cache_ids)}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    boundaries = json.loads((base_output / 'HISTORICAL_IMAGE_BOUNDARY.json').read_text())
    prior_pixels = set(boundaries['blocked_pixel_sha256'])
    prior_files = set(boundaries['blocked_file_sha256'])
    prior_coco = {int(x) for x in boundaries['blocked_coco_ids']}
    receipts = {}
    first_id = next(iter(side_by_id))
    first = download_original_image(side_by_id[first_id], cache_root, None)
    if first.get('download_failed'):
        write_json(output / 'INITIAL_NETWORK_PROBE_FAILED.json', first)
        raise ValueError('First original COCO image failed; bulk download not started')
    receipts[first_id] = first
    host = 's3' if first.get('actual_download_url', '').startswith('https://s3.amazonaws.com/') else None
    for side in selected_seed[0]['sides']:
        if side['coco_image_id'] not in receipts:
            receipt = download_original_image(side, cache_root, host)
            if receipt.get('download_failed'):
                write_json(output / 'INITIAL_NETWORK_PROBE_FAILED.json', receipt)
                raise ValueError('First complementary image pair failed; bulk download not started')
            receipts[side['coco_image_id']] = receipt
    completed = len(receipts)
    progress_path = output / 'DOWNLOAD_PROGRESS.json'
    atomic_progress(progress_path, {'status': 'downloading_original_seed_images',
                                    'images_completed': completed, 'initial_planned_images': len(side_by_id)})
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        for phase, image_ids in [('fixed_blind_review_images', review_image_ids),
                                 ('remaining_original_images', set(side_by_id) - review_image_ids)]:
            futures = {executor.submit(download_original_image, side_by_id[image_id], cache_root, host): image_id
                       for image_id in sorted(image_ids) if image_id not in receipts}
            for future in as_completed(futures):
                image_id = futures[future]
                receipts[image_id] = future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(side_by_id):
                    atomic_progress(progress_path, {'status': phase,
                                                    'images_completed': completed,
                                                    'initial_planned_images': len(side_by_id),
                                                    'download_failures': sum(bool(row.get('download_failed'))
                                                                             for row in receipts.values())})
                    print(json.dumps({'event': 'download_progress', 'completed': completed,
                                      'planned': len(side_by_id)}, ensure_ascii=False), flush=True)
            if phase == 'fixed_blind_review_images':
                review_receipts = [receipts[image_id] for image_id in review_image_ids]
                has_invalid = any(receipt.get('download_failed') for receipt in review_receipts)
                has_invalid = has_invalid or any(receipt.get('sha256') in prior_files or
                                                 receipt.get('pixel_sha256') in prior_pixels
                                                 for receipt in review_receipts)
                if not has_invalid:
                    blind_review_sheets(early_review_manifest(selected_seed, receipts, review_salt,
                                                             excluded_images, excluded_qids),
                                        output, review_salt, excluded_images, excluded_qids)
                    write_json(output / 'BLIND_REVIEW_READY.json', {
                        'status': 'fixed_sample_images_ready_before_remaining_bulk_download',
                        'training_pairs': 16, 'screen_pairs': 8,
                        'cohort_path': str((output / 'BLIND_REVIEW_COHORT.json').resolve()),
                        'remaining_bulk_content_boundary': 'pending; final cohort identity must remain equal',
                        'original_labels_on_sheets': False, 'confirm_displayed': False})
                    print(json.dumps({'event': 'blind_review_ready', 'training_pairs': 16,
                                      'screen_pairs': 8}, ensure_ascii=False), flush=True)
    selected = {split: [] for split in FINAL_TARGETS}
    skipped, used_pixels, used_files, used_coco, used_qids = [], set(), set(), set(), set()
    for split, target in FINAL_TARGETS.items():
        for pair in priority[split]:
            if len(selected[split]) == target:
                break
            identifiers = [side['coco_image_id'] for side in pair['sides']]
            for side in pair['sides']:
                image_id = side['coco_image_id']
                if image_id not in receipts:
                    if len(receipts) >= 2500 or len(list((cache_root / 'images').glob('*.jpg'))) >= 2500:
                        raise ValueError('Maximum2500 original-image downloads reached; stop without expansion')
                    receipts[image_id] = download_original_image(side, cache_root, host)
            pair_receipts = [receipts[image_id] for image_id in identifiers]
            reasons = []
            if any(receipt.get('download_failed') for receipt in pair_receipts):
                reasons.append('original_image_download_failed')
            else:
                files = [receipt['sha256'] for receipt in pair_receipts]
                pixels = [receipt['pixel_sha256'] for receipt in pair_receipts]
                qids = {side['question_id'] for side in pair['sides']}
                if any(int(image_id) in prior_coco for image_id in identifiers):
                    reasons.append('historical_coco_identity')
                if set(files) & prior_files:
                    reasons.append('historical_file_sha256')
                if set(pixels) & prior_pixels:
                    reasons.append('historical_pixel_sha256')
                if len(set(files)) != 2 or len(set(pixels)) != 2:
                    reasons.append('same_image_content_within_pair')
                if set(files) & used_files or set(pixels) & used_pixels or set(identifiers) & used_coco:
                    reasons.append('selected_image_content_reused')
                if qids & used_qids:
                    reasons.append('selected_question_id_reused')
            if reasons:
                skipped.append({'split': split, 'pair_id': pair['pair_id'], 'reasons': reasons,
                                'image_receipts': pair_receipts})
                continue
            selected[split].append(pair)
            used_files.update(receipt['sha256'] for receipt in pair_receipts)
            used_pixels.update(receipt['pixel_sha256'] for receipt in pair_receipts)
            used_coco.update(identifiers)
            used_qids.update(side['question_id'] for side in pair['sides'])
        if len(selected[split]) != target:
            write_json(output / 'POST_DOWNLOAD_SKIPS.json', skipped)
            raise ValueError('Post-download image boundary leaves an undersized fold; no rules relaxed')
    image_map, split_rows, pair_rows = {}, {}, {}
    for split, pairs in selected.items():
        flat_rows, links = [], []
        for pair in pairs:
            question_ids = {}
            for side in pair['sides']:
                receipt = receipts[side['coco_image_id']]
                image_map[side['image_id']] = {**receipt, 'source_split': 'official_coco_train2014',
                                               'pixel_hash_contract': 'sha256(str(RGB.size).encode()+RGB.tobytes())'}
                row = {key: side[key] for key in ('question_id', 'image_id', 'question', 'target', 'answers',
                                                  'prompt_suffix', 'source_question_id', 'question_type',
                                                  'coco_image_id', 'answer_vote_count')}
                row.update({'task': 'human_visual_yesno', 'source_split': split,
                            'label_source': 'official_vqa_v2_train2014_human_answers',
                            'complementary_pair_id': pair['pair_id'],
                            'negative_rule': 'original human answer; not missing object annotation'})
                flat_rows.append(row)
                question_ids[side['target']] = side['question_id']
            links.append({'pair_id': pair['pair_id'], 'positive_question_id': question_ids['yes'],
                          'negative_question_id': question_ids['no'],
                          'question': pair['sides'][0]['question'],
                          'component_id': pair['final_component_id'],
                          'question_tags': pair['question_tags']})
        split_rows[split], pair_rows[split] = flat_rows, links
    manifest = {
        'schema_version': 'rarevlm.vlm_data.v2', 'training_allowed': False,
        'purpose': 'Official complementary same-question opposite-human-answer supervision; '
                   'semantic admission pending, no capability gain claimed.',
        'source_lock_sha256': sha_file(output / 'SOURCE_LOCK.json'),
        'selection_plan_sha256': sha_file(output / 'FINAL_SELECTION_PLAN.json'),
        'builder_sha256': sha_file(Path(__file__)), 'images': image_map,
        'splits': {'train_all': split_rows['train'], 'train_10k': split_rows['train'],
                   'dev': split_rows['screen'], 'final': split_rows['confirm']},
        'complementary_pairs': pair_rows,
        'notes': ['Official VQA-v2 train2014 source, exact original question text preserved.',
                  'Each pair includes one >=9/10-vote yes and one >=9/10-vote no original human answer.',
                  'Full eligible official graph components fixed before final question filters.',
                  'Components over200 images excluded; final component hash split70/15/15.',
                  'Selected pairs share no COCO ID, question ID, raw file hash or RGB pixel hash.',
                  'All listed historical image-ID and file/pixel boundaries excluded.',
                  'No new model prediction used to select examples.',
                  'Blind assistant review pending; no expert annotation claim.',
                  'Pretraining and perceptual near-duplicate overlap are unknown.']}
    manifest['manifest_sha256'] = manifest_content_sha(manifest)
    manifest_path = output / 'MANIFEST.json'
    write_json(manifest_path, manifest)
    read_manifest(manifest_path, verify_images=True)
    for split, links in pair_rows.items():
        by_id = {row['question_id']: row for row in split_rows[split]}
        for pair in links:
            positive, negative = by_id[pair['positive_question_id']], by_id[pair['negative_question_id']]
            if (positive['target'], negative['target']) != ('yes', 'no'):
                raise ValueError('Complementary pair label linkage mismatch')
            if positive['question'] != negative['question'] or positive['image_id'] == negative['image_id']:
                raise ValueError('Complementary pair exact-question/distinct-image contract failed')
    review = blind_review_sheets(manifest, output, review_salt, excluded_images, excluded_qids)
    write_json(output / 'POST_DOWNLOAD_SKIPS.json', skipped)
    write_json(output / 'FINAL_SELECTION_BUILDER_SOURCE_METADATA.json',
               {'builder_sha256': sha_file(Path(__file__)), 'source_snapshot': 'FINAL_SELECTION_BUILDER_SOURCE.py'})
    source_snapshot = output / 'FINAL_SELECTION_BUILDER_SOURCE.py'
    if source_snapshot.exists() and source_snapshot.read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Final selection source snapshot changed')
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    complete = {'status': 'data_prepared_semantic_admission_pending', 'training_allowed': False,
                'manifest_path': str(manifest_path), 'manifest_sha256': sha_file(manifest_path),
                'manifest_core_content_sha256': manifest['manifest_sha256'],
                'selected_pairs': {split: len(pairs) for split, pairs in selected.items()},
                'selected_questions': {split: len(rows) for split, rows in split_rows.items()},
                'selected_images': len(image_map),
                'downloaded_original_images': sum(not row.get('download_failed') for row in receipts.values()),
                'downloaded_original_bytes': sum(row.get('bytes', 0) for row in receipts.values()),
                'shared_cache_original_images': len(list((cache_root / 'images').glob('*.jpg'))),
                'data_revision': revision,
                'post_download_skipped_pairs': len(skipped), 'manifest_reader_actual_files_verified': True,
                'blind_review': review, 'gpu_used': False, 'model_prediction_used_for_selection': False}
    write_json(output / 'PREPARATION_COMPLETE.json', complete)
    atomic_progress(progress_path, complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-downloads', type=Path, default=OUTPUTS /
                        'vlm_deep_code_inspection_v4/vqa_official')
    parser.add_argument('--output', type=Path, default=OUTPUTS /
                        'vlm_complementary_rank_v1/data_preparation')
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    parser.add_argument('--stage', choices=('inventory', 'select', 'revision2'), default='inventory')
    parser.add_argument('--download-workers', type=int, default=12)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.stage in ('select', 'revision2'):
        select_and_download(args)
        return
    output = args.output.resolve()
    if not output.is_relative_to(PRIVATE) or not args.official_downloads.resolve().is_relative_to(PRIVATE):
        raise ValueError('All sources and outputs must stay in the persistent workspace')
    if output != (OUTPUTS / 'vlm_complementary_rank_v1/data_preparation').resolve():
        raise ValueError('This version is scoped to its registered data-preparation directory')
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / 'official_sources'
    downloads.mkdir(exist_ok=True)
    questions_doc, question_source = existing_official_archive(args.official_downloads, 'Questions')
    questions = {row['question_id']: row for row in questions_doc.pop('questions')}
    annotations_doc, annotation_source = existing_official_archive(args.official_downloads, 'Annotations')
    annotations = {row['question_id']: row for row in annotations_doc.pop('annotations')}
    if set(questions) != set(annotations) or len(questions) != 443757:
        raise ValueError('Unexpected official VQA-v2 train2014 question/annotation identity set')
    pair_path, pair_receipt = pair_archive(downloads)
    with zipfile.ZipFile(pair_path) as archive:
        member, = archive.namelist()
        official_pairs = json.loads(archive.read(member))
    if len(official_pairs) != 200394:
        raise ValueError('Unexpected official Training complementary pair count')
    exclusions = collect_boundaries([OUTPUTS / item for item in DEFAULT_EXCLUSIONS] +
                                    [path.resolve() for path in args.exclude_manifest])
    write_json(output / 'HISTORICAL_IMAGE_BOUNDARY.json', exclusions)
    blocked_ids = set(exclusions['blocked_coco_ids'])
    graph = UnionFind()
    seen_pairs = set()
    rejected = Counter()
    candidates = []
    for first_id, second_id in official_pairs:
        first_q, second_q = questions[first_id], questions[second_id]
        first_a, second_a = annotations[first_id], annotations[second_id]
        for q, a in ((first_q, first_a), (second_q, second_a)):
            if q['image_id'] != a['image_id']:
                raise ValueError('Question and annotation image identities differ')
        image_ids = (int(first_q['image_id']), int(second_q['image_id']))
        if image_ids[0] == image_ids[1]:
            raise ValueError('Official complementary pair unexpectedly uses one image')
        graph.union(*image_ids)  # Complete official graph, before every candidate filter.
        pair_key = tuple(sorted((first_id, second_id)))
        if pair_key in seen_pairs:
            rejected['duplicate_unordered_official_pair'] += 1
            continue
        seen_pairs.add(pair_key)
        if normalized(first_q['question']) != normalized(second_q['question']):
            rejected['different_original_question_after_case_whitespace_only'] += 1
            continue
        labels = (consensus(first_a), consensus(second_a))
        if any(target is None for target, count in labels):
            rejected['not_both_yesno_with_at_least_9_votes'] += 1
            continue
        if {target for target, count in labels} != {'yes', 'no'}:
            rejected['not_opposite_yesno'] += 1
            continue
        if set(image_ids) & blocked_ids:
            rejected['historical_coco_image_identity_overlap'] += 1
            continue
        sides = []
        for q, a, (target, votes) in zip((first_q, second_q), (first_a, second_a), labels):
            coco_id = f'{int(q["image_id"]):012d}'
            sides.append({'question_id': f'vqa2_train2014:{q["question_id"]}',
                          'source_question_id': q['question_id'], 'coco_image_id': coco_id,
                          'image_id': f'coco2014:{coco_id}', 'question': q['question'],
                          'question_type': a['question_type'], 'target': target,
                          'answer_vote_count': votes,
                          'answers': [answer['answer'] for answer in a['answers']],
                          'prompt_suffix': '\nAnswer yes or no.',
                          'image_url': f'https://images.cocodataset.org/train2014/'
                                       f'COCO_train2014_{coco_id}.jpg'})
        sides.sort(key=lambda side: side['target'])
        candidates.append({'pair_id': f'vqa2_train2014_pair:{pair_key[0]}:{pair_key[1]}',
                           'official_question_ids': list(pair_key), 'sides': sides,
                           'normalized_question': normalized(first_q['question']),
                           'question_tags': question_tags(first_q['question']),
                           'source_question_type_pair': [first_a['question_type'], second_a['question_type']],
                           'semantic_review': 'unreviewed_original_human_labels',
                           'image_content_boundary': 'pending_after_download',
                           'training_allowed': False})
    image_components, components = graph.components()
    component_by_id = {row['component_id']: row for row in components}
    eligible_graph = UnionFind()
    for row in candidates:
        a, b = (int(side['coco_image_id']) for side in row['sides'])
        component_id = image_components[a]
        if image_components[b] != component_id:
            raise AssertionError('Official pair crosses its own full graph component')
        row['official_full_component_id'] = component_id
        row['provisional_full_component_split'] = component_by_id[component_id]['split']
        eligible_graph.union(a, b)
    _, eligible_components = eligible_graph.components()
    candidates.sort(key=lambda row: row['pair_id'])
    write_jsonl(output / 'CANDIDATE_PAIRS.jsonl', candidates)
    write_json(output / 'OFFICIAL_FULL_GRAPH_COMPONENTS.json', components)
    write_json(output / 'FILTERED_GRAPH_DIAGNOSTIC_COMPONENTS.json', eligible_components)
    by_split = {split: [row for row in candidates if row['provisional_full_component_split'] == split]
                for split in ('train', 'screen', 'confirm')}
    counts_by_component = Counter(row['official_full_component_id'] for row in candidates)
    tag_counts = Counter(tag for row in candidates for tag in row['question_tags'])
    likely_visible = [row for row in candidates if
                      'inference_or_context_review_needed' not in row['question_tags'] and
                      'other_question_review_needed' not in row['question_tags']]
    full_component_tops = [{**{k: v for k, v in row.items() if k != 'image_ids'},
                            'eligible_pairs': counts_by_component[row['component_id']]}
                           for row in components[:20]]
    sources = {'source_page': SOURCE_PAGE, 'questions': question_source,
               'annotations': annotation_source,
               'complementary_pairs': {**pair_receipt, 'member': member},
               'builder_sha256': sha_file(Path(__file__)),
               'historical_boundary_sha256': sha_file(output / 'HISTORICAL_IMAGE_BOUNDARY.json')}
    write_json(output / 'SOURCE_LOCK.json', sources)
    summary = {
        'status': 'inventory_only_awaiting_selection_and_split_decision',
        'training_allowed': False, 'images_downloaded': 0, 'gpu_used': False,
        'source_questions': len(questions), 'official_pairs': len(official_pairs),
        'unique_official_question_pairs': len(seen_pairs), 'rejected_sequential_counts': dict(rejected),
        'eligible_candidates_before_visual_review_and_pixel_boundary': pair_counts(candidates),
        'historical_boundary': {'coco_ids': len(blocked_ids),
                                'pixel_sha256': len(exclusions['blocked_pixel_sha256']),
                                'file_sha256': len(exclusions['blocked_file_sha256'])},
        'question_types_per_side': dict(Counter(side['question_type'] for row in candidates
                                                for side in row['sides']).most_common()),
        'question_prefix_3_words_per_pair': dict(Counter(' '.join(row['normalized_question'].split()[:3])
                                                        for row in candidates).most_common()),
        'question_tags_per_pair_nonexclusive': dict(tag_counts.most_common()),
        'question_text_only_visible_review_candidates': pair_counts(likely_visible),
        'question_text_only_visible_review_candidates_by_provisional_full_component_split': {
            split: pair_counts([row for row in likely_visible if
                                row['provisional_full_component_split'] == split])
            for split in by_split},
        'official_full_image_graph': {'components': len(components), 'images': len(graph.parent),
                                      'largest_20': full_component_tops},
        'filtered_eligible_image_graph_diagnostic_only': {
            'components': len(eligible_components), 'images': len(eligible_graph.parent),
            'largest_20': [{k: v for k, v in row.items() if k != 'image_ids'}
                           for row in eligible_components[:20]],
            'warning': 'This graph is not silently substituted for the official full graph split.'},
        'provisional_official_full_component_hash_split': {
            'salt': SALT, 'component_identity': 'sha256 of sorted complete component image IDs',
            'fold': 'sha256([salt, component_id]) first16hex modulo100; '
                    '0..69 train, 70..84 screen, 85..99 confirm',
            'counts': {split: pair_counts(rows) for split, rows in by_split.items()}},
        'requested_target_pairs': {'train': 600, 'screen': 256, 'confirm': 256},
        'requested_training_budget': {'maximum_pair_draws': 800, 'maximum_human_qa_exposures': 1600},
        'requested_maximum_new_images': 2500,
        'target_feasible_by_count_under_full_component_split': all(
            len(by_split[split]) >= target for split, target in
            {'train': 600, 'screen': 256, 'confirm': 256}.items()),
        'semantic_and_content_limitations': [
            'Original official human consensus is not new visual verification.',
            'Question tags only prioritize review and do not admit subjective or inferred labels.',
            'Known COCO identities are removed now; image file/pixel hashes must be checked after download.',
            'No photograph is downloaded until the parent selects a protocol.',
            'Pretraining and perceptual near-duplicate overlap remain unknown.']}
    write_json(output / 'INVENTORY.json', summary)
    examples = {}
    for tag in tag_counts:
        tagged = [row for row in candidates if tag in row['question_tags']]
        examples[tag] = sorted(tagged, key=lambda row: content_sha(['review_examples', row['pair_id']]))[:20]
    write_json(output / 'QUESTION_REVIEW_EXAMPLES.json', examples)
    print(json.dumps({key: summary[key] for key in (
        'status', 'official_pairs', 'eligible_candidates_before_visual_review_and_pixel_boundary',
        'historical_boundary', 'question_text_only_visible_review_candidates',
        'provisional_official_full_component_hash_split',
        'target_feasible_by_count_under_full_component_split')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
