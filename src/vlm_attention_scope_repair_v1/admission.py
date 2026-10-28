"""Bind new visual reviews to exact training inputs before applying v2 admission."""

from vlm_deep.common import content_sha
from vlm_semantic_admission_v1.admission import extract_training_rows
from vlm_semantic_admission_v2 import build_training_view


def apply_visual_reviews(source, prior, previous, reviews):
    rows = {q['question_id']: q for q in extract_training_rows(source)}
    seen = set()
    for review in reviews:
        qid = review['question_id']
        if qid in seen or qid not in rows:
            raise ValueError('Visual review must identify a unique source training QA')
        seen.add(qid)
        row = rows[qid]
        image = source['images'][row['image_id']]
        if (review['source_split'] != 'train'
                or review['pair_id'] != row['complementary_pair_id']
                or review['target'] != row['target']
                or review['question_sha256'] != content_sha(row['question'])
                or review['image_sha256'] != image['sha256']
                or not review.get('evidence', '').strip()):
            raise ValueError('Visual review input/evidence differs from source training QA')
    # A new uncertainty veto must survive an earlier supported review. Neither
    # model errors nor scores are passed to the admission function.
    return build_training_view(source, prior, list(previous) + list(reviews))
