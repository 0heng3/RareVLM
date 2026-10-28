"""Repair v1 candidates without equating annotation absence to visual absence.

Keep all original categories and sizes. Instance positives remain candidates for
semantic review; unknown negatives have no target and cannot enter SFT as `no`.
This transformation cannot manufacture reliable negative supervision.
"""
from collections import Counter


def repair_rows(manifest, excluded_rows):
    keys = ('image_id', 'pixel_sha256', 'file_sha256', 'coco_image_id')
    excluded = {key: {str(row[key]) for row in excluded_rows} for key in keys}
    candidates, unknown, images = [], [], {}
    for original in manifest['rows']:
        if any(str(original[key]) in excluded[key] for key in keys):
            continue
        row = dict(original)
        row['question_id'] = original['question_id'].replace('object_presence_v1:', 'object_presence_v3:')
        row['question_id_namespace'] = 'rarevlm.object_presence.v3:image_category'
        row['ancestor_question_id'] = original['question_id']
        row['supervision_eligible'] = False
        row['label_source'] = 'official_coco2017_instances_positive_evidence_only'
        if original['target'] == 'yes':
            if not original['bbox_evidence']:
                raise ValueError('Positive candidate has no instance evidence')
            row.update(target='yes', presence=True, label_status='annotation_positive_pending_semantic_review',
                       label_rule='noncrowd_instance_evidence_no_size_filter',
                       small_only=all(box['coco_area_bin'] == 'small' for box in row['bbox_evidence']))
            candidates.append(row)
        elif original['target'] == 'no':
            # Neither a caption omission nor an annotation omission proves `no`.
            row.update(target=None, presence=None, label_status='unknown',
                       label_rule='annotation_absence_is_not_verified_visual_absence')
            unknown.append(row)
        else:
            raise ValueError('Unexpected ancestor target')
        images[row['image_id']] = manifest['images'][row['image_id']]
    return candidates, unknown, images


def summary(candidates, unknown, images):
    return {'images': len(images), 'positive_candidates': len(candidates),
            'unknown_quarantined': len(unknown), 'supervised_no': 0,
            'positive_category_count': len({row['category_id'] for row in candidates}),
            'positive_categories': dict(sorted(Counter(row['category_name'] for row in candidates).items())),
            'small_only_positive_candidates': sum(row['small_only'] for row in candidates),
            'positive_instance_area_bins': dict(Counter(box['coco_area_bin'] for row in candidates
                                                        for box in row['bbox_evidence'])),
            'training_allowed': False,
            'reason': 'Candidate coverage repaired; semantic approval and verified negatives are still required'}
