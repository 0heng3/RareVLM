"""Posthoc official-word lookup; no prediction scoring, model input, or label changes."""
import json
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json,sha_file

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_pixel_transfer_diagnosis_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    if read(ROOT/'COMPLETION.json')['status']!='completed':raise ValueError('Full registered diagnosis must finish first')
    data=read(ROOT/'DATA.json');catalog=read(ROOT/'TRAIN_IMAGE_WORD_CATALOG.json')['images']
    qa={q['origin_question_id']:q for q in data['views']['full']};results={};examples={}
    for role in ('initial','pixel'):
        full=read(ROOT/role/'full/PREDICTIONS.json');crop=read(ROOT/role/'oracle_crop/PREDICTIONS.json')
        counts={key:0 for key in ('range_or_format','same_image_other_word','other_content_or_label')}
        rescued=dict(counts);rows=[]
        for a,b in zip(full,crop):
            if a['origin_question_id']!=b['origin_question_id']:raise ValueError('Unpaired cohort')
            if a['casefold_exact']:continue
            row=qa[a['origin_question_id']]
            matches=[q for q in catalog[a['source_image_id']]
                if q['annotation_id']!=row['annotation_id'] and q['text'].casefold()==a['prediction'].casefold()]
            kind=('range_or_format' if a['bounded_target_present'] else
                  'same_image_other_word' if matches else 'other_content_or_label')
            counts[kind]+=1;rescued[kind]+=int(b['casefold_exact'])
            rows.append({'origin_question_id':a['origin_question_id'],'source_image_id':a['source_image_id'],
                'target':a['target'],'full_prediction':a['prediction'],'crop_prediction':b['prediction'],
                'full_error_type':kind,'crop_correct':b['casefold_exact'],
                'other_annotation_matches':[{'annotation_id':q['annotation_id'],'text':q['text'],'points':q['points']} for q in matches]})
        assert sum(counts.values())==256-sum(q['casefold_exact'] for q in full)
        results[role]={'full_errors':sum(counts.values()),'error_types':counts,'oracle_crop_rescued_by_type':rescued}
        examples[role]=rows
    result={'status':'completed','scope':'Posthoc descriptive classification on training-only256; no scoring/input/label changes',
        'matching':'Literal casefold equality to another official word; does not prove model attention or pure localization causality',
        'source_catalog_sha256':sha_file(ROOT/'TRAIN_IMAGE_WORD_CATALOG.json'),
        'source_code_sha256':sha_file(Path(__file__)),'models':results,'errors':examples}
    write_json(ROOT/'POSTHOC_WORD_ERROR_TYPES.json',result)
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
