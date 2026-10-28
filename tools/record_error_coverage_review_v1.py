"""Persist manually supplied original-image judgments; never infer support automatically."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_error_coverage_mining_v1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,required=True);parser.add_argument('--count',type=int,default=8);args=parser.parse_args()
    raw=(ROOT/'TRAINING_CANDIDATES.json').read_bytes();candidates=json.loads(raw);data=json.loads((ROOT/'DATA.json').read_text())
    path=ROOT/'VISIBLE_REVIEW.json'
    review=json.loads(path.read_text()) if path.exists() else {'reviewer':'assistant original-image inspection; not user/expert relabeling','rows':[],'training_allowed':False}
    digest=hashlib.sha256(raw).hexdigest()
    if review.get('candidate_sha256',digest)!=digest:raise ValueError('Original candidate order changed')
    by={q['review_index']:q for q in review['rows']}
    for item in json.load(sys.stdin):
        if candidates[item['review_index']]['question_id']!=item['question_id']:raise ValueError('Review/candidate ID differs')
        if type(item['supported']) is not bool or not item['note']:raise ValueError('Need explicit verdict and visible evidence')
        if item['review_index'] in by and by[item['review_index']]!=item:raise ValueError('Refusing silent replacement of an existing judgment')
        by[item['review_index']]=item
    if sorted(by)!=list(range(len(by))):raise ValueError('Review the fixed candidate prefix without skipping')
    review.update(rows=[by[i] for i in sorted(by)],candidate_sha256=digest,reviewed=len(by),supported=sum(q['supported'] for q in by.values()))
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    batch=[]
    for i,q in enumerate(candidates[args.start:args.start+args.count],args.start):
        batch.append({'review_index':i,'question_id':q['question_id'],'question':q['question'],'target':q['target'],'prediction':q['prediction'],'image_path':data['images'][q['image_id']]['path']})
    print(json.dumps({'reviewed':review['reviewed'],'supported':review['supported'],'classes':dict(Counter(q['error_type'] for q in review['rows'] if q['supported'])),'batch':batch},ensure_ascii=False))


if __name__=='__main__':main()
