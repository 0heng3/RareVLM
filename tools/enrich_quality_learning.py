"""Explain real saved-batch tensors on CPU; original study source is immutable."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import DualAdapter
from clip_retrieval.features import read_image
from vlm_probe.evaluate import now,sha_file,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);out=root/'learning'
    torch.set_num_threads(2)
    feature=torch.load(root/'features/train.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text());index=json.loads((root/'IMAGE_INDEX.json').read_text())['offsets']
    handles={};records=[];cards=[]
    with np.load(root/'selection/indices.npz') as z:selection={k:z[k] for k in z.files}
    for branch in ['clean','noisy']:
        condition=branch+'_score4';run=root/'runs'/condition/'seed42'
        first=json.loads((run/'FIRST_REAL_UPDATE.json').read_text());anchors=np.array(first['anchor_ids']);neg=np.array(first['negative_ids'])
        ids=np.concatenate([anchors,neg]);i=int(anchors[0]);selected=selection[condition][ids].reshape(-1)
        m=DualAdapter();m.load_state_dict(torch.load(run/'final.pt',map_location='cpu',weights_only=True));m.eval();m.requires_grad_(False)
        ti=F.normalize(feature['image_features'][ids],dim=-1);tt=F.normalize(feature['text_features'][torch.as_tensor(selected)],dim=-1)
        with torch.inference_mode():
            zi=m.image_features(feature['image_features'][ids]);zt=m.text_features(feature['text_features'][torch.as_tensor(selected)])
            pt=F.softmax(ti[0]@tt.T/cfg['temperature'],dim=0)
            ps=F.softmax(zi[0]@zt.T/cfg['temperature'],dim=0)
        sha=np.asarray(feature['image_sha256']);positive=np.repeat(sha[ids]==sha[i],4)
        target=positive/positive.sum()
        # Per-query I2T CE+KL derivative wrt normalized-temperature logits; not full averaged loss gradient.
        gradient=2*ps.numpy()-target-pt.numpy()
        rows=[]
        for j in range(4):
            c=int(selected[j]);rows.append({'caption':feature['captions'][c],
                'cosine':float(ti[0]@tt[j]),'teacher_probability':float(pt[j]),
                'final_student_probability':float(ps[j]),'supervised_target':float(target[j]),
                'per_query_ce_plus_kl_logit_derivative':float(gradient[j])})
        row=manifest['splits']['train'][i];image=read_image(row,index,handles);image.thumbnail((640,420))
        name=f'{branch}_actual_seed42_anchor.jpg';image.save(out/name,quality=90)
        table=''.join('<tr><td>'+html.escape(x['caption'])+'</td>'+''.join(f'<td>{x[k]:.5f}</td>' for k in ['cosine','teacher_probability','final_student_probability','supervised_target','per_query_ce_plus_kl_logit_derivative'])+'</tr>' for x in rows)
        cards.append(f'<article><h3>{branch}：seed42第一步真实锚图＋Final@10已训练adapter</h3><img src="{name}"><p>锚图索引{i}；候选384条、正例{int(positive.sum())}条。概率在完整384候选上softmax，表中仅展示本图保留的4条。这些是CPU重算，不冒充保存的GPU第一步概率／训练历史梯度。</p><table><tr><th>实际保留描述</th><th>cosine</th><th>P_teacher</th><th>P_final_student</th><th>监督target</th><th>单query logit导数</th></tr>{table}</table></article>')
        records.append({'branch':branch,'condition':condition,'seed':42,'image_index':i,'source_image_sha256':row['image_sha256'],
            'picture':name,'first_actual_gpu_update_sha256':sha_file(run/'FIRST_REAL_UPDATE.json'),
            'final_checkpoint_sha256':sha_file(run/'final.pt'),'candidate_shape':[32,384],'positive_count':int(positive.sum()),
            'shown_rows':rows,'teacher_positive_mass':float(pt[torch.as_tensor(positive)].sum()),
            'student_positive_mass':float(ps[torch.as_tensor(positive)].sum()),
            'scope':'CPU reconstruction from saved actual step0 indices plus final fixed adapter;per-query derivative without symmetry/batch scaling;not attention or saved training trajectory'})
    for f in handles.values():f.close()
    intro='<h2>用真实张量理解CE＋KL</h2><p>CE把概率目标均匀分给同图正描述：这里各为1/4。单个query的CE导数是P_student−P_gt；KL导数是P_student−P_teacher。lambda=1时总导数为2P_student−P_gt−P_teacher（下表省略整体对称／batch平均系数）。负导数表示梯度下降希望该logit上升，正导数希望下降。</p><p>筛选会同时改变哪些描述进入监督，以及384候选上的教师分布。评分高只说明教师一致；过滤真实细节会减少覆盖。观察保留率、原始分支准确率和压力测试准确率，三者不能替换。</p>'
    page=out/'学习页.html';text=page.read_text();marker='<!-- actual_tensor_explanation -->'
    if marker not in text:page.write_text(text.replace('</html>',marker+intro+''.join(cards)+'</html>'))
    write_json(out/'ACTUAL_TENSOR_EXPLANATION.json',{'completed_at':now(),'new_gpu_predictions':0,'records':records,
        'original_study_sources_unchanged':True,'personal_mastery':'pending'})
    print('ACTUAL_TENSOR_LEARNING_ADDED',len(records),'real batches,zero GPU predictions')


if __name__=='__main__':main()
