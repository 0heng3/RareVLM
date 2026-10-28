#!/usr/bin/env python3
"""CPU-only read-only diagnostics of archived initial/SFT/DPO predictions.

Writes only the independent failure-review directory. Does not change scores,
annotations, thresholds, frozen scientific sources, or training/eval artifacts.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics
import textwrap

from PIL import Image, ImageDraw, ImageFont
from vlm_deep.common import NORMALIZER, consensus
from vlm_posttrain_pope.scoring import metrics, official_prediction, validity_diagnostic

BASE = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
ROOT = BASE / 'vlm_public_posttrain_diagnostic_v1'
OUT = BASE / 'vlm_public_posttrain_failure_review_v1'
ROLES = ('initial', 'sft', 'dpo')
CONTRASTS = (('dpo_minus_initial','initial','dpo'),
             ('sft_minus_initial','initial','sft'), ('dpo_minus_sft','sft','dpo'))

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def image_path(manifest_path, manifest, image_id):
    raw = Path(manifest['images'][image_id]['path'])
    candidates = [raw] if raw.is_absolute() else [manifest_path.parent / raw,
        manifest_path.parent.parent / raw]
    for p in candidates:
        if p.exists(): return str(p.resolve())
    raise FileNotFoundError(candidates)

def length_summary(rows):
    values = [r['generated_nonpad_tokens'] for r in rows]
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), maximum=max(values),
                total=sum(values),
                caveat='saved generated_nonpad_tokens includes generated end token; no tokenizer or model rerun')

def confusion(row):
    return ('T' if row['official_prediction'] == row['label'] else 'F') + ('P' if row['official_prediction']=='yes' else 'N')

def compact(row):
    return {k:row[k] for k in ('prediction','score','generated_nonpad_tokens')}

TEXT_OBSERVATIONS = {
    '5796': ('supported_answer_completion', '原图广告可见“Smooth as Ocean-Rocked Whiskies of Old”。DPO补全原回答截短的后半段；这次变长对应评分改善。', 'high'),
    '6143': ('transcription_variant_scoring_risk', '瓶签大字是带变音符的 Mjöd。mjod→mjoed 涉及ASCII转写变化；单凭字面共识分从1到0不能确定事实识别恶化，mjoed等价性仍属局部推断。', 'medium'),
    '16726': ('image_supported_answer_scope_expansion', '封面同时可见 The Great War 和 AMERICAN FRONT。新回答含图支持的副标题，但参考票数主要集中于较短the great war，1→0.3含答案范围/共识敏感性。', 'high'),
    '18020': ('target_localization_or_answer_content_error', '原图可见数字键，包括底部区域的0；三模型回答由0变channel 10。channel 10没有得到参考支持，符合目标定位/按钮内容错误；不由单例证明普遍定位退化。', 'medium'),
    '20949': ('brand_transcription_error', '原图袖部可见XTERRA，裤腿可见VORTEX 4。initial的xterra虽仅得0.3仍有图像支持；DPO的xtecc不对应清晰品牌文字，符合转写错误。', 'high'),
    '25421': ('name_scope_expansion_with_ambiguous_extra_descriptor', '右侧两条竖幅清晰可见FRIDAYS；DPO将bar b q附加进餐厅名称。图中竖幅上方细字有限，不能可靠判断附加词究竟来自相邻描述还是读取错误。确定的是参考名称范围不含该附加部分。', 'medium'),
    '29766': ('image_supported_answer_scope_expansion', '背景招牌可见Spring Water。spring→spring water得到图支持，但短形式有7票、长形式2票，原共识分1→0.6；这并非视觉事实变错的直接证据。', 'high'),
    '32899': ('image_supported_answer_scope_expansion', '机身可见MONGOLIAN和较小的AIRLINES。mongolian→mongolian airlines扩写得到图支持，参考集中于mongolian，1→0明显受参考覆盖影响。', 'high'),
}
POPE_CASE_IDS = ['pope_random_217','pope_random_715','pope_random_1423',
    'pope_random_366','pope_random_2026','pope_adversarial_2',
    'pope_random_1935','pope_popular_1935','pope_random_2311','pope_random_231']
POPE_OBSERVATIONS = {
    'pope_random_217': ('partial_person_miss', '低视角滑板图上方可见部分腿/鞋，官方yes；DPO由yes变no。原图只呈人体局部，局部观察支持“局部实体漏判”候选。', 'medium'),
    'pope_random_715': ('small_object_uncertainty', '厨房里有许多小型厨具，官方spoon=yes。联系页无法可靠定位勺子；保留TP→FN事实计数，视觉机制不确定。', 'low'),
    'pope_random_1423': ('small_clock_recovery', '床头枕头上方可见黑色数字钟，DPO由no变yes符合图像；属于FN→TP例。', 'high'),
    'pope_random_366': ('plate_bowl_confusion_candidate', '图中三明治放在白色盘子上，官方bowl=no；DPO从no变yes。局部观察符合餐盘/碗混淆候选，不能排除遮挡细节。', 'medium'),
    'pope_random_2026': ('scene_object_false_positive_removed', '图中为货车/树木场景，未见明确盆栽；DPO由yes变no符合官方标签，属于FP→TN例。', 'medium'),
    'pope_adversarial_2': ('tiny_object_absence_uncertainty', '雪坡人物很小，官方backpack=no。DPO由yes变no是FP→TN；原图不足以把背部细节作为强视觉核验。', 'low'),
    'pope_random_1935': ('same_semantic_input_batch_context_instability', '人群与伞的图中瓶子细节难辨，官方bottle=yes。此题同图同问跨类别有相反翻转，未填充input/pixel SHA相同，而padding长度379/380不同；仅能报告输出随批次上下文不一致，不证明padding是唯一原因。', 'low'),
    'pope_popular_1935': ('same_semantic_input_batch_context_instability', '与random_1935完全同图同问题。initial这里为no、DPO为yes，与random版本相反；不能把两种翻转当独立图像证据。瓶子局部可见性仍不确定。', 'low'),
    'pope_random_2311': ('format_only_shortening', '聚会图中人物明确可见；initial长句、SFT Yes.、DPO yes均被官方解析为yes且全得1。invalid→valid只改变完整二元回答格式。', 'high'),
    'pope_random_231': ('format_only_shortening_relative_sft', '桌边右侧可见人的手臂；initial与DPO为yes、SFT为完整Yes, there is...长句，官方标签和得分完全不变。', 'high'),
}

def export_cases(text_changes, pope_changes, data):
    text_cases=[dict(r) for r in text_changes
        if abs(r['predictions']['dpo']['score']-r['predictions']['initial']['score'])>1e-12]
    assert {r['question_id'] for r in text_cases}==set(TEXT_OBSERVATIONS)
    pmap={r['question_id']:r for r in pope_changes}
    pope_cases=[dict(pmap[qid]) for qid in POPE_CASE_IDS]
    for task,rows,notes in [('text',text_cases,TEXT_OBSERVATIONS),('pope',pope_cases,POPE_OBSERVATIONS)]:
        for r in rows:
            label,note,confidence=notes[r['question_id']]
            r['assistant_local_visual_observation']={'reviewer':'assistant',
                'scope':'selected original-image/contact-sheet local observation; not independent human ground truth',
                'mechanism_candidate':label,'note_zh':note,'confidence':confidence,
                'changes_frozen_score_or_annotation':False}
            r['dpo_minus_initial_score']=r['predictions']['dpo']['score']-r['predictions']['initial']['score']
            if task=='pope':
                r['saved_input_evidence']={role:{k:data['pope'][role][r['question_id']][k]
                    for k in ('input_tokens','padded_input_tokens','batch_start','question_prompt_sha256',
                              'input_ids_sha256','padded_input_ids_sha256','image_grid_thw','pixel_values')}
                    for role in ROLES}
    cases={'schema_version':'rarevlm.posttrain.failure_cases.v1',
        'selection':{'text':'all eight DPO-initial score-changing Textdev questions, exhaustive not favorable cherry-picking',
            'pope':'ten fixed representative rows covering TP/FN, FP/TN, same semantic query across categories, and formatting'},
        'display_note_zh':'以下展示为保存的原图/缩略联系页，不是重新运行processor的模型输入，不是attention，也没有新增GPU预测。',
        'text':text_cases,'pope':pope_cases,
        'text_supported_scope_expansion_summary':{'qids':['16726','29766','32899'],
            'gross_loss_score_sum':-2.1,'fraction_of_all_dpo_initial_gross_score_loss':2.1/5.4,
            'interpretation':'selected assistant local observations support answer-scope/reference-consensus sensitivity; not semantic-equivalence proof or a revised benchmark score'}}
    write('CASES.json',cases)
    font=ImageFont.truetype('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf',19)
    for name,rows in [('TEXT',text_cases),('POPE',pope_cases)]:
        w,h=700,650
        canvas=Image.new('RGB',(w*2,h*((len(rows)+1)//2)),'white')
        draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows):
            x=(i%2)*w;y=(i//2)*h
            with Image.open(r['image_path']) as orig:
                im=orig.convert('RGB'); im.thumbnail((w-25,430)); canvas.paste(im,(x+(w-im.width)//2,y+15))
            pred=' | '.join(k+': '+v['prediction']+' ('+str(round(v['score'],2))+')' for k,v in r['predictions'].items())
            lines=[r['question_id']]+textwrap.wrap(r['question'],63)+textwrap.wrap(pred,62)
            lines+=textwrap.wrap('refs: '+str(r.get('reference_counts',r.get('label'))),66)
            draw.multiline_text((x+12,y+450),'\n'.join(lines[:9]),font=font,fill='black',spacing=3)
        canvas.save(OUT/f'{name}_CONTACT.png')
    cards=[]
    for task,rows in [('TextVQA：全部8条分变',text_cases),('POPE：代表病例',pope_cases)]:
        cards.append('<h2>'+html.escape(task)+'</h2>')
        for r in rows:
            e=html.escape
            ref=r.get('reference_counts',r.get('label'))
            table=''.join('<tr><td>'+e(role)+'</td><td>'+e(v['prediction'])+'</td><td>'+str(v['score'])+'</td><td>'+str(v['generated_nonpad_tokens'])+'</td></tr>' for role,v in r['predictions'].items())
            cards.append('<article><h3>'+e(r['question_id'])+'</h3><p>'+e(r['question'])+'</p><img src="'+Path(r['image_path']).as_uri()+'"><p>完整参考/票数：'+e(str(ref))+'</p><table><tr><th>模型</th><th>完整回答</th><th>原分数</th><th>非pad生成token</th></tr>'+table+'</table><p><strong>助手局部观察：</strong>'+e(r['assistant_local_visual_observation']['note_zh'])+'</p><details><summary>原图绝对路径与记录</summary><pre>'+e(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></article>')
    title='SFT/DPO失败病例：真实原图与冻结预测'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#fafafa;color:#222}article{background:white;padding:24px;margin:24px 0;border:1px solid #ddd;border-radius:8px}img{max-width:100%;max-height:760px;object-fit:contain}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:10px;text-align:left}pre{white-space:pre-wrap;word-break:break-all}p{line-height:1.6}</style><h1>'+title+'</h1><p>'+cases['display_note_zh']+'</p><p>Text包含DPO相对initial全部1升7降；POPE类别共享500图。观察为助手局部审查，不修改参考、评分或门槛。</p>'+''.join(cards)+'</html>'
    (OUT/'病例学习页.html').write_text(page)

def main():
    OUT.mkdir(exist_ok=True)
    text_path = BASE / 'vlm_deep_v2/prepared/manifest.json'
    pope_path = BASE / 'vlm_posttrain_pope_v1/data/manifest.json'
    tm, pm, sl = read(text_path), read(pope_path), read(ROOT/'SLICE_LOCK.json')
    tr = {r['question_id']:r for r in tm['splits']['dev']}
    pr = {r['question_id']:r for r in pm['records']}
    sf = {r['question_id']:r for r in sl['features']}
    data, provenance = {}, {'text_manifest':{'path':str(text_path),'sha256':sha(text_path)},
        'pope_manifest':{'path':str(pope_path),'sha256':sha(pope_path)},
        'slice_lock':{'path':str(ROOT/'SLICE_LOCK.json'),'sha256':sha(ROOT/'SLICE_LOCK.json')},
        'diagnostic_script':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))}}
    for task, refs in [('dev',tr), ('pope',pr)]:
        data[task] = {}
        for role in ROLES:
            path=ROOT/'runs'/role/'seed42'/task/'predictions.jsonl'
            rows=[json.loads(line) for line in path.open()]
            qmap={r['question_id']:r for r in rows}
            assert len(qmap)==len(rows)==len(refs)
            assert qmap.keys()==refs.keys()
            for qid,row in qmap.items():
                assert row['image_id']==refs[qid]['image_id']
                if task=='dev':
                    assert abs(consensus(row['prediction'],refs[qid]['answers'])-row['score']) < 1e-12
                else:
                    assert row['official_prediction']==official_prediction(row['prediction'])
                    assert row['validity_diagnostic']==validity_diagnostic(row['prediction'])
                    assert row['score']==float(row['official_prediction']==refs[qid]['label'])
            data[task][role]=qmap
            provenance[f'{task}_{role}']={'path':str(path),'sha256':sha(path),'records':len(rows)}
        for qid in refs:
            for field in ('image_sha256','question_prompt_sha256','input_ids_sha256','pixel_values','image_grid_thw'):
                assert all(data[task][role][qid][field]==data[task]['initial'][qid][field] for role in ROLES)
    text={'questions':len(tr),'unique_images':len({r['image_id'] for r in tr.values()}),
          'slice_counts':sl['slice_counts'], 'models':{},'contrasts':{}}
    for role in ROLES:
        rows=list(data['dev'][role].values())
        text['models'][role]={'score_percent':100*statistics.mean(r['score'] for r in rows),
            'zero':sum(r['score']==0 for r in rows),'partial':sum(0<r['score']<1 for r in rows),
            'full':sum(r['score']==1 for r in rows),'generation_length':length_summary(rows)}
    text['persistence']={'raw_answer_identical_all_three':sum(len({data['dev'][role][qid]['prediction'] for role in ROLES})==1 for qid in tr),
        'zero_score_all_three':sum(all(data['dev'][role][qid]['score']==0 for role in ROLES) for qid in tr),
        'not_full_score_all_three':sum(all(data['dev'][role][qid]['score']<1 for role in ROLES) for qid in tr)}
    all_text_changes=[]
    for qid,ref in tr.items():
        preds={role:data['dev'][role][qid] for role in ROLES}
        if len({r['prediction'] for r in preds.values()})>1:
            all_text_changes.append({'question_id':qid,'image_id':ref['image_id'],
                'question':ref['question'],'answers':ref['answers'],
                'reference_counts':dict(Counter(ref['answers'])),
                'target':ref['target'], 'slice':sf[qid]['sft_target_answer_frequency'],
                'target_train_qa_frequency':sf[qid]['train_all_sft_target_qa_frequency'],
                'image_path':image_path(text_path,tm,ref['image_id']),
                'predictions':{role:{**compact(preds[role]),'normalized':NORMALIZER(preds[role]['prediction'])} for role in ROLES}})
    for name,before,after in CONTRASTS:
        rows=[]
        for qid,ref in tr.items():
            b,a=data['dev'][before][qid],data['dev'][after][qid]
            bn,an=NORMALIZER(b['prediction']),NORMALIZER(a['prediction'])
            delta=a['score']-b['score']
            # Lexical containment is a diagnostic flag, not a semantic judgment.
            retained=bool(bn and bn!=an and (' '+bn+' ') in (' '+an+' '))
            rows.append({'qid':qid,'image_id':ref['image_id'],'slice':sf[qid]['sft_target_answer_frequency'],
                'raw_changed':b['prediction']!=a['prediction'],'norm_changed':bn!=an,
                'delta':delta,'before':b['score'],'after':a['score'],
                'length_delta':a['generated_nonpad_tokens']-b['generated_nonpad_tokens'],
                'before_normalized_contained_in_after':retained})
        def summary(sub):
            n=len(sub); total=sum(r['delta'] for r in sub)
            return {'questions':n,'unique_images':len({r['image_id'] for r in sub}),
                'raw_answer_changed':sum(r['raw_changed'] for r in sub),
                'normalized_answer_changed':sum(r['norm_changed'] for r in sub),
                'score_changed':sum(abs(r['delta'])>1e-12 for r in sub),
                'improved':sum(r['delta']>1e-12 for r in sub),
                'declined':sum(r['delta']< -1e-12 for r in sub),
                'full_to_zero':sum(r['before']==1 and r['after']==0 for r in sub),
                'zero_to_full':sum(r['before']==0 and r['after']==1 for r in sub),
                'gain_score_sum':sum(r['delta'] for r in sub if r['delta']>0),
                'loss_score_sum':sum(r['delta'] for r in sub if r['delta']<0),
                'net_score_sum':total, 'slice_delta_pp':100*total/n,
                'contribution_to_overall_delta_pp':100*total/len(tr),
                'longer_count':sum(r['length_delta']>0 for r in sub),
                'shorter_count':sum(r['length_delta']<0 for r in sub),
                'same_token_length_count':sum(r['length_delta']==0 for r in sub),
                'score_declines_longer':sum(r['delta']< -1e-12 and r['length_delta']>0 for r in sub),
                'score_declines_same_length':sum(r['delta']< -1e-12 and r['length_delta']==0 for r in sub),
                'score_declines_shorter':sum(r['delta']< -1e-12 and r['length_delta']<0 for r in sub),
                'lexical_core_retained_score_decline_qids':[r['qid'] for r in sub if r['delta']< -1e-12 and r['before_normalized_contained_in_after']]}
        text['contrasts'][name]={'overall':summary(rows),
            'by_slice':{s:summary([r for r in rows if r['slice']==s]) for s in sl['slice_counts']},
            'changed_rows':[r for r in rows if r['raw_changed'] or abs(r['delta'])>1e-12]}
    pope={'questions':len(pr),'unique_images':len(pm['images']),
        'shared_image_note':'Three categories reuse the same 500 images; category rows are not independent image sets.',
        'unique_image_question_pairs':len({(r['image_id'],r['question']) for r in pr.values()}),
        'models':{},'contrasts':{}}
    for role in ROLES:
        rows=list(data['pope'][role].values())
        pope['models'][role]={**metrics(rows),'generation_length':length_summary(rows)}
    semantic=defaultdict(list)
    for qid,ref in pr.items():semantic[(ref['image_id'],ref['question'])].append(qid)
    repeated={k:v for k,v in semantic.items() if len(v)>1}
    pope['same_image_question_repetition']={'repeated_unique_image_questions':len(repeated),
        'models':{role:{'parsed_answer_inconsistent_groups':sum(len({data['pope'][role][qid]['official_prediction'] for qid in ids})>1 for ids in repeated.values()),
            'raw_answer_inconsistent_groups':sum(len({data['pope'][role][qid]['prediction'] for qid in ids})>1 for ids in repeated.values())} for role in ROLES},
        'caveat':'Category replicates can differ in batch padding/context. This observation does not establish padding as the unique causal mechanism.'}
    all_pope_changes=[]
    for qid,ref in pr.items():
        preds={role:data['pope'][role][qid] for role in ROLES}
        if len({r['prediction'] for r in preds.values()})>1:
            all_pope_changes.append({'question_id':qid,'image_id':ref['image_id'],
                'question':ref['question'],'label':ref['label'],'category':ref['category'],
                'image_path':image_path(pope_path,pm,ref['image_id']),
                'predictions':{role:{**compact(preds[role]),
                    'official_prediction':preds[role]['official_prediction'],
                    'strict_valid_binary':preds[role]['validity_diagnostic']['strict_valid_binary'],
                    'confusion':confusion(preds[role])} for role in ROLES}})
    for name,before,after in CONTRASTS:
        rows=[]
        for qid,ref in pr.items():
            b,a=data['pope'][before][qid],data['pope'][after][qid]
            bv,av=b['validity_diagnostic']['strict_valid_binary'],a['validity_diagnostic']['strict_valid_binary']
            rows.append({'qid':qid,'image_id':ref['image_id'],'question':ref['question'],
                'category':ref['category'],'label':ref['label'],
                'raw_changed':b['prediction']!=a['prediction'],
                'label_changed':b['official_prediction']!=a['official_prediction'],
                'delta_correct':int(a['score'])-int(b['score']),
                'transition':confusion(b)+'->'+confusion(a),
                'before_valid':bv,'after_valid':av})
        def summary(sub):
            flips=[r for r in sub if r['label_changed']]
            fixed=[r for r in sub if not r['before_valid'] and r['after_valid']]
            return {'questions':len(sub),'unique_images':len({r['image_id'] for r in sub}),
                'raw_changed':sum(r['raw_changed'] for r in sub),'parsed_label_changed':len(flips),
                'parsed_label_changed_unique_images':len({r['image_id'] for r in flips}),
                'parsed_label_changed_unique_image_questions':len({(r['image_id'],r['question']) for r in flips}),
                'improved':sum(r['delta_correct']>0 for r in sub),
                'declined':sum(r['delta_correct']<0 for r in sub),
                'net_correct_delta':sum(r['delta_correct'] for r in sub),
                'confusion_transition_counts':dict(Counter(r['transition'] for r in sub)),
                'invalid_to_valid':len(fixed),
                'valid_to_invalid':sum(r['before_valid'] and not r['after_valid'] for r in sub),
                'invalid_to_valid_parsed_label_unchanged':sum(not r['label_changed'] for r in fixed),
                'invalid_to_valid_correctness_improved':sum(r['delta_correct']>0 for r in fixed),
                'invalid_to_valid_correctness_declined':sum(r['delta_correct']<0 for r in fixed)}
        changed=[r for r in rows if r['raw_changed'] or r['label_changed'] or r['before_valid']!=r['after_valid']]
        groups=defaultdict(list)
        for r in changed:groups[(r['image_id'],r['question'])].append(r)
        group_rows=[{'image_id':k[0],'question':k[1], 'qids':[r['qid'] for r in v],
            'categories':[r['category'] for r in v], 'transitions':dict(Counter(r['transition'] for r in v)),
            'net_correct_delta':sum(r['delta_correct'] for r in v)} for k,v in groups.items()]
        image_groups=defaultdict(list)
        for r in rows:image_groups[r['image_id']].append(r)
        image_changes=[{'image_id':k,'net_correct_delta':sum(r['delta_correct'] for r in v),
            'changed_qids':[r['qid'] for r in v if r['label_changed']]} for k,v in image_groups.items() if any(r['label_changed'] for r in v)]
        pope['contrasts'][name]={'overall':summary(rows),
            'by_category':{s:summary([r for r in rows if r['category']==s]) for s in pm['categories']},
            'changed_rows':changed,'unique_image_question_change_groups':group_rows,
            'image_cluster_correctness_changes':image_changes}
    output={'schema_version':'rarevlm.posttrain.prediction_failure_diagnostics.v1',
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'scope':'CPU-only retrospective diagnosis; seed42 fixed checkpoints; no new generation, training, relabeling, or threshold changes',
        'verification':{'qid_alignment':True,'image_and_question_alignment':True,
            'input_hashes_matching':True,'all_4851_text_scores_recomputed_with_frozen_scorer':True,
            'all_27000_pope_labels_scores_and_validity_reparsed':True},
        'provenance':provenance,'text':text,'pope':pope,
        'limitations':['Descriptive retrospective associations, not a causal attribution of DPO failure.',
            'Lexical containment is not semantic correctness. Strict binary validity is a format diagnostic.',
            'Only selected cases receive assistant local visual observation, not full-dataset human annotation.',
            'Original frozen analysis confidence intervals and promotion/data-admission gates remain authoritative.']}
    write('PREDICTION_DIAGNOSTICS.json',output)
    write('TEXT_CHANGED_RESPONSES.json',all_text_changes)
    write('POPE_CHANGED_RESPONSES.json',all_pope_changes)
    export_cases(all_text_changes,all_pope_changes,data)
    print(json.dumps({'text_models':text['models'],
        'text_contrasts':{k:v['overall'] for k,v in text['contrasts'].items()},
        'pope_models':{k:v['overall'] for k,v in pope['models'].items()},
        'pope_contrasts':{k:v['overall'] for k,v in pope['contrasts'].items()},
        'text_changed_all_roles':len(all_text_changes),'pope_changed_all_roles':len(all_pope_changes)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
