"""Archive the fixed visual audit, then re-mine existing responses under v1 repair."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
from vlm_deep.common import read_manifest, consensus
from vlm_probe.evaluate import sha_file, lock_file, now
from vlm_alignment_signal.common import write_json
from vlm_preference_review.audit import audit_gate, fixed_sample
from vlm_preference_review.repair import repair_pair, pair_reasons
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT

ROOT=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
SOURCE=ROOT/'vlm_preference_data_v1'
REVIEW=ROOT/'vlm_preference_review_v1'
REPAIR=ROOT/'vlm_preference_repair_v1'

# Explicit judgments from the eight saved contact sheets plus native-size images.
# Non-clear means uncertain or unreliable as a factual preference, not proven false.
NOTES = [
('clear_visual_preference','右侧涂鸦轮廓读作RUDY；rejected增加/改变字母。'),
('question_or_entity_ambiguity','画面包含尺子数字与浅刻痕；actual numbers所指不明，不能确认偏好对象。'),
('clear_visual_preference','唱片书脊可见专辑标题；Sex Pistols是艺人名称，不是所问专辑名。'),
('clear_visual_preference','原尺寸查看红白交通标识有禁止停车/停留文字，支持no方向。'),
('clear_visual_preference','车头编号可读EH200-901；rejected漏掉0。'),
('clear_visual_preference','滑行飞机有British Airways文字与尾翼涂装；不是Canada Jazz。'),
('clear_visual_preference','原尺寸图标下写GReader；rejected重复r。'),
('clear_visual_preference','上层箱体起首ABDY可见；rejected只有3字母。'),
('clear_visual_preference','白衫背后Miraflores可见；rejected多插入字母。'),
('clear_visual_preference','封面曲目LES DEUX CONCERTOS POUR FLUTE；Mozart是作曲家。'),
('clear_visual_preference','原尺寸右侧灰砖竖排more可见，food出现在另一处。'),
('clear_visual_preference','原尺寸交通灯右侧绿色招牌KELLY可见。'),
('clear_visual_preference','冰箱门巧克力酱瓶Hershey品牌可辨，非Miller。'),
('answer_scope_or_partial','袋子标FOOD MARKET，food为正确文本子串；不宜当不同事实负例。'),
('clear_visual_preference','左上22.2%，右上53.5%；问题明确左上。'),
('clear_visual_preference','原尺寸前中黄色包装写m&m；rejected为无关字符。'),
('clear_visual_preference','底本The Visual Display of Quantitative Information；Visual Explanations是上一本。'),
('question_or_entity_ambiguity','artist可指作曲家Erik Satie或演奏者Aldo Ciccolini，二者都印在封面；rejected又有拼写差异。'),
('clear_visual_preference','衣服PAIA字样，rejected漏字母。'),
('answer_scope_or_partial','图中既是parking meter也写Broken Meter；宽泛问题下负例语义仍正确。'),
('clear_visual_preference','原尺寸报纸黑框可见SALVAR，sindicatos出现在别处文字。'),
('clear_visual_preference','画面中央球场有Lakers标志，与Kings不同队。'),
('question_or_entity_ambiguity','OLEG写在纸条上，但不能据纸条断定它确为图中手表品牌。'),
('clear_visual_preference','队徽包含C A R P字母，rejected缺A。'),
('clear_visual_preference','背部TAEKWONDO可读；rejected漏字母。'),
('question_or_entity_ambiguity','what kind既可能问名称Lust，也可能问固体香水类型solid；两者均有图文支持。'),
('clear_visual_preference','原尺寸长分针指向VII，非XII。'),
('clear_visual_preference','封面左上Peter Jones Enterprise Academy全名；单字母e不完整。'),
('clear_visual_preference','作者Irati Goikoetxea可读；rejected改变姓中一个字母。'),
('clear_visual_preference','标牌MARGAUX可辨；rejected是另一名称。'),
('clear_visual_preference','瓶身ABITA品牌可见；rejected错读字母。'),
('answer_scope_or_partial','招牌OVEN FRESH；oven只是同一真实短语的子串，非明确不同事实。'),
('clear_visual_preference','原尺寸球衣交织T/C徽标可辨，非T/S。'),
('clear_visual_preference','球衣正面号码22，非27。'),
('semantic_alias','瓶身可口可乐；Coke与Coca-Cola在此指同一饮料，零分不构成语义错误。'),
('unresolved_visual_detail','停车位地面数字不够清晰；未用十票共识替代视觉确认。'),
('clear_visual_preference','瓶标清楚标1994，非1997。'),
('answer_scope_or_partial','墙上Fastrack Bicycles；Fastrack是实际品牌核心，负例未明显指错。'),
('clear_visual_preference','原尺寸红标瓶下方1 pint可辨，非8。'),
('clear_visual_preference','短白标瓶KNOB CREEK；Baker另在右侧瓶子上。'),
('clear_visual_preference','书封红色ELLROY可见；American Tabloid为白色标题。'),
('clear_visual_preference','封面Maria Chakhtoura，rejected漏i。'),
('clear_visual_preference','第二页右上CCXX；rejected读了其他文字。'),
('clear_visual_preference','黄色标牌字母L，非F。'),
('clear_visual_preference','原尺寸窗口标题Disk可读；rejected不是窗口标题。'),
('clear_visual_preference','衫背LONE STAR字样，rejected改为LOST。'),
('clear_visual_preference','上层右起第二本KILL KID；rejected把L错读I。'),
('clear_visual_preference','瓶标文字8 hours保护期，72是另一处数字。'),
('question_or_entity_ambiguity','Yogi Berra名字在照片下方引语，Yankees也是球衣队伍；所属人物与队伍/引语对象易混，未外部查球衣归属。'),
('clear_visual_preference','玻璃杯Pieve Vecchia可辨；rejected将c错成s。'),
('clear_visual_preference','原尺寸右瓶下方CONTENTS 354mL清楚可读，非375。'),
('question_or_entity_ambiguity','标签同时有生产者Jean-Marc Burgaud与Morgon Cote du Py酒款/产区；brand所指有歧义。'),
('clear_visual_preference','主表盘时分针约10:08，非3:27。'),
('answer_scope_or_partial','Peace Tea完整品牌，Peace是实际包装品牌核心；不当明确不同事实。'),
('clear_visual_preference','左侧边缘GPS标识可读，menu在底部。只确认标识，不验证实际按钮功能。'),
('clear_visual_preference','原尺寸左侧黄色牌的白箭头向右，非向左。'),
('clear_visual_preference','原尺寸星期窗THU，支持Thursday，非Friday。'),
('answer_scope_or_partial','被拍物为名为The Great Bear的地铁地图；map也是合理物体描述。'),
('clear_visual_preference','粉罐标MARZEN，非MARCHEN。'),
('clear_visual_preference','鞋舌OSIRIS品牌清晰，非twenty eight。'),
('clear_visual_preference','原尺寸机身EI-IMN可读，rejected把M错为N。'),
('clear_visual_preference','前景背对镜头球衣清楚为6；14在背景另一球员处。'),
('clear_visual_preference','设备下框OQO可读；设备被问题叫手机不改变品牌字样辨读。'),
('clear_visual_preference','原图旋转仅作展示后SAMSUNG清晰，拒答缺乏依据；展示变换未进模型。'),
]

def render(records, out):
    font=ImageFont.truetype('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf',18)
    for page in range(8):
        sheet=Image.new('RGB',(1600,1600),'white');draw=ImageDraw.Draw(sheet)
        for local,rec in enumerate(records[page*8:page*8+8]):
            p=rec['pair'];left=local%2*800;top=local//2*400
            image=Image.open(rec['image_path']).convert('RGB');image.thumbnail((395,365));sheet.paste(image,(left,top+30))
            lines=f"#{rec['index']} QID {p['question_id']}\nQ: {p['question']}\nC: {p['chosen']}\nR: {p['rejected']}\nVotes: {p['chosen_normalized_votes']}/10"
            y=top+8
            for line in lines.splitlines():
                for part in textwrap.wrap(line,34):draw.text((left+402,y),part,font=font,fill='black');y+=24
            draw.line((left,top+399,left+799,top+399),fill='gray')
        sheet.save(out/f'confirm_page{page}.jpg',quality=94)

def main():
    plan=json.loads((REVIEW/'AUDIT_PLAN.json').read_text());sample=json.loads((REVIEW/'SAMPLE.json').read_text())
    if len(NOTES)!=64:raise ValueError('Incomplete explicit observations')
    records=[{'question_id':x['pair']['question_id'],'category':cat,'note':note,'image_path':x['image_path'],
              'source_display':f'page{x["index"]//8}.jpg','reviewer':'assistant','independent_human_label':False}
             for x,(cat,note) in zip(sample,NOTES)]
    completed_at=json.loads((REVIEW/'ANALYSIS.json').read_text())['completed_at'] if (REVIEW/'ANALYSIS.json').exists() else now()
    report={**audit_gate(records,plan),'categories':dict(Counter(x['category'] for x in records)),
            'source_plan_sha256':sha_file(REVIEW/'AUDIT_PLAN.json'),'completed_at':completed_at,
            'optimizer_updates':0,'training_allowed':False,'training_launched':False}
    lock_file(REVIEW/'VISUAL_REVIEW.json',{'records':records,'summary':report})
    write_json(REVIEW/'ANALYSIS.json',report)
    if report['gate_passed']:raise ValueError('This registered repair expects failed design audit;use separate training protocol if otherwise')
    cfg=json.loads((PROJECT_ROOT/'configs/vlm_preference_data_v1.json').read_text())
    data=json.loads((SOURCE/'DATA_ROSTER.json').read_text());rows={x['question_id']:x for x in data['questions']}
    manifest=read_manifest(Path(cfg['manifest_path']));images=manifest['images']
    excluded={r['image_id'] for s in ['dev','final'] for r in manifest['splits'][s]}
    badpixel={images[i]['pixel_sha256'] for i in excluded};badbytes={images[i]['sha256'] for i in excluded}
    design=set(plan['selected_question_ids']);pairs=[];counts=Counter();seen=set()
    REPAIR.mkdir(exist_ok=True)
    binding={'protocol_sha256':sha_file(PROJECT_ROOT/'docs/VLM_PREFERENCE_REPAIR_V1_PROTOCOL.md'),
             'sources':{str(p.relative_to(SOURCE_ROOT)):sha_file(p) for p in sorted((SOURCE_ROOT/'vlm_preference_review').glob('*.py'))},
             'parent_preferences_sha256':sha_file(SOURCE/'analysis/PREFERENCES.json'),
             'parent_roster_sha256':sha_file(SOURCE/'DATA_ROSTER.json'),'parent_analysis_sha256':sha_file(SOURCE/'analysis/ANALYSIS.json'),
             'design_plan_sha256':sha_file(REVIEW/'AUDIT_PLAN.json'),'design_review_sha256':sha_file(REVIEW/'VISUAL_REVIEW.json'),
             'manifest_sha256':sha_file(Path(cfg['manifest_path'])),'predictions_sha256':{},
             'tool_sha256':sha_file(Path(__file__))}
    for role in cfg['roles']:
        directory=SOURCE/'roles'/f'role{role}';path=directory/'predictions.jsonl';digest=sha_file(path)
        if json.loads((directory/'COMPLETE.json').read_text())['predictions_sha256']!=digest:raise ValueError('Source generation changed')
        binding['predictions_sha256'][str(path)]=digest
        for line in path.read_text().splitlines():
            record=json.loads(line);row=rows[record['question_id']];im=images[row['image_id']]
            if row['question_id'] in seen:raise ValueError('Duplicate source question')
            seen.add(row['question_id'])
            if row['source_split']!='train' or row['image_id'] in excluded or im['pixel_sha256'] in badpixel or im['sha256'] in badbytes:
                raise ValueError('Training/evaluation leakage')
            if consensus(record['greedy']['prediction'],row['answers'])!=record['greedy_score']:raise ValueError('Greedy rescore differs')
            pair=repair_pair(row,record);counts['source_questions']+=1
            if pair:
                counts['eligible_before_design_exclusion']+=1
                if row['question_id'] in design:counts['design_excluded']+=1
                else:pairs.append(pair)
    if len(seen)!=10000 or len({images[x['image_id']]['pixel_sha256'] for x in rows.values()})!=10000:raise ValueError('Source population incomplete')
    lock_file(REPAIR/'INPUT_LOCK.json',binding)
    lock_file(REPAIR/'PREFERENCES.json',{'pairs':pairs,'training_allowed':False,'semantic_confirmation':'pending',
                                      'design_question_ids_excluded':sorted(design), 'label_source':'ten-reference weak supervision'})
    selected=fixed_sample(pairs,'rarevlm-semantic-confirm-repair-20261015:',64,exclude=design)
    confirm_plan={'sample_size':64,'population_size':len(pairs),'selected_question_ids':[x['question_id'] for x in selected],
                  'selection':'sha256(rarevlm-semantic-confirm-repair-20261015:qid);old design64 excluded',
                  'clear_fraction_min':.90,'wilson95_lower_min':.80,'uncertain_is_not_clear':True,
                  'source_preferences_sha256':sha_file(REPAIR/'PREFERENCES.json'),'reviewer':'assistant;not independent human annotation'}
    lock_file(REPAIR/'CONFIRM_PLAN.json',confirm_plan)
    confirm=[{'index':i,'pair':x,'answers':rows[x['question_id']]['answers'],
              'image_path':str(Path(data['image_root'])/images[x['image_id']]['path'])} for i,x in enumerate(selected)]
    lock_file(REPAIR/'CONFIRM_SAMPLE.json',confirm);render(confirm,REPAIR)
    result={'status':'awaiting_fixed_semantic_confirmation','created_at':now(),'source_questions':counts['source_questions'],
            'generated_responses_reused':50000,'new_gpu_generations':0,'candidate_pairs':len(pairs),
            'eligible_before_design_exclusion':counts['eligible_before_design_exclusion'],'design_excluded':counts['design_excluded'],
            'training_allowed':False,'optimizer_updates':0,'old_audit_gate_passed':False,'input_lock_sha256':sha_file(REPAIR/'INPUT_LOCK.json')}
    write_json(REPAIR/'STATUS.json',result)
    print(json.dumps({'old_audit':report,'repair':result},ensure_ascii=False))
    for x in confirm:
        p=x['pair'];print(x['index'],p['question_id'],p['question'],'C=',repr(p['chosen']),'R=',repr(p['rejected']))

if __name__=='__main__':main()
