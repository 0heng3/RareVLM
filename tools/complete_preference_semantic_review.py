"""Archive the independent confirmation and build an actual-image learning page."""
from collections import Counter
import html
import json
from pathlib import Path
import os
from vlm_alignment_signal.common import write_json
from vlm_probe.evaluate import lock_file, sha_file, now
from vlm_preference_review.audit import audit_gate
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT

ROOT=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
REVIEW=ROOT/'vlm_preference_review_v1'; REPAIR=ROOT/'vlm_preference_repair_v1'

NOTES = [
('clear_visual_preference','海报AMERICAN SPEEDWAY；rejected将末尾Y错为R。'),
('clear_visual_preference','红书TO JE SLOWO BOZE（原字有附加符号）；rejected将BOZE错为BAZE。'),
('clear_visual_preference','招牌THE SPEAKING TREE；rejected改变专名词形，不只是去冠词。'),
('clear_visual_preference','海报MOREIRA DA SILVA；rejected字母顺序不同。'),
('clear_visual_preference','球衣背部姓起首R可见，非Z。'),
('clear_visual_preference','海报底部KOPPELARIJ，rejected插入E。'),
('question_or_entity_ambiguity','原尺寸只见页内For Better or Worse栏题，另有Good/Bad Luck小标题；整本书名无封面依据。'),
('clear_visual_preference','电脑下框msi可见，非Microsoft。'),
('clear_visual_preference','地图图标绿色区域G可见，非S。'),
('clear_visual_preference','地面近处LOUDEST；DOUBT在后墙，不在所问近处地面。'),
('clear_visual_preference','手机短信日期含2008，非2009。'),
('clear_visual_preference','原尺寸右上铅笔12584可辨；rejected中间数字不同。'),
('clear_visual_preference','白衣人物号码23，非29。'),
('clear_visual_preference','长指针指向表外50附近，18来自日期窗不是指针所指。'),
('clear_visual_preference','瓶身Nordeast品牌写法可见，非Northeast。'),
('clear_visual_preference','挡风玻璃CYPRUS RALLY，rejected插入I。'),
('clear_visual_preference','瓶标Weihenstephaner可辨，rejected插入S。'),
('clear_visual_preference','蓝横幅大写H，小写b；问题问大写字母。'),
('clear_visual_preference','球衣正面2可见，非25。'),
('clear_visual_preference','上层白书TINTYPE可见，rejected多O。'),
('clear_visual_preference','封面SCOTT TUROW，rejected多R。'),
('clear_visual_preference','击球者背部LOLLIS可见，rejected漏L。'),
('clear_visual_preference','硬币年份2000，非2006。'),
('clear_visual_preference','冰箱下层Silk盒装奶，非Driftwood。'),
('clear_visual_preference','HENCE左侧白字YOWK，非ORCHID。'),
('clear_visual_preference','瓶颈MOET竖排最靠上字母T，E在其下。'),
('clear_visual_preference','滑垒者背号4，14并非该人。'),
('question_or_entity_ambiguity','封面同时有LIVE RECORDINGS OF OUTSTANDING MUSICIANS与G. VERDI REQUIEM；系列/专辑标题层级混杂，不确认负例错误。'),
('clear_visual_preference','左边水瓶Dasani蓝标可辨，非Dakota。'),
('clear_visual_preference','投手球衣35，非39。'),
('clear_visual_preference','堆栈底部The Book of Dogs，非You Are Here。'),
('question_or_entity_ambiguity','原尺寸屏幕同时有Rental和Rauchen，两者均R开头；未指具体区域，rejected也满足问题。'),
('clear_visual_preference','休息区人物衣上TEXAS可见，非Keesley；问题问衣上名称而非本人姓名。'),
('clear_visual_preference','绿色球衣号码50，非60。'),
('clear_visual_preference','黄衫REALITY STAR，非EQUALITY STAR。'),
('clear_visual_preference','人物腿间NBA图标，无额外C。'),
('clear_visual_preference','大指针朝12，非6。'),
('clear_visual_preference','原尺寸书封作者Gosse可见，非Gasse。'),
('clear_visual_preference','页首大标题THE FIRST TIME，rejected是正文附近长表达。'),
('clear_visual_preference','从左第二瓶SILVER TREE，Square One在第三瓶。'),
('clear_visual_preference','桶面数字倒置，按题意恢复方向为28，非82。'),
('clear_visual_preference','尺子最右可见数字9，非8。'),
('clear_visual_preference','原尺寸橙框内true清楚可读，拒答不符合可读文字。'),
('clear_visual_preference','箱体PENNZOIL，rejected漏N。'),
('clear_visual_preference','广告品牌FEATHERS，THE SHIELD WATCH为款式名。'),
('clear_visual_preference','原尺寸卡片Boat Trip 2006，非2005。'),
('unresolved_visual_detail','原尺寸显示器品牌字体似含连字；chosen belnea与品牌字样是否完全一致仍不确定，未以8票替代确认。'),
('clear_visual_preference','中间金发女孩球衣11，6在右侧另一人。'),
('clear_visual_preference','照片底部手写George Reid，Police officer是职业而非姓名。'),
('clear_visual_preference','邮戳1889可见，非1899。'),
('clear_visual_preference','原尺寸传送带电池包装DURACELL可读，非DUMBA。'),
('clear_visual_preference','黄色区域1991，1990在绿色区。'),
('clear_visual_preference','右瓶Schöfferhofer；rejected另插入R，不只是去重音。'),
('clear_visual_preference','球衣33，非31。'),
('semantic_alias','瓶标CUSQUEÑA；去附加符号cusquena与cusqueña指同一品牌，不能当正负事实偏好。'),
('clear_visual_preference','原尺寸蓝饮水机顶部Aquafina，非Great Value。'),
('clear_visual_preference','最后中间瓶LEUTE，Rochefort在左侧另一瓶。'),
('answer_scope_or_partial','rejected在同一文字核心上混合一字母拼写和多行跨度；可能是格式/截取偏好，不当纯事实负例。'),
('clear_visual_preference','硬币面值1，非10。'),
('clear_visual_preference','从左第二块点阵屏呈V形，非U形。'),
('question_or_entity_ambiguity','原尺寸表带多处COACH清楚，未见ROLEX；8票chosen缺乏可见品牌依据。'),
('clear_visual_preference','SAMSUNG前三字母SAM，非SAN。'),
('clear_visual_preference','原尺寸橙色牌MASSAGE；另一黄色牌含Dont be silly，区域不同。'),
('question_or_entity_ambiguity','原尺寸前桶NET WT 250 LB可见，未找到225；chosen与可见数字冲突，不能以10票确认。'),
]

def main():
    binding=json.loads((REPAIR/'INPUT_LOCK.json').read_text())
    for name,digest in binding['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Repair science source changed')
    plan=json.loads((REPAIR/'CONFIRM_PLAN.json').read_text())
    if sha_file(REPAIR/'PREFERENCES.json')!=plan['source_preferences_sha256']:raise ValueError('Repair candidates changed')
    sample=json.loads((REPAIR/'CONFIRM_SAMPLE.json').read_text())
    design=json.loads((REVIEW/'AUDIT_PLAN.json').read_text())['selected_question_ids']
    if set(plan['selected_question_ids']) & set(design):raise ValueError('Confirmation overlaps design')
    if len(NOTES)!=64:raise ValueError('Incomplete confirmation observations')
    records=[{'question_id':x['pair']['question_id'],'category':cat,'note':note,'image_path':x['image_path'],
              'source_display':f'confirm_page{x["index"]//8}.jpg','reviewer':'assistant','independent_human_label':False}
             for x,(cat,note) in zip(sample,NOTES)]
    completed_at=json.loads((REPAIR/'ANALYSIS.json').read_text())['completed_at'] if (REPAIR/'ANALYSIS.json').exists() else now()
    result={**audit_gate(records,plan),'categories':dict(Counter(x['category'] for x in records)),
            'completed_at':completed_at,'candidate_pairs':199,'confirmation_disjoint_from_design':True,
            'training_allowed':False,'training_launched':False,'optimizer_updates':0,'new_adapter_count':0,
            'decision':'do_not_train_repaired_pool;keep_failed_gate;no further tuning_on_confirm',
            'source_plan_sha256':sha_file(REPAIR/'CONFIRM_PLAN.json')}
    if result['gate_passed']:raise ValueError('Audit passed:freeze a separate training protocol instead of using this failure report')
    lock_file(REPAIR/'VISUAL_REVIEW.json',{'records':records,'summary':result})
    write_json(REPAIR/'ANALYSIS.json',result)
    write_json(REPAIR/'STATUS.json',{'status':'completed_quality_gate_failed','updated_at':now(),
              'source_questions':10000,'generated_responses_reused':50000,'candidate_pairs':199,
              'optimizer_updates':0,'training_launched':False,'input_lock_sha256':sha_file(REPAIR/'INPUT_LOCK.json')})
    write_json(REPAIR/'COMPLETION.json',{'status':'completed','completed_at':completed_at,
              'analysis_sha256':sha_file(REPAIR/'ANALYSIS.json'),'quality_gate_passed':False,'optimizer_updates':0})
    old=json.loads((REVIEW/'ANALYSIS.json').read_text())
    portable={'schema_version':'rarevlm.preference_semantic_review.v1','source_questions':10000,
              'source_generations':50000,'raw_pairs':1480,'original_conservative_pairs':749,
              'original_audit':old,'repair_pairs_before_design_exclusion':217,'repair_design_excluded':18,
              'repair_pairs':199,'repair_confirmation':result,'new_gpu_generations':0,'optimizer_updates':0,
              'limits':['two different candidate populations;fractions not a paired improvement estimate',
                        '128 distinct assistant-reviewed train images;not full human annotation or model validation',
                        'no dev/test model predictions;no accuracy or RLHF success claim']}
    write_json(PROJECT_ROOT/'results/vlm_preference_semantic_review_v1_metrics.json',portable)
    learning=REPAIR/'learning';learning.mkdir(exist_ok=True)
    esc=html.escape
    sections=[]
    for title,rows,observations in [('首轮固定审查',json.loads((REVIEW/'SAMPLE.json').read_text()),
                                  json.loads((REVIEW/'VISUAL_REVIEW.json').read_text())['records']),
                                 ('修复后独立确认',sample,records)]:
        cards=[]
        for x,o in zip(rows,observations):
            p=x['pair'];rel=os.path.relpath(x['image_path'],learning)
            color='#167047' if o['category']=='clear_visual_preference' else '#9e3825'
            cards.append(f'<article data-kind="{o["category"]}"><a href="{rel}"><img loading="lazy" src="{rel}" alt="源图 {esc(p["image_id"])}"></a>'
                         f'<div><b>QID {esc(p["question_id"])}</b><p>{esc(p["question"])}</p>'
                         f'<p>chosen：<code>{esc(p["chosen"])}</code><br>rejected：<code>{esc(p["rejected"])}</code></p>'
                         f'<strong style="color:{color}">{esc(o["category"])}</strong><p>{esc(o["note"])}</p>'
                         '<details><summary>十份原参考答案</summary>'+esc(json.dumps(x['answers'],ensure_ascii=False))+'</details></div></article>')
        sections.append(f'<h2>{title}</h2>'+''.join(cards))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RareVLM：偏好得分为什么不等于正确偏好</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f5f6f8;color:#18232e}article{display:grid;grid-template-columns:42% 1fr;gap:25px;padding:20px;background:white;border:1px solid #d5dce4;margin:18px 0;border-radius:10px}img{max-width:100%;max-height:350px;object-fit:contain}code{background:#edf0f5;padding:2px 5px;overflow-wrap:anywhere}pre{white-space:pre-wrap}button{padding:10px;margin:5px}@media(max-width:650px){article{grid-template-columns:1fr}}</style>
<h1>从真实图片理解：评分差异为何不能直接拿来做DPO</h1>
<p>10000训练图→50000回答→1480规则偏好→749保守候选。首轮固定抽查50/64明确（78.13%），未过门控。新规则重挖217对，排除设计集18对后199对；未参与写规则的64对确认56对明确（87.50%），仍未过90%及Wilson95%下界80%的双门槛。两批不同总体，不能据此声称筛选准确率显著提高。</p>
<p>本页128个不同训练图的判断均由助手看图完成，不是全量独立人审，不代表用户已掌握；不确定与歧义不计明确，也不都叫已证实错误。0新增生成、0优化器更新、0新adapter、无dev/test提分。</p>
<h2>代码与梯度连接</h2>
<p>标准sigmoid DPO使用同一图像/问题下chosen与rejected的assistant序列logp之差，并减去静态初始SFT参考差。prompt、视觉与padding位置不进logp；回答结束token按原mask保留。</p>
<pre>d = (logp_chosen - logp_rejected) - (ref_chosen - ref_rejected)
loss = -logsigmoid(0.1 * d)
# 在初始policy == reference点：loss = log(2) ≈ 0.693147
# 对logp_chosen的导数=-0.05；对logp_rejected的导数=+0.05</pre>
<p>因此把Coca-Cola/Coke或cusquena/cusqueña当正负例，就会在没有事实区别时压低一种正确表达。8～10票参考也可能与图中文字冲突。原真实8对GPU前后向合同验证过loss与LoRA梯度；这里解释初始点的数学方向，不冒充本轮训练结果。实际训练必须用固定初始参考，不能每步current_actor.detach。</p>
<p>源码：<code>src/vlm_alignment_signal/loss.py::dpo_loss</code>、<code>src/vlm_preference_review/repair.py::repair_pair/pair_reasons</code>、<code>src/vlm_preference_review/audit.py::audit_gate</code>。官方公式参考 <a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">TRL DPO</a>。</p>
<p>图片为原图展示，点击看原尺寸；缩放不是模型输入、注意力图或新增推理。所有生成均沿用原整图BF16/1024记录。</p>
<button onclick="document.querySelectorAll('article').forEach(x=>x.hidden=x.dataset.kind==='clear_visual_preference')">只看风险案例</button><button onclick="document.querySelectorAll('article').forEach(x=>x.hidden=false)">显示全部128例</button>
'''+''.join(sections)+'</html>'
    (learning/'学习页.html').write_text(document)
    print(json.dumps({'original':old,'confirmation':result,'portable_results':str(PROJECT_ROOT/'results/vlm_preference_semantic_review_v1_metrics.json'),
                      'learning_page':str(learning/'学习页.html')},ensure_ascii=False))

if __name__=='__main__':main()
