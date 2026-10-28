"""Preserve the failed factual-data gate; admit only a declared noise diagnostic."""
from collections import Counter
import json
from pathlib import Path
from vlm_preference_review.audit import wilson
from vlm_probe.evaluate import sha_file, now, lock_file
from vlm_alignment_signal.common import write_json
from rarevlm_paths import PROJECT_ROOT

ROOT=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_public_preference_v1')
# The criterion is a visibly grounded factual preference, not complete response truth.
# Human segment corrections can leave shared speculation/incorrect clauses behind.
NOTES=[
('clear','站立持桨板与女孩图文有依据，修正man/riding a wave；并不验证运动经验推测。'),
('clear','桌面贴纸可见，画面未见所称椅子；用户喜欢贴纸的推断未验证。'),
('clear','球员单手持拍、一脚离地，未见第二人；修正多处实体/姿态。'),
('clear','栏杆旁多人和左侧伞可见，较four/right说法吻合；局部纠错非全段真值。'),
('clear','墙上蓝色短柄刮水器可见，非门。'),
('clear','行人在侧边行走而非一群等过街；车内驾驶者细节未独立确认。'),
('ambiguous_posture','猫蜷卧/蹲坐的loaf姿态，两种口语描述均可能，不确认为事实负例。'),
('clear','喷水可见；体重秤并非测已倒入厕具的水重量。厕具具体类别仍有残余措辞问题。'),
('clear','三人只有左侧穿白外套，修正都穿白袍；大学名称猜测未确认。'),
('semantic_equivalent','next to dining table/at dining table与逗号变化，无明确视觉事实偏好。'),
('clear','远近车流明显超过七辆，at least13是可核对的保守下界。'),
('semantic_equivalent','chef hat/toque、headwear/headgear为同义改写，不是明确事实纠错。'),
('unresolved_detail','鱼缸plants相对sand的改动在当前图中不够明确，不能以人类来源代替确认。'),
('clear','部分牛未朝向手风琴，修正all watching；和谐共处等叙述为推测。'),
('clear','蛋糕Pamie与Happy Birthday可辨，非Posey。'),
('unresolved_motion','静态卡车图不能明确分辨行驶/路边停放，两回答均否认送货。'),
('clear','长椅在铺装路边而非草上。'),
('clear','可见玻璃杯及坐姿，补充与问题相关的视觉内容；两版都正确提到黑手机。'),
('clear','狗旁边是蛋糕造型玩具，不是第二只玩具狗。'),
('clear','空中是海鸥，未见所述风筝。'),
('clear','微波炉在冰箱上方、电炉在右，修正布局与不存在的烤面包机。'),
('clear','手拿食物并竖拇指，非喝粉色饮料。'),
('clear','蓝天飞机图未出现太阳；白天不等于可见太阳。'),
('clear','脚部模糊，拒绝确认雪靴比肯定更有图像依据。'),
('clear','桌上电脑工作姿态，非满桌食物正在吃饭。'),
('clear','仅一人戴眼镜，另一副在桌上。'),
('clear','可见近旁四人观看，更多人分组；修正八人全观看及滑板数，局部计数为助手观察。'),
('semantic_equivalent','sliding/riding与holding/gripping同义，均双手抓板。'),
('clear','两人脚踩双板滑雪板，非全体站单板。'),
('clear','车牌末尾59G，而非731。'),
('unresolved_action','低头接近草地可见，但静图不能清楚确认正在进食，未将姿态推断当定论。'),
('clear','盒中至少九个可见甜甜圈，修正无依据的精确12；并不确认所有叙事推断。'),
('clear','水在低于两侧地面的水渠内，非漫过土路。'),
('clear','画面没有所称地面人物和多车，修正这些幻觉；起降方向不据静图确认。'),
('clear','近处大巴在右、远巴在左，修正左右与额外车辆/行人细节。'),
('ambiguous_posture','趴在bodyboard上也属于riding，两版near/in middle of wave难作明确负事实。'),
('clear','可见至少五头而非九头；dangerous environment是未被确认的新增推测。'),
('clear','裙子在衣架上挂着而非折叠摆放。'),
('shared_or_conflicting_error','图为螺旋桨军机，两版都称jet，不能将此作为可信飞机类型纠错。'),
('clear','白色遮阳帽带帽檐，非头带。'),
('subjective_question','best place取决于猫偏好；玻璃桌可见，但不能从图确认最优睡处。'),
('subjective_question','heavy set/overweight判断缺少客观标注，不作明确事实偏好。'),
('clear','四人球衣含红、黑红、棕白格纹，较只红黑更完整。'),
('clear','仅少量可见人物，去除several people散布等不支持描述；不保证整段全真。'),
('shared_or_conflicting_error','图片有多个装酒杯；两版空杯/红白酒计数均可疑，不能确认chosen完整正确或更优。'),
('semantic_equivalent','mounted on a pole/placed on a pole，无明确事实变化。'),
('clear','后视镜反射棕色狗，非棕色车。'),
('clear','明确前后景及手按额头，避免直接断言两人婚姻角色；角色真实情况未验证。'),
('clear','泥地为枝条，未见所加鸟群。'),
('unsupported_relationship','chosen新增might be father，图片不能确认亲子关系；rejected只说man有依据。'),
('clear','可见五头吃草牛，非六头。'),
('clear','问题要求详细描述，水果和蜡烛等可见内容比单词Cake更充分；个别水果种类未全确认。'),
('clear','空中翼伞与伞下人可见，修正普通kites描述。'),
('clear','车头编号117，非174。'),
('clear','手中长蓝带可见，非锐利刀。'),
('clear','画面有龙与角落文字，修正shark及龙栖石的细节；长篇情绪解读不是事实标注。'),
('clear','燃气灶头可见，非电炉。'),
('clear','厨房地面有瓷砖网格，非木地板。'),
('clear','未见长颈鹿吐舌，修正明显吐舌；共享的吸湿解释未验证且不当事实真值。'),
('clear','为滑板动作而非雪板，修正主要活动；季节仅凭衣着不能确定。'),
('clear','可见画和工具，去掉无依据的多个花瓶/两杯；不是整段无幻觉证明。'),
('clear','人物朝镜头微笑，未见所加桌面饮料杯。'),
('clear','无法从空拖车及店招推断运巧克力去巴黎，chosen更谨慎。'),
('semantic_equivalent','at least two/outdoor area与two/courtyard范围表达差，不能作明确事实负例；人物性别未独立核实。'),
]

def main():
    audit_path=ROOT/'semantic_audit/FROZEN_64.json';audit=json.loads(audit_path.read_text())
    assert len(NOTES)==len(audit['samples'])==64
    records=[{'index':i,'question_id':r['question_id'],'category':kind,'note':note,
              'reviewer':'assistant','independent_human_truth':False,
              'scope':'direction of local factual preference;not complete response correctness'}
             for i,(r,(kind,note)) in enumerate(zip(audit['samples'],NOTES))]
    clear=sum(x['category']=='clear' for x in records);interval=wilson(clear,64)
    passed=clear/64>=.9 and interval[0]>=.8
    review={'status':'completed','records':records,'clear':clear,'n':64,'clear_fraction':clear/64,
            'wilson95':interval,'clear_fraction_min':.9,'wilson95_lower_min':.8,
            'factual_quality_gate_passed':passed,'categories':dict(Counter(x['category'] for x in records)),
            'frozen_audit_file_sha256':sha_file(audit_path),
            'limits':['assistant judgments;not independent human reannotation','local corrections may retain shared unsupported claims',
                      'equivalent pairs are not proven reversed labels;not_clear is not a measured flip rate']}
    assert passed is False
    lock_file(ROOT/'semantic_audit/VISUAL_REVIEW.json',review)
    # Preserve the original factual-quality decision as rejected. No old pool is admitted.
    write_json(ROOT/'DATA_ADMISSION.json',{'status':'rejected_factual_quality_gate','training_allowed':False,
              'train_manifest_sha256':sha_file(ROOT/'TRAIN_MANIFEST.json'),'factual_quality_gate_passed':False,
              'semantic_review_sha256':sha_file(ROOT/'semantic_audit/VISUAL_REVIEW.json')})
    # Different scientific question, independently declared before optimizer work:
    # can standard DPO learn useful behavior from imperfect public human edits?
    # This cannot promote a reliable-data or clean-alignment claim.
    protocol=PROJECT_ROOT/'docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md'
    if not protocol.exists():raise ValueError('Diagnostic protocol must precede its admission')
    admission={'status':'admitted','scope':'noisy_public_preference_diagnostic',
               'training_allowed':True,'factual_quality_gate_passed':False,'automatic_promotion_allowed':False,
               'train_manifest_sha256':sha_file(ROOT/'TRAIN_MANIFEST.json'),
               'heldout_manifest_sha256':sha_file(ROOT/'HELDOUT_MANIFEST.json'),
               'semantic_review_sha256':sha_file(ROOT/'semantic_audit/VISUAL_REVIEW.json'),
               'diagnostic_protocol_sha256':sha_file(protocol),
               'rationale':'full public human segment-correction data with observed uncertainty;matched SFT/DPO diagnostic,not a qualified clean-preference run',
               'excludes_old_failed_pools':['vlm_preference_data_v1','vlm_preference_repair_v1'],
               'limits':['do not lower or overwrite the failed factual-data gate','no automatic extra seeds/formal benchmarks/resume promotion',
                         'POPE and Text generation decide measured behavior;training loss or heldout pair ranking alone is insufficient']}
    lock_file(ROOT/'DIAGNOSTIC_ADMISSION.json',admission)
    print(json.dumps({'clear':clear,'n':64,'wilson95':interval,'factual_quality_gate_passed':passed,
                      'diagnostic_only_admission':str(ROOT/'DIAGNOSTIC_ADMISSION.json')},ensure_ascii=False))

if __name__=='__main__':main()
