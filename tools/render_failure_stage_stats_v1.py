"""Read-only failure-stage counts and real image examples from saved evidence."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
from PIL import Image,ImageOps
from rarevlm_monitor_raster import Canvas,INK,MUTED,TEAL,RED,BLUE
from rarevlm_paths import PROJECT_ROOT

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_failure_stage_stats_v1'


def render():
    read=lambda p:json.loads(p.read_text());s=read(ROOT/'SUMMARY.json');state={'observed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')};c=Canvas(1680);c.header(state,'失败发生在哪一步：运行异常与性能未达标分开计数')
    c.card((30,104,1490,200),'#fff8ec');c.text((49,115),'本轮5个队列作业均完成；两组均卡在原生/罕见答案保护。',24,'#95611b');c.text((49,157),'最近两轮4/4候选未过rare门槛；不是独立seed失败率，也不能与历史作业数直接相加。',17,MUTED)
    c.card((30,220,745,870));c.text((49,236),'本轮逐步骤统计',24)
    for x,label in [(49,'步骤'),(342,'单位'),(538,'通过'),(631,'失败')]:c.text((x,286),label,18,MUTED)
    for i,row in enumerate(s['current_step_counts']):
        y=331+i*49;c.draw.line((49,y-9,725,y-9),fill='#dbe4e8');c.text((49,y),row['step'],18);c.text((342,y),row['unit'],15,MUTED);c.text((551,y),row['passed'],20,TEAL);c.text((645,y),row['failed'],20,RED if row['failed'] else MUTED)
    c.wrap((49,791),'同一worker覆盖多个步骤；模型、作业和检查数量不能跨行相加。CPU为保存的最终通过检查，开发期未落盘错误另列。',670,16,MUTED,max_lines=3)
    c.card((766,220,1490,870));c.text((785,236),'历史运行失败：6条执行记录',24)
    h=s['historical_queue_records'];c.text((785,285),'131目录 / 124队列状态 / 231去重执行记录',18,MUTED);c.text((785,322),'历史快照：219完成，6失败，2中止，4标running。',18)
    labels=[('evaluation_image_read','评测读取图片',2),('artifact_contract_handoff','工件交接/契约检查',2),('post_eval_metadata','评测后元数据检查',1),('pre_update_numeric_check','首次更新前数值检查',1)]
    for i,(key,label,expected) in enumerate(labels):
        value=h['failed_by_stage'].get(key,0);y=375+i*91;c.text((785,y),label,19);c.draw.rounded_rectangle((785,y+34,1370,y+52),radius=4,fill='#e1e8ea');c.draw.rounded_rectangle((785,y+34,785+value/2*585,y+52),radius=4,fill=RED);c.text((1400,y+24),value,22,RED)
    c.wrap((785,748),'5/6位于评测、元数据或工件交接；1/6在首次优化器更新前被拦下。这里未发现核心优化器OOM/NaN报错。',664,17,MUTED,max_lines=3)
    c.text((785,841),'其中2条陈旧；2条是本次诊断采集时状态，现已完成。',14,MUTED)
    c.card((30,890,1490,1013));c.text((49,905),'性能结果：原生512仅+0.117pp；rare366为−0.656pp，95%区间跨0。',22)
    c.text((49,950),'SFT与KL的512条回答字符串完全相同；整体提升与罕见答案保持均未达到目标。',19,RED)
    c.text((30,1034),'本轮8个rare丢分病例中的3个明确错误（原图/原分数保留）',22)
    r=BASE/'vlm_native_stable_kl_v1';data=read(r/'data/MANIFEST.json');meta={q['question_id']:q for q in data['evaluations']['seen_validation']};before={q['question_id']:q for q in read(r/'baseline/seen_validation/PREDICTIONS.json')};after={q['question_id']:q for q in read(r/'runs/sft/internal/seen_validation/PREDICTIONS.json')}
    notes=[('21295','帽子写COLTS，回答变成calgary。'),('14780','可见Heidi Rewell，回答额外写入e。'),('26752','问题指定蓝色区域，回答读取相邻区域。')]
    for i,(qid,note) in enumerate(notes):
        x=30+i*494;y=1080;c.card((x,y,x+474,y+500));q=meta[qid]
        with Image.open(data['images'][q['image_id']]['path']) as raw:im=ImageOps.contain(raw.convert('RGB'),(444,224))
        c.image.paste(im,(x+15+(444-im.width)//2,y+15+(224-im.height)//2))
        c.wrap((x+15,y+254),q['question'],444,16,max_lines=3);c.wrap((x+15,y+336),before[qid]['prediction']+' → '+after[qid]['prediction'],444,18,RED,max_lines=2);c.wrap((x+15,y+399),note,444,17,MUTED,max_lines=3);c.text((x+15,y+469),'q'+qid+' · 两组出现同样丢分',14,MUTED)
    c.text((30,1600),'其余病例含答案范围/实体歧义与遮挡风险，不能将全部负分归为识字失败或修改评分。',17,MUTED);c.footer(state);c.image.save(ROOT/'失败步骤与真实案例.png')
    print(json.dumps({'status':'passed','figure':str(ROOT/'失败步骤与真实案例.png'),'source':str(ROOT/'SUMMARY.json')},ensure_ascii=False))


if __name__=='__main__':render()
