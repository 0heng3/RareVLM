"""Offline PNG views from the exact dashboard payload; no browser dependency."""
import base64
import io
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'
BG = '#f4f6f5'; INK = '#172a3d'; MUTED = '#607080'; LINE = '#dbe4e8'; TEAL = '#127467'; BLUE = '#316f9e'; RED = '#b24645'
LABELS = {'native_hard': '训练纠错128（拟合）', 'literal_validation': '内部OCR256', 'seen_validation': '原生问答512', 'retention': '保持集64',
          'singleword': '单词答案', 'multiword': '多词答案', 'rare_1_5': '罕见答案366', 'transfer_dev': '整图问答1617'}


class Canvas:
    def __init__(self, height):
        self.image = Image.new('RGB', (1520, height), BG); self.draw = ImageDraw.Draw(self.image); self.fonts = {}
    def font(self, size):
        if size not in self.fonts: self.fonts[size] = ImageFont.truetype(FONT, size)
        return self.fonts[size]
    def text(self, xy, value, size=20, color=INK):
        self.draw.text(xy, str(value), font=self.font(size), fill=color)
    def card(self, box, color='white'):
        self.draw.rounded_rectangle(box, radius=12, fill=color, outline=LINE, width=1)
    def wrap(self, xy, value, width, size=18, color=INK, max_lines=4):
        x,y=xy; line=''; lines=[]
        for token in re.findall(r"\n|[A-Za-z0-9_./:'?—-]+|[ ]+|.", str(value)):
            if token=='\n' or self.draw.textlength(line+token,font=self.font(size))>width:
                if line: lines.append(line.rstrip())
                line='' if token=='\n' else token.lstrip()
            else:line+=token
        if line:lines.append(line)
        for i,line in enumerate(lines[:max_lines]): self.text((x,y+i*(size+9)),line+('…' if i==max_lines-1 and len(lines)>max_lines else ''),size,color)
    def header(self, state, title):
        self.text((30,18),'RAREVLM / RESEARCH OBSERVATORY',14,TEAL); self.text((30,44),title,32)
        self.text((1030,32),'只读数据快照 · '+state['observed_at'][11:19],18,MUTED)
        self.text((1030,61),state['observed_at'][:10]+' · Asia/Shanghai',15,MUTED)
    def footer(self, state):
        y=self.image.height-31
        self.text((30,y),'采集：'+state['observed_at']+'  |  离线数据渲染，与HTML共用同一份状态。',14,MUTED)
        self.text((1090,y),'不加载模型 · 不改变任何实验',14,MUTED)


def number(v, digits=3): return f'{v:.{digits}f}' if isinstance(v,(int,float)) else '未评测'


def curve(c, box, roles, kind, title):
    x0,y0,x1,y1=box;c.card(box);c.text((x0+20,y0+14),title,22)
    left=x0+60;right=x1-25;top=y0+70;bottom=y1-50
    values=[p[kind] for r in roles for p in r['curve']['points'] if p.get(kind) is not None];maximum=max(max(values)*1.12,1e-8) if values else 1
    for i in range(4):
        value=maximum*i/3;yy=bottom-(bottom-top)*i/3;c.draw.line((left,yy,right,yy),fill=LINE,width=1);c.text((x0+10,yy-12),f'{value:.2f}',14,MUTED)
    for r in roles:
        pts=[(left+(p['step']-1)/31*(right-left),bottom-p[kind]/maximum*(bottom-top)) for p in r['curve']['points'] if p.get(kind) is not None]
        color=TEAL if r['id']==roles[0]['id'] else BLUE
        if len(pts)>1:c.draw.line(pts,fill=color,width=3)
        for x,y in pts:c.draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    for step in [1,8,16,24,32]:c.text((left+(step-1)/31*(right-left)-8,bottom+7),step,14,MUTED)
    legend_x=x0+20
    for index,role in enumerate(roles):
        c.text((legend_x,y1-28),role['label'],14,TEAL if index==0 else BLUE)
        legend_x+=c.draw.textlength(role['label'],font=c.font(14))+24


def render(state, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runtime=state.get('runtime_errors',[]);role_labels={r['id']:r['label'] for r in state['roles']};control_id=state['roles'][0]['id']
    c=Canvas(1260);c.header(state,'实验运行总览')
    pending_holdout = state.get('source_holdout_validation') and not state['source_holdout_validation'].get('completion')
    c.card((30,104,1490,180),'#fff8ec');c.text((49,116),'运行异常' if runtime else '新验证等待空卡' if pending_holdout and state.get('state')=='waiting' else '数据准备中' if state.get('state')=='preparing' else '主实验结束' if state['completion'] and state.get('active_jobs') else '运行已完成' if state['completion'] else '运行中',23,RED if runtime else TEAL)
    c.text((270,116),'研究结论：'+state['research'],23,'#95611b')
    c.text((49,151),'训练是否正常、门槛是否通过、原因是否证实，分别显示。',16,MUTED)
    cards=[('训练更新',' / '.join(str(r['step'])+'/32' for r in state['roles']),' / '.join(r['label'] for r in state['roles'])),('固定纠错样本',str(state['dataset']['selected'])+' 条',str(state['dataset']['unique_targets'])+'个归一化答案'),('看图审核',str(state['dataset']['supported'])+' / '+str(state['dataset']['reviewed']),'支持 / 已审；非全池错标率'),('评测图像交叉',str(state['dataset']['overlap']),'图像与像素；未评测项不填0')]
    for i,(label,value,note) in enumerate(cards):
        x=30+i*370;c.card((x,196,x+350,311));c.text((x+17,207),label,17,MUTED);c.text((x+17,233),value,29);c.text((x+17,278),note,15,MUTED)
    for i,s in enumerate(state['stages']):
        x=30+i*246;color='#fbf4e8' if s['state']=='not_admitted' else 'white';c.card((x,327,x+230,427),color)
        c.text((x+13,336),f'{i+1:02d}  '+s['label'],17);c.text((x+13,368),'未放行' if s['state']=='not_admitted' else f"{s['value']}/{s['total']}",18,TEAL if s['state']=='completed' else '#95611b')
        c.draw.rounded_rectangle((x+13,405,x+217,411),radius=3,fill=LINE)
        width=204*min(1,s['value']/s['total'])
        if width:c.draw.rounded_rectangle((x+13,405,x+13+width,411),radius=3,fill=TEAL)
    c.card((30,445,750,811));c.text((49,459),'GPU与任务',22)
    if state['gpu']['status']=='observed':
        for i,g in enumerate(state['gpu']['devices']):
            y=501+i*77;c.text((49,y),f"GPU {g['index']} · {g['name'].replace('NVIDIA ','')}",18)
            c.text((49,y+29),f"利用率 {g['utilization']:.0f}%  |  显存 {g['used_gib']:.2f}/{g['total_gib']:.0f} GiB  |  计算进程 {len(g['processes'])}",16,MUTED)
    else:c.wrap((49,506),'GPU信息未采集，不能视为空闲。',660,19,MUTED)
    if state.get('active_jobs'):
        c.text((49,634),'运行：'+' / '.join(j['label'] for j in state['active_jobs']),14,TEAL)
    elif state.get('auxiliary_runs'):
        a=state['auxiliary_runs'][-1];c.text((49,634),'辅助：'+a['name']+' · '+a['status']+' · '+a['detail'].split('；')[0],16,'#95611b')
    elif state.get('batch_diagnostic'):
        d=state['batch_diagnostic'];c.text((49,634),f"零更新诊断 {d['new_generations']}/96：选定6题跨组差异 {d['batch_disagreements']}/6",16,'#95611b')
    for i,r in enumerate(state['roles']):
        y=669+i*57;c.text((49,y),f"{r['label']}  {r['step']}/32 · {r['phase']}",18,TEAL if i==0 else BLUE)
        c.text((49,y+27),f"训练 {number(r['train_seconds'],1)} 秒；峰值 {number(r['peak_gib'],2)} GiB；{number(r['trainable']/10000,2)}万更新参数",15,MUTED)
    independent = bool(state.get('independent_validation'))
    full_task = independent or state.get('study') in ('vlm_late_visual_dpo_v1','vlm_late_rank_matched_dpo_v1','vlm_stvqa_replay_v1')
    c.card((770,445,1490,811));c.text((789,459),'完整原生问答1617 · VQA分/%' if full_task else '原生问答512 · VQA分/%',22)
    for i,r in enumerate(state['scores']['transfer_dev' if full_task else 'seen_validation']):
        y=510+i*47;c.wrap((789,y),r['label'],150,15,max_lines=2)
        if r['value'] is not None:
            width=r['value']/100*400;c.draw.rounded_rectangle((950,y+3,950+width,y+22),radius=4,fill=TEAL if r['id']==control_id else BLUE if r['id'] in role_labels else '#c7d7dc');c.text((1365,y-2),number(r['value']),17)
        else:c.text((1335,y),'未评测',16,MUTED)
    holdout = state.get('source_holdout_validation',{})
    c.text((789,766),f"新来源验证：{holdout['completed']}/{holdout['expected']}生成+GT前向，0更新" if holdout else '各集合分别比较，结合罕见答案与保持门槛判断。',16,MUTED)
    curve(c,(30,829,750,1199),state['roles'],'ce','逐QA日志汇总：正确答案CE')
    c.card((770,829,1490,1199));c.text((789,844),'实际未通过项',22)
    for i,f in enumerate(state['failures'][:3]):
        y=890+i*84;role=role_labels[f['role']]
        c.text((789,y),role+' · '+f['label'],18);c.text((1250,y),f"{f['delta_pp']:+.3f} pp",20,RED)
        requirement=f"门槛 ≥ {f['threshold']:+.1f} pp"+('且区间下界>0' if f['cohort']=='transfer_dev' else '')
        c.text((789,y+28),requirement+f"；95% [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}]",15,MUTED)
    if not state['failures']:c.wrap((789,900),runtime[0]['tail'][-500:] if runtime else '尚未生成失败项，等待完整评测。',640,18,MUTED,max_lines=7)
    c.text((789,1167),'区间跨0时，不能称已证明显著退化。',16,MUTED);c.footer(state);c.image.save(output/'总览.png')

    c=Canvas(1570);c.header(state,'错误与证据：先定位失败项，再判断原因')
    c.card((30,104,1490,231),'#fff8ec');c.text((49,119),'运行异常：请核对原始日志，不自动重启。' if runtime else '本轮：指标未达门槛，未扩训；具体机制尚未单独证实。' if state['failures'] else '本轮暂无已确认指标失败；下面保留监督审核与已有完整案例。',23,'#95611b')
    for i,f in enumerate(state['failures'][:3]):c.text((49,160+i*21),role_labels[f['role']]+' · '+f['label']+f" {f['delta_pp']:+.3f} pp，门槛 ≥ {f['threshold']:+.1f} pp"+('且区间下界>0' if f['cohort']=='transfer_dev' else ''),15,MUTED)
    cases=[x for x in state['cases'] if x['status']=='score_loss' and x.get('rare')][:3]+[x for x in state['cases'] if x['id'] in ('t-49','t-64','t-70')]
    for i,q in enumerate(cases[:6]):
        x=30+(i%3)*494;y=251+(i//3)*620;c.card((x,y,x+474,y+600));c.text((x+15,y+13),'评测得分下降' if q['group']=='评测变化' else '监督暂不准入',20,RED if q['group']=='评测变化' else '#95611b')
        with Image.open(io.BytesIO(base64.b64decode(q['image'].split(',',1)[1]))) as raw:im=ImageOps.contain(raw.convert('RGB'),(444,200))
        c.image.paste(im,(x+15+(444-im.width)//2,y+53+(200-im.height)//2))
        c.wrap((x+15,y+266),q['question'],444,17,max_lines=3)
        c.wrap((x+15,y+354),'原标签：'+q['target'],444,16,max_lines=2)
        c.wrap((x+15,y+405),('起点：'+q['before']+' → '+q['prediction']) if 'before' in q else ('实际回答：'+q['prediction']),444,16,max_lines=2)
        c.wrap((x+15,y+468),q['reason'],444,16,MUTED,max_lines=4)
        c.text((x+15,y+568),'q'+q['qid']+' · '+('尚未逐图重标' if q['group']=='评测变化' else '助手审核，非专家重标注'),14,MUTED)
    c.text((30,1510),'历史KeyError：已独立恢复汇总，0重训/0重新生成；不作为当前训练故障。',17,MUTED);c.footer(state);c.image.save(output/'错误原因.png')

    c=Canvas(1160);c.header(state,'训练曲线与完整内部指标')
    curve(c,(30,112,750,510),state['roles'],'ce','正确答案CE（训练损失）');curve(c,(770,112,1490,510),state['roles'],state.get('regularizer','dpo'),'本轮只有标准CE，无附加损失' if state.get('regularizer')=='none' else '回放QA的KL（SFT权重为0）' if state.get('regularizer')=='kl' else '偏好QA的DPO（SFT权重为0）')
    c.card((30,530,1490,1097));c.text((49,546),'训练拟合与泛化分列；数值均读取现有完整分析。',22)
    columns=[(960+280*i,r) for i,r in enumerate(state['roles'])]
    for x,t in [(49,'指标 / 集合'),(650,'起点')]+[(x,r['label']+'32') for x,r in columns]:c.text((x,596),t,18,MUTED)
    metrics=state['analysis'].get('metrics',{})
    for i,key in enumerate(LABELS):
        y=642+i*51;c.draw.line((49,y-9,1460,y-9),fill=LINE,width=1);c.text((49,y),LABELS[key],18)
        for x,r in [(650,'start')]+[(x,r['id']) for x,r in columns]:c.text((x,y),number(metrics.get(r,{}).get(key)),18,INK if key!='native_hard' else MUTED)
    c.footer(state);c.image.save(output/'训练曲线.png')
    return {'status':'passed','renderer':'Pillow offline data views','overview_png':str(output/'总览.png'),'browser_runtime_test':'not run; use HTML for interaction'}
