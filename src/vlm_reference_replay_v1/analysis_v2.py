import html
import os
from urllib.parse import quote
from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import write_json
from .common import ROOT,TRANSFER,read
from .study import RESULT,PAGE,PROTOCOL

ROLES=("plain","soft_kl","hard_replay")

def analyze():
    from vlm_grpo_iteration_v1.study import paired
    cal=read(ROOT/'CALIBRATION.json')
    if not cal['training_allowed']:
        result={'status':'completed_diagnostic_no_training','calibration':cal,'goal_complete':False}
        write_json(ROOT/'COMPLETION.json',result);return result
    records={role:{name:[q for q in read(ROOT/'runs'/role/name/'PREDICTIONS.json') if q['evaluation_scored']]
                   for name in ('training_context','seen_validation','retention','transfer_dev')} for role in ('initial',)+ROLES}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={role:{name:percent(rows) for name,rows in groups.items()} for role,groups in records.items()}
    comparisons={};gates={};slices={}
    for cohort in ('seen_validation','transfer_dev'):
        for reference in ('initial','plain','hard_replay'):
            value=paired(records[reference][cohort],records['soft_kl'][cohort]);key=cohort+':'+reference
            comparisons[key]=value;gates[key+':overall']=value['delta_pp']>=.5 and value['ci95'][0]>0
            for field,values in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in records['initial'][cohort]}))]:
                for label in values:
                    delta=paired([q for q in records[reference][cohort] if q[field]==label],[q for q in records['soft_kl'][cohort] if q[field]==label])
                    slices[key+':'+label]=delta;gates[key+':'+label]=delta['delta_pp']>=(.5 if reference=='initial' and label=='rare_1_5' else -.5)
    for reference in ('initial','plain','hard_replay'):gates[reference+':retention']=scores['soft_kl']['retention']-scores[reference]['retention']>=-.5
    full={role:sum(q['vqa_score']>=.999999 for q in groups['training_context']) for role,groups in records.items()};gates['training_net32']=full['soft_kl']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],[q for q in records['soft_kl']['seen_validation'] if q['question_id'] in protected]);gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    result={'status':'completed','passed':all(gates.values()),'scores_percent':scores,'comparisons':comparisons,'slices':slices,'gates':gates,
            'training_full':full,'calibration':cal,'previously_full_multiword':keep,'goal_complete':False,
            'next':'Joint pass permits separate confirmation only;no automatic extra steps/seeds/formal/CV.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT未用图像dev1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 教师KL与加强CE完整结果\n\n'+table+'\nβ='+str(cal['beta'])+'；联合门槛='+str(result['passed'])+'。单seed内部观察，不称盲测/三seed稳定。\n\n[协议](VLM_REFERENCE_REPLAY_V1_PROTOCOL.md) · [原图与损失原理](learning/教师KL保留与加强CE回放.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result

def render(result=None):
    prior=read(TRANSFER/'POSTRUN_TRANSFER_REVIEW.json');data=read(ROOT/'DATA.json');relative=lambda path:quote(os.path.relpath(path,PAGE.parent),safe='/._-')
    saved={role:{q['question_id']:q for q in read(ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json')} for role in ('initial',)+ROLES if (ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json').exists()}
    cards=[]
    for case in prior['examples']:
        body=''
        for role,values in saved.items():
            if case['question_id'] in values:
                q=values[case['question_id']];body+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (role,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>dev q{case["question_id"]}</h2><img loading="lazy" src="{relative(case["image"])}" alt="前轮多词降分原图"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}；旧merger回答：{html.escape(case["candidate"])}</p><p>这是事后讲解病例，不改标签，不据这些问题选β或加入训练。</p><table><tr><th>组</th><th>本轮实际生成</th><th>VQA</th></tr>{body}</table></article>')
    status='准备/运行中，尚无新提分结果。' if result is None else html.escape(str(result['scores_percent']))+'；联合通过='+str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>教师KL与加强CE</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:18px;margin:18px 0;border:1px solid #cbd5e1}img{width:100%;max-height:440px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>纠错与保留：软分布还是加强硬标签</h1><p>'+status+'</p><pre>两新模型同数据/起点/64步/β，qv1843200参数。\nL_base = (Σ8hard CE + Σ8replay CE) / 16\nsoft_KL = L_base + β · mean_replay KL(p_初始 || p_当前)\nhard_replay = L_base + β · mean_replay CE_GT\nKL方向是初始教师→当前学生，完整词表151936/T=1。\nβ仅由训练checkpoint16/32/48的梯度范数比中位数校准，截在.1–100。\n缓存128批/512QA完整GT前缀分布，输入/labels/grid/pixels字节必须匹配。\n相同β不保证梯度/Adam步长或算力相同，强CE可初始非0、KL近0合法。\n训练前恢复原SFT，不从校准状态继续；无新奖励/策略梯度。\nseen512与SFT未用图像dev1617分列，均须超过初始、普通CE、加强CE。\n以下保存原图与实际预测，分数变化不自动等于语义优劣。</pre>'+f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_reference_replay_v1/loss.py")}">KL代码</a> · <a href="{relative(SOURCE_ROOT/"vlm_reference_replay_v1/reference.py")}">梯度校准</a></p>'+''.join(cards)+'</html>')
