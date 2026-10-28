"""Post-hoc read-only 2x2 result review, with real saved answers and images."""
from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_grpo_iteration_v1.study import paired

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_stvqa_replay_v1'
OLD = ROOT.parent / 'vlm_visual_preference_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'


def main():
    dest = ROOT / 'result_review_v1'
    if (dest / 'COMPLETION.json').exists():
        raise FileExistsError('Completed review is reusable; do not repeat it')
    paths = {'start': ROOT / 'baseline/transfer_dev/PREDICTIONS.json',
             'old_ce': OLD / 'runs/ce/transfer/transfer_dev/PREDICTIONS.json',
             'old_dpo': OLD / 'runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json',
             'new_ce': ROOT / 'runs/ce/transfer/transfer_dev/PREDICTIONS.json',
             'new_dpo': ROOT / 'runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json'}
    protected = {str(p): sha_file(p) for p in list(paths.values()) +
                 [ROOT / 'ANALYSIS.json', ROOT / 'DECISION.json', ROOT / 'data/MANIFEST.json']}
    write_json(dest / 'PLAN.json', {'status': 'frozen', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Post-hoc 2x2 interaction and saved-case learning; not a new success gate or a causal mechanism diagnosis',
        'contrast': '(new_DPO-new_CE)-(old_DPO-old_CE)',
        'interval': 'Same paired 6000 image-cluster bootstrap/seed as frozen analyzer; unadjusted exploratory interval',
        'case_rule': 'Fixed q2586/q11608 plus first CE regression and first DPO rescue in original evaluation order',
        'protected': protected, 'new_updates': 0, 'new_generations': 0})
    records = {key: read(path) for key, path in paths.items()}
    identity = lambda rows: [(q['question_id'], q['image_id'], q['input_sha256']) for q in rows]
    if any(identity(rows) != identity(records['start']) for rows in records.values()):
        raise ValueError('Saved predictions do not share question/image/input identity')
    manifest = read(ROOT / 'data/MANIFEST.json')
    metadata = {q['question_id']: q for q in manifest['evaluations']['transfer_dev']}
    effects = {}
    for prefix in ('old', 'new'):
        effects[prefix] = [{**a, 'vqa_score': b['vqa_score'] - a['vqa_score']}
                          for a, b in zip(records[prefix + '_ce'], records[prefix + '_dpo'])]
    interaction = paired(effects['old'], effects['new'])
    transitions = {}
    for before, after in [('start', 'new_ce'), ('new_ce', 'new_dpo'),
                          ('old_ce', 'new_ce'), ('old_dpo', 'new_ce')]:
        deltas = [b['vqa_score'] - a['vqa_score'] for a, b in zip(records[before], records[after])]
        transitions[before + '_to_' + after] = {
            'up': sum(v > 1e-9 for v in deltas), 'down': sum(v < -1e-9 for v in deltas),
            'unchanged': sum(abs(v) <= 1e-9 for v in deltas), 'net_score': sum(deltas)}
    aligned = [{**metadata[a['question_id']], 'models': {
        role: {'prediction': rows[i]['prediction'], 'vqa_score': rows[i]['vqa_score']}
        for role, rows in records.items()}} for i, a in enumerate(records['start'])]
    write_json(dest / 'PAIRED_CHANGES.json', aligned)
    scores = {k: 100 * sum(q['vqa_score'] for q in rows) / len(rows) for k, rows in records.items()}
    rare = [q for q in aligned if 1 <= q['answer_frequency'] <= 5]
    rare_scores = {k: 100 * sum(q['models'][k]['vqa_score'] for q in rare) / len(rare) for k in records}
    removed = sum(q['models']['new_ce']['vqa_score'] > q['models']['start']['vqa_score'] + 1e-9
                  and q['models']['new_dpo']['vqa_score'] < q['models']['new_ce']['vqa_score'] - 1e-9
                  for q in aligned)
    result = {'status': 'completed', 'n': len(aligned), 'image_clusters': interaction['image_clusters'],
              'scores_percent': scores, 'full_rare342_scores_percent': rare_scores,
              'data_loss_interaction': interaction, 'transitions': transitions,
              'ce_improved_questions_reduced_by_dpo': removed,
              'primary_study_decision_unchanged': read(ROOT / 'DECISION.json')['decision'],
              'inference_scope': 'Single-seed, reused internal evaluation; post-hoc interaction is exploratory. No unique data/gradient/architecture cause established.',
              'new_updates': 0, 'new_generations': 0, 'goal_complete': False}
    write_json(dest / 'ANALYSIS.json', result)
    picked = [('2586', '起点与SFT均正确，DPO后下降'), ('11608', 'SFT改善，DPO后回到旧错误')]
    first_regression = next(q for q in aligned if q['models']['new_ce']['vqa_score'] < q['models']['start']['vqa_score'] - 1e-9)
    first_rescue = next(q for q in aligned if q['models']['new_dpo']['vqa_score'] > q['models']['new_ce']['vqa_score'] + 1e-9)
    picked += [(first_regression['question_id'], 'SFT也有退化：保留反例'),
               (first_rescue['question_id'], 'DPO也有改善：保留反例')]
    labels = {'start': '共同起点', 'old_ce': '旧裁词 CE', 'old_dpo': '旧裁词 CE+DPO',
              'new_ce': '新整图 SFT', 'new_dpo': '新整图 SFT+DPO'}
    canvas = Image.new('RGB', (1600, 1690), '#f4f6f5'); draw = ImageDraw.Draw(canvas)
    def text(x, y, message, size=23, color='#172a3d'):
        draw.text((x, y), str(message), font=ImageFont.truetype(FONT, size), fill=color)
    def wrap(x, y, message, width, size=21, limit=3):
        line = ''; lines = []; font = ImageFont.truetype(FONT, size)
        for char in message:
            if draw.textlength(line + char, font=font) > width:
                lines.append(line); line = char
            else:
                line += char
        if line: lines.append(line)
        for i, line in enumerate(lines[:limit]): text(x, y + i * (size + 9), line, size)
    text(30, 20, '整图监督有效信号与DPO负向效果：真实保存预测', 32)
    text(30, 78, '1617题 / 1024图；事后内部分析，非三seed确认；原评分、门槛和权重未改。', 22)
    text(30, 122, f"SFT {scores['start']:.3f}% → {scores['new_ce']:.3f}%；加DPO → {scores['new_dpo']:.3f}%", 26)
    text(30, 164, f"2×2交互 {interaction['delta_pp']:+.3f}pp，探索95% [{interaction['ci95'][0]:+.3f}, {interaction['ci95'][1]:+.3f}]", 24)
    cases = []; articles = []
    for i, (qid, title) in enumerate(picked):
        q = next(row for row in aligned if row['question_id'] == qid)
        image_path = Path(manifest['images'][q['image_id']]['path'])
        if sha_file(image_path) != manifest['images'][q['image_id']]['sha256']:
            raise ValueError('Case image differs from the recorded source')
        cases.append({'question_id': qid, 'title': title, 'question': q['question'], 'target': q['target'],
                      'image_path': str(image_path), 'image_sha256': sha_file(image_path), 'models': q['models']})
        x = 25 + (i % 2) * 790; y = 225 + (i // 2) * 695
        draw.rounded_rectangle((x, y, x + 765, y + 675), radius=12, fill='white')
        text(x + 17, y + 12, f'q{qid}  {title}', 23)
        with Image.open(image_path) as raw:
            image = ImageOps.contain(raw.convert('RGB'), (725, 285))
        canvas.paste(image, (x + 20 + (725 - image.width) // 2, y + 55 + (285 - image.height) // 2))
        wrap(x + 20, y + 351, q['question'], 720, limit=2)
        wrap(x + 20, y + 419, '原参考：' + q['target'], 720, limit=2)
        for j, role in enumerate(('start', 'new_ce', 'new_dpo')):
            row = q['models'][role]
            wrap(x + 20, y + 480 + j * 53, f"{labels[role]}：{row['prediction']}  [VQA {row['vqa_score']:.1f}]", 720, size=19, limit=2)
        text(x + 20, y + 644, '原图等比展示；不是注意力、模型输入重建或新人工真值。', 16)
        image_link = html.escape(str(image_path.relative_to(ROOT.parent)))
        articles.append(f'<article><h2>q{qid} {html.escape(title)}</h2><img src="../../{image_link}"><p>{html.escape(q["question"])}</p><p>原参考：{html.escape(q["target"])}</p><pre>{html.escape(json.dumps(q["models"], ensure_ascii=False, indent=2))}</pre></article>')
    text(30, 1640, '显示缩小可能影响小字观察；不能据缩略图判坏标注。完整答案来自原预测；未进行新的模型生成。', 20)
    canvas.save(dest / '真实回答变化.png')
    write_json(dest / 'CASES.json', cases)
    table = '\n'.join(f'|{labels[k]}|{scores[k]:.6f}|{rare_scores[k]:.6f}|' for k in records)
    report = f'''# 整图回放结果与下一行动

两臂32/32更新、四项评测全部exit0；训练共64更新、评测5154新生成。主控制器23:44完成评测后因错误导入paired而汇总失败，已用独立CPU入口修复；原脚本/错误日志/权重/预测均保持，恢复没有重新训练或生成。

|模型|完整1617 VQA/%|其中罕见答案342 VQA/%（描述）|
|---|---:|---:|
{table}

新SFT对起点+.828695pp，95%[+.193496,+1.502531]；对旧CE+.723562pp，95%[+.136614,+1.345682]；对旧强DPO+.395795pp，95%[−.129728,+.926282]。这是预定次要比较的正向线索，不将其事后改为主候选。新DPO对同数据SFT−.723562pp，95%[−1.358561,−.110691]，对起点仅+.105133pp，未达主门槛。

四组同起点/同训练预算的事后交互为(newDPO−newCE)−(oldDPO−oldCE)={interaction['delta_pp']:+.6f}pp，探索95%[{interaction['ci95'][0]:+.6f},{interaction['ci95'][1]:+.6f}]。区间沿原6000图像簇bootstrap，不校正历史多重试验或包括训练种子方差，不证明唯一梯度/标注/容量原因。

SFT对起点29升13降、净+13.4分；DPO对新SFT11升22降、净−11.7分。SFT改善题中{removed}题在加入DPO后得分下降。训练纠错128的满分题数SFT37、DPO38，额外偏好目标没有表现出更好的完整任务泛化；拟合与迁移不可混淆。罕见342的点值另表，没有据此追加长尾显著提升声明。

## 已执行的下一行动

1. 独立恢复汇总，保留原失败；核对42个保护文件未变。
2. 读取全部四组预测，完成数据策略×偏好损失交互分析与四个真实正反案例；0新GPU/0更新/0生成。
3. 按冻结DECISION关闭本固定数据替换训练分支，不扫DPO权重/LR/数据混合、不追加步数/种子，不将次要SFT信号包装成达标。

下一次GPU训练需有区别于本配置的新数据或监督机制证据，再独立预注册同数据SFT/历史强控制、完整任务/长尾保持与停止门槛。现有结果更支持优先解释SFT收益及DPO交互，尚不支持为了RL关键词直接启动PPO/GRPO。当前没有启动新的训练，完整强基线/三seed/外部能力目标仍未完成。简历、科研、Agent、CI不改、不push。

[原预注册协议](../../packages/RareVLM_minimal_20261001/docs/VLM_STVQA_REPLAY_V1_PROTOCOL.md) · [四组原始变化](result_review_v1/PAIRED_CHANGES.json) · [图像学习页](result_review_v1/学习页.html)
'''
    (ROOT / 'RESULT_REVIEW.md').write_text(report)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SFT与DPO真实结果</title><style>body{max-width:1400px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}article,section{background:white;padding:20px;margin:22px 0}img{max-width:100%;max-height:550px;object-fit:contain}pre{white-space:pre-wrap}</style><h1>SFT与DPO：从原理到真实回答</h1><section><p>CE提高已知正确回答的概率；DPO提高正确与负回答相对冻结参考的序列似然差。两目标同用不能保证泛化更好。实际视觉LoRA为30个qkv、r12/alpha24、60个FP32因子1843200参数。</p><pre>L=(Σ16 CE + λΣ8 DPO)/16；λ=0或1\nDPO=−log sigmoid(0.1×[(logπchosen−logπrejected)−(logπrefchosen−logπrefrejected)])</pre><p>本轮SFT完整1617从87.198516%升至88.027211%，DPO仅87.303649%。SFT尚未显著超过历史强DPO；主候选门槛未过，关闭此配置扩训。以下是保存回答的事后教学病例，原参考不是新增人工真值，也不是注意力解释。</p><p><a href="../RESULT_REVIEW.md">全部指标、区间与下一行动</a> · <a href="../../../packages/RareVLM_minimal_20261001/tools/run_visual_preference_v1.py">冻结训练源码：train与analyze</a></p></section><img src="真实回答变化.png">' + ''.join(articles) + '</html>'
    (dest / '学习页.html').write_text(page)
    if any(sha_file(path) != digest for path, digest in protected.items()):
        raise ValueError('Saved scientific results changed')
    write_json(dest / 'COMPLETION.json', {'status': 'completed', 'protected_files_unchanged': True,
        'new_updates': 0, 'new_generations': 0, 'case_count': len(cases)})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
