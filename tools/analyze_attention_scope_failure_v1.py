"""Explain completed scope-study failure using saved predictions and training traces."""

from collections import Counter, defaultdict
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from vlm_deep.common import NORMALIZER
from vlm_human_sft.evaluate import binary_answer


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
STUDY = OUTPUTS / 'vlm_attention_scope_v1'
ROOT = STUDY / 'failure_analysis_v1'
REPORT = PACKAGE / 'docs/VLM_ATTENTION_SCOPE_V1_FAILURE_ANALYSIS.md'
PAGE = PACKAGE / 'docs/learning/QKVO失败原因_真实错题与收益分解.html'


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values)} if values else {'n': 0}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-probe', action='store_true')
    args = parser.parse_args()
    if args.refresh_probe:
        result = read(ROOT / 'ANALYSIS.json')
        result['fixed_train_probe'] = read(ROOT / 'fixed_train_probe_v1/ANALYSIS.json')
        probe_root = ROOT / 'fixed_train_probe_v1'
        probe_qa = read(probe_root / 'TRAIN_EVAL.json')['splits']['dev']
        historical = {str(q['question_id']): q for q in jsonlines(OUTPUTS / 'vlm_complementary_rank_v1/initial/train/human/predictions.jsonl')}
        generated = {role: {str(q['question_id']): q for q in jsonlines(probe_root / 'runs' / role / 'generation/predictions.jsonl')}
                     for role in ('initial', 'qv', 'qkvo')}
        differences = []
        for q in probe_qa:
            qid = str(q['question_id'])
            old, new = historical[qid], generated['initial'][qid]
            if old['prediction'] != new['prediction']:
                differences.append({'question_id': qid, 'historical_prediction': old['prediction'],
                    'fresh_prediction': new['prediction'], 'historical_score': old['primary_score'],
                    'fresh_score': new['primary_score'],
                    'old_padded_tokens': old.get('padded_input_tokens'), 'fresh_padded_tokens': new.get('padded_input_tokens'),
                    'same_unpadded_input_tokens': old['input_tokens'] == new['input_tokens'],
                    'same_question_image_hash': old['question_sha256'] == new['question_sha256'] and old['image_sha256'] == new['image_sha256']})
        extra = {'fresh_initial_vs_historical_textual_differences': differences,
                 'generation_changes_vs_fresh_initial': {}}
        for role in ('qv', 'qkvo'):
            extra['generation_changes_vs_fresh_initial'][role] = {
                group: {'corrected': sum(generated[role][str(q['question_id'])]['primary_score'] > generated['initial'][str(q['question_id'])]['primary_score'] for q in probe_qa if q['probe_group'] == group),
                        'broken': sum(generated[role][str(q['question_id'])]['primary_score'] < generated['initial'][str(q['question_id'])]['primary_score'] for q in probe_qa if q['probe_group'] == group)}
                for group in ('historically_wrong', 'correct_anchor')}
        result['fixed_train_probe_extras'] = extra
        result['cpu_decomposition_execution'] = {'model_loads': 0, 'gpu_forwards': 0, 'training_updates': 0}
        result['fixed_train_probe_execution'] = {'base_model_loads': 2, 'checkpoint_states': 3,
            'new_generation_qa_model_observations': 240, 'prompt_only_readout_qa_model_observations': 240,
            'training_updates': 0, 'confirmation_used': False}
        result.pop('model_loads', None)
        result.pop('gpu_forwards', None)
        write(probe_root / 'POST_READOUT_BREAKDOWN.json', extra)
        write(ROOT / 'ANALYSIS.json', result)
        render(result, read(ROOT / 'REVIEWED_CASES.json'))
        print(json.dumps(result['fixed_train_probe'], ensure_ascii=False, indent=2))
        return
    original = read(STUDY / 'SCREEN_ANALYSIS.json')
    manifest = read(STUDY / 'data/TRAINING_MANIFEST.json')
    human_train = {str(q['question_id']): q for q in manifest['splits']['train_all'] if q['study_source'] == 'human'}
    old_initial_path = OUTPUTS / 'vlm_complementary_rank_v1/initial/train/human/predictions.jsonl'
    old_initial = {str(q['question_id']): q for q in jsonlines(old_initial_path)}
    train_wrong = {qid: old_initial[qid]['primary_score'] == 0 for qid in human_train}
    wrong_by_label = Counter(q['target'] for qid, q in human_train.items() if train_wrong[qid])
    training_summary = {'human_unique_qa': len(human_train), 'initial_wrong_qa': sum(train_wrong.values()),
        'initial_wrong_by_label': dict(wrong_by_label),
        'derived_initial_train_accuracy_percent': 100 * (1 - sum(train_wrong.values()) / len(human_train)),
        'scope': 'old raw-FP32 initial train predictions restricted to unchanged admitted rows; new zero-output graph had 1024/1024 baseline parity; not new Final training accuracy'}

    manifests, predictions = {}, {}
    for task in ('human', 'text'):
        manifests[task] = read(STUDY / f'data/SCREEN_{task.upper()}.json')
        predictions[task] = {role: jsonlines((STUDY / 'initial' if role == 'initial' else STUDY / 'runs' / role)
                                           / task / 'predictions.jsonl') for role in ('initial', 'qv', 'qkvo')}
    gold = {task: {str(q['question_id']): q for q in m['splits']['dev']} for task, m in manifests.items()}
    mapped = {task: {role: {str(q['question_id']): q for q in values} for role, values in models.items()}
              for task, models in predictions.items()}
    transitions = {}
    for role in ('qv', 'qkvo'):
        initial, after = predictions['human']['initial'], predictions['human'][role]
        corrected = [b for a, b in zip(initial, after) if b['primary_score'] > a['primary_score']]
        broken = [b for a, b in zip(initial, after) if b['primary_score'] < a['primary_score']]
        transition_counts = Counter(binary_answer(a['prediction']) + '->' + binary_answer(b['prediction'])
            for a, b in zip(initial, after) if binary_answer(a['prediction']) != binary_answer(b['prediction']))
        transitions[role] = {'corrected': len(corrected), 'broken': len(broken), 'net': len(corrected) - len(broken),
            'corrected_by_label': dict(Counter(gold['human'][str(q['question_id'])]['target'] for q in corrected)),
            'broken_by_label': dict(Counter(gold['human'][str(q['question_id'])]['target'] for q in broken)),
            'binary_transition_counts': dict(transition_counts),
            'textual_prediction_changes': sum(a['prediction'] != b['prediction'] for a, b in zip(initial, after))}
    agreement = {}
    incremental = {}
    for task in ('human', 'text'):
        qv, qkvo = predictions[task]['qv'], predictions[task]['qkvo']
        different = []
        for a, b in zip(qv, qkvo):
            if a['prediction'] == b['prediction']:
                continue
            init = mapped[task]['initial'][str(a['question_id'])]
            different.append({'question_id': a['question_id'], 'initial_prediction': init['prediction'],
                'initial_score': init['primary_score'], 'qv_prediction': a['prediction'], 'qv_score': a['primary_score'],
                'qkvo_prediction': b['prediction'], 'qkvo_score': b['primary_score'],
                'score_delta_vs_qv': b['primary_score'] - a['primary_score']})
        agreement[task] = {'qa': len(qv), 'same_prediction': len(qv) - len(different),
                           'agreement_percent': 100 * (len(qv) - len(different)) / len(qv)}
        incremental[task] = different

    reference = read(read(PACKAGE / 'configs/vlm_attention_scope_v1.json')['baseline_config'])
    parent = read(reference['text_manifest'])
    frequency = Counter(NORMALIZER(q['target']) for q in parent['splits']['train_all'])
    text_changes = []
    for before, after in zip(predictions['text']['initial'], predictions['text']['qkvo']):
        delta = after['primary_score'] - before['primary_score']
        if abs(delta) < 1e-12:
            continue
        b, a = NORMALIZER(before['prediction']).split(), NORMALIZER(after['prediction']).split()
        shortened = bool(a) and len(a) < len(b) and any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))
        q = gold['text'][str(after['question_id'])]
        text_changes.append({'question_id': after['question_id'], 'before': before['prediction'], 'after': after['prediction'],
            'score_delta': delta, 'shorter_span_already_present_before': shortened,
            'training_target_frequency': frequency[NORMALIZER(q['target'])]})
    net_text = sum(q['score_delta'] for q in text_changes)
    shorter_net = sum(q['score_delta'] for q in text_changes if q['score_delta'] > 0 and q['shorter_span_already_present_before'])
    assert math.isclose(net_text / 512 * 100, original['text_slices']['initial']['overall']['delta_pp'], abs_tol=1e-10)

    online = {}
    for role in ('qv', 'qkvo'):
        by_group = defaultdict(list)
        last = {}
        conditional_wrong = Counter()
        source_exposure = Counter()
        with (STUDY / 'runs' / role / 'train/INPUT_TRACES.jsonl').open() as stream:
            for line in stream:
                q = json.loads(line)
                source_exposure[q['source']] += 1
                by_group[q['source']].append(q['qa_ce'])
                if q['source'] != 'human':
                    continue
                qid, label = str(q['question_id']), q['target']
                hardness = 'initial_wrong' if train_wrong[qid] else 'initial_correct'
                by_group[label].append(q['qa_ce'])
                by_group[hardness].append(q['qa_ce'])
                by_group[hardness + '_' + label].append(q['qa_ce'])
                signed = q['first_yes_minus_no_margin'] * (1 if label == 'yes' else -1)
                if signed <= 0:
                    conditional_wrong[label] += 1
                last[qid] = {'target': label, 'initial_wrong': train_wrong[qid], 'update': q['update'],
                             'signed_margin': signed, 'qa_ce': q['qa_ce']}
        online[role] = {'source_exposures': dict(source_exposure),
                       'online_qa_ce': {k: describe(v) for k, v in by_group.items()},
                       'online_nonpositive_target_margin_exposures': dict(conditional_wrong),
                       'last_exposure_initial_wrong': {label: {
                            'qa': sum(v['initial_wrong'] and v['target'] == label for v in last.values()),
                            'nonpositive_margin_qa': sum(v['initial_wrong'] and v['target'] == label and v['signed_margin'] <= 0 for v in last.values()),
                            'signed_margin': describe([v['signed_margin'] for v in last.values() if v['initial_wrong'] and v['target'] == label])}
                            for label in ('yes', 'no')},
                       'hard_ce_fraction_of_human_ce': sum(by_group['initial_wrong']) / sum(by_group['human']),
                       'warning': 'online teacher-forcing 3-token QA mean and variable last-exposure times; not fixed Final generation or first-answer NLL'}

    selected = [('human', 'vqa2_train2014:46112002', '车型与前后方位需要辨认；近端可见类似门的结构，本次不据此认定原no错标。'),
        ('human', 'vqa2_train2014:419979014', '可见一头体型较大且有明显象牙的象，未见幼象；原no有图像依据，两组却都改答yes。'),
        ('human', 'vqa2_train2014:143164000', '草地散布落叶、光影，未见明确成组条纹；原no有可见依据，两组都改答yes。'),
        ('human', 'vqa2_train2014:3286000', '蓝色杯体明显失焦，内容物或液面不能可靠分辨；保留原no，不把此例单独认定为确定视觉退化。'),
        ('human', 'vqa2_train2014:126798002', '击球左右手判断需要持棒和身体朝向关系；原yes为9/10共识，Q/V修复而候选未保住；本次不重标。'),
        ('human', 'vqa2_train2014:333461001', '多名滑雪者可见手套/连指手套，部分手被遮挡；未见明确裸手反证，不能把原yes机械判成噪声。'),
        ('text', '14476', '原图左下方按键可辨P；初始和候选答p，Q/V答a。增量收益来自保留原本正确答案。')]
    cases = []
    for task, qid, observation in selected:
        q = gold[task][qid]
        cases.append({'task': task, 'question_id': qid, 'question': q['question'], 'target': q['target'],
            'answers': q['answers'], 'image_path': manifests[task]['images'][q['image_id']]['path'],
            'predictions': {role: {key: mapped[task][role][qid][key] for key in ('prediction', 'primary_score')}
                            for role in ('initial', 'qv', 'qkvo')},
            'assistant_original_image_observation': observation,
            'scope': 'posthoc screen explanation only; no relabeling, denominator edits, or training reuse'})
    result = {'status': 'completed_saved_records_only', 'model_loads': 0, 'gpu_forwards': 0, 'training_updates': 0,
        'training_initial_difficulty': training_summary, 'visual_transitions': transitions,
        'qv_qkvo_prediction_agreement': agreement, 'qkvo_incremental_cases_vs_qv': incremental,
        'text_score_decomposition': {'net_score_points': net_text, 'shorter_existing_span_score_points': shorter_net,
            'fraction_of_net_points': shorter_net / net_text,
            'scope': 'descriptive answer-span changes, not a causal percentage or a denial of valid answer-selection improvement', 'cases': text_changes},
        'online_training': online,
        'frozen_screen_comparisons': original['comparisons'], 'frozen_text_intervals': original['text_slices'],
        'scope_limits': ['Single continuation seed42, repeatedly exposed internal screen.',
                         'No new Final training-set inference; old 36/45 residual result belongs to an earlier study.',
                         'No full-logit readout here: discrete no-to-yes flips do not prove uniform margin translation.',
                         'All cases keep original labels; uncertain screen images do not become training data.']}
    write(ROOT / 'ANALYSIS.json', result)
    write(ROOT / 'REVIEWED_CASES.json', cases)
    render(result, cases)
    print(json.dumps({'training_initial_difficulty': training_summary, 'visual_transitions': transitions,
        'agreement': agreement, 'text_score_decomposition': result['text_score_decomposition'],
        'online': online}, ensure_ascii=False, indent=2))


def render(a, cases):
    t, o = a['visual_transitions'], a['online_training']
    lines = ['# Q/K/V/O容量对照：失败原因分析', '',
        '前半分析复用已完成实验的预测、训练追踪与原图；后追加三个固定checkpoint各80题的只读生成和首token概率诊断。全程0参数更新、不使用确认结果，不改标签、阈值和历史置信区间。' if a.get('fixed_train_probe') else '本轮只读已完成实验的预测、训练追踪与原图，0新模型/GPU/训练；不改标签、阈值和历史置信区间。', '',
        '## 直接结论', '',
        '主要问题是共同CE训练中的纠错/保持取舍没有改善；新增K/O未带来额外视觉纠错。本轮lambda_rank=0，不能套用旧排序损失的失败机制。没有发现新增模块未更新或优化器空转。', '',
        *(['新增固定训练探针进一步表明：候选仍错31道历史原错，10道困难no全部未纠正，alias首答NLL约1.391→1.692；40个已会锚点保持全对。困难负例在训练题内尚未学好，不能只用泛化失败解释，也不能直接从小探针认定全量欠拟合或排除局部过拟合。', ''] if a.get('fixed_train_probe') else []),
        '| 模型 | 修复原错 | 破坏原对 | 净增 | GT=yes修复/破坏 | GT=no修复/破坏 |', '|---|---:|---:|---:|---|---|']
    for role in ('qv', 'qkvo'):
        v = t[role]
        lines.append(f'|{role}|{v["corrected"]}|{v["broken"]}|{v["net"]}|{v["corrected_by_label"].get("yes",0)}/{v["broken_by_label"].get("yes",0)}|{v["corrected_by_label"].get("no",0)}/{v["broken_by_label"].get("no",0)}|')
    lines += ['', '初始512题中488对、24错。候选纠正6/24原错，却破坏5/488原对，净增仅1题。no类的4个新错在两组完全相同，不能把这4题退化归因于新增K/O；候选相对控制只是少保住了一个对左手击球的修复。', '',
        '候选binary翻转为10次no→yes、1次yes→no；这是离散输出变化，不证明全部logit做统一平移。原本正确no被破坏，抵消了yes纠错，才是主指标与类别保护失败的直接原因。', '',
        '## 新增容量实际产生了什么', '',
        f"两组Human预测{a['qv_qkvo_prediction_agreement']['human']['same_prediction']}/512一致，Text{a['qv_qkvo_prediction_agreement']['text']['same_prediction']}/512一致。K/O已实质更新，但多出参数没有在这组筛选题上产生额外视觉正确答案。不能据此否定所有更宽LoRA，只能说本数据与200步预算下没有证据支持它。", '',
        '## 训练信号：不是只看平均loss', '',
        f"修复池1186QA的初始原错为{a['training_initial_difficulty']['initial_wrong_qa']}题，按GT分布{a['training_initial_difficulty']['initial_wrong_by_label']}，初始已会约{a['training_initial_difficulty']['derived_initial_train_accuracy_percent']:.2f}%。此数由原始初始训练预测限制到同一准入子集，不是新Final训练准确率。", '',
        '|角色|在线human三token CE均值|在线Text CE均值|原错占human CE值|', '|---|---:|---:|---:|']
    for role in ('qv', 'qkvo'):
        v = o[role]
        lines.append(f'|{role}|{v["online_qa_ce"]["human"]["mean"]:.6f}|{v["online_qa_ce"]["text"]["mean"]:.6f}|{v["hard_ce_fraction_of_human_ce"]*100:.2f}%|')
    lines += ['', '以上是同一排期的在线路径均值，不是固定cohort收敛曲线；CE包含首答案、结束和换行。不能用小均值声称训练原错已经拟合，也不能把曝光少或loss份额当参数梯度份额。所有人工曝光800yes/800no，但原错难例30yes/10no，数量均衡不等于困难监督或梯度均衡。当前证据提示困难纠错信号与目标分布值得优先研究，尚未隔离为唯一因果。', '',
        '## Text收益的分解', '',
        '候选相对初始Text增加3.0个VQA得分点，即+.585938pp，原探索性图像簇95%CI[+.058703,+1.294831]。其中2.1分来自原答案已经包含正确词串、输出改为较短span（yale、macintosh、spring）；这是有效答案选择改善，但不能直接包装为新OCR识字能力。2.1/3.0=70%是描述性分解，不是因果贡献率。', '',
        '相对Q/V多出的1.0分全部来自键盘p：初始p正确→Q/V改错a→候选保留p。其余3条Text字符串差异均仍得0分。额外收益体现避免一处遗忘；候选对Q/V总体+.195313pp的区间下界为0，未确认额外OCR提升。', '',
        '## 原图与数据质量边界', '',
        '已实际查看7原图。幼象、草地条纹等有原GT的可见依据；蓝色杯体明显失焦，巴士前后方位/车型语义需要辨认。评测存在可判性差异，但没有证据把全部失败归为错标。本次不后验删题、改标签或将screen错题回填训练。', '',
        '## 后续应做与不应做', '',
        '- 停止本轮容量分支，不继续扫rank、解冻模块或延长epoch来碰同一screen。修复执行链不等于已找到提分配方。',
        '- 下一立项优先面向可靠训练难例的实体＋属性/状态/关系辨别及已会能力保持，先以实际Adam位移/匹配CE检查作用方向，再用独立生成验收；不得仅因yes/no数量均衡就认为监督充分。',
        '- 已补固定训练错题的两Final首答案NLL与生成，见下表；不把该80题富集诊断当总体训练准确率，也不搬用旧实验36/45残错。' if a.get('fixed_train_probe') else '- 若需判断当前是否训练欠拟合，先在固定训练错题上测两Final的真实首答案NLL与生成。现有在线最后曝光记录不能替代它；不得搬用旧实验36/45残错当本轮结果。',
        '- 尚未证明数据噪声、训练长度、视觉encoder冻结或LoRA容量中的任何一项是唯一主因。当前最强证据是共同CE后的纠错/保持取舍，而不是框架故障。', '',
        '[完整分解JSON](/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_attention_scope_v1/failure_analysis_v1/ANALYSIS.json) · [真实原图与收益图解](learning/QKVO失败原因_真实错题与收益分解.html)']
    probe_html = ''
    if a.get('fixed_train_probe'):
        probe = a['fixed_train_probe']['results']
        lines += ['', '## 新增零更新固定训练探针', '',
            '看Final输出之前固定40历史原错＋40hash保持锚点，共80题、40yes/40no。三模型使用完全相同batch4/1024/greedy32，另做target-free dense首token读出。初始重新生成，避免拿不同批次的旧结果直接拼增益。不是总体训练准确率或独立能力收益。', '',
            '|模型|历史原错组答对/40|其中yes答对/30|no答对/10|保持锚点答对/40|原错canonical NLL|原错alias NLL|',
            '|---|---:|---:|---:|---:|---:|---:|']
        html_rows = ''
        for role in ('initial', 'qv', 'qkvo'):
            v = probe[role]
            line = f'|{role}|{v["historically_wrong_all"]["generation_correct"]:.0f}|{v["historically_wrong_yes"]["generation_correct"]:.0f}|{v["historically_wrong_no"]["generation_correct"]:.0f}|{v["correct_anchor_all"]["generation_correct"]:.0f}|{v["historically_wrong_all"]["mean_canonical_first_nll"]:.4f}|{v["historically_wrong_all"]["mean_alias_first_nll"]:.4f}|'
            lines.append(line)
            html_rows += f'<tr><td>{role}</td><td>{v["historically_wrong_yes"]["generation_correct"]:.0f}/30</td><td>{v["historically_wrong_no"]["generation_correct"]:.0f}/10</td><td>{v["correct_anchor_all"]["generation_correct"]:.0f}/40</td><td>{v["historically_wrong_all"]["mean_canonical_first_nll"]:.4f}</td><td>{v["historically_wrong_all"]["mean_alias_first_nll"]:.4f}</td></tr>'
        lines += ['', 'Canonical NLL与词形集合alias NLL分开记录，避免将Yes→yes的格式适配全解释为新知识。实际答对数仍以greedy生成评分为主，读出条件概率不替代生成。该探针只解释训练错题学习状态，不能推导增加epoch必然泛化。',
            '新初始在40历史原错中有1题答对，与旧缓存不同，因此使用本次同批次初始作对照；Q/V实际纠正7题，Q/K/V/O纠正8题，40锚点均保持。重新组批可能影响BF16临界数值，具体机制未进一步隔离；不把该初始差异算作训练收益。',
            '候选仍错31道历史原错，10道no题全部未纠正。no的alias首答NLL约1.391→1.675/1.692，词形聚合后结论仍相同。这不能只解释为训练已充分拟合而泛化失败，也不能据此直接认定所有数据错标、所有SFT无效或完全排除局部过拟合。']
        probe_html = f'<section><h2>补齐Final训练错题核对</h2><p>固定40历史原错和40保持锚点，三模型同批次重新生成，0训练更新；不是总体训练准确率。</p><table><thead><tr><th>模型</th><th>原错yes</th><th>原错no</th><th>锚点答对</th><th>原错首答NLL</th><th>alias NLL</th></tr></thead><tbody>{html_rows}</tbody></table><p>新初始重新组批后答对1个历史原错，因此实际纠正是Q/V 7题、Q/K/V/O 8题；不计作9题新增训练纠正。候选仍错31题，10个no全未纠正，不能只归因于泛化。以实际生成判断纠错，NLL用于说明概率变化，canonical/alias分开避免词形混淆。</p></section>'
    REPORT.write_text('\n'.join(lines) + '\n')
    cards = ''
    for q in cases:
        pred = ' / '.join(f'{role}: {v["prediction"]} ({v["primary_score"]:.1f})' for role, v in q['predictions'].items())
        cards += (f'<article><img src="{html.escape(q["image_path"],quote=True)}" alt="原始评测图"><h3>{html.escape(q["question"])}</h3>'
                  f'<p>原GT={html.escape(q["target"])}；{html.escape(pred)}</p>'
                  f'<p>{html.escape(q["assistant_original_image_observation"])}</p></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QKVO失败原因与真实案例</title>
<style>body{{font:16px/1.75 system-ui;background:#f2f5f7;color:#183141;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}section,article{{background:white;padding:20px;border-radius:12px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:300px;object-fit:contain;background:#eef2f5}}pre{{padding:14px;background:#eaf0f4;overflow:auto}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:9px;border-bottom:1px solid #dde3e7;white-space:nowrap}}a{{color:#07609b}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style><main>
<h1>参数加倍，为什么没有超过强CE控制？</h1><p>本页渲染只读已保存结果；另已完成固定训练题的零更新推理诊断。不改原GT或分母。筛选Q/V纠正7题、破坏5题；Q/K/V/O纠正6题、破坏5题。共同新增错误抵消了大部分纠正，额外容量没有补上这一缺口。</p>
<section><h2>把数量代入验收问题</h2><pre>初始：488题正确，24题错误
候选：修复6个原错 - 破坏5个原对 = 净增1题
QA增长 = 1/512 × 100 = 0.1953125个百分点
候选no类：0修复 - 4新增错 = -1.5625个百分点
相对Q/V：0额外纠正 - 1个未保住的修复 = -1题</pre><p>所有初始正确题中只损伤约1%，就抵消了5/6的新纠正。因此平均loss更低不保证决策取舍更好。两组no错误完全相同，不能把它们单独算作K/O造成。</p></section>
<section><h2>Text增长也要拆开解释</h2><pre>相对初始净+3.0个得分点：
  yale law library → yale                 +1.0
  team hambrick → a's                     +1.0
  macintosh classic ii → macintosh        +0.7
  spring water → spring                   +0.4
  brachs → brach's                        -0.1
相对Q/V净+1.0分：初始p → 控制a → 候选p</pre><p>缩短为正确span是有效答案选择改善，但不证明视觉识字本身增强。相对控制的额外1分来自保住原本正确的P，不是额外学会一个初始不认识的字符。</p></section>
{probe_html}<section><h2>原始图像与三模型预测</h2><p>这是事后解释screen案例，不是新专家标注或训练材料；网页缩放不是模型输入重建。</p><div class="grid">{cards}</div></section>
<section><h2>证据能支持什么</h2><p>新增K/O已更新，两组Human输出511/512一致。当前数据与预算下没有额外视觉纠错证据；执行链正常。数据可见性、难例监督与已会能力保持需要进一步隔离，不能宣布全部失败都是数据噪声或容量问题。</p><p><a href="../VLM_ATTENTION_SCOPE_V1_FAILURE_ANALYSIS.md">完整分析与限制</a> · <a href="../../tools/analyze_attention_scope_failure_v1.py">逐题分解代码</a></p></section></main></html>'''
    if any(not Path(q['image_path']).is_file() for q in cases):
        raise ValueError('Missing original case image')
    PAGE.write_text(page)


if __name__ == '__main__':
    main()
