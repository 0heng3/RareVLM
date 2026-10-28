"""Analyze fixed Final200 checkpoints without changing training or old scores.

The selected training cohort is enriched for initial errors; its accuracy is
not an estimate of accuracy on all training data. Conditional prefix readouts
describe mechanisms and never replace the saved actual-generation metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from rarevlm_paths import PROJECT_ROOT
from vlm_human_sft.evaluate import binary_answer
from vlm_fixed_error_readout.readout import classify_pair


ROLES = ("initial", "ce", "ce_rank")
TASKS = ("train_selected", "screen")
KINDS = ("canonical", "alias")
NEAR_BOUNDARY = 0.125
NUMERIC_FIELDS = (
    "canonical_margin", "alias_margin", "conditional_yes_probability",
    "canonical_conditional_yes_probability", "canonical_yes_log_probability",
    "canonical_no_log_probability", "yes_alias_log_probability", "no_alias_log_probability",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha_file(path)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_rows(path, rows, *, expected_sha=None, prefix=False):
    path = Path(path)
    if expected_sha is not None and sha_file(path) != expected_sha:
        raise ValueError(f"Frozen source predictions changed: {path}")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if len(values) != len(rows):
        raise ValueError(f"Incomplete fixed cohort: {path}: {len(values)} != {len(rows)}")
    if len({row["question_id"] for row in values}) != len(rows):
        raise ValueError(f"Duplicate question IDs: {path}")
    for index, (gold, value) in enumerate(zip(rows, values)):
        if (value["question_id"] != gold["question_id"]
                or value["image_id"] != gold["image_id"]):
            raise ValueError(f"Question/image order differs from fixed manifest: {path}:{index}")
        if prefix:
            if any(not math.isfinite(float(value[field])) for field in NUMERIC_FIELDS):
                raise ValueError(f"Nonfinite fixed-checkpoint readout: {path}:{index}")
            if (value["top1_binary"] not in (None, "yes", "no")
                    or not isinstance(value["top1_token_id"], int)
                    or value["top1_token_id"] < 0 or not isinstance(value["top1_decoded"], str)):
                raise ValueError(f"Malformed full-vocabulary top-one readout: {path}:{index}")
            if any(not 0 <= float(value[field]) <= 1 for field in
                   ("conditional_yes_probability", "canonical_conditional_yes_probability")):
                raise ValueError(f"Conditional probability outside [0,1]: {path}:{index}")
        elif value["primary_score"] not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Expected unchanged binary primary_score: {path}:{index}")
    return values


def decision(margin):
    return "yes" if margin > 0 else "no" if margin < 0 else None


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def describe(values):
    values = list(values)
    return {"n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def pearson(left, right):
    if len(left) != len(right) or not left:
        return None
    left_mean, right_mean = mean(left), mean(right)
    a = [value - left_mean for value in left]
    b = [value - right_mean for value in right]
    denominator = math.sqrt(math.fsum(value * value for value in a)
                            * math.fsum(value * value for value in b))
    return math.fsum(x * y for x, y in zip(a, b)) / denominator if denominator else None


def agreement(left, right):
    comparable = [(a, b) for a, b in zip(left, right) if a in ("yes", "no") and b in ("yes", "no")]
    return {"qa": len(left), "both_binary_valid": len(comparable),
            "agree": sum(a == b for a, b in comparable),
            "disagree": sum(a != b for a, b in comparable),
            "not_comparable": len(left) - len(comparable),
            "readout_unresolved": sum(a is None for a in left),
            "actual_generation_unresolved": sum(b is None for b in right),
            "definition": "agree/disagree only among rows where both sides give yes/no; ties and nonbinary full-vocabulary tokens are not silently counted as agreement"}


def pair_diagnosis(pair, prefix, generated):
    positive, negative = pair["positive_question_id"], pair["negative_question_id"]
    generation = {"positive": generated[positive], "negative": generated[negative]}
    both_correct = all(value["primary_score"] == 1 for value in generation.values())
    result = {"pair_id": pair["pair_id"], "component_id": pair["component_id"],
              "question": pair["question"], "positive_question_id": positive,
              "negative_question_id": negative,
              "anchor_group": pair.get("anchor_group"),
              "generation": {"both_correct": both_correct,
                             "correct_qa": sum(value["primary_score"] for value in generation.values()),
                             "positive_prediction": generation["positive"]["prediction"],
                             "negative_prediction": generation["negative"]["prediction"],
                             "positive_primary_score": generation["positive"]["primary_score"],
                             "negative_primary_score": generation["negative"]["primary_score"]},
              "readouts": {}}
    for kind in KINDS:
        m_pos = float(prefix[positive][kind + "_margin"])
        m_neg = float(prefix[negative][kind + "_margin"])
        classification = classify_pair(m_pos, m_neg)
        if classification["both_binary_correct"]:
            label = "readout_both_correct"
        elif classification["ordered"]:
            label = "ordered_but_threshold_wrong"
        else:
            label = "order_reversed_or_tied"
        result["readouts"][kind] = {
            **classification, "positive_margin": m_pos, "negative_margin": m_neg,
            "classification": label,
            "near_boundary": abs(m_pos) <= NEAR_BOUNDARY or abs(m_neg) <= NEAR_BOUNDARY,
            "positive_near_boundary": abs(m_pos) <= NEAR_BOUNDARY,
            "negative_near_boundary": abs(m_neg) <= NEAR_BOUNDARY,
            "readout_both_correct_but_generation_failed": classification["both_binary_correct"] and not both_correct,
            "generation_correct_but_readout_failed": both_correct and not classification["both_binary_correct"],
        }
    return result


def summarize(pairs, pair_values, prefix, generated):
    values = [pair_values[pair["pair_id"]] for pair in pairs]
    qids = [pair[key] for pair in pairs for key in ("positive_question_id", "negative_question_id")]
    if len(qids) != len(set(qids)):
        raise ValueError("Diagnostic groups require unique QA IDs")
    failed = [value for value in values if not value["generation"]["both_correct"]]
    both_correct = len(values) - len(failed)
    correct_qa = sum(generated[qid]["primary_score"] for qid in qids)
    result = {"pairs": len(pairs), "qa": len(qids),
              "images": len({generated[qid]["image_id"] for qid in qids}),
              "components": len({pair["component_id"] for pair in pairs}),
              "generation": {"correct_qa": correct_qa, "qa_accuracy_percent": percentage(correct_qa, len(qids)),
                             "both_correct_pairs": both_correct, "failed_pairs": len(failed),
                             "pair_accuracy_percent": percentage(both_correct, len(pairs)),
                             "metric": "unchanged saved primary_score, not conditional readout"},
              "readouts": {}}
    actual_binary = [binary_answer(generated[qid]["prediction"]) for qid in qids]
    for kind in KINDS:
        classes = Counter(value["readouts"][kind]["classification"] for value in values)
        fail_classes = Counter(value["readouts"][kind]["classification"] for value in failed)
        margins = [float(prefix[qid][kind + "_margin"]) for qid in qids]
        result["readouts"][kind] = {
            "classification_counts_all_pairs": {label: classes[label] for label in
                ("readout_both_correct", "ordered_but_threshold_wrong", "order_reversed_or_tied")},
            "classification_counts_actual_generation_failed_pairs": {label: fail_classes[label] for label in
                ("readout_both_correct", "ordered_but_threshold_wrong", "order_reversed_or_tied")},
            "strictly_ordered_pairs": sum(value["readouts"][kind]["ordered"] for value in values),
            "order_reversed_or_tied_pairs": sum(not value["readouts"][kind]["ordered"] for value in values),
            "ordering_ties": sum(value["readouts"][kind]["tie"] for value in values),
            "boundary_tie_pairs": sum(value["readouts"][kind]["boundary_tie"] for value in values),
            "boundary_tie_qa": sum(margin == 0 for margin in margins),
            "near_boundary_pairs": sum(value["readouts"][kind]["near_boundary"] for value in values),
            "near_boundary_qa": sum(abs(margin) <= NEAR_BOUNDARY for margin in margins),
            "readout_both_correct_but_generation_failed_pairs": sum(
                value["readouts"][kind]["readout_both_correct_but_generation_failed"] for value in values),
            "generation_correct_but_readout_failed_pairs": sum(
                value["readouts"][kind]["generation_correct_but_readout_failed"] for value in values),
            "decision_vs_actual_generation": agreement([decision(margin) for margin in margins], actual_binary),
            "margin": describe(margins),
            "positive_margin": describe(prefix[pair["positive_question_id"]][kind + "_margin"] for pair in pairs),
            "negative_margin": describe(prefix[pair["negative_question_id"]][kind + "_margin"] for pair in pairs),
            "gap": describe(value["readouts"][kind]["gap"] for value in values),
            "center": describe(value["readouts"][kind]["center"] for value in values),
        }
    result["full_vocabulary_top1_vs_actual_generation"] = agreement(
        [prefix[qid]["top1_binary"] for qid in qids], actual_binary)
    result["canonical_vs_alias"] = agreement(
        [decision(prefix[qid]["canonical_margin"]) for qid in qids],
        [decision(prefix[qid]["alias_margin"]) for qid in qids])
    return result


def comparison(pairs, before, after, before_prefix, after_prefix, before_generation, after_generation):
    qids = [pair[key] for pair in pairs for key in ("positive_question_id", "negative_question_id")]
    corrected = sum(not before[pair["pair_id"]]["generation"]["both_correct"]
                    and after[pair["pair_id"]]["generation"]["both_correct"] for pair in pairs)
    broken = sum(before[pair["pair_id"]]["generation"]["both_correct"]
                 and not after[pair["pair_id"]]["generation"]["both_correct"] for pair in pairs)
    qa_corrected = sum(before_generation[qid]["primary_score"] == 0
                       and after_generation[qid]["primary_score"] == 1 for qid in qids)
    qa_broken = sum(before_generation[qid]["primary_score"] == 1
                    and after_generation[qid]["primary_score"] == 0 for qid in qids)
    result = {"pairs": len(pairs), "qa": len(qids),
              "generation": {"corrected_pairs": corrected, "broken_pairs": broken,
                             "net_corrected_pairs": corrected - broken,
                             "corrected_qa": qa_corrected, "broken_qa": qa_broken,
                             "net_corrected_qa": qa_corrected - qa_broken},
              "readouts": {}}
    for kind in KINDS:
        result["readouts"][kind] = {}
        for field in ("gap", "center"):
            left = [before[pair["pair_id"]]["readouts"][kind][field] for pair in pairs]
            right = [after[pair["pair_id"]]["readouts"][kind][field] for pair in pairs]
            result["readouts"][kind][field + "_delta"] = describe(b - a for a, b in zip(left, right))
            result["readouts"][kind][field + "_pearson_initial_final"] = pearson(left, right)
        left = [float(before_prefix[qid][kind + "_margin"]) for qid in qids]
        right = [float(after_prefix[qid][kind + "_margin"]) for qid in qids]
        result["readouts"][kind]["margin_delta"] = describe(b - a for a, b in zip(left, right))
        result["readouts"][kind]["margin_pearson_initial_final"] = pearson(left, right)
    return result


def changed_cases(task, role, rows, manifest, before_prefix, after_prefix, before_generation,
                  after_generation, before_pairs, after_pairs):
    actual_changes, conditional_changes = [], []
    for row in rows:
        qid = row["question_id"]
        initial, final = before_generation[qid], after_generation[qid]
        actual_before, actual_after = binary_answer(initial["prediction"]), binary_answer(final["prediction"])
        changed_kinds = [kind for kind in KINDS if decision(before_prefix[qid][kind + "_margin"])
                         != decision(after_prefix[qid][kind + "_margin"])]
        item = {"task": task, "role": role, "question_id": qid, "image_id": row["image_id"],
                "image_path": str((Path(manifest["path"]).parent / manifest["images"][row["image_id"]]["path"]).resolve()),
                "question": row["question"], "target": row["target"],
                "pair_id": row["complementary_pair_id"],
                "initial_prediction": initial["prediction"], "final_prediction": final["prediction"],
                "initial_binary": actual_before, "final_binary": actual_after,
                "initial_primary_score": initial["primary_score"], "final_primary_score": final["primary_score"],
                "score_delta": final["primary_score"] - initial["primary_score"],
                "canonical_margin": {"initial": before_prefix[qid]["canonical_margin"],
                                     "final": after_prefix[qid]["canonical_margin"]},
                "alias_margin": {"initial": before_prefix[qid]["alias_margin"],
                                 "final": after_prefix[qid]["alias_margin"]},
                "initial_top1_decoded": before_prefix[qid]["top1_decoded"],
                "final_top1_decoded": after_prefix[qid]["top1_decoded"],
                "conditional_decision_changed": changed_kinds}
        if (initial["prediction"] != final["prediction"] or actual_before != actual_after
                or initial["primary_score"] != final["primary_score"]):
            actual_changes.append(item)
        if changed_kinds:
            conditional_changes.append(item)
    pair_changes = [{"pair_id": pair_id, "initial": before_pairs[pair_id], "final": value}
                    for pair_id, value in after_pairs.items()
                    if value["generation"]["both_correct"] != before_pairs[pair_id]["generation"]["both_correct"]]
    return {"actual_generation_changes": actual_changes,
            "conditional_decision_changes": conditional_changes,
            "actual_pair_correctness_changes": pair_changes}


def findings(results, comparisons):
    lines = []
    for role, label in (("ce", "CE控制"), ("ce_rank", "CE＋排序")):
        train = comparisons["train_selected"][role]
        screen = comparisons["screen"][role]
        fixed_train = train["initial_wrong"]["generation"]["corrected_pairs"]
        broken_train = train["initial_correct"]["generation"]["broken_pairs"]
        fixed_screen = screen["initial_wrong"]["generation"]["corrected_pairs"]
        broken_screen = screen["initial_correct"]["generation"]["broken_pairs"]
        lines.append(f"{label}：45个原错训练对修复{fixed_train}个、45个已会训练对新增错{broken_train}个；筛选24个原错对修复{fixed_screen}个、232个已会对新增错{broken_screen}个。")
        if fixed_train < 45:
            lines.append(f"{label}的固定Final200仍未修复{45-fixed_train}/45个原错训练对，不能将全部失败只归因为泛化；这是特定训练错例的未修复观察，不是总体训练集准确率。")
        else:
            lines.append(f"{label}已修复本次选中的全部原错训练对；其筛选迁移仍须由独立筛选修复与新增错误判断，不能由训练拟合替代。")
    rank = results["screen"]["ce_rank"]["all"]
    failed = rank["generation"]["failed_pairs"]
    for kind in KINDS:
        classes = rank["readouts"][kind]["classification_counts_actual_generation_failed_pairs"]
        lines.append(f"排序候选筛选的{failed}个真实生成失败对，在{kind}读出下：{classes['ordered_but_threshold_wrong']}个已排对但阈值未双正确，{classes['order_reversed_or_tied']}个逆序/并列，{classes['readout_both_correct']}个读出双正确而生成仍失败。")
    if any(rank["readouts"][kind]["readout_both_correct_but_generation_failed_pairs"] for kind in KINDS):
        lines.append("存在条件读出与实际生成不一致，必须保留读出路径、近边界数值和词形因素的解释边界；不能把全部生成错误强行归入阈值偏移。")
    lines.append("gap、center和margin相关系数均为固定checkpoint的配对描述，不隔离数据、损失、学习率或架构因果；本轮没有验证校准、新损失或新训练一定有效。")
    return lines


def render_report(root, analysis):
    lines = ["# 固定Final200错误分型：零更新诊断", "",
             "本轮仅读取初始模型与原CE/CE＋排序Final200，新增首token前向与必要训练题生成；0训练更新、不进入确认集。条件二元读出不替代原生成指标，不回写历史失败结论或置信区间。", "",
             "## 真实生成与原错修复", "",
             "| 题组 | 模型 | QA正确/总数 | 双全对/总对 | 原错修复 | 原全对新增错 |",
             "|---|---|---:|---:|---:|---:|"]
    for task, task_label in (("train_selected", "训练锚点＋批次补齐"), ("screen", "完整筛选")):
        for role in ROLES:
            value = analysis["results"][task][role]["all"]
            gen = value["generation"]
            changes = analysis["comparisons"][task].get(role)
            corrected = changes["initial_wrong"]["generation"]["corrected_pairs"] if changes else "—"
            broken = changes["initial_correct"]["generation"]["broken_pairs"] if changes else "—"
            lines.append(f"| {task_label} | {role} | {gen['correct_qa']:g}/{value['qa']} | {gen['both_correct_pairs']}/{value['pairs']} | {corrected} | {broken} |")
    lines += ["", "训练主诊断分母是45个初始错对和45个初始全对保持锚点；另80对仅为复用原四题批次而补齐。上表训练整组为富集样本，不能代表全部600个训练对。筛选原错/原全对分母分别为24/232。", "",
              "## 排序、阈值与生成不一致", "",
              "定义m=logit(yes)−logit(no)，gap=m正图−m负图，center=(m正图+m负图)/2。严格排序需gap>0；两图条件判定均正确需m正图>0且m负图<0。两者不是同一条件。alias读出用预先固定的yes/no词形集合概率质量，单独统计。", "",
              "| 题组/模型 | 读出 | 真实生成失败对 | 已排对但阈值错 | 逆序/并列 | 读出双对但生成失败 | 边界tie对 | 近边界对 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for role in ROLES:
            value = analysis["results"][task][role]["all"]
            for kind in KINDS:
                diagnostic = value["readouts"][kind]
                classes = diagnostic["classification_counts_actual_generation_failed_pairs"]
                lines.append(f"| {task}/{role} | {kind} | {value['generation']['failed_pairs']} | {classes['ordered_but_threshold_wrong']} | {classes['order_reversed_or_tied']} | {classes['readout_both_correct']} | {diagnostic['boundary_tie_pairs']} | {diagnostic['near_boundary_pairs']} |")
    lines += ["", "近边界固定为任一题|margin|≤0.125，仅用于描述；不据此调决策阈值。边界tie、排序tie独立记录，条件决策不以tie作正确。", "",
              "## 初始到Final的描述变化", "",
              "| 题组/模型 | 读出 | gap Δ均值 | gap Δ中位数 | center Δ均值 | center Δ中位数 | margin Pearson |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for role in ("ce", "ce_rank"):
            for kind in KINDS:
                value = analysis["comparisons"][task][role]["all"]["readouts"][kind]
                corr = value["margin_pearson_initial_final"]
                correlation = f"{corr:.6f}" if corr is not None else "未定义"
                lines.append(f"| {task}/{role} | {kind} | {value['gap_delta']['mean']:.6f} | {value['gap_delta']['median']:.6f} | {value['center_delta']['mean']:.6f} | {value['center_delta']['median']:.6f} | {correlation} |")
    lines += ["", "## 证据支持的结论", ""] + [f"- {line}" for line in analysis["findings"]]
    lines += ["", "## 复现与边界", "",
              "- 初始训练预测按原batch4完整派生，原batch_start和耗时保留为来源记录；两个Final仅补生成固定340题。三模型筛选生成均复用原512题结果。",
              "- 首token读出为prompt-only、BF16底座/FP32 LoRA、无全局autocast；canonical/alias/全词表top1与真实生成的一致性均保存于ANALYSIS.json，非二元top1和tie不暗算为一致。",
              "- 本次是单初始化、固定checkpoint、事后探索性诊断，不是新训练增益、总体训练准确率、三种子稳定性或校准有效性结论；未读取确认集结果，未重算旧置信区间。",
              f"- 完整数据：[ANALYSIS.json]({root / 'ANALYSIS.json'})；逐对分型：[PAIR_DIAGNOSES.jsonl]({root / 'PAIR_DIAGNOSES.jsonl'})；升降病例：[CHANGED_CASES.json]({root / 'CHANGED_CASES.json'})。", ""]
    report = PROJECT_ROOT / "docs/VLM_FIXED_ERROR_READOUT_V1_RESULTS.md"
    report.write_text("\n".join(lines))
    return str(report)


def analyze(root) -> dict:
    root = Path(root).resolve(strict=True)
    cohort_path = root / "data/COHORT.json"
    cohort = read(cohort_path)
    source_references = {"cohort": file_reference(cohort_path), "readouts": {}, "generation": {}}
    manifests, gold, prefix, generated, pair_results, grouped_pairs = {}, {}, {}, {}, {}, {}
    for task in TASKS:
        ref = cohort["manifests"][task]
        if sha_file(ref["path"]) != ref["sha256"]:
            raise ValueError("Fixed diagnostic manifest changed")
        manifest = read(ref["path"])
        manifests[task] = {**manifest, "path": ref["path"]}
        gold[task] = manifest["splits"]["dev"]
        if len(gold[task]) != ref["qa"]:
            raise ValueError("Fixed manifest population differs")
        gold_map = {row["question_id"]: row for row in gold[task]}
        pairs = cohort["pairs"][task]
        pair_qids = [pair[key] for pair in pairs for key in ("positive_question_id", "negative_question_id")]
        if len(pair_qids) != len(set(pair_qids)) or set(pair_qids) != set(gold_map):
            raise ValueError("Pair definitions do not cover the fixed QA population exactly")
        for pair in pairs:
            if (gold_map[pair["positive_question_id"]]["target"] != "yes"
                    or gold_map[pair["negative_question_id"]]["target"] != "no"):
                raise ValueError("Positive/negative pair definitions disagree with original labels")
        prefix[task], generated[task], pair_results[task] = {}, {}, {}
        for role in ROLES:
            path = root / f"runs/{role}/{task}/readout.jsonl"
            values = load_rows(path, gold[task], prefix=True)
            prefix[task][role] = {row["question_id"]: row for row in values}
            source_references["readouts"].setdefault(role, {})[task] = file_reference(path)
            source = cohort["generation_sources"][role][task]
            gen_path = Path(source["path"]) if source["reused"] else root / f"runs/{role}/train_generation/predictions.jsonl"
            gen_values = load_rows(gen_path, gold[task], expected_sha=source.get("sha256"))
            generated[task][role] = {row["question_id"]: row for row in gen_values}
            source_references["generation"].setdefault(role, {})[task] = {
                **file_reference(gen_path), "reused": source["reused"]}
            pair_results[task][role] = {pair["pair_id"]: pair_diagnosis(
                pair, prefix[task][role], generated[task][role]) for pair in pairs}
        if task == "train_selected":
            grouped_pairs[task] = {"all": pairs, **{label: [pair for pair in pairs if pair["anchor_group"] == label]
                for label in ("initial_wrong", "initial_correct", "batch_neighbor")}}
        else:
            grouped_pairs[task] = {"all": pairs, **{label: [pair for pair in pairs if
                pair_results[task]["initial"][pair["pair_id"]]["generation"]["both_correct"] == correct]
                for label, correct in (("initial_wrong", False), ("initial_correct", True))}}
        expected = {"initial_wrong": 45, "initial_correct": 45, "batch_neighbor": 80} if task == "train_selected" else {"initial_wrong": 24, "initial_correct": 232}
        if any(len(grouped_pairs[task][key]) != count for key, count in expected.items()):
            raise ValueError(f"Fixed anchor counts differ: {task}")
        for label, correct in (("initial_wrong", False), ("initial_correct", True)):
            if any(pair_results[task]["initial"][pair["pair_id"]]["generation"]["both_correct"] != correct
                   for pair in grouped_pairs[task][label]):
                raise ValueError("Initial-generation anchor identity changed")

    results = {task: {role: {group: summarize(pairs, pair_results[task][role], prefix[task][role], generated[task][role])
                            for group, pairs in grouped_pairs[task].items()} for role in ROLES} for task in TASKS}
    comparisons = {task: {role: {group: comparison(pairs, pair_results[task]["initial"], pair_results[task][role],
                                    prefix[task]["initial"], prefix[task][role], generated[task]["initial"], generated[task][role])
                                for group, pairs in grouped_pairs[task].items()} for role in ("ce", "ce_rank")} for task in TASKS}
    changes = {"scope": "Full changed predictions, scores and conditional decisions relative to the same initial checkpoint; not a handpicked success subset",
               "cases": {task: {role: changed_cases(task, role, gold[task], manifests[task],
                    prefix[task]["initial"], prefix[task][role], generated[task]["initial"], generated[task][role],
                    pair_results[task]["initial"], pair_results[task][role]) for role in ("ce", "ce_rank")} for task in TASKS}}
    pair_path = root / "PAIR_DIAGNOSES.jsonl"
    with pair_path.with_suffix(".jsonl.tmp").open("w") as stream:
        for task in TASKS:
            for role in ROLES:
                for value in pair_results[task][role].values():
                    stream.write(json.dumps({"task": task, "role": role, **value}, ensure_ascii=False, allow_nan=False) + "\n")
    pair_path.with_suffix(".jsonl.tmp").replace(pair_path)
    write_json(root / "CHANGED_CASES.json", changes)
    analysis = {"status": "completed", "study": "vlm_fixed_error_readout_v1",
                "training_updates": 0, "confirm_evaluated": False, "capability_gain_claim": False,
                "scope": "Exploratory single-initialization fixed-checkpoint diagnostic; selected train is not overall training accuracy; actual generation scores remain unchanged; no new confidence interval or threshold optimization",
                "definitions": {"margin": "yes-minus-no canonical logit or alias-set log mass",
                                "strict_order": "m_positive > m_negative",
                                "both_conditional_correct": "m_positive > 0 and m_negative < 0",
                                "near_boundary_absolute_margin": NEAR_BOUNDARY,
                                "gap": "m_positive - m_negative", "center": "(m_positive + m_negative)/2",
                                "readout_both_correct_but_generation_failed": "Separate inconsistency; not assigned to threshold failure",
                                "score": "unchanged saved actual-generation primary_score"},
                "results": results, "comparisons": comparisons,
                "findings": findings(results, comparisons), "source_references": source_references,
                "artifacts": {"pair_diagnoses": str(pair_path), "changed_cases": str(root / "CHANGED_CASES.json")}}
    analysis["report"] = render_report(root, analysis)
    write_json(root / "ANALYSIS.json", analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps({"status": result["status"], "findings": result["findings"],
                      "report": result["report"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
