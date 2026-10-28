#!/usr/bin/env python3
"""Export a completed official V2 pilot, without rerunning science or generation.

Missing official screen/COMPLETION => JSON pending, no report/metrics overwrite.
Optional bounded waiting reads only the two declared process IDs and progress.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
PRIVATE = Path("/opt/data/private")
ARMS = ("object_only", "object_replay")
PROCESS_IDENTITIES = {379307: "run_object_adapt_study.py", 380501: "vlm_object_study.worker"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private(path):
    path = Path(path).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError("artifact outside persistent workspace")
    return path


def read(path):
    path = private(path)
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def overall(rows):
    matches = [row for row in rows if row["slice"] == "overall"]
    if len(matches) != 1:
        raise ValueError("official analysis lacks a unique overall result")
    return matches[0]


def process_snapshot():
    values = []
    for pid, expected in PROCESS_IDENTITIES.items():
        proc = Path("/proc") / str(pid)
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            values.append({"pid": pid, "present": True, "expected_command_matches": expected in command,
                "state": stat[0], "starttime_ticks": stat[19]})
        except FileNotFoundError:
            values.append({"pid": pid, "present": False})
    return values


def pending(root):
    screen = root / "analysis/pilot_seed42/screen/ANALYSIS.json"
    completion = root / "COMPLETION.json"
    absent = [str(path) for path in (screen, completion) if not path.is_file()]
    screen_complete = False
    if screen.is_file():
        try:
            screen_complete = read(screen).get("status") == "complete"
        except json.JSONDecodeError:
            pass
    if not absent and screen_complete:
        return None
    progress = root / "runs/object_replay/seed42/text/PROGRESS.json"
    value = read(progress) if progress.is_file() else {}
    return {"status": "pending_official_completion", "missing_final_files": absent,
        "official_screen_complete": screen_complete,
        "replay_text_completed_questions": value.get("completed_questions"),
        "replay_text_total_questions": value.get("total_questions"),
        "specific_processes": process_snapshot(), "reports_written": False,
        "new_model_forwards": 0, "new_optimizer_updates": 0}


def protected_files(lock, config_file):
    result = {str(config_file): lock["config_file_sha256"]}
    for key in ("source_sha256", "protocol_files_sha256", "data_files_sha256", "model_files_sha256", "initializer_files_sha256"):
        for name, digest in lock[key].items():
            if name in result and result[name] != digest:
                raise ValueError("study lock contains conflicting hashes: " + name)
            result[name] = digest
    if str(Path(__file__).resolve()) in lock["source_sha256"]:
        raise ValueError("report exporter unexpectedly belongs to scientific source lock")
    return result


def verify(files):
    failed = [name for name, digest in files.items() if not Path(name).is_file() or sha(name) != digest]
    if failed:
        raise ValueError("protected science/data/model/CV bytes changed: " + repr(failed))
    return {"checked_files": len(files), "mismatches": [],
        "source_data_model_protocol_config_and_cv_unchanged": True}


def pp(value):
    return "undefined" if value is None else f"{value:+.6f}"


def interval(value):
    return "undefined" if value is None else "[" + ", ".join(f"{point:.6f}" for point in value) + "]"


def link(path, label, anchor=""):
    target = os.path.relpath(path, PROJECT / "docs")
    return f"[{label}]({target}{anchor})"


def materialize(config_file, *, refresh_learning=True):
    config_file = private(config_file)
    cfg = read(config_file)
    root = private(cfg["output_root"])
    missing = pending(root)
    if missing is not None:
        return missing
    screen_path = root / "analysis/pilot_seed42/screen/ANALYSIS.json"
    completion_path = root / "COMPLETION.json"
    analysis, completion = read(screen_path), read(completion_path)
    lock_path = root / "STUDY_INPUT_LOCK.json"
    lock = read(lock_path)
    if (analysis.get("schema_version") != "rarevlm.object_presence.screen.v1"
        or analysis.get("stage") != "pilot_order_seed42" or analysis.get("seeds") != [42]
        or analysis.get("config_sha256") != sha(config_file)
        or completion.get("study_input_lock_sha256") != sha(lock_path)
        or completion.get("analysis_files_sha256", {}).get(str(screen_path)) != sha(screen_path)):
        raise ValueError("official screen/completion/config/study bindings differ")
    if completion.get("status") != "completed_negative" or analysis.get("eligible_candidates") != []:
        raise ValueError("this V2 result exporter expects the official completed-negative pilot; do not invent a different outcome")
    recovery = None
    if completion.get("recovery_file"):
        recovery_path = private(completion["recovery_file"])
        if sha(recovery_path) != completion.get("recovery_sha256"):
            raise ValueError("CPU recovery receipt changed")
        recovery = read(recovery_path)
        if (recovery.get("new_updates") != 0 or recovery.get("new_predictions") != 0
            or recovery.get("original_assert_same_inputs_scorers_bootstrap_and_gates_unchanged") is not True
            or recovery.get("reader_global_restored") is not True):
            raise ValueError("recovery exceeded its metadata-only scope")
    files = protected_files(lock, config_file)
    evidence = {str(screen_path): sha(screen_path), str(completion_path): sha(completion_path), str(lock_path): sha(lock_path)}
    if recovery is not None:
        evidence[str(recovery_path)] = sha(recovery_path)
        for name, digest in ((recovery["original_status_snapshot"], recovery["original_status_sha256"]),
            (str(root / "controller.log"), recovery["original_controller_log_sha256"]),
            (recovery["wrapper_source_file"], recovery["wrapper_source_sha256"])):
            evidence[name] = digest
    for source in analysis["source_data"].values():
        for kind in ("predictions", "evaluation", "input_lock"):
            name, digest = source[f"{kind}_file"], source[f"{kind}_sha256"]
            if sha(name) != digest:
                raise ValueError("official analyzed evidence bytes changed: " + name)
            evidence[name] = digest
    train, cells, preflights = {}, {}, []
    formal_updates = discarded_updates = 0
    for arm in ARMS:
        path = root / "runs" / arm / "seed42/train/TRAIN_RESULT.json"
        value = read(path)
        if (value["status"] != "completed" or value["completed_updates"] != 181
            or value["actual_object_exposures"] != 2888 or value["actual_replay_exposures"] != (1448 if arm == "object_replay" else 0)):
            raise ValueError("V2 completed training exposure/update contract differs")
        formal_updates += value["completed_updates"]
        keys = ("completed_updates", "actual_object_exposures", "actual_replay_exposures", "training_active_seconds",
            "model_load_seconds", "preflight_seconds_current_session", "session_wall_seconds",
            "peak_gpu_allocated_gib", "peak_gpu_reserved_gib", "mean_object_ce", "mean_replay_ce",
            "training_object_input_tokens", "training_replay_input_tokens", "training_object_visual_tokens",
            "training_replay_visual_tokens", "training_object_answer_tokens", "training_replay_answer_tokens",
            "training_object_microbatches", "training_replay_microbatches", "compute_accounting")
        train[arm] = {key: value.get(key) for key in keys}
        evidence[str(path)] = sha(path)
        for name, digest in value["adapter_files_sha256"].items():
            evidence[str(path.parent / "adapter" / name)] = digest
    for path in sorted(root.glob("runs/**/EVALUATION.json")):
        if path.parent.name not in ("object", "text"):
            continue
        value = read(path)
        cells[f'{value["condition"]}_{path.parent.name}'] = {"path": str(path), **value}
        evidence[str(path)] = sha(path)
    expected = {"initial_object", "object_only_object", "object_replay_object", "object_only_text", "object_replay_text"}
    if set(cells) != expected or any(value["status"] != "completed" for value in cells.values()):
        raise ValueError("V2 must contain all five newly generated official cells")
    fresh_predictions = sum(value["questions"] for value in cells.values())
    if fresh_predictions != 4260 or formal_updates != 362:
        raise ValueError("V2 complete prediction/update counts differ")
    eval_engineering_generated = 0
    for path in sorted(root.glob("preflight/**/PREFLIGHT.json")):
        value = read(path)
        eval_engineering_generated += len(value["records"])
        evidence[str(path)] = sha(path)
    for path in sorted(root.glob("**/PREFLIGHT_HISTORY.jsonl")):
        for raw in path.read_text().splitlines():
            discarded_updates += json.loads(raw)["discarded_engineering_optimizer_updates"]
        evidence[str(path)] = sha(path)
    if discarded_updates != 4:
        raise ValueError("V2 actual discarded engineering optimizer-step count differs")
    for path in sorted(root.glob("**/train/PREFLIGHT.json")):
        value = read(path)
        preflights.append({"path": str(path), "status": value["status"],
            "peak_allocated_gib": value["peak_allocated_gib"], "peak_reserved_gib": value["peak_reserved_gib"],
            "discarded_engineering_optimizer_updates": value["discarded_engineering_optimizer_updates"],
            "formal_optimizer_updates": value["formal_optimizer_updates"],
            "initial_parameters_restored_exactly": value["initial_parameters_restored_exactly"],
            "rng_restored": value["rng_restored"],
            "model_runtime_buffers_and_rope_cache_restored": value["model_runtime_buffers_and_rope_cache_restored"]})
        evidence[str(path)] = sha(path)
    for name in ("STATISTICS_REVIEW.json", "TEXT_OBJECT_ONLY_REVIEW.json", "MATCHED_FIRST_UPDATE_REVIEW.json", "OBJECT_VAL_CASE_REVIEW.json", "OBJECT_VAL_CASE_REVIEW.html"):
        path = root / "independent_review" / name
        if path.is_file():
            evidence[str(path)] = sha(path)
    if list((root / "runs").glob("*/seed0/train/TRAIN_RESULT.json")) or list((root / "runs").glob("*/seed200/train/TRAIN_RESULT.json")) or list((root / "runs").glob("**/pope/EVALUATION.json")):
        raise ValueError("unexpected seed expansion/new POPE contradicts completed-negative pilot")
    all_files = {**files, **evidence}
    guard_before = verify(all_files)
    arms = {}
    for arm in ARMS:
        obj = analysis["contrasts"][f"{arm}_minus_initial_object"]
        text = analysis["contrasts"][f"{arm}_minus_initial_text"]
        arms[arm] = {"object_overall": overall(obj), "object_by_category": obj[1:],
            "text_by_slice": {row["slice"]: row for row in text},
            "class_balanced": overall(analysis["class_balanced_diagnostics"][arm]["contrasts"]),
            "gate": analysis["gates"][arm], "training": train[arm]}
    history = analysis["source_data"]["initial_42_text"]
    metrics = {"schema_version": "rarevlm.object_adapt.v2.final_metrics.v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "trial_status": completion["status"], "goal_achieved": False, "goal_status": "active",
        "scope": "seed42 pilot; fixed SFT42 initialization; scientific screen failed; no automatic new trial",
        "config_file_sha256": sha(config_file), "study_input_lock_sha256": sha(lock_path),
        "official_screen_path": str(screen_path), "official_screen_sha256": sha(screen_path),
        "official_completion_path": str(completion_path), "official_completion_sha256": sha(completion_path),
        "arms": arms, "initial_object_metrics": cells["initial_object"]["metrics"],
        "historical_initial_text": {"questions": 1617, "new_generation_questions": 0, "provenance": history,
            "score_percent": arms["object_only"]["text_by_slice"]["overall"]["before_score_percent"]},
        "accounting": {"new_adapters": 2, "formal_optimizer_updates": formal_updates,
            "discarded_engineering_optimizer_updates": discarded_updates,
            "new_official_model_x_qa": fresh_predictions, "new_object_model_x_qa": 3 * 342,
            "new_text_model_x_qa": 2 * 1617, "reused_initial_text_model_x_qa": 1617,
            "evaluation_engineering_generated_model_x_qa": eval_engineering_generated,
            "new_formal_pope_predictions": 0, "replication_seeds_run": [],
            "statistical_unit": "image cluster; model×QA and seed×image are not independent observations",
            "model_forward_total": None, "model_forward_total_note": "not instrumented; includes real training/backward/checkpoint recomputation and engineering forward/generation; exporter itself has zero new forwards"},
        "evaluation_resources": {key: value["resources"] for key, value in cells.items()},
        "training_preflights": preflights, "controller_session_seconds": completion.get("current_controller_session_seconds"),
        "completion": completion, "source_guard_before": guard_before,
        "cpu_recovery": {key: recovery.get(key) for key in ("recovery_reason", "original_workflow_started_at",
            "original_failure_at", "original_workflow_wall_seconds", "original_workflow_time_scope",
            "cpu_recovery_seconds", "protected_file_count", "wrapper_source_sha256", "per_file_text_normalization",
            "new_updates", "new_predictions", "new_model_forwards")} if recovery else None,
        "evidence_files_sha256": evidence, "exporter_new_model_forwards": 0,
        "exporter_new_optimizer_updates": 0, "exporter_is_outside_scientific_lock": True,
        "limits": ["local object improvement does not establish Text slice protection or full POPE transfer",
            "single order seed42 and public/exposed POPE, not three-seed or blind evidence",
            "v1 57/64 and v2 61/64 assistant quality reviews have different populations, not paired improvement or full human truth",
            "extra Text replay changes supervised information and compute; no equal-compute or speedup claim",
            "report completion is not user goal success; no new POPE and no seed expansion"]}
    result_table, slice_table, resources = [], [], []
    for arm in ARMS:
        entry = arms[arm]
        obj, text = entry["object_overall"], entry["text_by_slice"]["overall"]
        result_table.append(f'| {arm} | {obj["after_score_percent"]:.6f} | {pp(obj["delta_pp"])} | {interval(obj["image_cluster_bootstrap_95_pp"])} | {pp(obj["one_sided_98_75_lower_pp"])} | {text["after_score_percent"]:.6f} | {pp(text["delta_pp"])} | {pp(text["one_sided_98_75_lower_pp"])} |')
        for label in ("unseen_0", "rare_1_5", "medium_6_20", "frequent_gt20"):
            row = entry["text_by_slice"][label]
            slice_table.append(f'| {arm} | {label} | {row["questions"]} | {pp(row["delta_pp"])} | {interval(row["image_cluster_bootstrap_95_pp"])} |')
        t = train[arm]
        resources.append(f'| {arm} | {t["actual_object_exposures"]} | {t["actual_replay_exposures"]} | {t["training_active_seconds"]:.3f} | {t["session_wall_seconds"]:.3f} | {t["peak_gpu_allocated_gib"]:.3f} / {t["peak_gpu_reserved_gib"]:.3f} | {t["training_object_input_tokens"]} / {t["training_replay_input_tokens"]} |')
    failed = {arm: [key for group in (entry["gate"]["primary_checks"], entry["gate"]["point_protection_checks"]) for key, value in group.items() if not value] for arm, entry in arms.items()}
    review = root / "independent_review"
    recovery_note = ""
    timing_note = f'控制器本次session约{completion.get("current_controller_session_seconds",0):.3f}秒。'
    if recovery is not None:
        normalized = sum(row.get("normalized_categories", 0) for row in recovery["per_file_text_normalization"])
        recovery_note = f'''原控制器在全部新预测完成后，因新汇总器把历史Text的空`category`与新记录缺省字段视为不同而停止。原{link(root/'RECOVERY_ORIGINAL_STATUS.json','失败STATUS')}与controller.log原哈希保留。独立CPU恢复先由冻结reader完整验证，再只在内存Text副本中规范化空类别别名（移除{normalized}个已有空key）；object/POPE、全部非category字段、分数、原assert／评分／bootstrap／门槛及科学文件字节均不变。详见{link(recovery_path,'RECOVERY receipt')}；恢复新增更新、预测和model forward均为0。
'''
        timing_note = f'原工作流从LAUNCH至原控制器失败约{recovery["original_workflow_wall_seconds"]:.3f}秒；独立CPU恢复{recovery["cpu_recovery_seconds"]:.3f}秒，两者分列，不用短恢复耗时替代整轮。'
    report = f'''# 物体监督适配v2：局部物体改善未通过整体晋级门控

官方pilot筛选及COMPLETION已完成，trial状态为`{completion["status"]}`。两个候选均未达到预注册联合门槛，未追加seed0/200，未新增POPE；用户的真实提升目标仍active，脚本完成不等于目标成功。

{recovery_note}

## 固定实验与计数

同一完整TextVQA-SFT42起点、原BF16基座／FP32语言q/v LoRA（1,843,200可训练参数）、视觉上限1024、Final@1 epoch、lr5e−6、microbatch4×accumulation4=object16。两组各2888个object监督QA／181正式更新，Replay每更新另加8题Text，共1448额外QA，两类逐题均值独立归一；末尾object8题分母为8。

本轮新增2个adapter、362正式更新、4个已丢弃并精确恢复的工程optimizer step。正式新预测为3个object×342＋2个Text×1617＝**4260 model×QA**；原始initial Text1617题严格历史复用、0新增生成。评估工程预检另有**{eval_engineering_generated} model×QA生成**，还有有限logit forward与训练前后向／checkpoint重算，不能将报告工具“0新forward”写成整个实验0 forward。上述QA量不是独立样本数。

object full val为342QA／143图，18类；平衡诊断256QA／139图取自同一完整预测，0额外生成。Text为1617QA／1024图。单固定checkpoint条件下按图像整簇6000次bootstrap，seed20261015，双臂×两个主端点采用单侧98.75%下界（alpha=.05/4）；95%区间与切片区间用于单独探索解释。

## 官方主结果

共同initial object为{arms["object_only"]["object_overall"]["before_score_percent"]:.6f}%，历史initial Text为{metrics["historical_initial_text"]["score_percent"]:.6f}%。差值均以pp计。

| 候选 | object accuracy% | object差值 | object95%CI | object单侧98.75%下界 | Text共识% | Text差值 | Text单侧98.75%下界 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(result_table)}

逐项失败门控（保留官方字段，不调阈值）：

```json
{json.dumps(failed,ensure_ascii=False,indent=2)}
```

object_only在新物体任务上达到局部accuracy改善门槛，但整体Text保持与罕见／高频监督答案保护是不同问题：整体Text通过或点差为正，不能代替rare／frequent点值保护。object_replay的object单侧下界未严格大于0，点差达到+1pp仍不足以晋级；Text数值与门控一律以上述官方JSON为准。平衡子集只作为附加grounding保护，不替代完整val或全9000题POPE。

## Text切片与边界

| 候选 | 实际SFT答案曝光切片 | QA数 | 差值pp | 图像簇95%CI |
| --- | --- | ---: | ---: | --- |
{chr(10).join(slice_table)}

rare/frequent是原32985QA实际SFT target文本曝光的代理，不是视觉语义类别或预训练频率。点保护失败不等于已证明切片总体显著退化；整体非劣也不能证明切片非劣。未通过联合筛选，因此0/200复制与POPE确认没有执行，不能宣称稳定泛化、通用保持或长尾改善。

## 真实成本与不等信息

| 条件 | object QA | 额外Text QA | 训练active秒 | 单分支session wall秒 | 训练allocated / reserved GiB | object / Text input token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(resources)}

{timing_note}原工作流时间包含排队、加载、输入／权重核验、预检、训练、评测和失败汇总，非纯GPU耗时；不以两个并行分支耗时相加冒充壁钟。训练active时间、分支session、评估缓存生成阶段和完整工作流壁钟分别保留。评估资源原值见便携JSON的`evaluation_resources`，首batch预热与加载等排除范围保留原字段。

Replay增加1448个真实Text监督、1,414,049个训练input token及额外前后向；同初始化、object顺序、更新数、参数量和视觉上限不代表等信息、等FLOPs或等算力。没有跨运行时加速／线上服务性能结论，也不把历史NF4到本轮BF16路径解释成纯dtype收益。

## 独立核验与真实病例

- {link(review/'OBJECT_VAL_CASE_REVIEW.html','病例1：可见鹦鹉仍被漏检','#case_01')}；{link(review/'OBJECT_VAL_CASE_REVIEW.html','病例2：遮挡车辆类别存在歧义','#case_02')}。保留原标签与官方得分；局部助手看图不是全量独立人类真值，也不证明小目标／遮挡的因果份额。
- {link(review/'STATISTICS_REVIEW.json','数学、图像簇与官方解析核验')}仅证明未发现阻断实现问题，不证明性能成功。
- {link(review/'TEXT_OBJECT_ONLY_REVIEW.json','完整object_only Text重评分与频次核验')}区分整体NI与rare/frequent点保护；不替代官方整轮判决。
- {link(review/'MATCHED_FIRST_UPDATE_REVIEW.json','两臂真实首更新输入匹配核验')}核验初始object输入，而不是全历程／等算力或用户已独立掌握的证明。
- {link(root/'learning/学习页.html','真实图像、token、逐题CE与回放学习页')}将实际数据／GPU记录与教学假设分开。

v1质量审查57/64被拒、v2新总体61/64通过；类别、可见大小与排除图像不同，两比例不是配对质量提升，也不是全量噪声率。上一轮DPO相对偏好改善未转化为POPE收益及FN瓶颈，是本研究的机制动机，不自动证明新物体监督能跨任务成功。

## 来源、验收与下一边界

官方{link(screen_path,'完整pilot分析')}与{link(completion_path,'COMPLETION')}保持原字节；该独立导出工具未修改src/config/protocol/STUDY/模型／预测／正式CV，未重跑CPU全套测试或GPU。源码、数据、模型、初始化和CV在导出前后按研究锁逐文件验证；详细哈希和范围在{link(PROJECT/'results/vlm_object_adapt_v2_metrics.json','便携指标JSON')}及导出receipt。

本trial保留负联合结果。新实验若继续，需要依据新的错误机制另立协议；本报告不改变原门槛、checkpoint、数据或目标状态，也不自动启动后续实验。
'''
    if refresh_learning:
        subprocess.run([sys.executable, str(PROJECT / "tools/build_object_adapt_learning.py"), "--config", str(config_file)], check=True)
    metrics["exporter_sha256"] = sha(Path(__file__))
    report_path = PROJECT / "docs/VLM_OBJECT_ADAPT_V2_RESULTS.md"
    metrics_path = PROJECT / "results/vlm_object_adapt_v2_metrics.json"
    write(report_path, report)
    write(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    guard_after = verify(all_files)
    metrics["source_guard_after"] = guard_after
    write(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    receipt = {"status": "completed", "trial_status": completion["status"], "goal_achieved": False,
        "official_files_modified": False, "report_file": str(report_path), "report_sha256": sha(report_path),
        "metrics_file": str(metrics_path), "metrics_sha256": sha(metrics_path),
        "source_guard_before": guard_before, "source_guard_after": guard_after,
        "new_model_forwards": 0, "new_optimizer_updates": 0,
        "exporter_sha256": sha(Path(__file__)), "whole_study_new_official_model_x_qa": fresh_predictions,
        "evaluation_engineering_generated_model_x_qa": eval_engineering_generated}
    write(root / "REPORT_EXPORT.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/vlm_object_adapt_v2.json")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--no-refresh-learning", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 480 or not 1 <= args.poll_seconds <= 45:
        parser.error("bounded wait0–480 seconds; poll1–45 seconds")
    config_file = private(args.config)
    root = private(read(config_file)["output_root"])
    deadline = time.monotonic() + args.wait_seconds
    while True:
        value = pending(root)
        if value is None:
            result = materialize(config_file, refresh_learning=not args.no_refresh_learning)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return
        print(json.dumps(value, ensure_ascii=False), flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(args.poll_seconds, remaining))


if __name__ == "__main__":
    main()
