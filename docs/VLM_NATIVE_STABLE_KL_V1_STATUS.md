# 修复数值路径后的SFT/KL实验

2026-10-26完整终态：两臂32更新/各960内部预测exit0，native512均93.671875%，对修复新初始+.117188pp/95%[−1.386719,+1.621094]；rare366均−.655738pp/区间跨0，未过点保护，停止扩原配置。hard满分8→46/45，OCR均94.921875%，512条两臂回答完全一致。新Final完整训练梯度诊断已完成，KL/CE11.1221%、cos−.112416、Final回放KL下降23.2593%，非未反传或已证明严重整体冲突；数据覆盖/回答范围迁移仍待单独验证。见outputs/vlm_failure_stage_stats_v1/REPORT.md、outputs/vlm_stable_kl_gradient_v1/REPORT.md。

最新首步检查（2026-10-26T00:36:57.143504+08:00）：两臂实际12/32、12/32；首CE均0.5203343463，actor/reference logp差0、初始KL0、KL-only梯度9.6344e−7，训练梯度有限，峰allocated约16.60GiB。新基线1216和64教师forward完整，OCR128/native127正确回放准入。当前约18秒/步，预计第一阶段还需15–25分钟；依实时工件，不保证质量提升。独立GPU24QA前后向32.922秒/0optimizer已过，非正式更新。证据STARTUP_HEALTH.json。

本轮活动输出：`outputs/vlm_native_stable_kl_v1`。实时状态以工件为准；本文为2026-10-26提交快照。

修复已验收：像素和patch相同、视觉block00首先产生分组数值偏差；逐图执行原HF视觉forward后，14题视觉输出相同，但原6个分歧病例只4题所有生成分组输出一致。条件启用FP32语言计算后，原6题和额外8题均批量/单例一致，0更新，R1/R2共236生成/145.645秒。不是总体稳定率或质量提升，原权重值/数据/评分不改，旧数值结果不回写。见`outputs/vlm_batch_origin_trace_v1`、`outputs/vlm_batch_numeric_repair_v1`。

新实验已冻结并提交，controller746611/start505535406。复用384QA/每臂32更新512 chosen，比较SFT与SFT+仅正确训练回放的全词表forward KL，lambda1；同修复后起点、数值策略和新基线。两片基线448/768共1216实际生成后，至少224/256且每域112个回放答对才准入64个teacher缓存forward；两臂首步同输入KL身份检查通过再更新。不再做旧DPO负forward，无新增dev训练，阴性不扩步或seed。

读取顺序：`baseline_queue/STATUS.json`、`baseline_{a,b}/PROGRESS.json`；`reference/REPLAY_ADMISSION.json`、`reference/PROGRESS.json`、`reference/COMPLETE.json`；`runs/{sft,sft_kl}/train/{INITIAL_KL_CHECK,FIRST_UPDATE,PROGRESS,TRAIN_COMPLETE}.json`；`SCREEN_ANALYSIS.json`、条件`TRANSFER_ANALYSIS.json`、`COMPLETION.json`。另`gpu_first_update_check_v1`为独立8QA teacher+16QA actor的零优化器GPU梯度检查，不算正式首步或能力收益。

只有候选原保护门槛全过且native512相对新起点≥+.5pp、相对新SFT≥+.25pp，才补三模型各1617；能力提升需对新起点和新SFT各≥+.5pp且正95%区间。完整强基线/三seed/外部目标未完成，不更新简历，不push。GPU按共享空闲UUID门控，不操作其他任务。

[冻结协议](VLM_NATIVE_STABLE_KL_V1_PROTOCOL.md) · [训练源码](../tools/run_native_stable_kl_v1.py) · [交互看板](LOCAL_ARTIFACTS.md)
