# 回放覆盖对照 v1：真实结果

状态：`completed_negative`。本轮在原预注册科学门槛处停止，未达到真实提升验收。 `goal_complete=false`。

唯一候选为 wide32；repeat8x4 仅是 seed42 探索对照，不能替补候选或追加失败控制的种子。两组物体流、Final 更新规则与每步32条 replay 曝光相同；wide32 扩展 unique 覆盖，control 重复原8条四次。两组教师新建成本和信息覆盖不同，不作等教师计算或等 FLOPs 声明。

以下点值、区间和门槛直接读取原 ANALYSIS，不重评分或重算 bootstrap。四个主端点只属于 wide32，alpha=.05/4；按图像整簇、6000次抽样，单侧98.75%下界。顺序 seed 先均值后抽图，sample SD 另列；model×QA 不能视为独立样本。

官方阶段：`pilot_order_seed42`，seeds=[42]。

| 端点 | initial % | wide32 % | Δ pp | 95%图簇CI | 单侧98.75%下界 |
|---|---:|---:|---:|---|---:|
| object/overall | 95.614 | 97.661 | 2.047 | [0.588, 3.846] | 0.552 |
| text/overall | 87.112 | 86.964 | -0.148 | [-0.465, 0.143] | -0.520 |
| text/rare_1_5 | 87.398 | 87.076 | -0.322 | [-0.855, 0.000] | -0.950 |
| text/frequent_gt20 | 89.281 | 88.801 | -0.480 | [-1.211, 0.000] | -1.432 |

官方 gate：`False`。

- `object_gain_at_least1pp`：`True`
- `object_lower98_75_gt0`：`True`
- `text_NI_lower98_75_gt_minus0_5`：`False`
- `object_recall_point_nondecrease`：`True`
- `object_specificity_point_at_least_minus0_5`：`True`
- `balanced_object_point_gt0`：`True`
- `text_rare_point_nondecrease`：`False`
- `text_frequent_point_nondecrease`：`False`

逐类 yes/no 均衡固定子集：256 QA，139 图，Δ=1.953pp。每类正负相等不表示各类题数相等；该子集不替代完整主端点，也不是宏平均。

单 seed 探索对照（不用于选择候选）：

| 比较 | Object Δ pp | Text Δ pp |
|---|---:|---:|
| repeat8x4_minus_initial | 2.339 | -0.118 |
| wide32_minus_repeat8x4 | -0.292 | -0.031 |

原证据：[ANALYSIS.json](LOCAL_ARTIFACTS.md)。

rare/frequent 是原32985条 train_all 实际监督目标的答案频次代理；点值保护与总体 Text .5pp 非劣检验分别报告。切片/类别区间是条件于固定 checkpoint 的探索区间，不称切片非劣或预训练未见。

| 已完成训练 | seed | object QA | replay QA | replay unique | 正式更新 | teacher新前向QA | teacher历史bankQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat8x4 | 42 | 2888 | 5792 | 1448 | 181 | 0 | 1448 |
| wide32 | 42 | 2888 | 5792 | 5792 | 181 | 5792 | 0 |

| 实际成本项 | 数量 |
|---|---:|
| 正式优化器更新 | 362 |
| 实际记录的临时工程更新（已丢弃） | 4 |
| 正式物体teacher-forced QA前后向 | 5776 |
| 正式回放teacher-forced QA前后向 | 11584 |
| 新教师缓存QA前向（按manifest去重） | 5824 |
| 历史教师bank QA（按manifest去重） | 1448 |
| 历史bank在本研究新增教师前向 | 0 |
| 通过工程预检的临时actor QA前后向 | 192 |
| 工程初始reference QA前向 | 128 |
| 工程临时更新后reference QA前向 | 128 |
| 正式新生成model×QA（完整保存cell） | 3918 |
| 预检新生成model×QA | 32 |
| 通过的无标签finite-logit QA前向 | 32 |
| 记录的子进程调用 | 8 |
| 记录的子进程失败 | 0 |

计划 pilot 的4个丢弃工程更新仅是计划数；上表来自全部实际 history，追加种子或重试会单列增加。历史1448题 bank 的原计算保留为历史，不能与本轮新教师 forward 相加。失败/中断尝试的未保存计算不凭计划推算，完整 calls 与失败 metadata 保留在便携 JSON。

| 训练臂/seed | active 秒 | teacher setup壁钟秒 | allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|
| repeat8x4/42 | 4805.368 | 4.538 | 8.730 | 11.285 |
| wide32/42 | 4895.225 | 4719.122 | 8.723 | 10.834 |

原启动至终态壁钟：—秒；当前控制器 session：11828.993秒。均包含相应队列、加载、训练、评测或CPU阶段，不作纯GPU时间／线上延迟解释。

历史 initial 的 Object/Text/external/POPE 均只引用原完整预测，本轮新增 baseline 生成0题。独立审查只按已有收据链接，不重做旧审查。

- [ENGINEERING_PREFLIGHT_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `fe031d6b9674d47ed9bd99253b0e82e4faea207f8c3d398983106cd25d877957`。
- [FIRST25_TIMING_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `edb4da0eceeb969dd6cbf7aa947adde2d727856948b083a5533e7289855fcad4`。
- [FIRST_UPDATE_STREAM_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `2ffc78bf9afbc617632f843edd773e4b586f3e19915d4a889a74922a3d2d0f7c`。
- [OLD_KL_PREFIX_CHANGE_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `9cad3713b3633370e48fea97a2d6c527031338f1917d1c1c2b6d419fa6a350f5`。
- [PARTIAL_CACHE_TIMING_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `f8baa1f98fad3b49f4c02974d8b6f77d97f29b528a20ebe0085456c16ec4bd98`。
- [REPEATED_CONTROL_PREDICTION_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `1c439176dcfa27bcd903280dbc7e171237a8363f15268073c095dc4ec9ac78a4`。
- [REPEATED_CONTROL_WEIGHT_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `b4542241513e49ca728c39b057a858d645793be967615e312909151b2e917509`。
- [WAIT_WIDE_COMPLETE_LEARNING_ONCE.STATUS.json](LOCAL_ARTIFACTS.md)，SHA `d5b52dd794ee8f15c1e7e4aa38cd6f03986860fc63b678d3f423d0a20217f490`。
- [WIDE_FIRST25_PROGRESS_SNAPSHOT.json](LOCAL_ARTIFACTS.md)，SHA `c585648a1eb54e8f17b925dd80ecea0600c51301bac4bb286299c2c44f83b34e`。
- [WIDE_FIRST25_TIMING_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `75db2e9ef66c8165a95b9b5e0288034a9cf2d63055bdde6f4e42093df4d53a9f`。
- [WIDE_FORMAL_START_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `b3cf2d0641cf20877caeacff9a43cd444f98e842efac579b9285769796f08bbf`。
- [WIDE_PILOT_PREDICTION_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `c2cf932ed7842549ca228e8cde18bf99da306b4d5f21a1c51d04323d4031f278`。
- [WIDE_PILOT_REVIEW_ERROR.json](LOCAL_ARTIFACTS.md)，SHA `2bfce79275b5672a1529913b3b626934b02f4eb14c4ff69592057d567bc108a5`。
- [WIDE_PILOT_REVIEW_ERROR_attempt1.json](LOCAL_ARTIFACTS.md)，SHA `2bfce79275b5672a1529913b3b626934b02f4eb14c4ff69592057d567bc108a5`。
- [WIDE_PILOT_REVIEW_RECOVERY.json](LOCAL_ARTIFACTS.md)，SHA `873539632700e6005f3eb916a0ef1a886354d21bf712c9103d69e491e0c6cdfa`。
- [WIDE_PILOT_WAIT_STATUS.json](LOCAL_ARTIFACTS.md)，SHA `4a7be03ed68afe0fb69dd8decd4ac786134f017e5b27205035b84807c44c58ef`。
- [WIDE_PILOT_WAIT_STATUS_attempt1.json](LOCAL_ARTIFACTS.md)，SHA `a6ea0b1b6f42d379c9b93b02a6b451987f7a23d57982b1c6294a5f4a2606b36b`。
- [WIDE_TRAINING_COMPLETION_OBSERVER.json](LOCAL_ARTIFACTS.md)，SHA `da1c92b28d875f312f3e6ef05feb9ed523d4fa9c9fc75b2efd1397a37f123c6e`。
- [WIDE_TRAINING_COMPLETION_RECOVERY.json](LOCAL_ARTIFACTS.md)，SHA `e212e234e71a877c2fa1de69102d59b343ea5a267a473a81e8ddb8a759ce10d1`。
- [WIDE_TRAINING_COMPLETION_REVIEW.json](LOCAL_ARTIFACTS.md)，SHA `5fbf1f27bead4ac7df9284fd62b72ef3bc2a3ca8af80faea3a32a35fa55ced6c`。

完整数值与成本元数据：[便携 JSON](LOCAL_ARTIFACTS.md)。导出器仅标准库读取元数据，0模型加载、0forward、0GPU、0新增预测或更新。
