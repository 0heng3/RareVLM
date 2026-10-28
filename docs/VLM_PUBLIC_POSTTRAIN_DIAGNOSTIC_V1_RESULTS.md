# 公开噪声偏好诊断：已完成

2026-10-15北京时间17:00完成全部GPU训练/评测，约1小时53分；最终汇总CLI参数错误已由独立工具补齐，冻结源码/门槛/73个模型与预测等文件不变，0重训练/重评测。SFT/DPO各223更新，3模型完整4851条Text＋27000条POPE预测及1317个heldout model×pair完成；总446正式更新与4个丢弃工程step分开记录。

初始/继续SFT/DPO Text87.111936/86.870748/86.839827%，POPE accuracy87.100000/86.933333/87.100000%。DPO对初始Text−.272109pp，条件95%图像簇区间[−.573931,−.006119]；POPE .000000pp，[−.122222,.111111]。提准和Text .5pp保持门槛均未过，数据事实gate=false、automatic promotion=false保留。当前完整结论及真实病例见[失败机制分析](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_FAILURE_ANALYSIS.md)，最终数字见[便携结果](../results/vlm_public_posttrain_diagnostic_v1_metrics.json)。下方是原启动快照，pending/运行中仅代表当时历史。

## 原启动记录

2026-10-15北京时间15:07启动控制器349070，15:08两组真实GPU工程预检通过，15:09两A100分别接续SFT/DPO分支。**这是运行快照，完整生成指标尚未出齐**；实时读`outputs/vlm_public_posttrain_diagnostic_v1/STATUS.json`，最终权威数字为同目录`analysis/ANALYSIS.json`和[便携结果](../results/vlm_public_posttrain_diagnostic_v1_metrics.json)。

## 数据及本轮问题

官方RLHF-V人类局部纠错5733原对→排121→5612通过真实双completion长度检查→每图一对4007→训练3568/heldout439。与已有train/dev/final、DocVQA/MMStar与POPE的文件/像素精确重叠0；不代表排除近重复或预训练重叠。来源与SHA/筛选在[固定协议](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md)。

固定64例助手审查仅49例明确事实偏好（76.5625%，Wilson95%[64.867,85.250]%），未通过90%及下界80%。五例主要为同义/范围改写；其余有姿态/动作未定、主观问题、共享/冲突错误、不可验证的亲子关系等。**非明确不等于错误标签，更不能当独立随机flip率；人类局部纠错不保证整段chosen无幻觉。** 原`DATA_ADMISSION.json`保留拒绝，原749/199候选不训练。

因此另行预注册本轮“公开不完美偏好上的SFT/DPO噪声诊断”，单独scope/准入记录，不改原门槛、不自动晋级可信数据/更多种子/正式benchmark或简历。目标是实测标准DPO能否在有这些风险的公开人类纠错上改善生成行为，而非假定数据可靠或DPO必然提分。

## 已完成工程证据

64项相关CPU合同、五个CLI导入通过。两组真实GPU预检使用工程LR1e-4临时更新一步后恢复全部LoRA/RNG并丢弃optimizer；正式LR为5e-6，不混为正式更新。后续正式runner各再做一次预检；两次独立预检+两次正式runner预检预期共4个丢弃工程step，完整历史单独计数。

| 独立工程预检 | 初始loss | LoRA梯度范数 | 临时更新后actor/reference最大logp差 | allocated峰 |
| --- | ---: | ---: | ---: | ---: |
| SFT | 1.370587 | .679348 | 2.722198 | 7.516 GiB |
| DPO | .693147 | 2.474591 | 1.183678 | 7.531 GiB |

两组静态reference保持，初始LoRA/RNG精确恢复、基座/视觉冻结权重哈希前后相同。DPO初始loss≈log2是正确初始参考的数学检查，不是性能提升。

15:12日志中SFT已25/223更新、400/3568 chosen曝光，累计训练174.66秒；15:13 DPO参考缓存600/3568。单epoch/同chosen/同顺序/223更新，DPO额外rejected和reference算力单列，不称等算力。粗估全轮训练+三模型完整评测2–3小时，实际取决于后续完整POPE生成；该估计不作硬件加速成果。

## 固定评测与解释边界

- initial、SFT、DPO完整TextVQA dev各1617，报告整体、原train_all目标频次切片与图像簇区间；DPO相对initial .5pp非劣保护，非跨任务全保持。
- 三模型完整POPE各9000，random/popular/adversarial共用500图；Accuracy/F1/precision/recall/yes_rate及invalid独立，6000次同图像簇抽样。先排10个训练同源图，mirror只取JPEG，标注沿用固定官方三JSON。
- 三模型heldout各439对teacher-forced logp；排序受回答长度影响，不是实际生成正确率。所有答案只进入训练/teacher-forcing或生成后计分，不进入无标签生成prompt。
- 即使预设POPE数字判据与Text保持通过，也只报告本次噪声诊断的条件生成结果；事实数据gate仍false、automatic promotion仍false。单初始化/seed42，不声称三种子稳定、RLHF完整PPO或GRPO训练。

真实数据与公式/源码学习页在本地`outputs/vlm_public_posttrain_diagnostic_v1/learning/学习页.html`，含64原图/问题/两回答/审查理由。旧失败记录保留，助手实现与运行不代表用户已独立掌握或贡献；正式简历/论文/Agent未改，不启CI、不自动push，已有任务未抢占或终止。
