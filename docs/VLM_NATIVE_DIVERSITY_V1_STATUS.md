# 纠错覆盖对照：完整验证未达标

独立固定Final评测已完成，两Final共3234新生成/0优化，全部exit0、0争用。完整1617起点87.198516%、前DPO128 87.631416%、repeat96 87.476809%、diverse192 87.229437%。候选对起点+.030921pp/95%[−.532940,+.588596]，对repeat96−.247372pp/[−.875724,+.394613]，对前DPO128−.401979pp/[−.991287,+.186341]，task_signal=false；均未证实策略收益或显著退化。

候选完整集16升15降、净+.5分；rare342净−1分/−.292398pp。512的+1.171875pp/rare366+1.366120pp没有迁移。512/1617 rare占比71.5%/21.2%，但完整rare也没有保持正点值，不能把比例差异当唯一原因。旧128诊断原门槛failed与全部旧科学文件保持。

已执行下一行动：补齐固定Final复核，归档RESULT_REVIEW与DECISION，关闭此96/192配置扩步/扫参/seed及同类矿工循环；保留前DPO128作较强历史点参考，不作已验证提升或部署。后续新训练须有新机制假设和独立匹配协议；本轮未放行更多训练。原强基线/三seed/外部总目标仍未完成，CV/科研/Agent/CI不改、不push。

[完整结果、原图与停止决定](LOCAL_ARTIFACTS.md)

以下保留此前初筛与启动记录：

# 纠错覆盖对照：完整初筛与独立验证

2026-10-26 19:11原研究完整结束：64更新/1920生成，两worker exit0、0争用。候选native512 94.726563%，对起点+1.171875pp/95%[+.390625,+2.148438]；rare366 94.371585%，+1.366120pp/[+.273224,+2.732240]，6项得分升0降。repeat96 native94.140625%，候选对其+.585938pp但CI下界0，尚不证明策略优胜。

候选旧128诊断33满分≥32，但VQA增益19.53125<20，故原screen_negative_no_expansion、expand_transfer=false保持。该诊断不是双方完整训练准确率，不能将未过该阈值等同真实任务失败。原OCR/原生问答点保护均过；ret64候选与起点同60/64、低于repeat96的61/64，点值不能证明非劣。

本轮用户要求后续行动，已独立登记并执行native_diversity_transfer_v1（20:05提交）：只读两份Final32，各1617，共3234新生成/0更新，原起点和前DPO128预测复用；不改原科学设置或升级原gate。仍要求完整主任务对起点与repeat96各≥+.5pp和正95%区间、守rare及强参考。它是事后固定模型内部验证，非外部/三seed，原总目标未完成。首读该新root queue/STATUS、runs/role/transfer/PROGRESS、ANALYSIS、COMPLETION，不重启原训练或加步。

[六项原预测与原图边界](LOCAL_ARTIFACTS.md) · [独立协议](VLM_NATIVE_DIVERSITY_TRANSFER_V1_PROTOCOL.md) · [原筛选终态](LOCAL_ARTIFACTS.md)

以下为原启动记录：

# 纠错覆盖96与192：当前研究

2026-10-26启动。3072题筛选已14:47完整完成，195候选/194target、0优化/双worker exit0/0争用。按固定顺序审核前152，70支持（64 single/6 multi）、82不准入或不确定、余43未审。采用支持的前64个新single且normalized target不在旧135支持池；助手非专家审核，非整个数据集错标率，原标注/评分不变。

已构造旧64 single/64 multi + 新64 single=192；控制按四题组取半：旧64/新32=96。两组变量曝光为96×2或192×1，再加完全相同的64次旧锚点；纠错均256曝光（旧192/新64，single160/multi96），另同native128/OCR128，均512 chosen/256 negative/32步。

两组均复用原冻结`run_visual_preference_v1.train()`，CE+DPO/β=.1/权重1、视觉LoRA1843200参数、语言冻结、1024整图+FP32语言、相同初始化/LR/优化器。不是新增原创损失，不把96×2与192×1隐去额外共享锚点，也不把QA相等称相同FLOPs。新题原SFT已曝光，不能称底座未见。

各unit独立reference→训练→内部960，空闲GPU动态接续。首读`outputs/vlm_native_diversity_v1/training_queue/STATUS.json`、`units/{repeat96,diverse192}/reference/{PROGRESS,COMPLETE}.json`、`runs/{repeat96,diverse192}/train/{FIRST_UPDATE,PROGRESS,TRAIN_COMPLETE}.json`，随后SCREEN_ANALYSIS、条件TRANSFER_ANALYSIS/COMPLETION。main/runs是对应unit工件的路径别名，不是额外训练。

原纠错128诊断有64/126条属于两组训练，不能把其成绩当两组完整训练准确率；其旧推进门槛单列。真正主信号仍是1617同时相对共同起点和新匹配控制≥+0.5pp与正图像簇95%区间，并保能力/rare，另报告前DPO128强参考。当前未有新能力结果，不自动加步/扫参/seed/正式测试，不改CV/科研/Agent/CI、不push。

[冻结协议](VLM_NATIVE_DIVERSITY_V1_PROTOCOL.md) · [实际数据与排程脚本](../tools/run_native_diversity_v1.py) · [看板](LOCAL_ARTIFACTS.md)

实际启动检查（2026-10-26T18:55:30.907865+08:00）：两组首DPO均0.6931471825、relative reference margin0、60/144 scope与冻结梯度None、实际视觉checkpoint、FP32保存一致；首CE0.491479/0.564934随题集不同而不同。当前步数/速度以PROGRESS为准，详见STARTUP_HEALTH。

冻结DATA_RECEIPT保留历史继承字段；本轮真实新增/排程说明另见DATA_RECEIPT_CLARIFICATION.json，不据继承的identical_chosen_schedule推断两组相同训练题，也未修改任何冻结输入。
