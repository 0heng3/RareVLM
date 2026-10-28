# 原生问答DPO：整段偏好与首个分歧决策

2026-10-26，新GPU forward前登记。上一96/192覆盖分支完整1617分别87.476809/87.229437%，共同起点87.198516%，候选16题升15题降净+.5分；保留失败结论，不扩原分支。较强历史128 DPO为87.631416%，亦未满足原能力标准。

**问题。** 序列DPO提高的是整段chosen相对rejected的reference-relative log概率。自回归greedy在共享前缀后的首个不同token处作局部决策。两者并不等价。假设：已有DPO残错中存在一定数量“整段绝对排序正确，但首个分歧仍偏向错误token”的案例；若确实存在，比较在首个分歧上的局部偏好损失才有针对性。本地机制假设，不宣称新原创方法、唯一失败原因或确定可提升泛化。

**固定输入。** 复用visual_preference_v1的128条已看图支持的native纠错QA、原chosen/rejected实际token、CPU输入合同及原排程中首次出现的32个4QA纠错批。不挖新数据，不看正式集，不回填dev错例。三个固定模型：同复合起点、匹配视觉CE Final32、视觉CE+DPO Final32；直接绑定原适配器SHA。数值、视觉预算、assistant mask、padding与原8序列TF批一致。生成结果复用各模型原native_hard128；不能把新TF批当原greedy的逐步运行trace。

**观测。** 每个模型仅32批/256条teacher-forced序列；三模型768序列、0新生成、0优化器/更新。记录正确与错误答案各自sum logP及token均值NLL；在两答案相同的实际token前缀后，取chosen路径上的同一logits向量比较首个不同token的logP差、GT词元全词表rank/top1。raw-logit差与logP差相等。EOS保留；若两完整监督序列没有可比较分歧，原样记录，不拼造token或静默剔除。

正文/终止token、标签有效位置和两个实际输入前缀必须对应；token比较只用最后一维；复用现有completion_logps计算整段分数。输出是固定checkpoint、训练集上的机制诊断，不能当总体训练准确率变化、独立迁移、统计确认或能力提升。

**判定先冻结。** 在DPO原生成未满分、且实际completion IDs仍与旧负例相同的题中，同时满足 `chosen_sum > rejected_sum` 与 `first_divergence_logit_margin < 0` 的题数>=16，才标为`local_decision_mismatch_supported`，允许另行登记CE+首分歧偏好 vs原匹配CE+DPO的短预算研究。报告所有分型、初始和CE同题结果，以及该类相对initial的序列/局部变化。不达到16就停止这条机制分支，不扫阈值、扩大诊断池或自动训练。

该判定只是研究准入，不是显著性检验，也不能证明该损失是完整1617失败的主要病因。即使达到，当前程序也不自动训练；下一训练仍须先冻结其唯一变化、CPU损失验证和原完整任务提升/保持标准。不修改旧scientific source、预测、门槛、CV/科研/Agent/CI，不push。

**资源。** 复用共享UUID队列：零compute、空闲>=40GiB、利用率<10%、三次15秒；每模型累计活动<=900秒、allocated<=65GiB，不终止其他任务。每4题原子落盘，可在同UUID批边界恢复。仅补新token对齐/分歧与排序统计的必要CPU验证，已过加载/mask/save审查不重做。

输出`outputs/vlm_native_divergence_readout_v1`，源码`tools/run_native_divergence_readout_v1.py`。完成后展示真实原图、原生成、整段与分歧数值；不把teacher-forced概率图冒充注意力。原三seed/强基线/外部总体目标仍未完成。
