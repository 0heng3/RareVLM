# 完整7B教师验证：能力差距成立，学生蒸馏收益尚未成立

2026-10-29 01:31:24完成，四worker均exit0/0争用、无长度删失。新生成1681次（完整1617+保持64），0更新/0GT前向/0adapter；原学生完整预测直接复用。controller从01:09:36到完成约21分48秒，四job active643.19/642.82/500.14/494.12秒，峰allocated最高32.854GiB；首轮冷权重加载约144秒，后轮缓存加载约4秒，均正常。

| 固定内部集合 | 强3B SFT VQA% | 完整7B教师 VQA% | 教师−学生pp | 配对图像簇95%区间 |
|---|---:|---:|---:|---|
| 完整1617 | 88.027211 | **93.537415** | **+5.510204** | **[+4.232778,+6.834321]** |
| 完整rare342 | 88.157895 | 92.280702 | +4.122807 | [+1.849599,+6.589765] |
| 保持64 | 95.312500 | 98.437500 | +3.125000 | [0,+7.812500] |

完整任务145涨37降，rare27涨8降，保持2涨0降；涨降为VQA部分分变化，不是完全正确数。主任务≥+.5pp/正95%，rare与保持点保护全部通过，`full_task_teacher_opportunity=true`。保持64的区间包含0，不能单独称保持显著提升。

## 可以证明什么

此前128道已见困难题教师105/128 literal并非仅有的优势；在同问题、图像、1024视觉预算、raw greedy32、EOS和评分下，冻结7B也明显优于当前强3B SFT的完整原生问答。支持选择该教师开展知识迁移研究，不支持“教师不够强”作为旧KD失败的完整解释。

模型规模、预训练权重、适配状态同时不同，不能作纯参数规模因果；1617是历史使用的内部开发集，不是外部盲测；固定模型图像簇区间不含训练seed差异或全部历史选择。旧3B KD仍86.846011%，`student_improved=false`、`goal_complete=false`，不能把93.537415%归为学生训练提升或新增简历算法收益。

## 已执行下一步：学生当前前缀上的蒸馏

参考[GKD原论文](https://arxiv.org/abs/2306.13649)和[TRL官方实现](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py)，旧GT-prefix KD只在正确历史下约束下一token，而学生自由生成会进入自己的历史。该差异是新假设，尚非唯一失败原因证明。实际固定103训练题的强起点保存轨迹中，47题/127个正文位置已偏离GT前缀，通过事前16题/32位置最低准入；4项新CPU检查（解析KL/梯度、词表padding排除、原始token/终止符、左padding逐QA平均）通过。

独立[新协议](VLM_ONPOLICY_KD_V1_PROTOCOL.md)固定原384QA/512曝光/32更新、强起点、联合204因子3686400参数、LR1e-4、T2/λ1、原mask103、调度及数值政策；只将辅助KL的状态换为学生当前greedy32生成的前缀，GT CE与真实EOS保持。教师在同一原始token/图像输入上提供软分布，不把错误生成当正确硬标签；原始IDs直接拼接，不重新分词。官方文本Trainer因head padding不同不能直接使用，本地沿用151665实际词表对齐，是部署greedy变体，非论文随机版本完全复现或PPO/GRPO。

新controller793427/start535195691于10:44:15提交，共32更新/至多206训练生成+206教师正文前向，固定Final32内部960+完整1617共2577评测。旧强起点与GT-prefix KD/hard/uniform/历史DPO控制全部复用，不重训关闭候选；额外前向计算与正文长度不同，不称等算力或唯一因素归因。主任务须对这些控制全部≥+.5pp/正95%，原rare/OCR/native/保持点保护保持。阴性关闭固定新配置，不扫参、扩步或seed。

提交后两A100均有其他任务/约72GiB占用，新研究等待共享空卡，未抢占；准入后预计训练20–30分钟、评测15–25分钟，总约40–60分钟，排队未知，以实际进度修正。CV/科研/Agent/CI不改，不自动push。

[教师评测协议](VLM_FULL_TEACHER_TRANSFER_V1_PROTOCOL.md) · [教师评测源码](../tools/run_full_teacher_transfer_v1.py) · [量化结果](../results/vlm_full_teacher_transfer_v1_metrics.json) · [真实结果与案例绘图](../tools/render_full_teacher_transfer_v1.py) · [新原始token与KL代码](../src/vlm_onpolicy_kd_v1/objective.py)

本地`outputs/vlm_full_teacher_transfer_v1/result_review_v1/`保留真实PNG、正反案例JSON、教师能力与学生前缀蒸馏的公式源码学习页；原分析/预测/权重不改。
