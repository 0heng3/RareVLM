# 回放分布修复 V1：运行入口

2026-10-25完整结论：收尾恢复已完成，`recovery_v1/COMPLETION.json`为`pilot_negative_no_expansion`，0恢复额外更新/生成。全部native1617 repair87.247990%、initial87.328386%、old control87.260359%；对initial−.080396pp/95%[−.835366,.650581]，对control−.012369pp/95%[−.562038,.551101]。因此回放分布修复未验证整体问答提升，不追加其order0/200或新OCRtest；原OCR正结果保持。下一项为[真实原生问答纠错V2](VLM_NATIVE_CORRECTION_V2_STATUS.md)，与本负方案分开。下文“待完成/预计”是历史快照。

2026-10-25 16:19更新：新训练64/64正常完成，active484.834秒，峰9.799GiB，1024 OCR/512 native曝光正确，冻结分区保持。内部960生成已齐；原生1617当前676/1617，按当时实测速度剩约10分钟生成，另加收尾汇总，预计16:30–16:33首轮结论。以下仅分项结果，不能先判整体成功。

| 已完成集合 | 共同initial | 原回放control42 | 修复回放repair42 |
| --- | ---: | ---: | ---: |
| 内部OCR256（%，非官方test3000） | 92.578125 | 94.531250 | 94.921875 |
| 内容词OCR243（%） | 93.004115 | 94.650206 | 95.061728 |
| native512（%） | 93.222656 | 93.750000 | 93.613281 |
| native512多词切片（%） | 95.273438 | 94.765625 | 94.882813 |
| 保持64（%） | 93.750000 | 93.750000 | 93.750000 |

多词相对initial差−.390625pp，回到原≥−.5点保护线内；比旧回放改善.117188pp，但未据此声称统计改善。native512对initial仅+.390625pp、对原回放−.136719pp，不能称整体收益。新官方test尚未评测。

四旧模型完整native1617补测已15:56结束，6468生成/0新训练/全部exit0/0争用：initial87.328386%、dense_language87.093383%、dense_visual87.081014%、sparse_language87.260359%。稀疏语言对initial−.068027pp，条件图像簇95%区间[−.835420,.717399]跨0：OCR收益未自动迁移成整图问答提升，也未证明显著退化。原始结果在`outputs/vlm_iiit_native_transfer_v1/COMPLETION.json`。

发现并版本化修复新增收尾检查的兼容性错误：DATA保持64均没有inference_input_sha256，而原worker在全部评测后无条件读取。实际64预测input_sha256与冻结initial逐题一致。当前worker继续自然完成生成；原源码/失败状态保留，独立[元数据收尾修复协议](VLM_IIIT_REPLAY_REPAIR_V1_METADATA_RECOVERY.md)和`recover_iiit_replay_repair_v1.py`已冻结、controller739047于16:18提交。保持64改引用原baseline实际SHA，其余2513项依旧严格用DATA SHA，0重训/重新生成。最终优先读取`outputs/vlm_iiit_replay_repair_v1/recovery_v1/{PILOT_ANALYSIS,COMPLETION}.json`；原队列可能显示已知KeyError，不能将该收尾错误当成训练失败。未通过原内部门槛不追加复现。

以下为提交时快照。

2026-10-25 15:32:23+08已提交，控制器737898；当时两张A100正在既有native1617补测，本任务等待共享空闲准入，无抢占。此处为提交快照，实时进度读下列工件。

准备通过：native256长度桶150/91/15→64/152/40，含the 0→18；947池取256，与评测图像/像素重叠0。实际CPU输入与assistant标签已检查；原OCR128各8次、native256各2次、64更新与损失不变。新候选尚无完整生成结果，不能写成提分。

- [协议](VLM_IIIT_REPLAY_REPAIR_V1_PROTOCOL.md) · [脚本](../tools/run_iiit_replay_repair_v1.py) · [真实图片、分布图、公式与代码](LOCAL_ARTIFACTS.md)。
- 实时工件在`outputs/vlm_iiit_replay_repair_v1/`：`pilot_queue/STATUS.json`、`units/repair42/runs/sparse_language/train/PROGRESS.json`、同目录`EVAL_PROGRESS.json`及`transfer/EVAL_PROGRESS.json`。
- 第一阶段结果`repair42_ANALYSIS.json`；全阶段终态`COMPLETION.json`。第一阶段不通过即结束，不自动长训或叠新损失。

第一阶段仅64新更新、2577新生成，预计获卡后30–35分钟，排队/IO另计。CPU准入和首步通过不作为能力收益。order0/200仅按已登记门槛有条件推进，是固定SFT42初始的数据顺序复现，非独立底座三seed。
