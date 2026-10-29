# 固定Final的来源保持验证：全部完成，来源泛化信号未过门槛

2026-10-28 17:28队列完整结束，8个分片exit0/0争用，共1024新生成+1024 GT前向、0更新。source exact起点195/256、旧CE196、旧DPO194、新SFT197、新DPO196；四个预定比较校正98.75%区间全部跨0。不能称稳定来源泛化提升，也不能称来源内DPO普遍损伤；原native1617主候选失败保持。

完整结果、NLL与真实原图解释已归档；停止本固定验证，不重跑或扩池。下一独立方向为完整7B教师信号准入，不自动学生训练。[完整结果](VLM_STVQA_HOLDOUT_TRANSFER_V1_RESULTS.md) · [便携指标](../results/vlm_stvqa_holdout_transfer_v1_metrics.json) · [新教师协议](VLM_FULL_TEACHER_SIGNAL_V1_PROTOCOL.md)。

以下为提交与排队时的历史快照，不代表当前仍在等待。

2026-10-28 16:52:34（北京时间）提交controller783452，start_ticks528765628。CPU准备通过：原固定256保持QA/256物理图/64批，旧新训练图文件与像素均0交叉；四个Final32实际路径/权重与原起点预测绑定。未重新编码旧输入、训练或修改原结果。

16:53队列为waiting_resources、8项均waiting_gpu、0/1024；两卡有其他compute任务、各约60GiB占用，未终止其他任务。此文为提交快照，实时以queue/STATUS和PROGRESS为准。获得两张空卡后约10–15分钟，单卡约20–30分钟；排队无法预估。

本轮只做固定模型评测：旧裁词CE/DPO、新整图CE/DPO各256生成+256 GT前向，总1024+1024，0optimizer/0新adapter。起点256+256精确复用。沿原循环/真实EOS/FP32 LoRA与language、1024视觉预算、原64批/两个shard；共享队列动态接续，不抢占。

要回答：新整图正监督的收益是否扩展到未训练的同来源场景；DPO是否也削弱来源内泛化，还是主要发生原任务取舍。不是继续为原失败配置调参；原DPO停止与完整强基线/三seed/外部目标保持。来源exact不与原native VQA混列，起点分数已知，非新盲测或官方ANLS。

最终完整四模型后报告四个预定exact比较的条件95%/四比较98.75%图像簇区间，以及正文/EOS NLL、删失。任何部分分数不用于挑权重或启动训练，完成后无自动扩训。当前权重均冻结，不宣称新增模型能力提升。

[固定协议](VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md) · [绑定与分析代码](../tools/run_stvqa_holdout_transfer_v1.py) · [实际原图与源码学习页](../../../outputs/vlm_stvqa_holdout_transfer_v1/来源保持_原理代码与实际回答.html) · [实时队列](../../../outputs/vlm_stvqa_holdout_transfer_v1/queue/STATUS.json)
