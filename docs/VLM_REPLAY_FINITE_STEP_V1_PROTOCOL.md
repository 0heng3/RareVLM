# 回放与纠错的有限步CE核验

2026-10-22事前登记，独立于已经完成并冻结的`vlm_replay_projection_v1`。后者在训练checkpoint16/32/48的next批次，hard/replay对真实Adam位移的cos均为负，0/3达到冲突门槛、投影修正0，因此不启动正式投影训练；不得改原门槛或据此排除所有数值/全局遗忘问题。

本次只补充必要的有限步边界：对相同3个历史checkpoint及next训练批次17/33/49，恢复各自原FP32参数/Adam动量，按原micro4×4和L_base累计梯度，clip1、原线性LR、一次真实Adam更新，比较**相同4个padded输入帧**的更新前后hard/replay各8题内容+真实EOS逐QA平均CE。更新前读数来自原梯度前向，更新后额外一次无梯度前向；保持model.train、visual.eval、dropout0、BF16底座/冻结FP32头/use_cache=False，读出和labels由冻结函数复用。

记录实际CE差及`2*g_source·δ`一阶预测（source梯度由Σ8CE/16定义，因此恢复每源QA均值需乘2），不把raw点积混成NLL变化。新增3次丢弃optimizer步、48QA梯度前向+48QA额外前向、0生成/正式更新/新adapter；最后恢复原SFT哈希并销毁模型。没有任何条件长训或按结果选系数，180秒活动上限/65GiB/共享GPU空闲准入，不修改前项源码/结果/PLAN/门槛。

两源实际CE均下降仅支持这些批次可被共同优化；若回放CE与一阶异号，仅记录有限步/数值近似差异，不能单因归为BF16框架bug、全局遗忘或证明某种投影有效。独立图像VQA才是能力指标；本次不声称提分、不读正式benchmark，不改CV。
