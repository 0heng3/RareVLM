# 最难词元先导：逐词元数值实现修正

2026-10-24。继承[V1科学协议](VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md)的全部数据、最大词元目标、32步预算、线性LR前缀与门槛。V1固定模型诊断完整通过：59错误中20词均值CE<=1但正文目标仍非首选，最难/平均NLL中位比2.967116，错误词EOS CE中位.000450629。诊断不重跑。

V1候选第一次optimizer.step前，新的`cross_entropy(logits.transpose(1,2))`路径与原`log_softmax(last_dimension)+gather`的均值误差超出1e-6而停止；没有FIRST_UPDATE/LATEST/adapter，也没有完成任何optimizer更新。该事实属于新增数值实现未满足既有合同，不是模型质量负结果，不能归因旧数据或底座。原源码/PLAN/日志/数据留在V1命名空间，运行中的mean32继续，不终止或重跑。

V2只将每词元NLL实现改成原序列logps所用的最后维`log_softmax+gather`，再作mask和mean/max聚合。保留原1e-6实测检查，不放宽阈值。第一次实际训练forward复现两种实现的差值，记录同一logits下的新旧数学一致性；0额外模型forward。复用同一局部CPU数值验证；新actual FIRST_UPDATE需沿用既有公式exact、有效有限梯度和输入SHA才可继续。

V2独立代码/PLAN/输出，DATA/CPU/SCHEDULE与V1逐字节相同，fresh原SFT、同seed/参数/目标/32更新，原目标没有改变。mean32的固定权重与960次新生成由V1进程完成后完整复用；记录其完成标记、输入锁、五队列预测SHA，hard128本就复用旧step32，不重新生成或评分。V2只新增候选32更新/1088生成，总研究新生成仍2048，未完成的失败首forward另记。

V2不调用V1控制器，不混接其失败队列锁；只用新的共享GPU候选队列，mean32完成后收取封闭预测并核对两命名空间DATA/CPU/SCHEDULE完全一致，复用原分析函数。准入/预算/32步推进与原能力门槛均不改，0自动长训/种子。结果和教学页区分V1成功诊断、V1未完成首步、V2候选质量。
