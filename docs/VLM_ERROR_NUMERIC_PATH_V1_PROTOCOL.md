# 临界token数值路径隔离 v1：固定单例，只读

2026-10-20登记。上一决策追踪实际发现：DPO32的the pointe（qid20675）在原训练batch8 teacher下，e对EOS的logit差为+.067114；原batch4真实cached greedy却为−.143963并提前结束。两次前缀相同但batch、长度、cache、数值执行路径不同，不能直接归因框架缺陷。

固定该单例和已保存DPO32适配器、原first-exposure四chosen+四rejected微批、图像和相同focal prefix；它是事后机制诊断，不是模型收益或代表性抽样。

原BF16执行与从相同BF16底座数值上转FP32的执行，各做full8 teacher、截断到focal预测位置的prefix8、相同padding/像素子集的prefix4、同prefix4的generate1四条件。完整序列/截断序列的同一focal prefix RoPE位置必须相同。全FP32执行同时可能改变SDPA内核选择，不能仅归于存储dtype。

读取原FP32 head前的focal hidden，用同一FP32 dot(weight[e]-weight[EOS])比较，以控制head矩阵形状；另报实际输出头的margin和hidden变化。prefix4生成仅1token，不加GT后续；其他行保留数值批形状，不作为新QA评测。原teacher复现值与上一记录单列。

0optimizer；FP32转换仅进程内、无模型保存，原权重/LoRA和全部历史源不变。LoRA哈希前后相同，不能把该单例路径变化称泛化提分、FP32训练成功或唯一失败原因。共享3×15秒空闲/UUID锁/40GiB规则、不终止其他任务，每worker300秒cap。

