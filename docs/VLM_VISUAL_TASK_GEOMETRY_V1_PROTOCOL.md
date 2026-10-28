# 视觉CE+DPO分源梯度及实际Adam位移诊断 v1

本轮仅判断当前视觉适配是否存在值得处理的回放冲突，不训练新模型。局部logit教师58/128失败、旧语言回放投影0/3及后层rank分支保持关闭，不恢复旧训练。

旧projection_v1测量旧640 QA/语言q-v；本轮固定较强历史visual_preference_v1的视觉qkv LoRA（60因子、1,843,200参数）、新128审核困难题+128原生回放+128 IIIT回放及CE+DPO目标。其可训练子空间、来源与目标不同，尚无该分源梯度/实际位移记录。不是以本轮重命名旧实验，不重查已有mask/加载/首步验收。

## 固定输入与预算

仅检查历史ce_dpo训练checkpoint8、16、24，各自下一update9、17、25；不选择更有利点。直接复用原MANIFEST/CPU_INPUTS/SCHEDULE及同batch reference CACHE，每点原4微批/16 chosen+8 rejected，逐条CE/DPO与原INPUT_TRACES中对应记录比较≤1e-5。

恢复各点的原FP32视觉参数、完整Adam状态、原LR调度；保持BF16视觉/FP32语言与LoRA、dropout0、TF32off、同batch/mask/EOS。语言与底座冻结。将目标分为hard CE、hard DPO、native replay CE、OCR replay CE，按原sum/16累计真实参数梯度，和为实际总梯度。偏好两项需独立autograd.grad，额外反传成本披露；不改loss权重。

clip1后执行原AdamW一步，记录实际FP32参数位移δ，以及各源g·δ和cos(g,δ)。再对同训练微批只前向，测量实际CE/DPO变化；局部方向预测不替代有限步读数。每点仅1次丢弃更新，之后恢复该checkpoint并校验LoRA哈希；最后恢复共同initial，销毁诊断模型。**共3丢弃更新、0正式更新、0新生成、0新adapter**。不用评测集或教师crop候选。

单卡通过共享空卡队列（零compute、40GiB、<10%、3×15s、UUID锁）；活动≤300秒、allocated≤65GiB。不抢占其他任务。共72序列梯度前向、72同批更新后前向；分开CE/DPO额外6次偏好batch反传、48批轨迹（其中24 chosen），记录进度与时间。原数据曾被SFT看到且历史训练QA，本轮不是新泛化证据。

## 事前机制门槛

每点同时要求：

1. cos(g_hardCE+hardDPO, δAdam)<−0.01；cos(g_nativeReplay+OCRReplay, δAdam)>0.01。
2. 同批实测hard CE+DPO均值下降≥0.001，replay CE均值上升≥0.001。

至少2/3点通过才报告该局部机制有信号；否则关闭本视觉回放冲突假设，不补点/改阈值/恢复旧投影。通过也不自动训练PCGrad或投影Adam；需要单独同数据/同预算/同监督权重对照并评测完整1617及全部长尾/保持门槛。任何方向、局部loss下降或丢弃step均不算能力提升，完整强基线/三seed/外部目标仍未完成。

## 算法学习依据

[PCGrad原论文](https://arxiv.org/abs/2001.06782)与[作者实现](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)逐任务求梯度，对负内积项执行`g_i -= (g_i·g_j)/||g_j||²*g_j`，随后交给优化器。该实现的原始梯度冲突不保证本地Adam实际位移对回放不利，所以本轮同时读δ与实测loss，不直接套用PCGrad。

[A-GEM原论文](https://arxiv.org/abs/1812.00420)用记忆任务约束新任务更新；旧本地QP是该思想的Adam位移适配，并非原创算法或标准A-GEM复现。本轮只建立是否有必要协调的证据。
