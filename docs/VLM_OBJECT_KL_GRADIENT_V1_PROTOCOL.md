# 固定训练检查点的物体／KL梯度诊断 v1

2026-10-15，在任何本诊断GPU前后向之前登记。源研究`vlm_object_kl_v1`已经完整结束并独立复核：物体342QA从95.614035%升至97.953216%（+2.339181pp，单侧98.75%下界+.591716），Text1617QA从87.111936%降至86.796537%（−.315399pp，下界−.648809）；rare/frequent点保护失败。保留原`completed_negative`、所有数据／模型／评分／门槛，不追加该候选种子或正式确认。用户真实提升目标继续active；诊断通过不代表该目标完成。

## 问题与解释边界

测量同一组1843200个FP32 LoRA参数上，物体CE梯度与固定初始模型全词表KL梯度的强度、夹角，以及已有Adam状态下的局部更新方向。原在线mean KL约.000203不能代替梯度强度，也不能证明生成能力保持。

`g_object = grad(mean_QA mean_assistant CE_object)`；`g_KL = grad(mean_QA mean_assistant KL(p_initial || p_actor))`，KL先对完整151936词表求和。T=1、lambda=1均继承原研究，hard replay CE仅detach诊断；不修改原loss或增大lambda。

记录两范数、`r=||g_KL||/||g_object||`、内积和cosine，保留原始数值。零范数cosine为null；初始数值控制提供同批KL梯度底噪。运行前固定解释guard为`max(10 * measured_initial_KL_gradient_norm, 1e-10)`，KL范数不超过guard时`interpretable_cosine=null`，原始非零cosine与范数仍保留；这是工程数值规则，不是统计显著性阈值。接近底噪的方向不得解释为冲突。固定三批的正负夹角不是总体概率或遗忘因果，不能将三点当独立重复检验。

## 固定检查点与训练输入

选择源研究的首个、中间、最后一个非Final保存点，不根据新分数或梯度挑点：

| 已完成检查点 | 下一个实际block（0起始） | 对应更新 | object／replay |
| ---: | ---: | ---: | ---: |
| 25 | 25 | 26 | 16／8 |
| 100 | 100 | 101 | 16／8 |
| 175 | 175 | 176 | 16／8 |

直接绑定源`SCHEDULE.json`、checkpoint实际state SHA／lineage／FP32 master／Adam／RNG及完整训练输入合同。不能用Final@181回看某批冒充当时历史梯度。本诊断是离线控制重算，不冒充原训练中保存的逐源梯度日志。

只使用原2888物体／32985回放训练池中的上述既定索引及GT前缀；不使用Text dev、外部确认或POPE的题目／答案计算梯度、选择样本或调参。保留原micro4、object16、replay8及独立逐QA均值，实际padding／IDs／mask／labels／pixels／grid必须与原相应teacher／actor输入一致。

## 固定教师、数值控制与无更新

继承原BF16基座／视觉冻结、q/v rank8 FP32 master、dropout0、checkpoint前向、SDPA、tf32关闭及处理器。每组只加载一个基础模型：先用完全相同的初始BF16舍入后FP32 LoRA，对本组各point的8条replay做初始parity，再装载相应checkpoint精确FP32 tensors进行分源前后向。

原teacher缓存保持GPU0生产者UUID和原完整FP32 logprob、原manifest及每批file/tensor/input SHA不变；新消费卡UUID独立记录。本诊断允许另一A100消费已验证的原缓存，不能改写原硬件绑定并冒称原研究恢复。新消费卡初始forward需对同一padded输入的既有GPU0 teacher分布验证`abs(KL)<=1e-5`且数值有限，记录差异。GPU0初始参照已存在，不再为此重复前向。

每点两源计算前恢复相同参数、RNG和本次可恢复的运行状态，分别清空梯度，采集完整FP32梯度后再清空。不建立优化器，不调用step，不保存新adapter，模型参数及原输入文件前后SHA必须不变。KL教师始终是原固定初始分布，不能取当前actor.detach充当教师。

新代码按组运行：`early_late=[25,175]`、`middle=[100]`。主诊断共3×(16+8)=72条训练QA前后向，初始消费卡控制共3×8=24条，总计**96条新QA前后向**（early_late64、middle32）；原缓存生产者24条参照为历史计算，不能再计为新调用。每组1次模型加载，共2次；0优化器更新、0生成评测。梯度检查点3组两源约44.24MB，另保存初始底噪梯度时总约66.36MB，实际字节和前后向／加载／CPU／壁钟成本单列。checkpoint重计算不当额外独立QA。

## Adam只读方向分析

读取各checkpoint实际`m/v/step/betas/eps/weight_decay`和下一步既定LR。复制张量后，在CPU按原顺序合并梯度、统一clip1，再计算下一Adam方向；不得修改源Adam或参数。比较“当前步加入KL”和“当前步省略KL”两种方向，并报告`g_object·delta_theta`、`g_KL·delta_theta`及方向差。

这是条件于同一历史m/v的单步近似。原始cosine不等于Adam实际方向；Taylor一阶预测不等于执行后的损失，也不是另一条完整训练轨迹。没有额外step或更新后forward，不声称复现了真实反事实效果。

## 准入、产物与后续

新配置`configs/vlm_object_kl_gradient_v1.json`，新模块`src/vlm_object_kl_gradient/`，新launcher`tools/run_object_kl_gradient_study.py`，输出`outputs/vlm_object_kl_gradient_v1`。新CPU数学和编排合同通过后，冻结新来源／配置／协议／原模型来源／三个checkpoint／schedule／teacher及所选批次，才允许GPU执行。原qualified训练数据及旧negative结果均只读继承，不以诊断绕过数据质量门槛。

共享`research_runtime.queue`按空闲40GiB、util<10%、零compute的三次15秒快照及UUID锁动态准入两组，不终止或抢占已有任务。allocated/reserved任一超过65GiB或OOM即工程失败，不静默降micro。两个组完成后纯CPU按原保存梯度独立复算汇总；所有状态、实际输入、各源梯度、初始parity、shadow方向及成本可追溯。观察超时不当进程终止，不盲目重启。

后续训练必须依据已核验结果另预注册；本协议不自动选lambda、执行更多epoch、变更底座或晋级正式评测。诊断没有heldout性能指标，不能写成性能提升、已解决遗忘、三种子稳定性、完整RLHF／GRPO或用户本人掌握。正式简历、论文、Agent保持，无CI或自动push；全部新文件与临时产物在private内。
