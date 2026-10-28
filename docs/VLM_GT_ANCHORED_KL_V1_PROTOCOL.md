# 真实答案监督锚定与分布保持：GT-CE+KL v1

2026-10-16事前登记。前置：stage34完整负结果与独立1959预测验收已完成。源码启发为已固定commit的LlamaFactory pref_ftx和TRL多loss SFT项；本实验不含偏好/DPO损失，不称复现RPO/MPO，不属于RL。

## 问题与唯一变化

既有generated-prefix KL在Object有收益，但rare/frequent点保护失败；单独训练visual merger没有Object提升。这些观察不证明唯一失败机制。本轮检验在**相同GT-prefix**下增加监督答案CE是否有助于同时实现Object收益与Text保持。

唯一候选 `gt_ce_kl32`，新鲜且不可选替的单seed控制 `gt_kl32`。两组均从同一SFT42开始，训练语言q/v rank8/alpha16 LoRA，144 FP32张量、1843200可训练元素。底座及视觉BF16冻结。独立固定SFT参考保留其LoRA，不使用disable_adapter替代。禁止用历史cached-GT或generated-prefix候选充作同输入控制。

每题先对assistant监督token取平均，再对问题取平均：

`L = mean_QA_object CE + mean_QA_replay KL(p_initial || p_actor) + gamma * mean_QA_replay CE_GT`

控制gamma=0，候选gamma=1，KL系数1、温度1，全151936词表求和。**CE_GT从相同actor log-probabilities计算并反向传播**，不detach，不额外增加actor/reference forward。GT labels只放在GT-prefix位置。每题等权而非全batch token等权；EOS和实际模板后缀保持原assistant mask。对比两臂只改变CE项；不宣称其他历史条件组成完整因子实验或实测等壁钟成本。

## 固定训练与输入

2888 object +5792不同Text训练回放/每种子，181完整Final更新，1epoch，object每步16（最后8）/replay32，micro4，lr5e-6、warmup.1、weight_decay0、clip1、gradient checkpointing。视觉上限1024、max_seq2048、无截断。共享初始canonical6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7；种子只影响训练顺序/回放选择。seed42严格引用已接受stage31 gt_ce32的8680完整输入合同和计划，不重新生成教师回答，不读取大型teacher bank。0/200用相同选择算法及真实GT经CPU编码，新增教师回答始终0。所有源、权重、配置、数据及完整计划锁定；检查动态时间只作遥测，不加入稳定身份。

## 工程准入

CPU合同覆盖真实consumer、CE梯度、归一、完整12字段legacy投影、静态/运行时参考声明与实测证据、Final汇总实际调用路径；通过后真实prepare，随后共享GPU队列准入。预检独立拆分Object/GTCE/KL梯度，初始KL-only每题abs<=1e-5、KL/Object梯度比<=.01；候选联合回放梯度非零合法。一次丢弃联合更新后检查正KL恢复梯度，精确恢复参数/RNG/buffers/rope/optimizer隔离。参考无梯度、无共享参数storage，构建不改变actor RNG。显存allocated/reserved实测，65GiB硬cap，不降batch或静默fallback。

共享队列每张卡3次15秒观测、>=40GiB空闲、util<10%、零compute、UUID锁、同role恢复同卡；不终止已有任务。双空闲A100可并行两臂，各自独立模型与参考。每25更新保存完整master/Adam/RNG/实际输入，181 Final唯一评测点，不依据dev选checkpoint。独立训练验收与全新预测评分在锁外执行，不修改冻结科学代码。

## 原完整成功范围不变

pilot42完整Object342/143图与Text1617/1024图；Object至少+1pp且单侧98.75%图像簇区间下界>0；Text整体下界>-.5pp；recall点值不降、specificity>=-.5pp、固定class-balanced256点差>0；Text rare342与frequent417点值均不降。6000次图像簇bootstrap/seed20261015。4主要终点alpha=.05/4，不代表所有历史实验FWER，点保护不代表切片NI。

只有唯一候选8/8通过才新训练0/200，三种子每题均值通过原8门槛、每种子Object正向与Text点差>=-.5；再每seed完整external4372（均衡3438）均值至少+.5pp/下界>0/逐seed正/原F1-recall-specificity-balanced保护；再每seed全POPE9000（三类别各3000、500图）至少+.5pp/下界>0/逐seed正/原F1-recall-specificity与逐类别保护。最后完整独立root验收。自动通过仅pending review。任一候选失败停止其扩展并保留负结果，用户真实提升目标未完成。公开数据预训练重叠未知，禁止宣称未见。

## 学习与成果

学习页连接上述公式、实际张量/源码、固定原图、GT/模型回答、分项loss/梯度/计时及完整实测。准备/工程成功不当算法收益；不据事后病例修改训练。当前CV、PDF、论文、Agent、CI和旧科学结果不改，不自动push。计划数据或预测不能当已完成数字。
