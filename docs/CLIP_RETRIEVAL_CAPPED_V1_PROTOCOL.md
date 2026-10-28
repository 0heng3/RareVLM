# 曝光受控的困难负例＋冻结教师蒸馏 v1

状态：预注册协议，指标以对应输出中的 STATUS/ANALYSIS 为准。

## 问题与固定条件

原 CE 混合困难负例测试 T2I 比随机负例低 2.153pp；最常用 1% 图承担约13.53%负例曝光，随机约1.64%。这是集中曝光的证据，未证明语义误负或其因果作用。已有随机负例＋KL 的三checkpoint作为强基线，dev T2I69.907±.343/I2T84.733±.404。

保持完整28000训练图/140000caption、冻结CLIP-B/32 FP32缓存、双侧132480参数adapter、32anchor/64额外负图、温度.07、AdamW lr.001、wd.0001、clip1、lambda1、10epoch/8750更新、Final@10与seeds0/42/200。新模型从与旧基线相同初始化训练，并非在旧蒸馏checkpoint上续训。直接调用旧contrastive_distill_loss，不改CE或KL。

## 三组及变化因素

| 组 | 额外负图 | 新训练 |
| --- | --- | --- |
| A distill | 64随机 | 0，绑定已有3checkpoint及两划分预测 |
| B mixed_distill | 32hard＋32随机 | 3 |
| C capped_distill | 同B，但每图每epoch最多8次hard入池 | 3 |

分数保持原batch-max：32anchor的冻结五caption均值向量对全部训练图相似度取最大，固定图索引处理并列。排除anchor、同源图SHA及已知相同规范化caption关联。未排除全部语义误负。C先排除本epoch hard计数已达8的图，再取最高32张；不足32时均匀补齐64总数并明确记录实际hard/random比例。随机部分不受hard上限约束，可再次选中曾达hard上限的合法图。总negative曝光仍完整报告，不能称每图最多8次所有负例。

锚图顺序、初始化、更新数及正样本曝光逐seed对齐；负例身份有意不同。C上限未触及时与B选择严格一致；改hard池时随机池合法范围和教师分布会连带变化，不宣称只有重复次数或单独KL梯度发生变化。checkpoint每250步保存模型、optimizer、计数矩阵、采样/anchor摘要；原GPU UUID与来源锁匹配才可恢复。

## 统计与门控

完成六训练/六新dev格，复用三个A dev格，每格1000图/5000caption全图库。主候选固定C，B作为机制对照，不据结果切换候选。两主端点各单侧alpha=.025、6000次配对图像整簇bootstrap，先平均三个固定checkpoint输出；seed样本SD单列，不把seed×image或5caption当独立样本。

1. C−A T2I R1均值至少+1pp，校正后的单侧下界>0。
2. C−A I2T R1单侧下界>−.5pp。
3. 至少2seed T2I正方向，两方向nDCG10点值至少−.5pp。

B−A及C−B的双向区间为点式探索，不能宣称全家族校正。全部dev门槛通过才完整test评估两个固定新arm各3格，复用A三格；否则不追加test，不调cap/lambda/epoch或加seed。test历史已曝光、预训练重叠未知，不称盲测。假设来自既往dev/test观察，统计不能消除这种探索性。

## 教学与验收

保存真实32anchor/96图/480caption输入合同、首个cap排除事件、两类曝光计数及全dev Top10。学习页展示真实训练照片、分数与前后计数、真实改进/退化检索例；缩略图仅展示，不冒充224编码输入或attention。预检包括原B采样逐项一致、C触顶排除/完整预算、状态恢复、真实CPU loss/有限梯度和baseline初始化。最终从Top10独立重算Recall/nDCG并核验来源、旧科学文件和正式简历锁。

GPU使用已有队列准入：3次15秒空闲采样、共享UUID锁、无抢占，按空闲卡动态接续；对时间保留竞争/查询异常记录。只更新独立实验和项目学习入口，正式简历不变，不启用CI，不自动push。

参考现有固定OpenCLIP DistillClipLoss源码与[Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224)的误负风险分析；本轮只检验曝光策略，不声称复现该论文算法。
