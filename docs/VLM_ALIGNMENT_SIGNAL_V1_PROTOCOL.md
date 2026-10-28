# 多模态偏好与奖励信号审计：冻结协议 v1

2026-10-15。根据训练／对齐岗位要求，先验证DPO偏好供给与GRPO奖励差异，不同时铺开RLHF/PPO、DPO、GRPO。SFT已有完整研究；新增审计不能记为已完成偏好训练／强化学习。JD计数与选择见独立`POSTTRAIN_JD_DECISION_20261015.md`。

## 数据与固定输入

复用完整train_all 32985QA，按固定SHA256规则每图取一题，再固定选512张不同训练图。排除已有dev/final图ID及文件／像素哈希，不依据模型错误或答案挑题。固定普通CE seed42最终q/v rank8 adapter，原权重BF16、整图视觉上限1024、原单词／短语后缀、最多32新token、不截断。参考答案和target只在生成后评分；生成输入仅图像和问题。视觉上限不代表每题实际token，逐题保存grid、长度和像素／inputIDs摘要。

两role各256图，沿用三次15秒空闲、>=40GiB、<10%利用率、零compute与共享UUID锁准入，不抢占。每题greedy一次＋固定temperature1/top_p.95/top_k0四次采样，共2560份实际生成。四采样通过同图同问题重复batch4实现，随机seed按qid固定；完整一题五预测写一个持久JSONL记录，断点只修复尾部不完整行／batch。源、配置、数据、权重、硬件、包锁不一致拒绝混用缓存。

## 两项数据供给指标

沿用冻结的十参考LOO VQA共识。DPO候选要求实际采样chosen得分>=.9，rejected得分=0，两个回答非空且规范化后不同；按得分／采样次序固定选一对。不构造随机错误答案、不使用greedy参考或target冒充模型采样。报告候选对比例、空回答、回答多样性、均分与greedy差异、实际target频次切片；这些是训练集诊断，不是泛化收益。

GRPO诊断 `A_i=(r_i-mean(r))/(std_population(r)+1e-8)`，全同奖励置0；报告非零奖励组比例、全正确／全错误／等分组。这里只实现奖励与优势统计，不实现在线rollout更新、clipped policy loss或GRPO训练；现代TRL的归一化／KL默认变化另标，不宣称复现其全栈。

DPO供给比例>=10%才建议扩充全train10k偏好数据；GRPO非零组>=10%仅为供给筛选，不足以直接启动RL。两阈值在运行前固定，不据结果调温度、组大小、题目或门槛；未通过不做audit小样本训练。通过后仍先检查语义偏好，再独立预注册完整数据、继续SFT强对照与DPO；不得以规则分数直接认定人类偏好正确。本轮无dev/test预测与参数更新。

## DPO数学与真实梯度合同

实现标准sigmoid DPO：`L=-logsigmoid(beta*((logp_chosen-logp_rejected)-(ref_chosen-ref_rejected)))`，beta=.1。logp为assistant回答token概率之和（包含原mask所定义结束token），不按回答长度平均；prompt／视觉／padding token忽略。

每role对最早最多4个有效真实采样偏好对做前后向检查。仅暂时开启原1843200个语言LoRA参数梯度，模型eval、dropout关闭；reference由同一固定初始policy的sequence logp detach获得，保证该初始化点的数学参考。初始loss应为log2、chosen logp导数为−beta/2、rejected为+beta/2，teacher无梯度；保存实际logp、mask数量、LoRA梯度范数、显存、输入合同。**不执行optimizer.step，不保存新adapter。** 这验证真实Qwen输入／mask／DPO梯度通路，不是优化、能力提升或独立reference模型训练流程的证明。

CPU合同检查独立逐token log概率、头部切片／ignore mask、DPO梯度符号／reference stop-grad、GRPO零组／平移缩放、偏好歧义与生成输入边界。完成后独立从五条原回答重评分全部512图，保护历史源码／配置／结果／权重和正式简历。

## 学习与边界

学习页用真实同图五回答、十参考答案、score／advantage／chosen-rejected解释SFT→偏好优化和组内奖励。来源原图缩略图不冒充模型输入或attention；语义审查由助手做局部观察，不能冒充用户掌握或独立人审标签。固定policy／一次采样预算的比例不代表市场频率、三训练seed稳定性或未来RL奖励质量。

原理参考[TRL DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer)和[TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)，上游源码独立固定SHA；本项目使用现有Qwen／processor／mask合同，不安装或迁移新TRL，不称完整官方框架复现。正式简历／论文／Agent保持，不使用CI，不自动推Git。
