# 完整多模态偏好数据构建：冻结协议 v1

2026-10-15。父审计512图／2560回答完成：85个规则偏好对（16.60%），149组非零奖励（29.10%）。8个最早GPU合同对局部助手视觉审查发现Wi-Fi/wifi等价、正确核心长引用及人类答案分歧；新增保守筛选仅保留40候选（7.81%），不修改原85统计、VQA评分、旧门槛或模型。40不是全量人工确认标签，排除45不代表45全部语义错误。

## 固定数据与采样

从原32985 train_all按父协议SHA顺序每像素图一题，固定10000不同训练图。已有dev/final ID与像素／文件SHA排除；不使用评测答案或按错误挑题。前512与父审计重叠，另行新生成，不当10512独立图像。普通CE SFT seed42 adapter、原BF16/1024、单词短语后缀、greedy1＋temperature1/top_p.95/top_k0 sample4、最多32新token不变，共计划50000实际生成。

复用已冻结`vlm_alignment_signal.generate`，本输出、配置、输入／硬件／源码锁独立；完整每题五输出断点恢复，逐qid固定sampling seed。关闭重复DPO梯度预检，0optimizer更新／0新adapter，不称正式DPO或GRPO训练。沿用三次15秒／零compute／40GiB／UUID共享锁，不抢占。

## 新质量筛选与准入边界

初候选仍按原采样分数chosen>=.9、rejected=0、非空和规范化不同，选点规则不改。随后在独立`quality.py`做训练数据筛选：chosen至少6/10规范化参考支持；排除至少3个unanswerable或chosen为拒答的歧义；排除去标点空白后与任一参考相同的rejected，及包含至少两字符参考核心的rejected。

这是保守精度优先筛选，含正确核心的否定／错误长句也可能被排除，仍可能漏掉同义表达。不能将筛选叫全量语义验证，不能用它替代或改写正式VQA计分。保存原全部采样／raw候选／排除理由／保守候选，分别统计，全部参考仅生成后评分。

本阶段只构建完整训练候选，`training_allowed=false`直至后续独立数据语义审核和正式训练协议。下一项拟以同一初始SFT、固定QA/正答案、Final固定更新预算比较继续SFT与DPO，额外reject/reference token和算力单列；参考必须静态初始policy缓存／模型，不能每更新后用current_actor.detach。效果需独立dev质量与能力保持门控，源代码／配置另立版本，未通过不救门控。

## 核验与学习

父8数学／mask／泄漏合同、8真实GPU前后向和4新增质量过滤合同已通过；本阶段只追加完整数据，旧结果保持。运行状态以STATUS／role PROGRESS／最终analysis为准，启动时尚无10000图结果或训练收益。学习材料解释词面score与语义偏好、reference静态约束和DPO/GRPO区别；真实原图展示不冒充attention或processor输入。

正式简历／论文／Agent保持，不启CI、不自动push、不把助手执行当本人掌握／独立贡献。[父审计协议](VLM_ALIGNMENT_SIGNAL_V1_PROTOCOL.md)与[JD分层统计](POSTTRAIN_JD_DECISION_20261015.md)保留。
