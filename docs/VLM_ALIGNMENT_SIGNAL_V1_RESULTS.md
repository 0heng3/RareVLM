# 多模态后训练：信号审计完成结果

[JD分层统计](POSTTRAIN_JD_DECISION_20261015.md) · [冻结协议](VLM_ALIGNMENT_SIGNAL_V1_PROTOCOL.md) · [便携指标](../results/vlm_alignment_signal_v1_metrics.json)

## 完成范围

完整train_all32985QA中按图SHA固定512不同训练图、每图一题；固定SFT seed42原BF16／1024。每题greedy1＋采样4，共2560真实回答。参考只生成后计分，旧dev/final ID／图像哈希排除。训练图已被SFT见过，不是泛化评测或4训练seed。

| 指标 | 结果 | 解释 |
| --- | ---: | --- |
| 原规则可用DPO对 | 85/512，16.60% | chosen≥.9、rejected=0且非空不同；只是词面共识偏好 |
| GRPO非零奖励组 | 149/512，29.10% | 固定policy一次采样的奖励差异，未做在线GRPO |
| 相同奖励组 | 363/512，70.90% | 奖励优势为零；KL／熵等项仍可能有梯度 |
| 全四次满分／全四次零分 | 317／29题 | 还有17题相同的中间分数 |
| 保守质量过滤后 | 40/512，7.81% | 45候选不符合新规则，不等于45全为错标 |
| 平均不同规范化采样回答 | 1.574 | 四回答共享模型／图像，不能当独立图像样本 |

greedy训练图共识88.164%，随机采样81.353%；差来自生成策略，0模型更新，不能称训练退化或提升。供应的Wilson95近似区间为原DPO13.63%—20.07%、奖励差异25.33%—33.18%，仅描述本固定模型／图像审计，不是就业市场比例或未来训练收益。

## 发现的质量问题

最早4对/role共8对做了实际GPU合同，局部助手看图：Wi-Fi/wifi语义等价却得分1/0；citizens与包含同核心的完整引用主要是回答跨度差；年份／无法判断的图存在人类答案分歧；品牌核心加额外文字/乱码与纯错误实体应分别分析。未做全85独立人审，不把这些例子的比例推广到全数据。
独立质量筛选要求chosen至少6/10规范化参考一致；排除拒答歧义、去标点空白等价和包含参考核心的rejected。原VQA评分、85候选、旧科学源码和门槛全部保持，4质量合同通过。保守规则仍可漏同义词／误排真错例，40仍是候选，不是真实人类偏好标注。

## 真实DPO数学与GPU工程检查

8真实chosen/rejected对使用原assistant mask求sequence logp，标准sigmoid DPO beta=.1；仅1843200个语言LoRA开启梯度。初始化reference由同固定初始forward detach，loss均.69314718≈log2；LoRA梯度L2范围.984—3.710且有限，基座／视觉无梯度，峰值allocated最高14.760GiB。**0optimizer.step、0新adapter，不是完成DPO训练或测得收益。**
这个reference detach写法只适用于本初始化点合同。实际DPO训练必须使用静态初始SFT reference模型／缓存；每步current_actor.detach会让参考随actor漂移，不能直接复制该合同作为训练器。prompt／视觉／padding不参与answer logp，logp用sum而非SFT的每题token均值。

8数学／mask／reference输入边界合同、4质量合同通过；2560输出全量重评分，272旧科学／模型权重／正式简历文件哈希保持。控制器649.68秒，首次启动至分析完成643.62秒；sampling allocated最高8.517GiB，不把单对梯度峰当完整训练batch性能。

## 决策与下一阶段

有必要补一项可信偏好数据与受控DPO研究，现有SFT训练已覆盖；不凭方向公告或词面奖励直接启动完整RLHF/PPO/GRPO。原供给门槛通过仅允许扩充数据；训练质量门槛仍false。下一项完整10000训练图／50000生成＋固定保守过滤，独立配置与输出；数据与语义准备后才另立继续SFT强对照／DPO训练及质量保持门控。
[完整数据构建协议](VLM_PREFERENCE_DATA_V1_PROTOCOL.md)保留计划与边界，启动不等于完成训练。正式简历／论文／Agent保持，不启CI、不自动push，个人贡献／掌握待确认。

真实图像／五回答／reward／advantage／DPO源码学习页：`/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_alignment_signal_v1/learning/学习页.html`。
