# 当前训练失败：代码、数据与优化的实测分因

本轮只读定位已完成：3个现有模型各32题真实greedy重放和64条原训练chosen/rejected序列前向，共96生成+192 teacher序列、0更新。三模型各32/32逐字复现原预测；初始teacher与原reference的32题打分最大绝对残差0。没有修改旧源码、数据标签、checkpoint或评测分数。

结论：当前不能继续笼统归为“数据问题”。已证实的是训练目标与实际生成决策之间仍有差距，另有负例时效性、字符串到token路径、监督末尾和数值路径等具体问题。当前未发现本轮token shift、batch归一化或reference实现错误；未证明唯一架构/容量瓶颈。

## 1. 正确路径还没有在真实greedy中胜出

DPO32还有23题未满分；23题的相对reference margin均改善，22题正确答案整段logP仍低于固定rejected。相对改善没有跨过绝对顺序，更不保证逐步argmax选择正确。相同GT前缀下的teacher读数与真实生成路径要分别报告。

|完整目标|DPO实际生成|真实首次分词路径分歧处的下一token概率|
|---|---|---|
|techne|technie|'tech': 5.21% / 'techn': 93.67%|
|sol|sor|'sol': 12.49% / 's': 38.01%|
|no turn|no|' turn': 21.47% / '<|im_end|>': 36.07%|
|the pointe|the point|'e': 46.06% / '<|im_end|>': 53.19%|
|gomes|cincinnati reds|'g': 2.35% / 'c': 21.67%|
|fghij|fghij|已满分；GT与实际路径分词不同，不能把BPE差异当语义错误|

这些是单个token/前缀路径的条件概率，不是整个单词或整段答案概率。例如Techne监督路径首段tech仅5.21%，真实分支techn为93.67%，仍会走向technie；不能因平均CE约.70就称已经学好。

训练代码src/vlm_error_supervision_sft_v1/train.py:75–92使用完整序列logP除assistant token数，再按16个chosen汇总；4个micro×4题与/16匹配，没有重复除batch。src/vlm_alignment_signal/loss.py:21–24为标准相对log-ratio sigmoid DPO，reference固定、无梯度，不是绝对概率排名约束。beta=.1时，在同起点处DPO对chosen logP的导数为−.05，而CE为−1/n；lambda=1不意味着两项梯度等强。这是标量logP导数，不是参数梯度或Adam贡献。

## 2. 全答案平均CE掩盖了内容仍难；末尾换行是次要契约问题

src/vlm_sft/common.py:95–101只mask prompt，assistant模板尾部全部监督；32题标签都包含答案 + <|im_end|> + 换行。真实生成在EOS停止，不生成EOS之后换行。

post-EOS换行平均占分母21.0193%。DPO Final平均完整CE=.537144，单独答案内容CE=.944622；内容token的负对数似然数值占99.21%左右。post-EOS CE在本次FP32数值计算中全部为0，因此不能声称“格式loss吞掉了训练”。去掉该换行主要改变各题缩放，不能承诺解决语义或泛化。

如后续修正边界，应保留EOS并独立登记。不能只把末尾换行labels改为−100而沿用旧keep=max(valid_count)+1：src/vlm_alignment_signal/loss.py:15的裁尾依赖有效标签连续到序列末尾，增加尾部ignore后可能丢首个目标。新实现应按首个有效标签位置保留其前一个hidden state，并用真实输入验证与完整logits loss一致；本轮未修改旧loss或训练。

## 3. 固定负例不完全对应当前greedy，重分词也会改变轨迹

32条registered rejected中29条与固定32题重测的初始答案相同；其余包括1题已正确、1个第三种错拼、1个格式差异。因此不是全部错误都来自负例失效，但少量反馈已不对准当前主错误。

在29条负文本相同案例中，2条实际生成content token IDs与训练重编码不同：walter为实际w+alter、canonical wal+ter；fghi为实际fg+hi、canonical f+ghi。DPO满分fghij也沿实际fg+hi+j生成，而监督GT是f+gh+ij。

src/vlm_probe/evaluate.py中的decode(...).strip()只保存字符串；后续prepare与common.samples重新encode rejected_target，无法保证保留真实生成token轨迹。这是offline文本训练契约的局限，不是所有DPO实现都算错；若要优化实际rollout，须保留原始response IDs、prompt/图像绑定、终止位置和生成策略。本轮新追踪已补真实IDs，不把解码字符串重新分词冒充原轨迹。

## 4. 训练/推理条件下有临界符号翻转，数值来源尚待隔离

初始coin/number两题在原batch8 teacher的正确-vs实际错token差为正，而batch4真实cached greedy仍选错；DPO the pointe的teacher e−EOS为+.067114，真实greedy为−.143963。

同一GT前缀比较仍同时改变了batch、序列长度、cache及数值执行路径，现阶段不能归因唯一BF16或框架bug。FP32 LoRA/head也不代表全网FP32：当前底座BF16，本地PEFT Linear.forward最后转回base result dtype。数值差足以影响临界题，但不能解释全部大负gap错误。

固定the pointe的8条件只读隔离已另登记并排队，比较full8/prefix8/prefix4/generate1及原BF16/全FP32执行。两卡当前有其他任务，遵守空闲准入、不抢占；尚未取得结果，不能先宣布FP32修好。FP32也会改变可能的SDPA内核，且该单例不代表泛化。

## 数据还是代码：当前证据分级

- 确认的数据问题：上一批52候选中18个不可靠偏好；当前有少量负例/真实轨迹不对应，32种错误覆盖仍有限。不能将这些计数外推全数据错标率。
- 确认的优化现象：相对margin改善但实际错误不翻转，内容CE仍偏高；当前32步仅改善少数训练题。不能凭此唯一判定该加LR、加步数或解冻视觉。
- 确认的实现契约限制：平均CE包含推理终止之后换行；文本重编码不保留真实rollout分词路径；teacher和生成路径对临界样例结论不同。
- 当前没有证据的结论：通用loss错位、实际未更新、加载同一adapter、唯一底座架构缺陷，或换GRPO必然改善。

下一算法行动应首先让可靠错例的实际生成决策得到可控改善，再测独立迁移。若考虑on-policy优化，需先确认可采样到正确轨迹、有可靠正负奖励，训练直接使用实际token IDs；不直接从负Final长训或把本次诊断当能力提升。

[执行前协议](VLM_ERROR_DECISION_TRACE_V1_PROTOCOL.md) · [全部实测摘要](LOCAL_ARTIFACTS.md) · [分因统计](LOCAL_ARTIFACTS.md) · [分词轨迹证据](LOCAL_ARTIFACTS.md) · [逐图、token与概率讲解](LOCAL_ARTIFACTS.md) · [单例隔离协议](VLM_ERROR_NUMERIC_PATH_V1_PROTOCOL.md)
