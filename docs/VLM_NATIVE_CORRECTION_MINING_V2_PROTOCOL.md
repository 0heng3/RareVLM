# OCR增强模型的真实问答纠错：V2筛选协议

2026-10-25，任何新预测前登记。回放修复完整1617为87.247990%，对initial87.328386%差−.080396pp/95%[−.835366,.650581]，对原sparse87.260359%差−.012369pp/区间跨0。未通过原推进，不增加其步数或seed。元数据收尾恢复完成、0重训/重生成，原失败日志保留；不能把算法负结果归咎该字段。

新假设：单词转写信号与正确native回放并未充分教会纠正真实问题条件下的答案。本轮从原train_all选择新的native问答错误监督，保留原问题、目标、答案和整图输入，不生成合成toy问题，不从dev错例补训练。最终看native1617提升与已有OCR收益保持；本阶段只建候选，0优化器更新，不冒充能力收益。

固定2048条：单词/多词各1024，目标归一化后≥8/10原标注共识、长度≤64字符/≤6词，排yes/no及unknown/unanswerable/unreadable，最多每target2、每图/解码像素1条。桶内固定qid哈希排序。排原dev/final与当前IIIT、回放、旧error-supervision矿工、严格native OCR1269池涉及图像/像素。原SFT曾见train_all，不称原SFT/预训练未见；该筛选未读新模型预测。不足2048停止、不降低筛选。

起点为已提升官方OCR的sparse_language Final64（复合SHA3af138…1a9cb87），不是原5d8模型。原30视觉B始终精确0，仅导出144个语言FP32因子至独立initial_adapter、原值逐张量保留；用原语言config及dropout0，1843200参数。CPU证明视觉B0，首GPU shard0对旧sparse原native512最前8条以相同batch4生成，completion IDs与prefix必须完全一致，否则停止，不借旧指标称新起点。8条是结构等价后的spot check，不称全量输出核验。新language hash与source composite hash均保留。

两片各1024，原BF16 base/FP32 head和LoRA、TF32off、greedy32/自然EOS、原整图1024视觉预算、batch4。源/数据/图像/环境先冻结，逐batch真实prefix SHA一致、前后模型hash不变，完整batch断点。每片1200秒/65GiB，遵守共享空闲3×15秒与UUID锁，已有任务不动。共2048矿工生成+8起点spot check，0训练。

错误候选须VQA≤.3、自然EOS、完整GT归一化文本不包含在预测内，固定SHA排序、每target≤2。最多审核前256原图：必须问题语义对应、完整答案可由画面支持、负答案明确错误；含糊、遮挡/标签问题或同样可接受的负例排除。至少128条/96target才准入独立后续训练，助手审核非专家真值，全部未审核候选不得自动当正确偏好，原GT和分数不改。

后续拟对比同起点同数据SFT和DPO+SFT；偏好参考模型必须是新的OCR增强起点，负答案用该起点实际completion IDs附到原prefix，不解码再分词。DPO标准序列sum log概率/β.1、辅助chosen CE按QA内token均值；新原生纠错不是重复上次单词ROI数据。训练必须在审核后独立冻结完整数据、source、短程预算与主终点，不能在此阶段自动长训、调用PPO/GRPO或按official test挑配置。

原理参考：[DPO原论文](https://arxiv.org/abs/2305.18290)、[TRL官方DPO说明](https://huggingface.co/docs/trl/dpo_trainer)。仅复用已安装的本地冻结数值函数，不安装最新TRL或将当前在线接口混进旧环境。简历、科研、Agent、CI不变，不push。
