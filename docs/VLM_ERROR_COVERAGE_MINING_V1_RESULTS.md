# 扩大原生错误覆盖：生成已完成，待语义审核

{'training': {'qa': 3072, 'vqa_percent': 93.51562500000004, 'full': 2862, 'length_censored': 0}, 'validation': {'qa': 512, 'vqa_percent': 93.22265625000001, 'full': 474, 'length_censored': 0}}

算法候选205，类型{'possible_spelling': 70, 'different_text_or_target': 126, 'extra_context': 7, 'empty': 2}；0已审核/0更新，training_allowed=false。至少128不同图像的可信错误及类别/target多样性才考虑后续独立强SFT对照，不凑负例/改GT。

[固定协议](VLM_ERROR_COVERAGE_MINING_V1_PROTOCOL.md) · [固定原图候选（生成时快照）](LOCAL_ARTIFACTS.md)

2026-10-21 13:24审核完成更新：按固定候选前184逐原图助手复核，131支持、53排除或不确定；首128不同图/125规范化target/每target最多2，至少两类各16，通过预登记数据准入。排除包括忠实完整品牌、网址/电话空格、转写/翻译、单位、边缘截断、多种有效答案及不可见；不是53个确认错标，亦非全池噪声率。原GT/标准分数/候选快照保持；COMPLETION内0已审为生成阶段历史记录，最新以TRAINING_ADMISSION.json、VISIBLE_REVIEW.json与REVIEWED_TRAINING.json为准。

已另预登记并提交覆盖128对分层重复32的普通CE研究，各64步/1024曝光，0旧Final接训。本页挖掘不当新增模型能力。[后续SFT协议](VLM_COVERAGE_SFT_V1_PROTOCOL.md) · [128原图及控制子集](LOCAL_ARTIFACTS.md)
