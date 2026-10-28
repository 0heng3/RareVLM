# 视觉适配残错：固定模型答案读出诊断 V1

登记时间：2026-10-26；在任何本诊断 GPU forward 前冻结。

问题：native_scope_v1 的视觉方案在 native512 初筛 +0.78125pp，但完整内部1617仅 +0.068027pp，95%区间跨0。它是否仍将初始错误答案排在正确监督答案之前？不能仅因训练128题尚未全对就直接换成DPO。

使用既有共同起点及 visual Final；保持原整图/1024、imagewise vision + FP32 language、FP32 adapter、提示与图像不变，eval/no_grad。只诊断原128条已准入的训练纠错QA，四题一批，按照原排程的首次出现顺序。每个模型分别在同一问题下对三个完整回答做teacher forcing：原chosen监督、初始生成的固定负例、视觉Final的实际生成。后两者直接附加原completion IDs（含实际EOS），不decode/re-encode；不完整EOS拒绝执行。相同负回答组可复用同一次forward。

记录三种回答的总logP与有效assistant token平均logP；chosen同时记录正文/EOS NLL、每个GT-prefix位置的top1和正确token对最强竞争token的logit margin。有效token及mask使用已有编码与CPU记录，尾部读出复用已验trajectory.readout_tail_length。sequence总分受长度影响；均值只作敏感性描述，不混作DPO默认目标。

预算：2个固定模型，每个128条chosen+128条初始negative+至多128条Final回答；最多768 teacher-forced QA序列，最多900秒/模型、65GiB peak allocated。**0 optimizer、0新自由生成、0奖励优化**；通过现有空闲UUID共享队列并行，不抢占其他任务。保存分批概率工件可断点续读；前后LoRA hash一致、无参数梯度。

输出对残错分别统计：(a)chosen对旧负例总分已较高；(b)chosen对当前错答总分已较高；(c)GT-prefix存在非top1 token；(d)正文/EOS与首个局部错误位置。总分排序并不保证greedy自由生成，不能由teacher-forcing改善宣称能力提升，也不能将本诊断等同历史训练梯度或唯一病因。

同一目录保存1617题事后完整升降与单/多词、频次切片，仅读原vqa_score，浮点变化阈值1e-9；这些新切片不改变旧主指标，不用于向训练回填验证答案。原1617为反复使用过的内部评测，非外部盲测；不执行新bootstrap替换旧区间。

后续决策：若多数残错对旧negative已排对，不再仅凭“负例存在”放行同静态负例DPO；若当前错答仍压过chosen，只说明存在候选排序空间，仍须可信训练偏好和匹配CE对照的新协议，不能保证泛化。该诊断结束不自动启动训练、扩步、加seed或读取正式测试。当前实测视觉方案未达到能力门槛，完整强基线/三seed/外部保持目标继续未完成。
