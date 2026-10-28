# 视觉侧纠错：匹配CE与CE+DPO V1

2026-10-26，在本轮新reference/训练forward前冻结。

**已有证据与假设。** native_scope_v1完整1617：共同起点87.198516%、language86.295609%、visual87.266543%，视觉相对起点+.068027pp/95%跨0。新固定模型读出768 teacher-forced序列、0优化证实：视觉Final的90条训练残错中，81条当前错答总logP仍高于chosen，76条均值也更高；不是大多数已正确排序、只受格式影响。此前语言DPO没有额外收益，本轮检验在冻结语言的视觉适配中，加入相对reference的答案偏好约束是否能增加泛化净收益。该证据仅给出排序空间，**不证明DPO必然有效或唯一病因**。

**唯一组间处理因素。** CE控制与CE+DPO候选同一共同复合起点、同原384训练QA、同32步排程/512 chosen曝光（hard128各2次、native128/OCR128各1次），每个preference microbatch均拼4chosen+4原始negative。两组均额外256 negative前向，CE组DPO权重0，候选1；不能直接用旧无negative的visual Final冒充唯一匹配控制。所有原数据和人工审核准入复用，不采新数据，不从验证案例补标签。

**模型与数值。** 原0109语言144 FP32因子冻结；视觉blocks2–31 attn.qkv rank12/alpha24的60 FP32因子可训练，共1843200参数。共享存储3686400，冻结底座和head。imagewise原vision/BF16、FP32 language/head/adapter、TF32off、dropout0，整图视觉上限1024、greedy32、batch4评测。视觉train模式启用已有非重入checkpoint；梯度仍经冻结decoder回传。源加载器与native_scope相同，不重新做旧数值修复审查。

**目标与优化。** 对每条chosen用有效assistant token（含实际EOS）的均值CE；prompt/pad/尾部忽略标签沿用已验CPU记录。偏好沿用已审核128条原纠错对、observed EOS实际negative IDs，不decode/re-encode。

`L = mean_QA CE + λ · mean_QA[hard · (−log σ(β[(logπ(y+)−logπ(y−))−(logπref(y+)−logπref(y−))]))]`。

总分DPO，β=.1、λ=0/1；不是长度归一化DPO或GRPO。AdamW lr2e−4、wd0、clip1、warm4/linear32、seed42、microchosen4/effectivechosen16完全一致；不扫β/λ/epoch。新reference必须为共同复合起点，以实际拼8的同batch计算32唯一batch/64排程键，固定后不更新。首个实际优化步核对同起点relative margin≤.05、DPO接近log2、CE两组共同值、视觉梯度有限、冻结分区无梯度及保存FP32母参数一致，不丢弃预检更新。

**预算与队列。** 新reference 256序列/0更新；两组各32更新、512chosen+256negative训练序列；各960内部生成。共享新起点960+1617预测从native_scope逐字节复用并登记哈希，0新起点生成。reference≤600秒，训练≤2400秒/组，内部评测≤1500秒/组，allocated≤65GiB，完整optimizer/RNG断点按1/8/16/24/32保存。原空闲UUID门控、3×15秒、≥40GiB/<10%/0compute和共享锁，不抢占、不终止其他任务。两个角色动态使用可用卡。

**推进与主指标。** 完成内部960/组后，候选hard满分≥32且相对起点VQA≥20pp；OCR/native512/单词/多词/rare1–5/ret64相对起点均≥−.5pp，并且OCR/native512/rare相对新CE也≥−.5pp，才补两Final各1617（3234新生成，起点1617复用）。这只是短预算可学/点保护筛选，不能称能力成功。512原来过+0.5筛选却未迁移，本轮不把它单独当获益证据。

主能力信号仍要求候选在完整1617相对**共同起点与新匹配CE控制均≥+.5pp、配对图像簇95%区间下界严格>0**，并守内部保护；同时报告总体与完整1617频次/单多词切片，后者标明探索边界。使用原paired实现/2000次图像簇bootstrap，单checkpoint条件区间；不是训练种子稳定性、全历史多重比较校正或外部盲测。所有内部评测历史使用过。失败停止该配置，不改变门槛或按部分结果修改设置。无自动追加更新/seed/官方IIIT test/外部测试；原强基线/三seed/跨任务目标未完成。

新生成总量：第一阶段1920，条件3234，最多5154；新训练64次更新。训练loss/reward/margin改善不替代完整能力评测。不改简历/科研/Agent/CI、不push。
