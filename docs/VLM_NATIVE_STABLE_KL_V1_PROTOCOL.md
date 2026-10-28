# 修复数值路径后的原生纠错 SFT / SFT+回放KL V1

2026-10-26，训练/评测输出前冻结。前轮native纠错SFT训练128题满分6→49，但rare366净−3.4 score；DPO48满分且无额外计分纠正。逐层定位17716首次差异在视觉block00，原像素与patch一致。本实验仅在`vlm_batch_numeric_repair_v1`修复验收通过后准入，选择其冻结稳定性规则确定的策略，不按VQA选数值模式。新起点、CE控制与KL候选都使用该策略，旧BF16历史数值不混为新基线。

## 固定数据与预算

完整复用前轮384 QA、CPU chosen标签和32更新顺序：128已看图支持的原生纠错各2次、128原生回放和128 IIIT训练词各1次，共512 chosen/臂；单步micro4/effective16，8hard+4native+4OCR。初始0109 OCR增强语言adapter，144 FP32 q/v因子1843200参数，视觉/底座/head冻结；lr2e−4、warm4/linear32、AdamW wd0/clip1/dropout0/TF32off、seed42。同一保存模型及processor1024/原问题/greedy32/原VQA与ASCII36指标。两组均不再做已无收益的DPO负序列forward；因此不能把与旧DPO的耗时差作为算法加速。32步后不加epoch、seed或调lambda。

## 回放教师与唯一算法变量

修复后起点先实际生成128 native+128 OCR训练回放，并生成全新内部baseline960题（hard128/OCR256/native512/ret64）。native满VQA或OCR ASCII36完全正确才允许该回放题的KL，CE仍保留全部384题。至少224/256且每域112/128可用，否则在0更新停止，不放松门槛。

固定教师为同一修复后的0109起点，GT-prefix上的全词表log_softmax。缓存对应原common64个micro的实际padded input/labels/grid及有效assistant（含真实EOS）token帧。全词表FP32日志概率safetensors逐batch保存，引用与actor使用同一chosen4分组；不从不匹配generated prefix拷贝标签，不使用dev错题、外部teacher或新答案。排除不正确回放后KL按原QA预算归一，不重新加权补足。

新基线分为a=hard128/OCR256/ret64（448）与b=native512/两域训练回放256（768）两队列worker，可并行，均存完整同cohort固定分组预测；参考阶段只逐字节合并，不重生成/评分。缓存教师以training=True加载、dropout0/visual.eval、同actor的非重入checkpoint和grad-enabled执行，立即detach保存，不反传/不优化；这样避免inference与grad-enabled计算路径差异混入首步KL身份检查。

`L_CE = (Σhard CE + Σnative CE + ΣOCR CE)/16`，CE为每QA有效assistant token均值。

`K_i = mean_t Σ_v p_ref(v|GT-prefix) [log p_ref − log p_actor]`。

控制：`L=L_CE`；候选：`L=L_CE + Σ已答对common K_i/16`，lambda=1固定，两组都计算该KL用于日志，唯一差异是反向损失系数0/1。每步CE源权重为.5/.25/.25；KL名义native/OCR各.25，实际按admitted flag减少。它旨在限制回放分布漂移，不保证自由生成边界或泛化。

首正式micro0复用同一forward检查：actor/ref有效token logp差≤1e−6、abs(KL)≤1e−6、KL-only梯度norm≤1e−5；额外autograd.grad不产生优化更新，两组同做。每步实际chosen输入/labels与原CPU哈希一致，有限梯度、参数量与FP32存储；1/8/16/24/32保存完整Adam/RNG边界，支持恢复。所有首步和Final工件以实际结果为准，不用loss下降证明能力。

## 评测与推进

每模型内部960，第一阶段总baseline1216+2×960=3136次新生成、64新更新，另64 teacher forward。指标保持整体/单多词/rare1–5/ret64与OCR分列，按图簇配对95%区间；OCR按filename-family簇。历史使用过的内部评测，非外部盲测；训练图原SFT曝光与父IIIT训练曝光保留披露。新增数值路径仅14固定训练题验收，不宣称全域bitwise稳定。

原学习门槛≥32hard满分且VQA+20pp、OCR和native整体/单/多/rare/ret各≥−.5pp保持；只有KL候选全部通过，并且native512对新初始≥+.5pp、对同数据新SFT≥+.25pp，才补新初始/新SFT/候选各1617（4851生成、0追加训练）。补测能力信号须对新初始与新SFT各≥+.5pp且正95%区间，并保持原切片门槛。无信号停止，不根据官方test、失败dev病例或新结果扩预算。

这是单seed探索性新算法对照，不完成原完整强基线/三seed/外部4372/全POPE目标，不更新简历或声称已提升。无论补测是否通过都不自动做官方OCRtest或扩seed。

## 资源

修复验收、协议/科学源码/数据/包/模型先锁定。reference3600秒（包括1216生成与64前向）、每臂训练1800秒/65GiB、内部评测1800秒、每1617评测2400秒。共享GPU原三次15秒/≥40GiB/利用率<10%/零compute/UUID锁，不抢占已有任务。队列异常自然保留工件，不偷偷修改冻结科学源或重启负结果。CV/科研/Agent/CI不改、不push。
