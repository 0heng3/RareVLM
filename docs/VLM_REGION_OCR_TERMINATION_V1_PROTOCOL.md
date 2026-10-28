# 区域转写结束符监督：单因素消融协议

本轮是在区域SFT完整负结果之后登记的新探索，保留所有旧结果与门槛。

## 问题与证据

旧候选TextVQA 85.605469，初始85.859375、原问答控制85.976563；Human94.921875，未通过联合门槛。两臂均完成200更新，训练35.0/35.3分钟，无执行故障。
固定64训练区域题由初始24道规范化正确变为候选31道，对照23道；原始严格匹配20→29。证明部分训练任务学会，不代表泛化提升。
候选相对初始19条Text输出变化，其中12条为原回答严格前缀，净损失1.0/总净损失1.3 score。相对控制8条前缀净损失1.7/总1.9。部分缩短有益，不使用统一长度阈值或改解码来补分。
固定同图训练QA的多词首词后EOS相对GT后续token logit，候选相对控制平均提高约2.78/3.08（共同11题/另一原问答12题）；是同前缀机制观察，不证明所有负迁移均由EOS因果导致。

## 唯一目标变化

只对region_aux的im_end及其后的尾部换行设置loss weight=0。区域文字token保持原梯度归一化，共同问答保持原完整CE：

L_old(q) = Σ全部assistant token CE / n_all
L_new(region_aux) = Σim_end之前文字token CE / n_all
L_new(common) = L_old(common)

分母仍是原完整assistant token数，不改为文字token数，避免同时增加文字梯度。输入labels、实际序列和可见图像均不改；将loss有效token数和普通完整CE另记入训练trace。该方法是标准token加权消融，不包装为原创算法或RL。

## 固定设置与对照复用

- 使用原1600图、1600共同QA+1600官方TextOCR单词QA、同序200更新、micro4/effective16；全部CPU实际输入hash复用并在训练逐题匹配。
- 原SFT42起步、FP32语言q/v LoRA1843200、BF16底座、FP32冻结head/CE，视觉冻结、dropout0、LR2e-5、20步warmup、TF32off。
- 候选从初始模型重新开始，不接旧负结果Final继续训。复用父实验已经完整完成、字节锁定、完全相同数值/解码/输入设置的initial、普通问答control和完整region CE三模型，避免重复训练/评测对照。
- 训练循环派生自冻结父版本，差异另存TRAINER_DERIVATION.diff；保留原保存恢复、首步保存逐值核验与完整输入记录。共享UUID零compute/40GiB/三次15秒准入，不终止其他任务。
- 累计训练active上限2700秒；Final@200，不选中间checkpoint，不扫学习率/权重/种子。

## 验收与下一步

候选生成同一Text512/Human512，raw greedy/repetition1/max32/batch4，固定同问图；三组缓存对照逐题对齐。
对initial、普通问答control、完整region CE均要求Text整体及rare1–5至少+0.5pp，整体6000次图像簇95%CI下界>0；Text frequent>20和Human overall/yes/no点差均>=−0.5pp。
频次仍为原32985条TextVQA监督目标，Human按既有互补组件聚类；不把三模型或QA当独立种子。全部升降保留。
Final额外评测原固定64训练区域题，分别报告raw exact、casefold exact及旧VQA-normalized口径，不将其当heldout提升。
本轮只登记200步探索筛选，失败停止、不确认、不扩训；通过后仍须独立确认和多seed，不能把筛选通过写为稳定能力或修改CV。旧yes/no分支+1pp/pair+2pp门槛不变。

## 启动前验证

新损失4项CPU数学检查通过：文字梯度与原CE逐值一致；终止符/尾部梯度为零；原分母保持；缺失/重复EOS或空文字拒绝。冻结诊断RESULTS、配置、源码、协议、测试、父输入与全部对照预测SHA后启动。该检查只验证实现，不能保证泛化成功。
