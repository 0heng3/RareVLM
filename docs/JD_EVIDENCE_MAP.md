# RareVLM：JD技术对齐与项目展示

更新：2026-10-14。只整理CV／RareVLM模块，不修改科研论文、Agent或正式简历。项目完成不等于本人已独立掌握／提出／实现；以下是可复核项目事实，个人描述须按实际参与确认。两周求职不等待所有扩展完成。

## 研究主线

围绕有限可训练参数与计算预算下的视觉语言适配，串起长尾分类、图文表示学习、两阶段检索和质量—成本评测。简历按具体JD选2–3条成果，不把CLIP/Qwen拆成多个重复项目或罗列所有负实验。

| JD技术要求 | 已完成证据 | 代码／结果 | 业务解释与边界 |
| --- | --- | --- | --- |
| 长尾学习、分类训练优化 | ImageNet-LT B32 CE OA/Few61.38/37.76→固定校准65.36/61.23；OA+3.98pp、Few+23.47pp | [分类结果](RESULTS.md) | 类别不均衡下头尾决策取舍；不外推iNat或工业罕见实体已解决 |
| Embedding、对比学习、蒸馏 | 完整28k图/140kcaption，132480参数，三seed；test T2I61.640→67.820、I2T74.533→79.833（+6.180/+5.300pp） | [蒸馏协议](CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md)、[指标](../results/clip_retrieval_distill_v1_metrics.json) | 保留冻结教师关系的图文表征适配；不是完整CLIP预训练/VLM压缩 |
| Rerank、细粒度匹配 | 固定Top50+BLIP，test T2I67.820→85.093、I2T79.833→95.067（+17.273/+15.233pp） | [重排结果](CLIP_RETRIEVAL_RERANK_V1_RESULTS.md)、[源码](../src/clip_retrieval_rerank/common.py) | 双塔先召回、联合图文匹配修正排序；新增模型/384信息，非等计算收益 |
| 推理性能、质量成本权衡 | DocVQA强配置2.385×阶段吞吐、ANLS91.843→91.841，门控通过；自适应重排真实pair−21.88%、matcher+route1.277×，500图确认双向R1 .5pp非劣通过 | [推理协议](VLM_STRONG_CONFIRM_V1_PROTOCOL.md)、[路由结果](CLIP_RETRIEVAL_ADAPTIVE_V1_RESULTS.md) | 真实离线测量、避免弱基线；后者30%成本门槛失败未新增test，非线上p95/全任务无损 |
| 分层候选预算、质量约束的推理优化 | 固定test实际pair312116→210397（−32.59%），同card同batch阶段1.476×；双向.5pp非劣通过，T2I点差−.207pp | [预算结果](CLIP_RETRIEVAL_BUDGET_V1_RESULTS.md)、[预算代码](../src/clip_retrieval_budget/route.py) | 直接对应检索链路计算成本；非严格无损、冷启动或线上p95，个人掌握须另核对 |
| VLM SFT／PEFT、跨任务评测 | Qwen2.5-VL3B，q/v rank8、1843200参数，32985QA；正式TextVQA79.560→81.138±.169（+1.578pp） | [正式协议](VLM_FORMAL_BF16_V1_PROTOCOL.md)、[指标](../results/vlm_formal_bf16_v1_metrics.json) | NF4语言量化/FP32训练与原权重BF16评测分开；MMStar能力保持/rare改善未证实 |

## 一手／官方来源与适用范围

- [阿里1688团队](https://liamding.cc/hiring.html)明确工业VLM、品商Embedding/Rerank，另有杭州在读硕博研究实习入口。团队方向覆盖不等于每个方向都有已核验日常实习HC。
- [国际站多模态实习原文](https://career.nankai.edu.cn/correcruit/content/id/118629.html)要求图／文／文档表征、检索链路延迟和吞吐、论文复现与落地分析。当前图文检索已有，DocVQA是问答，文档检索、亿级图库与线上高并发仍缺证据。
- [字节Seed官方项目页](https://seed.bytedance.com/en/seedearlycareer)含视觉表征、评测与推理系统等方向；实习毕业门槛2027年9月或以后，用户毕业年月未确认。它是方向级公告，不计为多个独立杭州实习JD，不宣称岗位可投资格。
- [腾讯多模态数据岗位](https://careers.tencent.com/jobdesc.html?postId=2083008230946160640)按既有本地官网API记录仅作数据生产/检索效率/服务技术参考；2年以上社招，本轮API未返回正文，不算实习或fresh开放核验。

以上是定向技术对齐，不给市场覆盖率，不以社会岗/方向广告凑分母。[SentenceTransformers官方两阶段文档](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)解释召回与重排的质量/成本分工；本项目仍使用固定CLIP/BLIP，未迁移其文本模型或将演示项目当实验证据。

## 本轮新增：0／20／50预算研究

上一轮保持质量仍需约70%T2I及几乎全部I2T各匹配50候选，实际pair仅减少21.88%。新增独立cal-only预算：旧fast阈值固定，较难查询Top50，中间区间Top20；只实际prefix进入matcher，尾部不伪造分数。15合同及12格独立复核通过；dev/test均通过固定门槛。test实际pair−32.59%、缓存matcher＋route1.476×，T2I85.093→84.887、I2T95.067持平，双向.5pp条件非劣通过。[完成报告](CLIP_RETRIEVAL_BUDGET_V1_RESULTS.md)保留统计和边界，不称无损／提准／线上部署。预设仍是30%真实pair削减、双向.5pp非劣、同card同batch三轮计时，未降低旧门槛。

## 图文数据质量研究（已完成）

独立[一致性筛选结果](CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md)：完整28k图、4条件×3seed共12新训练，原始／20%受控来源置换、score4／random4分别配对。clean dev T2I+0.227pp，全门控失败未新增test；noisy dev T2I+1.160pp，全门控失败未新增test。原始收益与受控错配鲁棒性分开，不把后者包装为干净数据提准。 错配组图搜文−.533pp、非劣下界−1.934pp，导致双向门控失败；nDCG点值改善不能替代R1保护。

已知置换保留率random20.04%→score0.393%，这是选择后的来源诊断，不是原始数据错标率。各分支初始化／采样序列／监督量严格一致，强clean-full5历史结果监督量更多并单列。对应多模态数据管线、筛选与科学评测；尚无工业商品或生成式数据合成增益。个人贡献／掌握、岗位资格限制沿用上文，正式CV保持。

## 岗位定向选择

- 检索／Embedding／Rerank：优先用三seed表征蒸馏、固定候选重排的test结果，加一项有质量约束的实际成本结果。
- 传统CV／长尾训练：突出ImageNet-LT头尾取舍、LA／固定校准及泛化失败边界。
- VLM SFT／训练评测：在能够讲清assistant mask、每题CE、LoRA插入和跨任务结果后使用已有SFT事实；不自动把执行记录当本人熟练。

这些方向仍是一份RareVLM项目，Agent和论文由用户另行负责；正式MD/PDF保持，候选事实清单仅用于准备。

## 边做边学的六项核对

1. 对一张图五caption画出[32,480]/[160,96]正负矩阵，说明多正平均CE和集合目标差异。
2. 手算一项KL梯度P_student−P_teacher，解释教师无梯度和lambda1的目标混合。
3. 解释512维双塔与BLIP[32,577,768]图像hidden／[32,64]文本／[32,2]ITM之间的成本差。
4. 用真实margin判断0/20/50；解释Top20尾部保留与有效Top1覆盖的区别。
5. 用已保存图像失败例说明细粒度服饰／语义相似与标注配对关系。
6. 比较三旧checkpoint SD、图像簇CI、.5pp非劣下界和三轮计时，不把它们混为三次新训练。

未验证能力保留为缺口：中文工业商品／文档检索、数据合成质量增益、RL/DPO/CPT、大规模训练、视频音频、在线vLLM/SGLang/TensorRT／TTFT/p95。没有明确高匹配JD与反馈前，不为补关键词同时扩展。


## 训练／偏好对齐／强化学习：2026-10-15

[分层JD统计与决策](POSTTRAIN_JD_DECISION_20261015.md)：4具体岗位微调1/4（仅加分）、RL职责0/4；2多模态方向公告后训练2/2、RL1/2。小便利样本不当市场覆盖或否定其他RL岗位；表示语义对齐与生成偏好对齐分开。

[已完成信号审计](VLM_ALIGNMENT_SIGNAL_V1_RESULTS.md)：512训练图／2560输出；85原规则DPO对、149非零奖励组，8真实DPO梯度合同通过，但0优化器更新。局部真实图像揭示Wi-Fi/wifi和答案跨度伪偏好；独立保守规则剩40，未作全量语义确认。不能写已DPO/GRPO训练或模型提分。

[完整偏好数据构建](VLM_PREFERENCE_DATA_V1_PROTOCOL.md)已启动10000训练图／计划50000生成，约3小时粗估，来源／质量边界和恢复独立锁；正式继续SFT／DPO须数据语义审核和独立训练协议，现无自动模型更新。原验证成果与正式简历保持，个人掌握／贡献待确认。
