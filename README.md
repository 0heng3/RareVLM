# RareVLM

[实验状态看板](docs/LOCAL_ARTIFACTS.md) · [最新运行总览](docs/LOCAL_ARTIFACTS.md) · [错误图像与原因](docs/LOCAL_ARTIFACTS.md) · [监控与快照说明](docs/monitor/README.md)

## 长尾视觉适配、多模态指令微调与图文检索

简体中文 · [English](README.en.md) · [项目架构](docs/ARCHITECTURE.md) · [研究与学习路线](docs/JOB_ROUTE.md) · [JD能力与证据](docs/JD_EVIDENCE_MAP.md)

RareVLM 研究有限可训练参数、视觉信息与计算预算下的视觉语言模型适配。项目包含冻结 CLIP 的长尾分类、校准与图文检索，以及 Qwen2.5-VL 的受控 SFT、生成式评测和推理质量—成本分析。各模块独立验证，不将分类、VQA和检索指标合成总分。

## 教师能力与蒸馏研究更新（2026-10-29）

完整7B教师在同协议内部1617题上取得 **93.537415%**，强3B SFT为88.027211%，差值 **+5.510204 pp**，95%图像簇区间[+4.232778,+6.834321]；长尾342题差值+4.122807 pp。该结果证明冻结教师的任务优势，不能写成学生微调收益。

三臂GT/均匀尾/教师KD完整结果均低于强起点，KD为86.846011%。组件恢复中，保留KD视觉、恢复语言后为87.909709%，仍未超起点。旧配置停止扩训。新学生生成前缀蒸馏固定原数据、范围和超参，仅改变辅助KL所看的历史；11:47完整验收后，新学生为87.031540%，对强起点−0.995671 pp（95% [−1.712673,−0.288533]），对旧KD仅+0.185529 pp、区间跨0，固定配置停止。实际251/710正文位置偏离GT前缀，证明机制已执行但未获得整体收益。详见[最终结果与轨迹诊断](docs/VLM_ONPOLICY_KD_V1_RESULTS.md)。

[教师完整结果](docs/VLM_FULL_TEACHER_TRANSFER_V1_RESULTS.md) · [三臂蒸馏结果](docs/VLM_FULL_TEACHER_KD_V1_RESULTS.md) · [组件恢复](docs/VLM_KD_COMPONENT_RESTORE_V1_RESULTS.md) · [新学生前缀蒸馏协议](docs/VLM_ONPOLICY_KD_V1_PROTOCOL.md) · [实际目标代码](src/vlm_onpolicy_kd_v1/objective.py)。这些均为历史使用的内部评测，尚无三seed稳定性或独立外部提升结论。以下保留历史结果。

## 最新结果：整图监督与同数据SFT/DPO（2026-10-28）

只把128个IIIT裁词正监督槽位替换为预测前固定的128张ST-VQA整图，保留原生纠错/回放、共同起点、视觉LoRA的1,843,200个更新参数及32步预算。两臂训练与全部5,154条新评测生成完成；SFT对起点出现正向信号，但预定DPO主候选未通过完整任务和强对照门槛。

| 同协议完整内部1617题 / 1024图 | VQA分（%） |
|---|---:|
| 共同起点 | 87.198516 |
| 历史裁词CE | 87.303649 |
| 历史裁词CE+DPO强对照 | 87.631416 |
| 新整图SFT | **88.027211** |
| 新整图SFT+DPO | 87.303649 |

新SFT对起点 **+0.828695 pp**，条件图像簇95%区间 **[+0.193496,+1.502531]**；对历史强DPO仅+0.395795 pp、区间跨0。新DPO比同数据SFT低0.723562 pp，95%区间[−1.358561,−0.110691]。这是单seed、已历史使用的内部评测，不能事后更换主候选或宣称三seed稳定、外部泛化及长尾问题已解决。原固定配置停止扩训。

[完整结果与限制](docs/VLM_STVQA_REPLAY_V1_RESULTS.md) · [便携指标](results/vlm_stvqa_replay_v1_metrics.json) · [冻结训练入口](tools/run_stvqa_replay_v1.py) · [独立CPU汇总恢复](tools/resume_stvqa_replay_report_v1.py)。下一项已登记为[四固定Final的256图来源保持验证](docs/VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md)，只评测、不更新权重，不将尚未验收的结果写成提升。[登记摘要](results/vlm_stvqa_holdout_transfer_v1_registration.json)。以下按日期保留历史研究记录。

2026-10-25 20:14：[原生问答纠错SFT/DPO对照](docs/VLM_NATIVE_CORRECTION_V2_STATUS.md)已提交。两次3584条固定筛选得到251候选，全部助手看图后135支持，按原序选128（127归一化答案、66单词/62多词），配128 native和128 OCR回放，各32步/512chosen曝光；两组同起点、同数据和负forward，只差DPO权重。保留已完成OCR收益的起点，先缓存匹配参考概率，再两GPU并行；当前无新Final量化提升结论，不以实现/拟合代替能力。支持/排除是候选审核，不是全池错标率或专家重标注。[真实图片与监督质量](docs/LOCAL_ARTIFACTS.md) · [公式、固定训练图与代码](docs/LOCAL_ARTIFACTS.md)。

2026-10-25 19:24：[回放修复完整结果](docs/VLM_IIIT_REPLAY_REPAIR_V1_RESULTS.md)未通过整体问答提升，native1617为87.248%（初始87.328%、原OCR增强模型87.260%，差值区间均跨0），停止该方案扩训；原官方OCR 94.967%→96.933%的收益不变。当前转向[真实原生问答纠错V2](docs/VLM_NATIVE_CORRECTION_V2_STATUS.md)：保留OCR增强起点，固定2048条原图/原问题/高共识答案，先筛错并看图审核；至少128可靠错误/96个不同答案才准入同数据32步SFT与DPO+SFT。已通过8条固定GPU输出一致性检查，两GPU正在筛选，尚无V2训练提升结论。下文为各日期历史记录。

2026-10-25最新：[IIIT5K完整结果](docs/VLM_IIIT_SCOPE_V1_RESULTS.md)中官方test3000初始94.9667%→稀疏语言LoRA96.9333%（+1.9667pp，条件filename-family簇95%区间[1.3805,2.5868]）。native512总体+.5273pp但区间跨0，多词−.5078pp未过点保护，联合目标未完成。后续[回放分布修复](docs/VLM_IIIT_REPLAY_REPAIR_V1_PROTOCOL.md)已15:32提交、等待共享空闲GPU：native256长度桶150/91/15→64/152/40，含the 0→18，OCR数据/CE/64步不变。第一阶段仅一组新训练，内部门槛通过才追加固定起点数据顺序复现；已曝光官方test不用于选配置。四旧模型各1617独立图像问答同步补测。[当前状态入口](docs/VLM_IIIT_REPLAY_REPAIR_V1_RESULTS.md) · [真实图片、分布条图、公式与源码](docs/LOCAL_ARTIFACTS.md)。后文较早“当前”描述为对应日期快照。

2026-10-25新增[官方场景文字监督×同参数适配研究](docs/VLM_IIIT_SCOPE_V1_PROTOCOL.md)：完整官方IIIT5K来源已归档，训练部分固定512不同词图/128分层子集、256内部验证与256答案词隔离native回放。三个条件分别比较128/512覆盖和语言/视觉适配，每组64步、1536QA、1843200更新参数；官方test3000只在全部Final64冻结后打开，四模型完整评测，预登记15840生成。内部256全量与预测前213助手支持子集分列，审核非专家真值；IIIT ASCII36评分与旧TextOCR字面评分分开。[当前状态与结果](docs/VLM_IIIT_SCOPE_V1_RESULTS.md) · [真实词图、LoRA公式与代码](docs/LOCAL_ARTIFACTS.md)。配置固定、无自动扩epoch/seed；未完成项不写成能力提升。

2026-10-24新[最难词元32步先导](docs/VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md)已通过固定模型诊断：59错误词中20词平均CE<=1但正文目标仍非最高，最难/均值NLL中位比2.967、错误词EOS CE中位.000451。V1新增ND交叉熵实现的数值检查在首个optimizer.step前拦下，0新更新；[V2修正](docs/VLM_LITERAL_TOKEN_FOCUS_V2_NUMERICAL_CORRECTION.md)沿用原log_softmax/gather路径，阈值和科学协议不变，候选已GPU准入。mean32复用原权重、其新960预测继续完成；max32只新训练32步并评1088题，0自动长训。[V2状态与结果](docs/VLM_LITERAL_TOKEN_FOCUS_V2_RESULTS.md) · [实际词图、概率与源码](docs/LOCAL_ARTIFACTS.md)。旧cosine文字与真实线性LR之差已记录；当前尚无新质量结果。

2026-10-24[任务匹配回放结果](docs/VLM_LITERAL_TASK_REPLAY_V1_RESULTS.md)已完整结束且未通过：纠错70/128、新词91.4063%（初始94.5313%）、旧词83.2031%（初始82.4219%、区间跨零）、native92.4805%（初始93.2227%）、保持93.75%。训练回放124→127/128不代表泛化；不扩该配置。[真实词图与回放原理](docs/LOCAL_ARTIFACTS.md)。

2026-10-23后续[大小写口径与SFT可学性对照结果](docs/VLM_LITERAL_CASE_SFT_V1_RESULTS.md)已完整结束且未通过：raw/canonical训练满分52/69题，新128词验证初始94.5313%→92.1875%/91.4063%；native512初始93.2227%→93.8672%/93.2813%，区间均未确认提升。统一大小写多拟合17题，未带来验证收益，不扩1617或种子。[真实图像、标签token与代码](docs/LOCAL_ARTIFACTS.md)区分格式修复、训练拟合和泛化；旧256完整保留，不用新集合更高起点当提分。

2026-10-23新[完整词可靠监督与偏好对照](docs/VLM_LITERAL_PREFERENCE_V1_PROTOCOL.md)已完成且未通过：训练拟合SFT22/DPO25题，固定256词验证82.0313%/82.4219%，native512为93.1641%/92.8320%，保持均92.1875%。未扩该配置。[实际图像、错误标签案例与公式](docs/LOCAL_ARTIFACTS.md)连接了监督审核与原生VQA迁移验收；当前尚无新训练能力结果，给定区域转写不称整图定位或原SFT未见图片。

2026-10-23：[原生文字双源监督诊断](docs/VLM_NATIVE_OCR_PROBE_V1_RESULTS.md)已完成，1269图初始96.5485%、q/v强回放96.0914%、MLP-KL96.6273%，没有达标的新能力提升；42错误候选不足128图/96target的训练数量条件。后续[原生小字尺度对照结果](docs/VLM_NATIVE_SMALLTEXT_V1_RESULTS.md)已完成：104小字94.2308%→94.2308%，104匹配大字98.6538%→97.6923%，尺度信号未通过；同模型同卡共416生成、0训练，仅6个新错误候选，不扩该配置；[实际输入与原图学习页](docs/LOCAL_ARTIFACTS.md)记录实际grid、视觉token和两种策略的真实回答。原图插值不增加原生信息，诊断不当新增训练收益，简历不变。

当前完整实验（2026-10-18）：[人工监督SFT结果](docs/VLM_HUMAN_SFT_REPAIR_V1_RESULTS.md)已完成，各36878次QA曝光/2305更新；人工yes/no初始95.570%、控制94.937%、候选95.992%，TextVQA分别86.753%、87.353%、87.372%，联合门槛未通过、不扩种/正式测试。[零更新失败回顾](docs/VLM_HUMAN_SFT_FAILURE_REVIEW_V1.md)确认7题纠正/5题新错；固定200步约33分钟已达96.20%，继续跑满未增加主指标点值。后续采用[小预算先行与长训放行规则](docs/EXPERIMENT_BUDGET_POLICY.md)，不把事后回放当新成功或已校准成功率。

当前算法行动（2026-10-17）：[完整SFT训练视觉预算对照](docs/VLM_SFT_RESOLUTION_V1_PROTOCOL.md)已完成且未达标。内部TextVQA初始87.112、256训练86.988、1024训练87.106；高预算相对新对照仅+.118pp，图像簇95%区间跨0，不扩种或扫参。[源码追踪与定向诊断](docs/VLM_FAILURE_CODE_REVIEW_V1.md)已完成：确认强制舍入LoRA会损失部分识别，独立FP32 LoRA入口通过真实权重/GPU生成检查。零新训练的匹配全量新口径为86.982→87.372（+.390pp，95%区间跨0）；同Final对原路径+.266pp。仍未达+.5标准，不改原负结果、不宣称稳定能力提升。

[第二轮分因诊断](docs/VLM_FAILURE_FACTORS_V2_RESULTS.md)进一步确认数据覆盖/标签语义错配：Object/POPE漏检79.4%在训练类别外、7.2%会被尺寸规则排除；真实32QA学习正控制中lr1e-4能拟合到100，两个小LR臂仅82.19，probe均未升。它证明链路能学并定位局部设置影响，不是新增能力提分；后续先补任务匹配监督与强SFT基线。

[修复实现与数据候选 v3](docs/VLM_TRAINING_REPAIRS_V3.md)：新 SFT 入口保留 FP32 LoRA、显式 FP32 CE，并核验保存重载；候选覆盖恢复至 80 类，9,576 条未验证负例改为未知、2,047 个仅小目标正例候选得以保留。候选尚未准入训练；修复验证不作为模型能力提升。

[深入代码排查 v4](docs/VLM_DEEP_CODE_INSPECTION_V4.md)：确认此前 8 QA 验证没有训练修正后的物体监督。现已从官方 VQA v2 train2014 接入 >=9/10 人一致的 3,893 条训练/474 条内部验证问答，保留原问题语义；16 张原图及 4 张跨版本图像抽查像素一致，实际编码与新数据回归通过。新数据包含属性/关系等 yes/no，不称 80 类纯存在性覆盖；本轮尚未新训练或提分。

CLIP 主实验与各多模态协议分别归档。2026-10-12归档快照中的Qwen深入研究为 **17 次训练、101 个跨协议评测单元**；后续研究在各协议另列，评测单元不是独立重复样本。项目保留负结果：频次加权未通过罕见答案门槛；区域策略超过双视图对照，但比同预算单整图低 **5.850 个百分点**。

2026-10-15新增[物体适配v2结果](docs/VLM_OBJECT_ADAPT_V2_RESULTS.md)与[固定初始模型全词表KL回放结果](docs/VLM_OBJECT_KL_V1_RESULTS.md)：v2两个候选均未通过联合筛选；新KL候选沿用相同训练输入，物体内部val从95.614%升至97.953%，Text从87.112%降至86.797%，整体保持和罕见／高频答案保护仍未通过。三项候选均没有据此追加种子或POPE，不称通用保持成功。外部1791原图／4372QA只完成初始基线96.821%，不是候选收益。[固定训练点梯度诊断已完成](docs/VLM_OBJECT_KL_GRADIENT_V1_RESULTS.md)：96条训练QA前后向、0优化器更新，三点KL/Object梯度范数比0.031/2.699/0.333，不能据小mean KL认定约束始终很弱，也未证明覆盖因果。最新[回放覆盖对照结果](docs/VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md)为`completed_negative`：wide32物体95.614→97.661%（+2.047pp），Text87.112→86.964%（−.148pp）；Text保持及rare/frequent点保护失败，5/8门槛通过。控制不可替补，未追加0/200或候选外部/POPE，真实提升目标仍active。[便携指标](results/vlm_object_replay_coverage_v1_metrics.json)。正式简历保持原状。

2026-10-16完成[生成答案硬CE回放](docs/VLM_OBJECT_BEHAVIOR_REPLAY_V1_RESULTS.md)及[生成前缀软KL回放](docs/VLM_GENERATED_PREFIX_KL_V1_RESULTS.md)：两者Object均为97.661%（相对共同初始+2.047pp），Text分别为87.062%与87.007%，均通过整体非劣效、未通过rare/frequent点保护，均为6/8门槛通过的完整负结果。新KL的1959条正式预测独立评分零差异；不扩该负候选的种子或确认集。下一项[固定0.5的LoRA坐标更新收缩](docs/VLM_LORA_COORDINATE_SHRINK_V1_PROTOCOL.md)已完成[完整评测与版本化恢复](docs/VLM_LORA_COORDINATE_SHRINK_V1_RESULTS.md)：Object +0.877pp、Text −0.130pp，仅4/8门槛通过，未优于原KL配置；完整1959条新预测独立评分零差异，保留负结果且不追加系数网格；完整三种子、external4372及POPE9000成功范围保持。

![CLIP结构：冻结图像与文本编码器，只训练66,240参数残差adapter；固定checkpoint校准只调整logits](assets/overview.png)

## 已验证的主要结论

1. **普通 CE 存在头尾取舍。** ImageNet-LT B/32 三种子中，Many 相对零样本提高 **15.06 pp**，Few 下降 **21.46 pp**。
2. **固定表征后仍能调整决策。** 验证集选择校准强度后，OA/Few 从 **61.38/37.76%** 变为 **65.36/61.23%**；校准不更新 adapter、编码器或文本原型。
3. **输入信息影响质量和成本。** 固定 SFT checkpoint 的视觉预算上限从256提高到1024，内部dev共识分提高 **11.905 pp**，摊销生成时间约增加 **4.26倍**。
4. **数值工程收益有明确范围。** 固定256预算，原权重FP32→BF16的纯生成吞吐提高 **2.44倍**、峰值allocated下降 **49.66%**；这些是推理结果。
5. **新候选不自动成为成功成果。** 温和频次加权与单区域分配均保留完整对照、失败样本和限制，没有把弱对照收益当成整体最优。

`pp` 表示百分点；跨任务指标、内部dev与公开benchmark、不同输入/数值协议不混合比较。

## 图文检索与推理复核：2026-10-14更新

完整Flickr30k WebDataset实际含29,000张源训练图；固定持出1,000图dev，28,000训练图，每图5caption。冻结FP32 CLIP-B/32表征，训练双侧残差adapter共132,480参数；两负采样策略各3seed、10epoch固定Final@10，正式test为1,000图/5,000caption。

| 完整test，均值±3seed样本SD | 文搜图Recall@1（%） | 图搜文Recall@1（%） |
| --- | ---: | ---: |
| 冻结CLIP（无seed SD） | 58.780 | 77.600 |
| 随机负样本 | 61.640 ± 0.461 | 74.533 ± 0.702 |
| 混合困难负样本 | 59.487 ± 0.647 | 73.633 ± 0.115 |

随机文搜图均值比冻结底座+2.860pp，同时图搜文−3.067pp，不能称双向改善。混合困难相对随机的文搜图差−2.153pp，条件图像簇95%区间[−3.260,−1.060]，三个seed均为负，主门槛失败。Recall采用hit-any正项命中，nDCG另列；预训练重叠未知，公开test已历史曝光。[检索协议](docs/CLIP_RETRIEVAL_V1_PROTOCOL.md) · [完整便携指标](results/clip_retrieval_v1_metrics.json) · [检索训练源码](src/clip_retrieval/train.py)

独立强配置正式复核共23,698条model×QA、0训练：原dense batch4→分段SDPA batch16，TextVQA/DocVQA/MMStar的三轮预处理+H2D+生成阶段吞吐比中位数为2.537/2.385/1.809×，仅DocVQA通过预设质量非劣门槛（ANLS91.843→91.841）。TextVQA/MMStar未通过；相对dense4的allocated增加约1.52–1.70GiB。该固定普通CE seed42 checkpoint与上方历史三seed SFT不是同一组adapter，不能混表或称所有任务无损、线上SLA、省显存。[强配置复核协议](docs/VLM_STRONG_CONFIRM_V1_PROTOCOL.md) · [完整便携指标](results/vlm_strong_confirm_v1_metrics.json)

[多正集合损失对照](docs/CLIP_RETRIEVAL_SET_V1_PROTOCOL.md)已完成六次完整训练和七个完整dev格，只改变图搜文正项聚合，保留文搜图loss及同一负池。原平均正项→集合正项的dev文搜图R1为64.153±.273→62.967±.630%，图搜文R1为78.233±.814→77.467±.513%，图搜文nDCG10为73.714→71.580。文搜图差−1.187pp，探索95%图像簇区间[−1.794,−.580]；图搜文差−.767pp区间[−2.267,.767]跨0。仅1/3seed图搜文为正，预设门槛失败，本轮未追加test。原基线三seed逐query指标与Top10完整复现；两方法初始化/锚点与负池序列/曝光SHA一致。不能仅因目标与hit-any指标接近而认为泛化会改善。[完整便携指标](results/clip_retrieval_set_v1_metrics.json)

配套精读固定OpenCLIP、BLIP、lmms-eval源码，提供真实480项logits/梯度、源图和召回排序。零GPU审计已核对两组各5000条保存TextVQA预测，评分差异超过1e−12的条目为0；这是评分函数对照，不证明完整框架/输入等价。训练监控有并发占用记录，未将训练时间包装成独占性能指标；已有任务未终止，正式简历未修改。

[冻结CLIP教师关系蒸馏](docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md)已完成并通过dev/test全部预设门槛：保留原多正目标，加同候选池双向KL、固定lambda1，新增三个完整训练并复用三个原checkpoint与四dev对照格；dev通过后才新评测七个完整test格。

| 完整test，均值±3seed样本SD | 文搜图Recall@1（%） | 图搜文Recall@1（%） | 图搜文nDCG@10（%） |
| --- | ---: | ---: | ---: |
| 冻结CLIP（无seed SD） | 58.780 | 77.600 | 70.087 |
| 原多正平均目标 | 61.640 ± 0.461 | 74.533 ± 0.702 | 71.146 ± 0.085 |
| 原目标＋冻结教师KL | **67.820 ± 0.122** | **79.833 ± 0.404** | **76.705 ± 0.195** |

对原目标，T2I增+6.180pp、探索95%图像区间[5.060,7.320]，I2T增+5.300pp、[3.467,7.133]；三统计端点各单侧alpha=.05/3。I2T对冻结点值+2.233pp的区间[−.067,4.533]跨0，单侧下界−.301高于预设−.5非劣界限，不能称显著超过冻结。三个种子对原目标双向均正，新test原对照逐query/Top10重现v1，匹配初始化/负池序列/曝光SHA；七test格均从Top10独立重评分核验。[完整便携指标](results/clip_retrieval_distill_v1_metrics.json) · [蒸馏损失源码](src/clip_retrieval_distill/loss.py)

该结果支持本协议中的整套训练正则配置；lambda1相当于监督与教师软目标混合，尚无均匀软目标/相似熵对照，不能把增益全部归于教师语义结构。公开test已历史曝光、pretrain重叠未知；不称盲测、长尾问题已解决、模型压缩、VLM/Reranker或工业部署。正式简历与PDF保持，个人贡献/掌握需独立核对。

## CLIP长尾主结果

采用 `{0,42,200}` 三种子均值；零样本为确定性参考。下表均为验证集选择的校准强度，列中为 **OA / Few（%）**。fixed view为固定训练视图，online views为在线裁剪与翻转。

| 数据集与配置 | 零样本 OA / Few | CE OA / Few | LA OA / Few | 固定CE＋验证集校准 OA / Few |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-LT · B/32 · fixed view | 59.42 / 59.22 | 61.38 / 37.76 | 65.97 / 59.01 | 65.36 / 61.23 |
| ImageNet-LT · B/16 · fixed view | 63.99 / 63.35 | 67.20 / 44.23 | 71.04 / 63.73 | 70.83 / 64.57 |
| ImageNet-LT · B/32 · online views | 59.42 / 59.22 | 61.54 / 35.81 | 66.64 / 59.28 | 66.11 / 62.90 |
| iNaturalist 2018 · B/32 · fixed view | 3.48 / 3.32 | 5.05 / 1.57 | 5.69 / 3.70 | 5.35 / 2.48 |

iNaturalist 使用8,142个官方科学名文本原型。在该冻结编码器/分类器协议下，OA仅约 **3.48–6.01%**，绝对识别能力有限。三个种子均选择α=0.25；固定α=1使OA下降1.05pp，校准强度不是通用常数。

![四种配置下CE相对零样本的头尾变化：点为种子，误差线为种子样本SD](assets/main_results.png)

[完整结果与边界](docs/RESULTS.md) · [28行汇总](results/main_results.csv) · [76条种子记录](results/seed_results.csv) · [核心adapter/loss](src/methods.py) · [校准代码](src/calibrate.py)

## Qwen多模态研究

### 早期工程子集

早期链路验证了图像互斥的数据准备、QLoRA SFT、回答生成和本地推理。4张V100训练1,843,200个语言LoRA参数，视觉编码器与底座冻结；语言权重NF4量化、FP32计算。1轮覆盖1,024张训练图、128张诊断图，每rank64次更新。

| 相同NF4/FP32路径，256张最终图 | 归一化VQA共识分（%） | 任一参考答案匹配（%） |
| --- | ---: | ---: |
| Pretrained instruct | 68.55 | 70.70 |
| QLoRA SFT, fixed final epoch | 71.52 | 75.39 |

增益 **+2.97 pp**，条件图像bootstrap区间 **[−0.20,6.25]**。该单种子工程子集和简化归一化不证明稳定收益或完整benchmark性能。[工程结果](docs/VLM_SFT_RESULTS.md) · [指标与验证](results/vlm_sft_metrics.json)

### 受控SFT与正式复核

深入研究比较相近参数预算的q/v rank8、q/k/v/o rank4和q/v rank4＋视觉merger。阶段1每策略10k QA、三个种子；按内部dev固定规则选两种策略，阶段2各使用32,985 QA、2,062次更新。原始主线共15次训练、67项评测，未按正式val挑选checkpoint。

| 阶段2，NF4/FP32，视觉上限256 | TextVQA正式val | DocVQA ANLS×100 | MMStar子类宏平均 |
|---|---:|---:|---:|
| 固定NF4底座 | 69.030 | 68.730 | 54.732 |
| q/v rank8，均值±种子SD | 72.166 ± 0.062 | 69.901 ± 0.176 | 54.018 ± 0.314 |
| q/v rank4＋merger，均值±种子SD | 72.389 ± 0.085 | 69.954 ± 0.066 | 54.941 ± 0.577 |

后续原权重BF16/1024正式复核采用底座与3个既有q/v checkpoint，完成 **12格、47,396条预测、0新训练**：

| 原权重BF16，视觉上限1024 | 底座 | SFT三checkpoint，均值±样本SD | 平均变化（pp） |
|---|---:|---:|---:|
| TextVQA，5,000 QA | 79.560 | 81.138 ± 0.169 | +1.578 |
| DocVQA，5,349 QA | 91.631 | 91.819 ± 0.019 | +0.187 |
| MMStar，1,500 QA、18子类宏平均 | 56.289 | 55.869 ± 0.245 | −0.420 |

![BF16/1024正式复核结果](assets/vlm_formal_bf16_v1/formal_quality.png)

TextVQA三个固定checkpoint的配对探索区间均高于零；DocVQA/MMStar区间跨零，不能证明等价或能力保持成功。实际监督target的rare/mid切片点值未改善，且该正式分析没有切片图像区间。训练来源包含V100/A100，公开val已有历史曝光，预训练重叠未知。

[深入SFT协议](docs/VLM_DEEP_V2_PROTOCOL.md) · [正式复核协议](docs/VLM_FORMAL_BF16_V1_PROTOCOL.md) · [正式便携指标](results/vlm_formal_bf16_v1_metrics.json)

### 视觉预算与数值路径

下表为相同stage2 q/v seed42 checkpoint的内部dev，每格1,617 QA/1,024图；与正式val分开。

| NF4/FP32视觉上限 | 底座共识分 | SFT共识分 | SFT纯生成QA/s |
|---|---:|---:|---:|
| 256 | 71.738 | 74.601 | 2.919 |
| 512 | 78.299 | 82.158 | 1.511 |
| 1024 | 83.562 | 86.506 | 0.686 |

1024−256的SFT增益为 **11.905pp [10.085,13.770]**，生成时间约4.26倍。实际平均视觉token分别为241.607/489.021/938.856，上限不是实际token数；底座也受益，不能归因于LoRA特有机制。

![视觉预算质量成本曲线](assets/vlm_optimization_v1/budget_quality_cost.png)

| 固定256预算、同一SFT checkpoint | 共识分 | 纯生成QA/s | 预处理＋H2D＋生成QA/s | allocated GiB |
|---|---:|---:|---:|---:|
| NF4/FP32 | 74.601 | 2.918 | 2.692 | 5.346 |
| 原权重FP32 | 74.904 | 3.285 | 2.996 | 14.317 |
| 原权重BF16 | 75.343 | 8.015 | 6.510 | 7.207 |

原权重FP32→BF16：纯生成2.44倍、三阶段吞吐2.17倍、allocated下降49.66%。分差+0.439pp，区间[−0.091,0.957]，未证明精度等价。统计排除模型加载、预检、写文件和队列开销；不能当成BF16训练或线上服务加速。

![固定预算数值路径质量成本曲线](assets/vlm_optimization_v1/precision_quality_cost.png)

独立联合诊断的BF16/1024 SFT结果为86.926、纯生成2.172QA/s、allocated8.652GiB；2.44倍仍属于固定256的FP32/BF16比较。[推理结果及代码](docs/VLM_OPTIMIZATION_RESULTS.md) · [机器可读指标](results/vlm_optimization_v1_metrics.json)

### 频次加权与区域分配：保留负结果

新普通CE与频次加权均使用同初始化、QA顺序、32,985 QA/2,062更新和1,843,200参数。统一BF16/1024内部dev，整体 **87.112→86.877**，rare **87.398→87.281**。预设rare至少+0.5pp的门槛未通过，没有自动扩增种子或正式评测。[加权结果](docs/VLM_WEIGHTED_SFT_RESULTS.md)

区域实验固定普通CE seed42最终adapter，不新增训练；四组共 **6,468条预测**，每题实际 **1,024个合并后视觉token**。CPU检测器只给文字框与置信度，不向VLM提供OCR识别文字或参考答案。图像等比例缩放、灰边填充；随机对照保持选区宽高与面积。

| 新letterbox协议，内部dev1,617 QA | 整体共识分（%） | rare共识分（%） |
|---|---:|---:|
| 单整图，1024 tokens | **84.886** | **82.310** |
| 双整图，512＋512 | 77.372 | 76.491 |
| 全图＋预测文字区域，512＋512 | 79.035 | 78.450 |
| 全图＋随机同形区域，512＋512 | 76.964 | 76.491 |

文字区域相对随机 **+2.072pp [0.752,3.368]**，相对双整图 **+1.664 [0.339,3.020]**，相对单整图 **−5.850 [−7.589,−4.133]**。`gate=true`只表示超过两个双视图控制，并非最优质量。rare相对弱对照的区间跨零，不能宣称长尾改善。

四组allocated均约8.420GiB，缓存VLM阶段吞吐1.74–1.79QA/s；1,024图检测额外554.52秒。等视觉token不等于等信息、总输入、FLOPs或端到端成本。旧native-resize分数不作为同协议基线。[区域结果](docs/VLM_REGION_TOKEN_RESULTS.md) · [区域指标](results/vlm_region_token_v1_metrics.json)

## 安装与本地验证

Python 3.10/3.11，在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e . --no-deps
python -m smoke
```

smoke使用合成数据，不下载图像或模型；检查CE/LA训练、验证选点、checkpoint、固定原型与固定checkpoint校准，输出不作为研究证据。CPU版PyTorch可先从[官方CPU wheel源](https://download.pytorch.org/whl/cpu)安装。

```bash
pip install -r requirements/dev.txt
python -m pytest -q
python tools/check_release.py
```

当前本地验证覆盖 **45项CPU契约**，包含assistant mask、数据隔离、评分、参数预算，以及双rank Gloo相对集中式目标的加权梯度核对。项目使用本地测试、CPU smoke和发布校验，不使用GitHub CI。

GPU研究另装 `requirements/vlm.txt`，按具体协议准备权重、数据、adapter、兼容数值环境和运行路径。原机配置是执行快照，克隆仓库并不自动获得全部实验资产。新任务先检查GPU占用，保留已有进程；本次结构整理不重跑GPU实验。

## 目录与运行入口

```text
RareVLM/
├── src/             CLIP实现与九个vlm_*研究模块
├── tools/           分析、图解学习页、导出与发布检查
├── tests/           CPU契约测试
├── configs/         原实验配置快照
├── requirements/    基础、开发与GPU依赖
├── results/         分协议指标、CSV和历史研究锁
├── assets/          实验图与图片来源记录
├── docs/            协议、结果、复现与学习路线
├── metadata/        源码来源、结构迁移映射和发布清单
├── pyproject.toml   包安装、模块发现与pytest配置
├── README.md        中文首页
├── README.en.md     英文首页
├── CITATION.cff
└── .gitignore
```

保留原有import和模块名：`import methods`、`python -m run_imagenet_lt`、`python -m vlm_region.pipeline`。源码集中在src，配置与输出以项目根为基准。研究过程按[学习路线](docs/JOB_ROUTE.md)连接基本原理、源码、真实图像与保存预测。

目录迁移没有重写旧配置、CSV/JSON结果或研究锁。`metadata/ARCHITECTURE_MIGRATION.json`记录旧/新路径与前后SHA；历史实验绑定提交`2049c0e`。源码改动会触发旧运行锁不匹配，精确历史恢复使用该提交的独立checkout，新运行创建独立输出与来源锁。[迁移说明](docs/ARCHITECTURE.md)

## 证据、边界与引用

- [实验指标](results/main_results.csv)、[种子记录](results/seed_results.csv)、[实验来源](results/provenance.json)与各协议JSON分别保存。
- [图片来源](assets/figure_provenance.json)、[源码来源](metadata/SOURCE_ORIGINS.json)、[历史代码锁](results/research_code_lock.json)、[迁移映射](metadata/ARCHITECTURE_MIGRATION.json)、[发布清单](metadata/PACKAGE_MANIFEST.json)支持复核。
- [深层错误分析](docs/ANALYSIS.md)记录错误转换、先验、margin、表征变化及因果结论限制。
- 单checkpoint图像簇bootstrap区间与三种子SD不同；探索区间没有多重比较校正。未见监督答案不代表预训练未见，也不等同视觉类别长尾。
- 模型、数据集、checkpoint、原始预测和包含原图的离线学习页留在本地；仓库发布源码、聚合指标、图表及学习页生成器。

项目受到TS-MOF/AES/GALE相关研究问题启发；这些科研材料不是本仓库已实现算法的自动归属。主矩阵比较CE/LA与既有校准，未声称新算法或SOTA。第三方代码保留来源和许可证信息。

## 最新机制对照：教师负样本关系

2026-10-14完成[概率置乱对照](docs/CLIP_RETRIEVAL_SHUFFLE_V1_RESULTS.md)：新增三个完整28k图/10epoch训练、三个完整内部dev评分格；六个历史checkpoint与七dev格只读复用。保留正项教师概率、负项概率集合、教师/混合目标熵与损失系数，只置乱负候选对应。原蒸馏减置乱的文搜图R@1为+0.800pp（条件95%图像区间[0.227,1.393]），图搜文+0.267pp（[−0.733,1.300]跨0）。两方向联合门槛未过，不新增test；不能把全部蒸馏收益归因负项语义关系，也不证明两目标等价。原完整test提升保留，正式简历未改。[协议](docs/CLIP_RETRIEVAL_SHUFFLE_V1_PROTOCOL.md)、[便携指标](results/clip_retrieval_shuffle_v1_metrics.json)、[损失实现](src/clip_retrieval_shuffle/loss.py)。后续图文重排另立协议，尚未启动。

## 图文重排：完整量化结果

2026-10-14完成[固定Top50的BLIP匹配重排](docs/CLIP_RETRIEVAL_RERANK_V1_RESULTS.md)：0新训练、三已有CLIP蒸馏checkpoint及一固定COCO匹配BLIP。完整test文搜图R@1 **67.820±.122→85.093±.023（+17.273pp）**，图搜文 **79.833±.404→95.067±.058（+15.233pp）**；条件图像区间[15.7265,18.880]/[13.033,17.467]，dev/test全部门槛通过。新增模型容量/384输入/BF16与计算，不是等计算的原创算法或工业效果；预训练重叠未知、test历史曝光。

test900000逻辑pair去重为312116实际pair，两A100各约75.6–75.8秒匹配，缓存评分allocated1.464GiB/卡；完整test流程247.8秒含输入/排队/预检/IO/评分，首次模型传输/代码准备/报告恢复另计，不当线上延迟。旧Top10精确复现、候选集合/R@50不变及六新格独立重评分通过。[协议](docs/CLIP_RETRIEVAL_RERANK_V1_PROTOCOL.md)、[便携指标](results/clip_retrieval_rerank_v1_metrics.json)、[匹配代码](src/clip_retrieval_rerank/common.py)。原计划与JD来源见[行动记录](docs/NEXT_ACTIONS_20261014.md)，正式简历未改。

软件引用见[CITATION.cff](CITATION.cff)。仓库许可证发布仍待维护者确认；第三方数据与权重按其来源分别获取。

## 曝光受控困难负例：2026-10-14完成

固定已有蒸馏损失，新增普通hard与每图每epoch最多8次hard入池两组，各三seed/完整28k图/10epoch。内部dev随机蒸馏→曝光受控：T2I69.907→70.033（+.127pp，95%条件图像区间[−.627,.887]），I2T84.733→86.000（+1.267pp，[.266,2.233]）。Top1%总负例曝光13.535%→4.692%，但文搜图未达预设+1pp门槛、联合gate失败，未新增test；不称双向正式提升或长尾已解决。

[完成报告](docs/CLIP_RETRIEVAL_CAPPED_V1_RESULTS.md) · [协议](docs/CLIP_RETRIEVAL_CAPPED_V1_PROTOCOL.md) · [便携指标](results/clip_retrieval_capped_v1_metrics.json) · [采样源码](src/clip_retrieval_capped/sampler.py)

## 自适应重排：2026-10-14完成

固定CLIP＋BLIP，500图cal选择margin路由、500图confirm评价，0新神经网络训练。确认集T2I86.587→86.533（−.053pp，下界−.400>−.5），I2T94.400持平，双向预设.5pp条件非劣通过。真实unique pair312780→244340（−21.88%），同卡batch32三轮缓存matcher＋route中位1.277×；不是线上或冷启动加速。30%pair削减门槛未达，联合gate失败、不追加test。

[完成报告](docs/CLIP_RETRIEVAL_ADAPTIVE_V1_RESULTS.md) · [协议](docs/CLIP_RETRIEVAL_ADAPTIVE_V1_PROTOCOL.md) · [便携指标](results/clip_retrieval_adaptive_v1_metrics.json) · [路由代码](src/clip_retrieval_adaptive/route.py)

## 按岗位要求阅读项目

- **Embedding／Rerank**：[三seed表征蒸馏](docs/CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md) → [固定Top50重排](docs/CLIP_RETRIEVAL_RERANK_V1_RESULTS.md) → [真实质量—成本路由](docs/CLIP_RETRIEVAL_ADAPTIVE_V1_RESULTS.md)。
- **VLM训练与评测**：[Qwen受控SFT与正式跨任务结果](docs/VLM_FORMAL_BF16_V1_PROTOCOL.md)，先掌握mask／每题loss／LoRA／能力保持的边界。
- **推理与实验工程**：[强配置复核](docs/VLM_STRONG_CONFIRM_V1_PROTOCOL.md)、来源锁、断点恢复、同卡计时及图像簇非劣。

[JD对齐表](docs/JD_EVIDENCE_MAP.md)将阿里具体检索方向与字节官方方向公告、腾讯历史社招参考分开；不把技术相关性当岗位资格／当前HC或本人已掌握。新增[0／20／50候选预算](docs/CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md)已完成，test实际pair−32.59%／缓存匹配＋route1.476×，双向.5pp条件非劣通过，详见完成报告。

## 0／20／50候选预算：2026-10-14完成

固定模型／Top50池、cal-only κ与旧fastτ，12新dev/test格、0新神经网络训练。test完整重排→预算：T2I85.093±.023→84.887±.012（−.207pp，条件95%图像簇[−.473,.060]），I2T95.067±.058保持；预设.5pp双向非劣通过。真实pair312116→210397（−32.59%），同card batch32三轮缓存matcher＋route中位1.476×，dev/test完整门槛均通过。新增计算优化有质量取舍，非无损、显著提准或线上服务加速；有效prefix覆盖单列。

[完整报告](docs/CLIP_RETRIEVAL_BUDGET_V1_RESULTS.md) · [固定协议](docs/CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md) · [指标](results/clip_retrieval_budget_v1_metrics.json) · [JD对齐](docs/JD_EVIDENCE_MAP.md)


本轮图文数据质量研究已完成：clean dev T2I+0.227pp，全门控失败未新增test；noisy dev T2I+1.160pp，全门控失败未新增test。12次训练，失败分支不新增test；详见[筛选结果](docs/CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md)。原始数据收益与错配压力测试分别报告，正式简历保持。 错配分支图搜文−.533pp、非劣保护失败，不替换已验证的完整蒸馏主方案。


2026-10-15后训练推进：[JD分层统计](docs/POSTTRAIN_JD_DECISION_20261015.md)、[信号审计结果](docs/VLM_ALIGNMENT_SIGNAL_V1_RESULTS.md)。完整10000训练图／50000生成已完成，1480原偏好对、749保守候选。固定语义抽查50/64明确（78.13%）；独立离线重挖217对，排除设计样本后199对，未参与写规则的确认56/64明确（87.50%）。两轮均未达到预设90%及Wilson95%下界80%，**未启动DPO／继续SFT、0参数更新**；两比例不是配对提分。详见[语义审查与真实案例](docs/VLM_PREFERENCE_SEMANTIC_REVIEW_V1_RESULTS.md)及[便携指标](results/vlm_preference_semantic_review_v1_metrics.json)。不把弱标签当人类偏好，不改简历。

上述749/199失败数据仍不训练。2026-10-15另启动[公开局部纠错偏好的噪声诊断](docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md)：官方RLHF-V去重/排评测重叠后3568训练对＋439 heldout。固定64助手审查49明确，事实质量gate仍失败，单独诊断scope不覆盖该结论；比较同一SFT起点、同chosen/223更新的继续SFT与标准DPO，另跑完整POPE9000与内部TextVQA1617及heldout概率诊断。两GPU工程预检、64 CPU合同已通过，训练/参考缓存正在进行，完整性能尚pending；不自动晋级/扩种子/改简历。[启动快照与解释边界](docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_RESULTS.md)记录真实数值，最终以新outputs ANALYSIS为准。

该噪声偏好诊断已完成：初始/继续SFT/DPO Text87.112/86.871/86.840%，POPE87.100/86.933/87.100%，未通过提分/能力保持门槛。DPO在439 heldout上74.03%相对偏好方向改善，但Text仅21条回答变化、8条分变；POPE少3误报又多3漏检，收益抵消。已用权重、真实照片与上游代码核对区分目标迁移、答案范围、数据风险和未证实假设；[失败分析](docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_FAILURE_ANALYSIS.md)与[完成报告](docs/VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_RESULTS.md)保留负结果，不自动追加训练/改正式简历。

视觉merger KL完整单seed复核：Object变化0pp、Text−0.024737pp，4/8门槛通过，保留负结果；见[结果与边界](docs/VLM_VISUAL_MERGER_KL_V1_RESULTS.md)。

GT-CE+KL完整独立复核：3918新预测0评分差异，Object+1.462pp/4张增分图，Text−.087pp，原门槛5/8，未证实综合能力提升。见[结果与真实案例](docs/VLM_GT_ANCHORED_KL_V1_RESULTS.md)。
