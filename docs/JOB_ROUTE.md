# RareVLM：岗位能力映射与算法学习路线

更新：2026-10-12（北京时间）。本页面向已有 Python/PyTorch 训练经验、希望结合源码与真实结果学习的读者。主问题是：在有限可训练参数、视觉信息与计算预算下，怎样提高细粒度视觉理解和罕见监督答案泛化，并评估通用多模态能力保持？CLIP 长尾分类与 Qwen 生成式问答属于同一项目的两个研究模块，各自保留数据、指标和协议。

公开仓库提供源码、执行配置快照及轻量指标。完整预测、checkpoint、实时状态和包含数据原图的离线学习页保存在本地归档，未随仓库发布。配置中的原机路径需要在准备数据、模型、adapter 与对应依赖后重新绑定；不能直接把原机执行命令当作 clean clone 后可运行的 GPU 教程。实验已完成不自动代表某位研究者已经掌握或独立实现全部环节，个人贡献应按实际参与范围表达。

## 已完成研究与关键结论

| 阶段 | 完成规模 | 可核验结论与限制 | 公开入口 |
| --- | --- | --- | --- |
| CLIP 长尾适配与固定校准 | ImageNet-LT B/32、B/16、在线增强及 iNaturalist 2018，三种子对照 | ImageNet-LT B/32 的 CE 使 Many +15.06 pp、Few −21.46 pp；固定 CE 后选校准强度可恢复部分尾类表现 | [CLIP结果](RESULTS.md)、[主指标](../results/main_results.csv) |
| Qwen 深入 A+B | 15 次训练、67 项评测、233,839 条保存预测重计分 | 全量 SFT 提高 TextVQA；LoRA 位置优势、通用能力保持尚未证实 | [冻结协议](VLM_DEEP_V2_PROTOCOL.md)、[主实验指标](../results/vlm_deep_v2_metrics.json) |
| 视觉预算与数值桥接 | 两组各 6 格完整内部 dev，共 19,404 条预测，0 新训练 | 视觉预算提高质量且增加成本；固定 256 的原 BF16 推理有实测吞吐/显存收益 | [结果与代码讲解](VLM_OPTIMIZATION_RESULTS.md)、[便携指标](../results/vlm_optimization_v1_metrics.json) |
| BF16 高预算联合诊断 | 4 格、6,468 条预测，0 新训练 | 固定 seed42 的 BF16/1024 同一配置为 86.926 分、2.172 纯生成 QA/s、8.652 GiB allocated | [联合结果](VLM_JOINT_BF16_RESULTS.md) |
| BF16/1024 正式复核 | 12 格、47,396 条预测，0 新训练 | TextVQA 三 checkpoint 收益方向一致；DocVQA/MMStar 区间跨零，能力保持未证实 | [正式协议](VLM_FORMAL_BF16_V1_PROTOCOL.md)、[正式指标](../results/vlm_formal_bf16_v1_metrics.json) |
| 温和频次加权 SFT | 新普通 CE 与加权各 seed42：32,985 QA、2,062 更新；dev 共 3,234 条预测 | rare 未达 +0.5 pp 推进门槛，保留负结果，不追加种子或正式集 | [加权结果](VLM_WEIGHTED_SFT_RESULTS.md)、[便携指标](../results/vlm_weighted_sft_v1_metrics.json) |
| 等实际视觉 token 区域分配 | 4×1,617＝6,468 条预测，0 新训练 | 文字区域超过两个双视图对照，但比单整图低 5.850 pp，当前方案未达到最佳质量 | [区域结果](VLM_REGION_TOKEN_RESULTS.md)、[便携指标](../results/vlm_region_token_v1_metrics.json) |

这些是跨协议评测单元及 model×QA 预测数，不能当作独立训练种子或统计样本数。工程预检也不计入科研训练成果。

### CLIP：整体收益与头尾取舍

冻结 CLIP 图文特征，训练 66,240 参数残差 adapter：`[B,512] → [B,64] → [B,512]`，再与固定文本原型计算 logits。ImageNet-LT B/32 的零样本/CE/验证集选强度的固定 CE 校准，整体准确率依次为 59.42/61.38/65.36%；Few 为 59.22/37.76/61.23%，Many 的 CE→校准为 76.10→69.87%。校准不更新参数，结果支持决策调整的作用，不能据此证明表征没有损伤或提出了新算法。

iNaturalist 2018 的 CE 为 5.05%，固定 α=1 校准降至 4.00%，验证集选 α=0.25 为 5.35%。不能把 ImageNet-LT 的有效强度视作普适规则。代码：[adapter与loss](../src/methods.py#L35)、[选点与训练](../src/run_imagenet_lt.py#L160)、[固定校准](../src/calibrate.py#L49)。

### Qwen：适配收益与跨任务边界

全量 `train_all` 为 32,985 QA / 20,929 图，每次 2,062 更新；此前 10k 阶段为 625 更新。数据与更新数共同变化，不能把收益仅归因于更多数据。q/v rank8 有 1,843,200 可训练参数；q/v rank4＋merger 为 1,835,008，参数预算接近但语言 rank 与位置同时改变。

历史 NF4/FP32/256 正式路径的三种子结果（均值±样本 SD）：

| 条件 | TextVQA 共识分 | DocVQA ANLS | MMStar 18子类宏平均 |
| --- | ---: | ---: | ---: |
| 同路径底座 | 69.030 | 68.730 | 54.732 |
| 全量 q/v rank8 | 72.166±0.062 | 69.901±0.176 | 54.018±0.314 |
| 全量 q/v rank4＋merger | 72.389±0.085 | 69.954±0.066 | 54.941±0.577 |

merger−q/v 的 TextVQA 平均差为 +0.223 pp，三个固定 checkpoint 的同图配对区间均跨零。尚缺 q/v rank4-only 控制；不能命名为已证明的 merger 增益。stage2 q/v seed0 在 V100 训练，其余五次在 A100，训练硬件并非独立控制因素。详见[底座与岗位决策](VLM_BACKBONE_JOB_DECISION.md)。

2026-10-11 20:06 完成的新原权重 BF16/1024 正式路径：

| 任务 | 底座 | q/v三checkpoint均值±样本SD | SFT−底座，pp |
| --- | ---: | ---: | ---: |
| TextVQA | 79.560 | 81.138±0.169 | +1.578 |
| DocVQA ANLS | 91.631 | 91.819±0.019 | +0.187 |
| MMStar 18子类宏平均 | 56.289 | 55.869±0.245 | −0.420 |

TextVQA 三 checkpoint 的图像配对探索区间均高于零，DocVQA/MMStar 均跨零。九项点式探索区间未作多重比较校正；SD 保留 V100/A100 混合训练谱系。公开验证集已有历史曝光，预训练重叠未知，不能称首次盲测或预训练未见。

按实际全量监督 target 曝光频次 0/1–5/6–20/>20，正式 TextVQA 的 SFT 平均变化为 +2.579/−0.118/−1.432/+2.284 pp；辅助参考答案覆盖口径的 rare 则为 +1.087 pp。切片无图像配对 CI，不能用整体收益宣称解决长尾，也不能把负点值直接断言为长尾遗忘。[公开频次图](../assets/vlm_formal_bf16_v1/formal_textvqa_frequency.png)与[正式指标](../results/vlm_formal_bf16_v1_metrics.json)保留两种口径。

### 视觉预算和 BF16：质量与计算一起报告

固定 NF4/FP32 checkpoint，256→1024 上限使内部 dev 的 SFT 分数 74.601→86.506，+11.905 pp，图像配对探索区间 [10.085,13.770]；纯生成吞吐 2.919→0.686 QA/s，生成摊销时间约增加 4.26 倍。底座也增加 11.824 pp，尚未证明 LoRA 特有的分辨率机制。

固定 256，原 FP32→原 BF16 的纯生成吞吐 3.285→8.015（2.44 倍），预处理/H2D/生成合计 2.996→6.510（2.17 倍），峰值 allocated 14.317→7.207 GiB（−49.66%）。分差 +0.439 pp 的区间 [−0.091,0.957] 跨零，未证明等价。以上为推理数值路径对照，无 BF16 新训练或线上服务结果。

随后 BF16 联合 dev 的固定 q/v seed42，在 256/512/1024 为 75.343/82.678/86.926；三阶段吞吐 6.510/3.804/1.873 QA/s，allocated 7.207/7.502/8.652 GiB。最高质量不自动是最佳业务配置。预算是上限，实际 token 逐题记录；原 1024 native-resize 平均视觉 token 为 938.856。详见[联合结果](VLM_JOINT_BF16_RESULTS.md)。

### 温和频次加权：目标权重变了，rare 没有改善

唯一权重候选从实际训练 target 频次计算：`r_i=clip((20/f_i)**0.25,1,2)`，全训练固定 `Z=mean(r_i)=1.5860100540591475`，`w_i=r_i/Z`，目标为 `mean_QA(w_i*ell_i)`。普通 CE 设置 `w_i=1`；不能按每 batch 权重均值重新归一。

两组保持初始化、QA 顺序、32,985 QA/2,062 更新、NF4/FP32/256 训练、1,843,200 参数一致，统一原 BF16/1024 完整内部 dev：

| 切片 | QA数 | 普通CE | 加权 | 变化pp | 95%图像配对探索区间pp |
| --- | ---: | ---: | ---: | ---: | --- |
| 整体 | 1617 | 87.112 | 86.877 | −0.235 | [−0.657,0.205] |
| 未见0 | 687 | 84.032 | 84.061 | +0.029 | [−0.747,0.815] |
| 罕见1–5 | 342 | 87.398 | 87.281 | −0.117 | [−0.357,0.000] |
| 中频6–20 | 171 | 93.626 | 93.626 | 0.000 | [−1.221,1.214] |
| 高频>20 | 417 | 89.281 | 88.417 | −0.863 | [−1.847,−0.167] |

训练 rare QA 份额 63.741%→目标权重份额 75.730%，这不是实际梯度占比。CPU 审计得到 17,227 个归一化 target，75 条归一化空串仍保留 raw 监督，raw/norm 众数差异 294 条。普通/加权训练 active 分别 8,551.30/8,531.97 秒，allocated 均 6.667 GiB；运行资源记录不等于独占基准。

2026-10-12 03:34 完成，五项探索区间来自 2,000 次图像整簇 bootstrap，单 checkpoint 对，不含再训练不确定性、未校正多重比较。rare 未达 +0.5 pp 的预设门槛，因此不扩增种子或正式评测。39 条回答文本变化中仅 21 条得分变化（8升、13降）；rare 下降全部来自 QA29766 的 `spring`→`spring water`（1.0→0.6）。需要区分视觉错误与标注/词面计分，不能从一个反例推断普遍罕见能力退化。[结果及来源](VLM_WEIGHTED_SFT_RESULTS.md)。

### 等实际视觉 token 区域分配：位置收益与全局信息损失

2026-10-12 21:44 完成，固定新普通 CE seed42 最终 adapter、原 BF16/1024、batch4、greedy32。单整图 896×896；其余两视图各 896×448，保比例 letterbox，实际视觉 token 全部为 1024 或 512+512。PaddleOCR/PaddleX 3.0.3、Paddle 3.0.0 的 PP-OCRv5_mobile_det 只检测位置，不向 VLM 输入识别文本或参考答案。随机区域保持 ROI 宽高/面积；三种双槽位条件逐题 prompt/input IDs/grid/padding 匹配。

| 条件 | 整体共识分% | 罕见监督答案分% |
| --- | ---: | ---: |
| 单整图 | 84.886 | 82.310 |
| 双整图 | 77.372 | 76.491 |
| 全图＋预测文字区域 | 79.035 | 78.450 |
| 全图＋随机同形区域 | 76.964 | 76.491 |

文字区域−随机为 +2.072 pp，探索区间 [0.752,3.368]；区域−双整图 +1.664 [0.339,3.020]；区域−单整图 −5.850 [−7.589,−4.133]。`gate=true` 仅要求超过两个双视图控制，不要求超过单整图，不能解释为整体优化成功。相对单整图的四频次切片点值均下降，尚无长尾改善证据。15 个点式探索区间未多重校正，固定单 checkpoint 不代表种子稳定性。

各条件 allocated 约 8.420 GiB，缓存 VLM 阶段吞吐约 1.74–1.79 QA/s；1024 图检测额外累计 predict API 554.52 秒（11,205 框、28 空检测）。检测、初始化、视图准备与 GPU 阶段分别计时，缓存吞吐不能当作新请求端到端速度。等视觉 token 不等于等总 input、FLOPs、独立信息量或成本。新 letterbox 与旧 native-resize 是不同输入协议，旧 87.112 不能作为本轮单整图基线。

区域相对单整图 79题升、186题降，其中118题由满分降至零分。下一步优先零 GPU 复用这些实际输入与预测，人工核对可读性、选区相关性、全局信息与问题类型；机制标签仍待人工审查。入口：[区域结果](VLM_REGION_TOKEN_RESULTS.md)、[视图实现](../src/vlm_region/views.py)、[输入合同](../src/vlm_region/data.py)、[配对分析](../src/vlm_region/analysis.py)。

## 岗位能力映射：历史 JD 样本

以下来源是截至 2026-10-11 的历史核验记录，本次公开整理未重新检索。样本只用于能力映射，不代表公司全部方向、个人申请资格或职位当前持续开放；岗位没有统一 TextVQA 录用分数线。

| 官方来源与岗位类型 | 核验日期/范围 | 相关能力 | 项目证据与缺口 |
| --- | --- | --- | --- |
| [阿里算法工程师—多模态](https://campus-talent.alibaba.com/campus/position/199907620025)，2027届校招 | 2026-10-10 官方公开职位接口正文；10-11 页面可访问但详情403，未重新确认正文或开放状态 | 图文/图视频、内容理解、模型实现、PyTorch、研究与数据分析 | 有受控适配、输入诊断、评测；视频与生成未验证，仍需机制消融 |
| [阿里国际站算法研究员—多模态](https://campus-talent.alibaba.com/campus/position/199909040001)，杭州研究型实习 | 同上，采用10-10历史正文 | 图文文档对齐、检索、长上下文、Agentic RAG、离线/在线实验、延迟吞吐 | 有图文文档任务和质量成本实测；检索Recall@K/nDCG、RAG与在线请求指标尚缺 |
| [字节电商LLM/MLLM](https://joinbytedance.com/search/7621874043991132421)，新加坡正式岗位 | 2026-10-11 官方页面正文可读，job code A115740A | 领域数据、SFT/PEFT、OCR、多任务/多语言、量化与推理工程 | 有SFT、频次审计、区域控制、数值工程；电商治理、多语言、剪枝/蒸馏/TensorRT未验证 |
| [腾讯微信安全内容理解](https://careers.tencent.com/jobdesc.html?postId=2099340394696458240)，广州/深圳社招 | 2026-10-11 [公开ByPostId接口](https://careers.tencent.com/tencentcareer/api/post/ByPostId?postId=2099340394696458240&language=zh-cn)可读，岗位更新时间2026-09-21 | SFT、RLHF/DPO、LoRA、标注—评估—部署、人机协同 | 有SFT/LoRA、错误分析与资源记录；偏好数据/DPO、审核指标、路由、视频及部署未验证 |

另有 2026-09-09 发布的[阿里国际站/Accio研究实习高校稿](https://job.hust.edu.cn/zpinfo3/2431817.htm)，包含杭州、多模态后训练、图文文档理解与训练工程，注明2028届及之后、至少3个月；这只是历史招聘来源，地点信息不等于具体办公区。

[AgentGuide](https://github.com/adongwanai/AgentGuide)及其[算法路线](https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/learning-roadmap-algorithm.md)、[项目交付指南](https://github.com/adongwanai/AgentGuide/blob/main/docs/03-practice/05-ship-agent-project.md)在2026-10-11被用作交付组织参考。其9周、1–2个项目、10–20个eval case是作者建议，不能当官方招聘门槛；README中的简历亮点数字未取得对应原始实验，不与本项目实测比较。项目补强应由真实误差和目标职责决定。

## 边做边学：原理、代码与可检查产出

每轮保留一条证据链：问题与假设 → 必要公式/张量形状 → 源码 → 公平对照 → 真实输入与预测 → 正负结果及后续决策。小输入检查用于理解代码，不冒充新算法实验或历史个人贡献。

| 学习单元 | 原理与实际代码 | 可检查产出 |
| --- | --- | --- |
| CLIP adapter与校准 | [methods.py](../src/methods.py)、[calibrate.py](../src/calibrate.py)：固定表征、残差更新、训练LA加log先验、固定校准减log先验 | 画张量链路，计算66,240参数；核对α仅在val选择，解释Many/Few取舍及iNat反例 |
| LoRA/QLoRA | [model.py](../src/vlm_deep/model.py)：`ΔW=(α/r)BA`，参数`r(d_in+d_out)`；NF4量化的是冻结语言权重 | q为2048→2048、v为2048→256；rank8每层32768+18432，36层合计1843200；区分低秩、插入位置和视觉trunk冻结 |
| assistant-only SFT | [encode/answer_loss](../src/vlm_deep/common.py)、[mask](../src/vlm_sft/common.py)：prompt标签−100仍作上下文，每题answer token平均后再题间平均 | 核对一条真实QA的input/labels，用两种答案长度说明逐题和逐token目标的区别 |
| 频次加权与DDP | [权重数据](../src/vlm_weighted/data.py)、[训练归约](../src/vlm_weighted/train.py)：固定全训练归一，valid sentinel，末步实际QA数 | 对照公式与真实权重；核对33条QA/3更新的恢复检查，解释相同QA不等于相同梯度 |
| 视觉token与区域控制 | [预算评测](../src/vlm_probe/evaluate.py)、[区域视图](../src/vlm_region/views.py)：patch14/merge2，`prod(image_grid_thw)//4` | 核对实际1024或512+512、原图和真实输入；解释双整图与随机同形控制分别排除什么 |
| 生成、评分与统计 | [TextVQA计分](../src/vlm_deep/common.py)、[区域分析](../src/vlm_region/analysis.py)、[正式分析](../tools/analyze_vlm_formal_bf16.py) | 核对保存预测与十参考LOO共识；1/2/3/4个匹配参考得0.3/0.6/0.9/1.0，图像簇bootstrap保留同图多题 |
| 数值路径与性能 | [加载器](../src/vlm_precision/evaluate.py)、[控制器](../src/vlm_precision/pipeline.py) | 区分NF4存储、compute dtype、allocated/reserved、生成/三阶段/完整壁钟；推理收益不能外推训练或线上服务 |

本地归档学习材料：`rarevlm_pilot/outputs/vlm_learning_v1/图解课堂.html`、`rarevlm_pilot/outputs/vlm_learning_v1/图解课堂.pdf`、`rarevlm_pilot/outputs/vlm_weighted_failure_review_v1/学习页.html`、`rarevlm_pilot/outputs/vlm_region_token_v1/learning/学习页.html`。它们包含真实源码、图片和保存预测，未随公共仓库发布；可从[区域学习页构建脚本](../tools/build_vlm_region_reader.py)了解生成方式。展示图、人工放大、权重增量图与教学计算器都须分别标注，不能当作模型输入、注意力或新增实验。

基础出处：[CLIP](https://arxiv.org/abs/2103.00020)、[LoRA](https://arxiv.org/abs/2106.09685)、[Qwen2.5-VL技术报告](https://arxiv.org/abs/2502.13923)、[lmms-eval TextVQA](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/textvqa/utils.py)、[PaddleOCR文字检测](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_detection.html)、[DPO](https://arxiv.org/abs/2305.18290)。上游主分支可能变化，实验以已保存 revision 与版本锁为准。

## 下一步：尚未完成的研究

| 优先级 | 可检验问题与有效对照 | 预期交付与边界 |
| --- | --- | --- |
| 当前优先：零GPU失败诊断 | 复用区域相对单整图的118个满分→零分病例，以及区域胜随机的病例 | 人工核对实际输入、文字可读性、ROI相关性、全局丢失和计分歧义；当前机制分类未完成，不先追加GPU长实验 |
| 等QA/更新数据选择 | 同训练池10k QA、625更新，固定随机/质量多样性/监督频次覆盖策略，三seed | 报告token、独立图数、长度与目标曝光；等QA不是等计算，尚未实现或训练 |
| 多任务能力保持 | 先建独立能力dev；通用视觉train混合，总预算匹配和TextVQA曝光匹配分别对照 | 评测集不进训练；区间跨零不证明保持，须预设可接受损失及相应检验；未执行 |
| 可靠偏好后训练 | 同SFT起点：不追加、chosen-only继续SFT、DPO | 10000图/50000回答已构建；749候选和独立修复199候选的语义质量门控均失败，未执行DPO；详见[审查结果](VLM_PREFERENCE_SEMANTIC_REVIEW_V1_RESULTS.md) |
| 服务级推理诊断 | 候选512/1024×并发1/4；每条件200请求×3重复＝2400请求 | p50/p95、首token时间、吞吐、失败率、allocated/reserved；设计未执行，现有QA/s不能当生产SLA |
| 容量/外部底座复现 | 有诊断动机后，同系列7B容量对照或独立新底座协议 | 重新核对processor、LoRA目标、参数与提示；底座变化非纯规模因素，尚未实测 |
| 检索方向分支 | 负采样、多正聚合、教师蒸馏、机制对照及固定Top50图文重排均已完成；后续先复用保存排名学习和诊断 | 重排test T2I+17.273/I2T+15.233pp，额外模型/信息/计算单列；SigLIP、文档检索与线上检索尚未执行 |

继续当前3B主线、按需借鉴开源模块；更换底座或训练框架由误差机制与实际收益决定。[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[ms-swift](https://github.com/modelscope/ms-swift)或[TRL VLM DPO](https://huggingface.co/docs/trl/dpo_trainer#training-vision-language-models)接入前，须核对冻结模块、LoRA计数、assistant mask、逐题loss、processor、OCR及恢复契约，不能把框架默认值当本项目协议。

2×A100与4×V100按单卡显存、精度支持、模型规模、并行方式和完整实测时间比较。DDP每卡保留完整模型，四卡16GiB不能当一张64GiB卡；已有同A100数值路径收益不是跨硬件速度对照。新任务检查已有进程并等待空闲准入，不抢占已有任务。新研究使用独立模块/配置/输出，正式评测不能回调已冻结权重、预算或checkpoint。

## 2026-10-13：岗位定向展示与批量推理补强

同一个RareVLM保留CV与VLM两种项目描述：CV突出CLIP固定模型校准的可核验正结果，VLM展示已有QLoRA/SFT、数据与正式评测。原简历不自动改写；个人参与及掌握程度必须独立核对。新增独立项目或更换底座需要新的岗位/误差证据，不为项目数量拆项。

两轮OCR已完成并保留负结果：原提示87.112，文字/位置方案85.733，紧凑纯文字83.605；覆盖和文本长度变化不能替代准确率改善。当前独立候选为[质量保护的离线批量推理协议](VLM_BATCHING_V1_PROTOCOL.md)：同checkpoint、原图、BF16/1024输入不变，比较FIFO4、FIFO16、bucket16与bucket32；每组同GPU重跑基线。5格完整dev与3轮同256QA计时尚不能预先写成提速成果；实时状态以本地`outputs/vlm_batching_v1/STATUS.json`为准。

学习顺序保持CLIP适配/校准 → Qwen的LoRA/assistant mask/逐题loss/数据隔离 → 新分桶、数值变化和配对非劣。学习与经授权实验同步推进；源码完成不代表用户已经独立掌握。新输入图解入口是本地`outputs/vlm_batching_v1/learning/准备学习页.html`，完成后补`学习页.html`，均使用真实输入记录及原图展示副本，不冒充注意力或生产服务。

启动后发现batch32容量预检超过65GiB门槛，原五格未全部执行；可行FIFO4/FIFO16/bucket16三格继续，另有同策略硬件复核。原失败状态保留，接续状态在`outputs/vlm_batching_v1/COMPLETION_STATUS.json`及`outputs/vlm_batching_hardware_repeat_v1/STATUS.json`，不能把缺失条件写成成功。随后定位该版本视觉SDPA的`[1,N,N]`稠密允许mask，独立[按cu_seqlens分段合批的SDPA协议](VLM_SEGMENTED_SDPA_V1_PROTOCOL.md)及源码已准备，CPU数学合同3项通过，排队等待GPU；尚未完成真实特征/质量/性能验证。该研究在batch4和16分别做同batch原始/分段对照，不把增batch效果全部归于注意力执行；实时状态在`outputs/vlm_segmented_sdpa_v1/STATUS.json`。

2026-10-14完成更新：分段SDPA同batch16内部dev通过预设条件门槛（速度中位5.354×、allocated27.220→10.180GiB、共识86.970→87.130；下界−.211pp）。batch4质量门槛未过；不称盲测/显著提分/无损。[CLIP图文检索和困难负样本](CLIP_RETRIEVAL_V1_PROTOCOL.md)已完成全部4.288GB/10档校验、6训练及14个dev/test评测格；完整test文搜图冻结58.780→随机61.640±.461→mixed59.487±.647，随机图搜文77.600→74.533±.702，存在双向取舍；mixed门槛失败。[dense4强配置正式确认](VLM_STRONG_CONFIRM_V1_PROTOCOL.md)完成6格23698条评分预测，仅DocVQA通过质量和性能门槛，离线阶段2.385×，allocated8.640→10.216GiB。两任务仍在原项目内，不凑独立项目、不改正式简历，不推广未实现的Reranker/线上部署经验。

## 2026-10-14：固定源码精读与多正目标研究

用户授权精读相关仓库并执行下一步行动。本轮固定上游OpenCLIP commit `7067b350050dc409bfdfdbb3f0cc6c9a1651c000`、BLIP `056a169437371659074aa2732649d5de3bffb4a8`、lmms-eval `a6f96fe9069e1054e27ebfc2863d6589987220a6`，归档关键源码和许可证。只借鉴相关机制，不替换整个框架或底座。

| 源码精读 | 要讲清的实际机制 | 本轮证据/行动 |
| --- | --- | --- |
| [OpenCLIP loss](https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/loss.py) | 对称CE、一对一标签、scale/gather，与本项目5caption多正矩阵不同 | 真实32锚图/64负图的[32,480]/[160,96]与两种正聚合梯度 |
| [BLIP retrieval](https://github.com/salesforce/BLIP/blob/056a169437371659074aa2732649d5de3bffb4a8/models/blip_retrieval.py) | 动量软目标、同ID多正、ITM逐query抽样；困难负例用于匹配分支 | 复用6份负例计数；seed42随机最大44、mixed1145，曝光集中不证明误负/因果 |
| [BLIP retrieval evaluation](https://github.com/salesforce/BLIP/blob/056a169437371659074aa2732649d5de3bffb4a8/train_retrieval.py) | 向量召回TopK、图文cross-attention匹配重排 | 已读机制，尚未在RareVLM执行重排，不包装为已实现 |
| [lmms-eval TextVQA](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/a6f96fe9069e1054e27ebfc2863d6589987220a6/lmms_eval/tasks/textvqa/utils.py) | 十参考LOO、答案归一化、可选OCR与问题格式 | 执行未修改的固定评分函数/class节点，对10000保存预测审计；>1e−12评分差异0，最大误差2.22e−16；两总分81.394/81.378一致。仅评分函数一致，像素/提示/整个框架未对齐 |

独立[多正集合损失协议](CLIP_RETRIEVAL_SET_V1_PROTOCOL.md)与`src/clip_retrieval_set/`于北京时间13:39完成六次训练和七个完整dev格。均匀平均正项CE对正集合概率NCE；只改图搜文正聚合，两方法3seed、完整28k图、10epoch、同初始化/正暴露/逐步负池。三seed原基线逐query指标及Top10与v1完全一致；两组初始化/负池序列/曝光SHA匹配。dev文搜图R1从64.153±.273变为62.967±.630（−1.187pp，95%图像区间[−1.794,−.580]），图搜文78.233±.814→77.467±.513（−.767pp，区间[−2.267,.767]跨0），图搜文nDCG73.714→71.580（−2.134pp）。1/3seed图搜文方向为正，原门槛失败，本轮不追加test，不调整损失/阈值。完整结果见[便携指标](../results/clip_retrieval_set_v1_metrics.json)；实时本地STATUS/COMPLETION均completed。这个受控负结果不能推广为所有正集合目标无效，也不代表新的性能提升。

学习页在本地`outputs/clip_retrieval_set_v1/learning/学习页.html`，已包含实际训练源图、五caption、480项logits/梯度、负曝光Lorenz、固定源码、三seed结果和前8图真实Top5。工程检查不当科研提分，缩略图不当attention或实际224输入，用户掌握仍需独立讲述核对。训练存在GPU0并发记录，不作独占训练性能证明；只用空闲资源准入，未停止其他任务。

## 冻结CLIP关系蒸馏：新三训练与历史对照

用户授权参考相关项目/实习执行下一步。[蒸馏协议](CLIP_RETRIEVAL_DISTILL_V1_PROTOCOL.md)已启动，借鉴固定OpenCLIP的DistillClipLoss，原多正CE加lambda1双向KL，教师同批次冻结特征无梯度；不复制其一对一监督标签。仅新增三个seed的完整训练，四dev对照格与三原checkpoint按SHA复用，不反复训练对照或冒充新独立重复。

本轮已于北京时间2026-10-14 14:33完成，dev全部门槛通过后固定配置完成新test七格，两划分均通过。完整test原mean→distill：T2I61.640±.461→67.820±.122（+6.180pp，95%条件图像区间[5.060,7.320]），I2T74.533±.702→79.833±.404（+5.300pp，[3.467,7.133]），I2T nDCG71.146→76.705。三seed双向均正；I2T相对冻结77.600的+2.233pp区间[−.067,4.533]跨0，预设非劣通过，不称显著超过冻结。新3训练、新dev3格/复用4格、新test7格，不能把14个逻辑格写成14次新评测或独立样本。各test格Top10独立重评分/原mean对v1全数组复现，源文件和简历保护通过。[便携指标](../results/clip_retrieval_distill_v1_metrics.json)

这为Embedding对齐、对比学习、蒸馏及受质量门控评测提供真实证据。lambda1单方向目标等价于`2CE((P_GT+P_T)/2,P_S)−H(P_T)`，因此尚未排除一般平滑正则化作用，不将整套配置收益全部归因教师语义关系；后续若研究机制，另预注册均匀软目标/相似熵对照，不能据test继续调lambda。三个数学合同/真实batch/上游CE−H=KL通过，学习页已含历史baseline分布、完整dev/test及8个固定真实Top5案例。训练/dev/test资源观察零查询失败、零并发记录，仅为观察范围，不称独占性能或线上部署。正式简历和PDF保持，个人理解与贡献另核对。

## 教师负关系置乱：机制证据的边界

2026-10-14 16:10完成[新机制对照](CLIP_RETRIEVAL_SHUFFLE_V1_RESULTS.md)：仅新三个seed完整28k图/10epoch/8750更新，原mean/distill六checkpoint及七dev格按SHA复用。新dev置乱T2I69.107±.147、I2T84.467±.208；原distill69.907±.343/84.733±.404。原减置乱的T2I+.800pp，6000图像簇95%区间[.227,1.393]，3seed均正；I2T+.267pp，[−.733,1.300]跨0，2seed正。两方向各+.5pp/校正单侧下界>0的联合门槛失败，不新评test、不调门槛；只能支持文搜图的负关系价值，不能归因全部语义收益或称两目标等价。置乱仍显著高于CE的探索区间不替代主门槛，正项教师置信度和其他正则机制未隔离。

四项数学/真实CPU/第一CUDA合同、完整初始化/采样流/曝光SHA、10格Top10独立重评分和96历史文件/正式简历保护通过；数值导出与NumPy JSON修复在独立工具和审计，原科学源码/协议不改。[便携指标](../results/clip_retrieval_shuffle_v1_metrics.json)、[概率置乱代码](../src/clip_retrieval_shuffle/loss.py)。本地学习页含真实源图、完整480概率及梯度、三seed结果和6个实际Top5病例，图与7张原图展示已检查。面向JD下一候选为Embedding后的图文匹配重排；文档检索、服务指标仍另立协议，不扩大到全部RL/音视频方向，不等待实验才投递，正式简历按实际掌握和贡献范围定向调整。

## 2026-10-14：图文匹配重排完成

[固定Top50重排](CLIP_RETRIEVAL_RERANK_V1_RESULTS.md)已完成0新训练、三旧CLIP蒸馏checkpoint与一COCO匹配BLIP；完整dev/test全部门槛通过。test文搜图67.820±.122→85.093±.023（+17.273pp，条件95%区间[15.7265,18.880]），图搜文79.833±.404→95.067±.058（+15.233，[13.033,17.467]）。新增模型容量、384输入/BF16与逐候选计算，不作等预算算法/SOTA/工业效果声明，公开集历史曝光与pretrain未知保留。

test900000逻辑pair去重为312116真实pair，两A100各156058pair匹配约75.6–75.8秒，缓存评分allocated1.464GiB/卡；完整流程247.8秒含准备/队列/原生预检/IO/评分，首次传输、代码和报告修复另计。六新重排格独立逐row重评分、原Top10/候选池/R50复现及54历史文件/简历保护通过，六准入队列观察0失败/0竞争。当前英文1000图库；文档检索、中文商品、多语言、视频与在线并发未验证。

学习入口：本地`outputs/clip_retrieval_rerank_v1/learning/学习页.html`，实际原图/384输入、ITM logit差、Top50上界、成功与退化Top5、完整质量/成本；[缓存匹配代码](../src/clip_retrieval_rerank/common.py)、[评分断点](../src/clip_retrieval_rerank/matcher.py)、[便携指标](../results/clip_retrieval_rerank_v1_metrics.json)。先独立掌握召回与联合匹配、概率/logit、缓存去重和配对统计，复用保存错误病例；正式简历按确认的掌握和贡献范围定向修改，不自动新增实验或推Git。

## 曝光受控hard＋蒸馏：完成证据与学习

六次全量训练与六新dev格已完成，随机蒸馏强基线三格只读复用。控制hard曝光后dev图搜文+1.267pp、文搜图+.127pp（区间跨零），联合门槛未通过、不新增test；Top1%曝光13.535%→4.692%仅是机制指标。适用于训练采样/Embedding/受控消融的技术讲述，不等于工业效果或已掌握。学习顺序：真实图和caption→batch-max分数→hard计数与排除事件→CE/KL共同变化→完整召回与区间。

[完成报告](CLIP_RETRIEVAL_CAPPED_V1_RESULTS.md)、[固定协议](CLIP_RETRIEVAL_CAPPED_V1_PROTOCOL.md)、[源码](../src/clip_retrieval_capped/sampler.py)。下一项自适应重排序尚未执行，正式简历保留用户确认的贡献范围。

## 自适应重排：真实计算与质量权衡

0新模型训练，3旧CLIP＋1BLIP；cal500/confirm500。确认双向R1 .5pp非劣通过，实际去重pair−21.88%、同卡batch32三轮缓存matcher＋route1.277×；预设30%pair目标未达，未新增test。适用于Embedding/Rerank/推理成本与严格对照讲述，非单模型线上SLA、工业数据效果或已掌握证据。学习：真实cosine差→阈值只用cal→query路由与pair去重→实际forward计时→图像簇非劣判断→细粒度服饰失败例。

[结果](CLIP_RETRIEVAL_ADAPTIVE_V1_RESULTS.md)、[协议](CLIP_RETRIEVAL_ADAPTIVE_V1_PROTOCOL.md)、[路由](../src/clip_retrieval_adaptive/route.py)。本地学习页有实际Top1照片对照；后续候选预算分层尚未执行。

## 当前JD入口与预算研究

完整的[JD技术—实验数字—源码—业务边界](JD_EVIDENCE_MAP.md)已整理，重点对齐杭州图文表征／Rerank与链路优化，SFT作为定向分支。字节官网实习毕业门槛为2027年9月或以后，用户毕业月份未确认；腾讯社会岗位不计为可投实习。

[0／20／50预算协议](CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md)已固定，15合同与资产绑定通过，控制器288476执行双卡实际full／prefix评分；cal-only规划pair削减33.33%只是准入，不是最终质量或速度结论。只cal选κ、不修改旧fastτ；有效prefix覆盖与保留Top50成员分开。

预算研究完成更新：dev/test门槛均通过，test实际pair−32.59%、缓存matcher＋route1.476×，双向.5pp非劣；T2I点值−.207pp及下界接近边界，不能无损宣传。[完成报告](CLIP_RETRIEVAL_BUDGET_V1_RESULTS.md)与[JD证据表](JD_EVIDENCE_MAP.md)已同步，初始running描述保留为历史准备。后续学习优先源码／图像／上限／统计四项，不自动把助手实施归为本人独立贡献。

## 图文数据质量能力补充（2026-10-14完成）

[一致性筛选结果](CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md)补充多模态数据管线／筛选与受控噪声评测：完整28k图、12次三seed匹配训练，原始score4相对random4文搜图+.227pp、区间跨零；20%来源置换压力测试文搜图+1.160pp（95%CI[.580,1.727]），图搜文−.533pp、.5pp非劣未通过。两分支全门控均失败，未新增test，不替换已验证主方案。已知置换保留率20.04%→.393%只是来源诊断，不是工业语义错标率或干净数据提准。

针对数据质量／表征岗位，可讲清训练集筛选、同预算随机控制、教师偏差、真实图片失败例和R1/nDCG不同目标；参见[JD证据映射](JD_EVIDENCE_MAP.md)。这属于同一RareVLM项目，正式简历保持，个人参与／掌握仍待确认。

## 生成偏好数据与质量门控（2026-10-15完成）

完整10000训练图/50000实际回答构建1480原偏好对、749保守候选。固定助手视觉审查50/64明确（78.125%）；机制修复重挖217对、排设计18对后199对，独立确认56/64明确（87.500%）。两个总体不同，不声称配对质量提升；均未过90%明确及Wilson95%下界80%，未启动SFT/DPO。真实风险含同义表达、答案范围、参考共识与图中文字冲突，不能将规则奖励或8～10票自动当人类正确偏好。

[语义审查报告](VLM_PREFERENCE_SEMANTIC_REVIEW_V1_RESULTS.md)、[独立协议](VLM_PREFERENCE_REPAIR_V1_PROTOCOL.md)、[指标](../results/vlm_preference_semantic_review_v1_metrics.json)、[重挖代码](../src/vlm_preference_review/repair.py)。本轮六CPU合同、128不同图像学习例、源锁与正式简历保护通过，0新GPU生成/0参数更新/无新增模型分数。对齐JD的数据构建、质量分析和评测能力可以讲述；尚无DPO优化结果、RLHF或GRPO训练经验。下一训练要先有可信偏好来源与静态reference协议，既有CLIP/SFT主成果保持，个人贡献按实际确认。

## 公开人类局部纠错的SFT/DPO诊断（启动，尚无性能结论）

RLHF-V官方5733人类局部纠错对，经来源/图像/长度与POPE隔离，训练3568＋heldout439；固定64只有49明确，数据事实gate失败保留。另注册[噪声偏好诊断协议](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md)，比较相同SFT42起点、同chosen曝光/223更新的继续SFT与标准DPO。原BF16基座＋FP32语言LoRA、静态初始SFT reference、assistant sum-logp及mask、可恢复checkpoint、真实GPU工程预检均显式实现；64 CPU合同通过。两A100训练/参考缓存已启动，完整Text1617与POPE9000的生成指标尚未出齐，不能预先写DPO提分或无噪声RLHF成功。此研究对应JD的训练/偏好优化、数据质量与可信评测；先讲清[启动记录](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_RESULTS.md)及真实图文学习，个人掌握和贡献按实际确认。

完成更新：本轮两次正式训练及三模型完整评测均完成，未验证提分。Text DPO相对初始−.272pp、POPE .000pp；数据质量、目标迁移、答案范围/字面评分与物体漏检主导的证据见[失败机制分析](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_FAILURE_ANALYSIS.md)。现在有实际标准DPO优化、静态reference/损失工程和负结果诊断记录；不能写成DPO准确率优化成功、完整RLHF/PPO/GRPO能力或用户本人已掌握。既有检索/蒸馏/预算指标仍为已验证主成果，正式简历保持。


2026-10-15物体适配与保持研究更新：[v2完整结果](VLM_OBJECT_ADAPT_V2_RESULTS.md)已保留两个候选的负联合结论：Only物体+1.754pp，Replay+1.170pp，但Text罕见／高频答案保护等门槛未过；仅单顺序seed42，0新增POPE。下一项[固定初始模型KL回放协议](VLM_OBJECT_KL_V1_PROTOCOL.md)只改变回放监督目标，新增外部1791图／4372QA用于确认，不用于训练或选参。相关JD能力是SFT、冻结／LoRA、静态教师蒸馏、逐QA损失归一、图像簇配对评测和错误分析；GT前缀上的监督KL不等于on-policy训练、RLHF／GRPO或模型压缩。尚无新KL效果或工业场景结论，不改正式简历、不把助手实现执行当用户本人掌握。学习顺序为原图漏检／答案漂移→CE与KL梯度→真实mask／缓存源码→完整保持与外部结果。


物体KL完成与梯度诊断更新：[KL结果](VLM_OBJECT_KL_V1_RESULTS.md)已完整核验：内部物体+2.339pp且区间通过，Text−.315pp／保持下界−.649pp、rare/frequent点保护失败，尚无通用保持或POPE收益。[固定训练点梯度诊断已完成](VLM_OBJECT_KL_GRADIENT_V1_RESULTS.md)，96条训练QA前后向、0更新，三点KL/Object范数比.031/2.699/.333、raw cos略负；条件Adam方向三点KL一阶项均负，不能推出普遍冲突、增lambda必要性或覆盖因果。CPU线程归约故障已用原2线程exact恢复，未放宽检查。

[回放覆盖预算对照已完成](VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md)，状态`completed_negative`（[便携指标](../results/vlm_object_replay_coverage_v1_metrics.json)）。两臂保持同初始模型/物体流/181步/lambda1，各5792回放曝光；wide32为5792unique，repeat8x4重复原1448unique，每步均值不把重复放大为4倍KL。候选新teacher与控制引用旧bank成本分列，批次方差亦变化，不称纯覆盖因果或等计算；repeat8x4仅pilot42探索，不可替补唯一候选wide32。

wide32内部物体95.614→97.661%（+2.047pp，单侧98.75%下界+.552）通过；Text87.112→86.964%（−.148pp，下界−.520）未过.5pp非劣，rare−.322/frequent−.480点保护失败，合计5/8通过、3项失败。Text有11题得分变化（4升7降）。两臂共2新adapter、362正式更新、4丢弃工程step和3918正式新生成model×QA；工程生成、teacher前向与训练曝光另计，model×QA不是独立样本。未扩0/200、候选外部或POPE，真实提升目标仍active。独立预测核验复用既有验收，CI直接引用官方图像簇分析、非独立重采样；科学源保持。对应JD的持续SFT／知识保持、低秩适配、缓存与可恢复训练、数据覆盖和严谨评测仍可作为学习证据，不称通用保持、POPE提分或用户本人已掌握。正式简历和已验证检索／校准主成果保持。
