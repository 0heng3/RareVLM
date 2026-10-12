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
| 可靠偏好后训练 | 同SFT起点：不追加、chosen-only继续SFT、DPO | 先审查有效答案/偏好噪声，再核对reference、beta、长度及曝光；未构建偏好集或执行DPO |
| 服务级推理诊断 | 候选512/1024×并发1/4；每条件200请求×3重复＝2400请求 | p50/p95、首token时间、吞吐、失败率、allocated/reserved；设计未执行，现有QA/s不能当生产SLA |
| 容量/外部底座复现 | 有诊断动机后，同系列7B容量对照或独立新底座协议 | 重新核对processor、LoRA目标、参数与提示；底座变化非纯规模因素，尚未实测 |
| 检索方向分支 | 与目标职责匹配时再建CLIP/SigLIP召回、困难负例、VLM重排 | 独立Recall@K/nDCG及延迟，当前VQA不能替代检索成果；未执行 |

继续当前3B主线、按需借鉴开源模块；更换底座或训练框架由误差机制与实际收益决定。[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[ms-swift](https://github.com/modelscope/ms-swift)或[TRL VLM DPO](https://huggingface.co/docs/trl/dpo_trainer#training-vision-language-models)接入前，须核对冻结模块、LoRA计数、assistant mask、逐题loss、processor、OCR及恢复契约，不能把框架默认值当本项目协议。

2×A100与4×V100按单卡显存、精度支持、模型规模、并行方式和完整实测时间比较。DDP每卡保留完整模型，四卡16GiB不能当一张64GiB卡；已有同A100数值路径收益不是跨硬件速度对照。新任务检查已有进程并等待空闲准入，不抢占已有任务。新研究使用独立模块/配置/输出，正式评测不能回调已冻结权重、预算或checkpoint。
