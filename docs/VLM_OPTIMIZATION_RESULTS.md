# RareVLM：已完成优化、岗位证据与下一步

研究快照：2026-10-11（视觉预算与数值桥接完成时）。项目已有可展示的多模态算法工程证据；研究创新和业务部署还需要独立证据，个人掌握及贡献另行核对。官方岗位没有公开规定达到某个TextVQA分数即合格。后续联合、正式、加权及区域研究的完成状态见[当前路线](JOB_ROUTE.md)。

便携指标：[vlm_optimization_v1_metrics.json](../results/vlm_optimization_v1_metrics.json)。图像原字节与来源哈希：[FIGURE_PROVENANCE.json](../assets/vlm_optimization_v1/FIGURE_PROVENANCE.json)。新实验均已完成，无新增训练。

## 1. 已完成哪些实验

| 实验 | 已完成内容 | 研究问题与边界 |
|---|---|---|
| 历史CLIP长尾主线 | ImageNet-LT B/32、完整B/16、在线crop/flip；iNaturalist2018；三种子CE/LA、固定CE校正与相关控制 | 监督适配的Many/Few取舍；冻结表征后决策干预。标准长尾分类，单独归档 |
| 早期Qwen工程pilot | 图像隔离的小规模QLoRA SFT、生成评测、本地推理、CPU契约 | 单种子子集与旧紧凑答案归一化；不能混入深入实验分数 |
| 深入A：数值路径 | 原权重/NF4 × 底座/早期pilot adapter，完整TextVQA官方val | 观察数值路径差；两个权重路径采用FP32，不是BF16训练 |
| 深入B1：参数分配 | q/v rank8、q/k/v/o rank4、q/v rank4＋merger，10k QA各三种子、每次625更新 | 约184万参数预算的分配；rank和位置同时变化，非单因素消融 |
| 深入B2：扩训练规模 | 冻结stage1 dev规则选前两策略，32,985 QA各三种子、每次2062更新 | 更多数据与更多更新共同变化；TextVQA、DocVQA、MMStar跨任务评测 |
| 视觉输入预算 | NF4/FP32；base与固定qv_r8 seed42；256/512/1024上限，六格各完整dev | 图像信息与计算成本干预，无权重更新；单固定adapter种子 |
| 数值工程桥接 | 固定256；base与同一adapter；NF4/FP32、原权重/FP32、原权重/BF16，六格完整dev | 固定adapter的数值路径迁移，无BF16重训；记录实际dtype、预测变化与资源 |

深入A+B共15次训练、67项评测；后两组新增12项完整dev评测、19,404条预测，合计该生成式深入主线15次训练、79个评测单元。CLIP与早期pilot另计；79不是独立训练种子数。新12格的内部dev为1,617 QA / 1,024图，不能与正式val分数混列。

## 2. 受控SFT与跨任务结果

全量训练三种子均值±样本SD。TextVQA为5,000 QA官方val共识分；DocVQA为5,349 QA的ANLS×100；MMStar为1,500 QA子类宏平均。

| 配置 | TextVQA | DocVQA | MMStar |
|---|---:|---:|---:|
| 固定NF4底座 | 69.030 | 68.730 | 54.732 |
| 全量q/v rank8 | 72.166±0.062 | 69.901±0.176 | 54.018±0.314 |
| 全量q/v rank4＋merger | 72.389±0.085 | 69.954±0.066 | 54.941±0.577 |

TextVQA相对同NF4路径底座分别增加3.136与3.359 pp。merger相对q/v仅0.223 pp；三个固定种子的同图配对区间均跨零，不足以确定优劣。merger策略也降低语言rank，归因还欠qv_r4-only控制。阶段2 q/v seed0在V100训练，其余五次在A100；不能隔离硬件影响。MMStar的小幅变化不证明通用能力恢复或普遍遗忘。

原路径核验：VERIFICATION.json（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_deep_v2/runs/VERIFICATION.json`；完整三种子分析：[公开主实验指标](../results/vlm_deep_v2_metrics.json)。原始源码与配置继续冻结。

## 3. 视觉预算：大幅收益伴随明确成本

固定NF4/FP32、batch4、提示、greedy32，只有最大视觉预算变化。表中分数为内部dev，吞吐为排除session首批后的纯生成QA/s。

| 上限 | 实际平均视觉token | 底座分数 | SFT分数 | SFT生成QA/s | SFT峰值allocated GiB |
|---|---:|---:|---:|---:|---:|
| 256 | 241.607 | 71.738 | 74.601 | 2.919 | 5.347 |
| 512 | 489.021 | 78.299 | 82.158 | 1.511 | 5.860 |
| 1024 | 938.856 | 83.562 | 86.506 | 0.686 | 7.892 |

![固定checkpoint的视觉预算质量与成本](../assets/vlm_optimization_v1/budget_quality_cost.png)

SFT的1024−256为+11.905 pp，按同图整簇配对bootstrap2,000次的95%探索区间为[10.085,13.770]。但每题摊销生成耗时约增加4.26倍，这不是等计算量算法收益。

底座同样增加11.824 pp；两者预算增益的交互仅0.080 pp，区间[-1.539,1.652]。结果支持当前输入预算影响表现，未证明LoRA特有机制。1024下97.90%的dev输入超过原训练768总token上限；推理未截断，但这是输入分布变化，未来高预算训练需要新的长度协议。

## 4. BF16：已测量的工程收益

固定256预算与同一个NF4训练的adapter，三条路径各完整内部dev。BF16将原权重、vision和LoRA转换为BF16，并可能改变SDPA内核，因此不声称只改变compute dtype或量化bytes相同。

| SFT数值路径 | dev分数 | 纯生成QA/s | 生成＋预处理＋H2D QA/s | 峰值allocated GiB |
|---|---:|---:|---:|---:|
| NF4/FP32 | 74.601 | 2.918 | 2.692 | 5.346 |
| 原权重/FP32 | 74.904 | 3.285 | 2.996 | 14.317 |
| 原权重/BF16 | 75.343 | 8.015 | 6.510 | 7.207 |

![固定256预算的精度与资源对照](../assets/vlm_optimization_v1/precision_quality_cost.png)

原BF16相对原FP32：纯生成2.440倍、三阶段合计2.173倍，峰值allocated降低49.66%。相对NF4，BF16纯生成2.746倍，但显存从5.346升到7.207 GiB，不能把两种对照混写成“比NF4更快且显存减半”。

BF16−原FP32共识分差+0.439 pp，配对区间[-0.091,0.957]；规范化预测一致率约96.60%。未预声明等价界限，CI跨零或高一致率不证明无损。模型加载、预检、图像核查、落盘和控制器开销不计入三阶段吞吐，不能称为端到端线上性能。监测未观察到同卡竞争，不能保证完全独占。

上述两矩阵本身没有测量BF16＋512/1024；86.506分与2.44倍加速仍属于不同配置。随后[联合诊断](VLM_JOINT_BF16_RESULTS.md)已完成四格，原BF16/1024固定SFT实测86.926分、2.172纯生成QA/s、8.652 GiB allocated。随后[正式BF16/1024三任务复核](VLM_FORMAL_BF16_V1_PROTOCOL.md)已完成12格、47,396条预测，新协议结果单独见[正式便携指标](../results/vlm_formal_bf16_v1_metrics.json)。

## 5. 长尾证据与真实失败案例

下面频次是全训练train_all每QA实际监督target的规范化计数，不是所有人类参考答案的覆盖次数，更不是视觉实体类别或模型预训练曝光。

| SFT目标频次 | dev题数 | SFT@256 | SFT@512 | SFT@1024 |
|---|---:|---:|---:|---:|
| 未见0次 | 687 | 67.540 | 78.486 | 84.250 |
| 少见1–5次 | 342 | 75.585 | 81.988 | 88.070 |
| 中频6–20次 | 171 | 82.281 | 90.526 | 89.591 |
| 高频>20次 | 417 | 82.278 | 84.916 | 87.674 |

未见目标切片增加16.710 pp是重要观察，仍有答案长度、问题类型、图像难度和标注歧义混杂；不能写作标准长尾benchmark或新长尾方法已经验证。

三个预声明教学样例的保存预测：路牌27482与店名10476的底座在256失败，512/1024成功，而SFT各预算均成功。钟表12257两角色三个预算都失败，1024仍答12:50，参考为12:58。增加像素未解决全部错误；个例解释不替代总体分数。

原图与按记录resize尺寸制作的展示图、预测和score见[公开视觉预算指标](../results/vlm_optimization_v1_metrics.json)。展示图不是processor张量或注意力热图。

## 6. 算法原理如何对应代码

LoRA用`DeltaW=(alpha/r)*B*A`限制更新秩，参数量是`r*(d_in+d_out)`；q/v rank8与q/k/v/o rank4均1,843,200参数，改变更新容量和覆盖位置。merger LoRA更新视觉到语言的映射，未解冻视觉trunk。实现：[vlm_deep/model.py](../src/vlm_deep/model.py)。

SFT只监督assistant回答，prefix标签为-100但仍作为模型上下文。loss先按每问题的有效回答token平均，再按问题平均；不能默默替换为全部token平均。实现：[answer_loss](../src/vlm_deep/common.py)。

本地Qwen2.5-VL是14×14 patch与2×2 merger，视觉token数由处理后网格计算：

```python
actual_visual_tokens = grid_t * grid_h * grid_w // 4
max_pixels = token_budget * 28 * 28
```

预算是上限，241.607/489.021/938.856是实测均值。较多像素能保留小字，但也改变编码、预填充与KV成本，不能将总耗时简单当作token平方。实现：[describe_input与evaluate_batch](../src/vlm_probe/evaluate.py)。

NF4路径的有效权重约为`Q(W)+(alpha/r)*B*A`；迁移到原BF16后为`W_BF16+(alpha/r)*B_BF16*A_BF16`，未进行BF16重新训练。低位存储不保证更快，FP32 compute与解量化有成本；A100 BF16 Tensor Core、SDPA路径和带宽可能共同影响吞吐，本轮没有分离各贡献。实现：[load_path](../src/vlm_precision/evaluate.py)。

完整图解教学保留于图解课堂（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_learning_v1/图解课堂.html`，其旧实验示例与新结果分别标记，不使用人工参考答案裁剪推理输入。

## 7. 岗位适配判断

| 岗位样本 | 已有可展示证据 | 仍需补的证据 |
|---|---|---|
| [阿里多模态](https://campus-talent.alibaba.com/campus/position/199907620025) | PyTorch模型实现、SFT、实验设计、数据分析、输入错误诊断 | 方法创新与机制消融；视频理解/生成未做 |
| [阿里国际站多模态](https://campus-talent.alibaba.com/campus/position/199909040001) | 图文文档感知、质量—成本曲线与推理性能 | 检索Recall@K/nDCG、重排、长上下文/RAG、独立服务延迟 |
| [字节电商MLLM](https://joinbytedance.com/search/7621874043991132421) | 领域SFT/PEFT、OCR相关感知、混合精度、量化对照、分布式恢复与跨任务评测 | 领域数据选择、多任务/多语言、业务验证；未做剪枝、蒸馏、TensorRT |
| [腾讯微信内容理解](https://careers.tencent.com/jobdesc.html?postId=2099340394696458240) | LoRA/SFT、数据隔离、评测、错误驱动改进与资源工程 | 可靠偏好数据/DPO、审核指标、人机路由、视频时序与部署 |

字节正文和腾讯[官方接口](https://careers.tencent.com/tencentcareer/api/post/ByPostId?postId=2099340394696458240&language=zh-cn)本轮可读取；腾讯更新时间2026-09-21。阿里正文沿用2026-10-10历史官方核验，本轮动态正文不可读取，不宣称今日确认招聘状态。岗位样本不代表公司全部岗位。

可公开展示的技术证据是模型适配、受控评测、信息预算诊断和数值推理工程；实验执行不自动等于个人掌握或独立贡献。研究岗仍欠新方法及严格算法对照，业务系统仍欠并发请求、p50/p95、长输入压力、真实领域指标和上线评测。

## 8. 2026-10-11阶段设计与后续状态

下表保留当时的行动设计。步骤1、2已经完成；步骤3后来增加双整图控制并完成四组区域实验，当前方案仍低于单整图。步骤4、5尚未执行，最新结论见[研究路线](JOB_ROUTE.md)。

| 顺序 | 具体实验 | 验收与学习内容 |
|---|---|---|
| 1 | 新独立runner：原权重BF16 × 512/1024 × base/固定qv seed42，共4个完整dev单元；与已完成BF16256组成曲线 | 真正验证高质量与快数值路径能否组合；同batch4/提示/greedy32，记录实际token、预测、paired CI、资源，不预先宣称组合收益 |
| 2 | 按dev冻结一个工程预算，base一次＋现有qv三个种子，正式TextVQA/DocVQA/MMStar，共12格47,396 QA | 新协议外部复核，不新增训练；保持旧三种子表。若研究正式预算收益，256与选定高预算两档共24格94,792 QA |
| 3 | 等总视觉token的信息分配：全图、全图＋自动预测文字区域、全图＋随机同形区域 | 从更多像素推进到同成本更有效；区域不读答案，检测成本计入，图像槽位/格式与实际总token匹配 |
| 4 | 等QA/更新数据选择：随机、质量/多样性、实际监督频次覆盖；10k QA、625更新、三种子 | 对应领域数据优化；报告实际token曝光和独立图数，控制长度/频次混杂，不能把等QA称等计算 |
| 5 | 通用视觉train混合，分别匹配总预算与TextVQA曝光；可靠偏好对后DPO与chosen-only继续SFT对照 | 能力保持、负迁移、偏好噪声；评测数据不进训练 |

[BF16联合诊断协议](VLM_JOINT_BF16_V1_PROTOCOL.md)与四格队列已完成，执行状态记录为STATUS（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_joint_bf16_v1/STATUS.json`。联合结果见[VLM_JOINT_BF16_RESULTS.md](VLM_JOINT_BF16_RESULTS.md)，结果未齐的阶段只记录进度。新模块/配置/结果目录不改已锁定的vlm_deep/vlm_a100/vlm_probe/vlm_precision，空闲GPU准入，不抢占其他任务。

若要证明高预算BF16与FP32等价，另补同预算原FP32对照并预声明容许差异；四格联合诊断本身不提供等价证明。正式评测不用于回调预算或挑checkpoint。

继续当前3B底座。容量对照用同系列7B，较新架构迁移可用官方Qwen3.5-4B，两者独立协议；目前大幅预算收益与BF16工程收益优先于整体重建。

成本先由四格新实测更新。旧stage2 qv seed42训练约2小时23分钟、2062更新，属于NF4/FP32与256预算；不能用推理2.44倍外推训练加速。高预算重训前需20–100更新有限loss/显存/恢复预检并重新固定最大总长度、精度、LoRA和优化器状态。

绘图规范及软件出处沿用两组原始FIGURE_PROVENANCE与analysis记录，含Kassis et al. (2026), Scientific Agent Skills, https://arxiv.org/abs/2609.00065；本页图片未重画、未修改数据。
