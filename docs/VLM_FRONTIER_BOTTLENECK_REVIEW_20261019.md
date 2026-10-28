# RareVLM增量训练瓶颈：前沿论文与源码对照

检索日：2026-10-19。重点为2025—2026多模态后训练，保留mDPO作为相关机制基础。本轮核对6篇核心论文，归档10个仓库固定提交的26份源码/配置/文档，精读其中与问题直接相关的函数；不是完整仓库复现或穷尽式综述。没有启动GPU训练、安装新框架、修改旧模型/科学设置或新增简历指标。CLIP和此前初次SFT的已有正成果保留，本页讨论其后的增量训练失败。

**判断：优先补充可验证的视觉纠错监督，再处理纠错与保持的优化取舍；数值设置需要固定，但不是剩余负提升的全部原因。视觉路径适配与图像条件目标值得检验，尚未证明是唯一瓶颈。不建议现在重写底座或仅替换RL算法名称。**

## 1. 瓶颈判断首先来自本地证据

|问题|量化证据|支持的判断与边界|
|---|---|---|
|纠错监督的验证覆盖有限|最新1174题训练视图，限制历史初始化缓存后原错33题：yes29/no4；其中有原图审核支持的原错6题：yes2/no4|不能把未逐图核验的27题认定为噪声，也不能把反复训练6题当成充分的泛化方案。这里是历史缓存统计，不是新组批准确率|
|纠错与遗忘抵消|512筛选题初始488对/24错；候选修复6题、破坏5题，净1题|当前问题不是完全没有纠错，而是没有保住收益。若仍破坏5题，相对初始净6需要纠正11题；还要超过更强Q/V控制，则要求更高|
|强基线与统计空间|Q/V控制490/512；相对它至少+1pp需要至少496/512；相对初始净8，若破坏5个初始正确题，需要纠正13/24原错|不是理论上不能提升，但比展示若干正例困难。反复使用的screen仍是探索证据，不能当独立确认|
|更宽语言LoRA没有新增视觉纠错|可训练参数184万→369万，两组Human预测511/512一致；候选比Q/V少保住1个修复|不支持此预算下继续堆语言LoRA容量；不否定所有更宽适配|
|可信负例更新仍不利|完整FP32参考的2个可信no首答NLL：0.85738→0.90408；yes：1.34756→1.21360|对小训练探针存在优化取舍，不能由此断言全训练唯一病因为梯度冲突|
|局部loss不等于能力|约束Adam临时点相对普通Adam，no NLL0.94146→0.90163，但4题生成仍0/4|候选保留为算法消融工具；没有资格直接宣布能力修复或进入长训|
|数值不是全部解释|首步复算梯度与保存Adam动量相对误差6.66e−8；完整FP32下首步一阶预测与实测吻合，但最终no仍退化|固定数值契约、保留小样本FP32参考；不继续把主要工作变成重复dtype审查|

对应本地证据：[最新源码/真实更新报告](VLM_ACTUAL_ADAM_AND_NUMERICS_V1.md)、[监督准入修复](VLM_ATTENTION_SCOPE_V1_CODE_REPAIRS.md)、[完整容量负结果](VLM_ATTENTION_SCOPE_V1_RESULTS.md)。历史缓存限制统计见`outputs/vlm_frontier_bottleneck_review_20261019/LOCAL_BOTTLENECK_COUNTS.json`。

特别需要收紧“模型忽略图像”的表述：初始同问异图成对准确率已90.625%，说明大量题目确实能随图像改变答案。**视觉依赖不足只是困难样本上的待验解释，不能概括整个模型没看图。**

## 2. 六篇论文真正能教给我们什么

|论文/时间|核心机制与可核对结果|对本项目的启示|
|---|---|---|
|[RL makes MLLMs see better than SFT / PIVOT](https://arxiv.org/html/2510.16333v2)，Junha Song等，ICLR2026|研究后训练怎样改变视觉表征；表1的Qwen2.5-VL-3B在作者设置下SFT62.8、GRPO65.9。PIVOT流程包含3M指令样本和20K偏好对，视觉端参与更新|不能把结果直接迁移到我们的语言LoRA。其LLaVA式Stage2源码同时开启vision/projector/language；视觉表示是否需要更新可作单独假设，不应同时换底座和数据|
|[Perception-R1](https://arxiv.org/html/2504.07954)，En Yu等，2025|强调感知任务复杂度、奖励与输出匹配；不是所有感知题都需要长思维链。官方grounding奖励使用IoU²|选择有明确坐标/文字真值的任务，形成可验证反馈；不要把当前yes/no题加上长CoT就当有效探索|
|[Visual-RFT](https://arxiv.org/html/2503.01785)，Ziyu Liu等，2025|把GRPO用于少样本细粒度分类、检测与定位，围绕任务真值定义奖励|可借鉴低样本、视觉感知、匹配SFT的实验结构；奖励代码仍需逐项核对，分类脚本的回退子串匹配不应照搬|
|[mDPO](https://aclanthology.org/2024.emnlp-main.460/)，Fei Wang等，EMNLP2024机制基础|同时比较回答偏好、图像条件偏好，并给chosen相对reference的奖励加软锚定；Bunny的MMHalBench分数DPO2.28、mDPO2.96，非准确率百分点|学会区分“答案更受偏好”与“真正使用图像”。软锚定不能保证每题概率不降；也不能把我们的旧跨图排序换名成mDPO|
|[PAPO](https://arxiv.org/html/2507.06448)，Zhenhailong Wang等，2025/2026版本|在RL正确性目标外增加原图/受扰图的感知差异项；论文表1 Qwen2.5-VL-3B overall47.92→49.92，即+2.00pp，作者另报相对增幅4.36%|与旧静态reference保持KL不同；它鼓励视觉条件差异。仍需防止仅对噪声敏感，论文avg@8/temperature1.0也不能当我们的greedy提升|
|[P²-DPO](https://arxiv.org/html/2606.03376v2)，Ruipeng Zhang等，2026|从增强/退化图像生成视觉相关偏好，再校准训练；区分“未感知到”和“已有视觉证据但后续处理出错”|对我们比再次收集纯文本改写偏好更有启发。但自生成chosen不自动正确；其参考模型采样含义也不能直接等同当前actor在线rollout|

这些是论文各自设置下的报告，不是本项目收益预测或同预算横评。优先级按错误机制与可执行性决定，不按算法热度决定。

## 3. 精读源码后的关键区别

以下链接均绑定本次归档提交；完整SHA、文件哈希和阅读范围在`SOURCE_READING_NOTES.json`。

### 3.1 PIVOT：更新视觉表征与语言LoRA是不同问题

[Stage2配置](https://github.com/junha1125/PIVOT/blob/1e54c7d0c71039216ace92d07b5e35e498d88948/scripts/train/stage2-dpo.sh#L20-L52)与[参数开关](https://github.com/junha1125/PIVOT/blob/1e54c7d0c71039216ace92d07b5e35e498d88948/llava/train/train_dpo.py#L1720-L1742)实际开启视觉、projector与语言参数。所读LLaVA式Stage2脚本不是论文表1的Qwen GRPO子实验，不能混同其配方。

我们最近只改变语言注意力q/v或q/k/v/o；旧视觉merger KL虽然动过视觉连接层，却是另一监督、目标、rank/初始化共同变化的负实验。它不等价于“可靠感知监督下更新视觉编码器”的检验，也不能直接推导下一次解冻必成功。

### 3.2 mDPO：三种关系，不只是再加一个排序loss

[官方trainer](https://github.com/luka-group/mDPO/blob/2221bb317a29dfe0f8d2c710cd8572e9d7ec101a/mdpo_trainer.py#L68-L124)显式计算visual-token-masked的chosen分支，以及回答、图像、chosen锚定三项损失。设ℓ为回答log-prob：

```text
回答项：比较同图同问的chosen / rejected，减去静态reference对应差值
图像项：比较同一chosen在原图 / 低信息图像条件下的log-prob差，减去reference差
锚定项：比较chosen当前log-prob与reference log-prob，施加软保持
```

这和旧`softplus(-(m正图−m负图))`不相同，但仅在现有二元数据上改写公式未必带来新视觉信息。官方代码还依赖Bunny式自定义返回值、`images`、`mask_visual_tokens`接口，不能原样接到本地Qwen的pixel/grid输入。应学习目标与输入条件，而非直接替换整个训练器。

### 3.3 PAPO：感知差异KL不是静态teacher保持KL

[actor损失代码](https://github.com/MikeWangWZHL/PAPO/blob/fb63bf4533b78da5a2bb17bc812375dcb620887d/verl/workers/actor/dp_actor.py#L320-L382)将感知KL以负号加入最小化目标，鼓励原图与受扰图条件差异；普通reference KL保持项则是另一条正号路径。受扰图log-prob也可能缓存或停止反传，移植时必须对应实际路径。

直接最大化图像差异可能只学会检测噪声。没有正确性奖励、真实视觉任务得分和保持约束时，它不是可直接用于本项目的万能辅助loss。

### 3.4 TRL：相同奖励的组不会凭epsilon产生任务梯度

[GRPO归一代码](https://github.com/huggingface/trl/blob/88365e546b5be84821f1a8bd9f167ef3230950e2/trl/trainer/grpo_trainer.py#L2780-L2812)先减组均值，再可选除标准差。若一组奖励全同，任务advantage为0；epsilon解决数值问题，不解决信号问题。

项目历史512个Text训练图、每图4次采样中，363组（70.90%）奖励相同。那是旧Text任务，不能直接当当前yes/no或未来TextOCR的比例；新RL立项仍需确认正确候选可采到、奖励有区分度。PPO增加价值模型也不会自动产生缺失的监督。

### 3.5 高star框架仍需核对具体分支

- [Qwen官方LoRA分支](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/train/train_qwen.py#L163-L179)先冻结全模型，再加q/k/v/o LoRA；tune_mm开关属于另一分支，且`dropout or 0.05`会覆盖显式0。不能仅凭命令行字段判断视觉端已解冻。
- [LLaMA-Factory默认值](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/src/llamafactory/hparams/finetuning_args.py#L569-L579)默认冻结vision tower与projector；[all目标解析](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/src/llamafactory/model/adapter.py#L229-L235)还会受视觉冻结参数过滤。`lora_target=all`不等于全视觉训练。
- [Visual-RFT分类回退分支](https://github.com/Liuziyu77/Visual-RFT/blob/2ffad63b25ddd79bfe25d3e046645401201c89d6/src/virft/src/open_r1/grpo_classification.py#L55-L90)采用双向子串判断。单独复算该布尔表达式时，空答案、`cat/caterpillar`也会通过。我们没有照搬它；本轮没有运行作者trainer，也不由此否定论文实验。

## 4. 是否更换底座或重建项目

**目前保留现有项目和Qwen2.5-VL-3B。** 真实首步梯度/Adam连接已验证，未发现必须重写框架才能训练的证据。迁移新版Trainer、换Qwen3-VL、换视觉编码器都会引入新变量，不能用来替代目前的监督与泛化问题。

可借鉴成熟框架的模块冻结、数据模板、reference与奖励实现；未来若确有多卡/在线采样工程需求，再建立独立适配分支。不要把这些仓库最新main直接装进已冻结环境。

仓库star为2026-10-19 API快照：

|仓库|star|本轮使用范围|
|---|---:|---|
|LLaMA-Factory|75,583|精读冻结与LoRA目标解析|
|verl（原volcengine/verl，现verl-project/verl）|23,928|归档核心算法源码作背景，不称完整训练复现|
|Qwen3-VL|20,086|精读兼容训练入口、LoRA分支|
|TRL|19,551|精读GRPO组奖励/优势逻辑，其余trainer归档作背景|
|ms-swift|15,888|README能力与选型扫描，未称精读全框架|
|Visual-RFT|2,280|任务奖励与分类回退逻辑|
|PR1|290|grounding奖励|
|PAPO|164|原/受扰图感知损失路径|
|mDPO|93|条件偏好和软锚定实现|
|PIVOT|8|视觉参数与Stage2配方|

论文作者仓库star低不表示论文无效，高star也不保证奖励或默认设置适用于我们。

## 5. 推荐下一实验：区域文字证据驱动的SFT

这是研究建议，**不是已准备完成或已启动的实验**。它回到RareVLM原本的文字理解/罕见监督答案主线，将二元VQA保留为保持与诊断任务，避免长期只优化几道no的NLL。

### 数据与监督

优先考察[TextOCR官方标注](https://textvqa.org/textocr/dataset/)：训练集714,770个word annotations、21,778图，并有文字四边形与转写。它不是大模型伪偏好，也不是中文商品或企业私域数据。

只能从当前TextVQA `train_all`图像白名单中选取；TextOCR与TextVQA图像ID对应，不能直接使用官方train全量。内部screen/confirm/final及其他已声明保留图必须排除。不可读/非英文标为`.`的词排除；按原图方向和`points`解释坐标，不凭`bbox`字段名猜xyxy或xywh。

建议把训练题变成“读取指定文字区域”的明确监督：输入完整训练图＋给定区域坐标，输出官方转写；保持普通TextVQA问答回放。区域标注是该训练任务的合法输入，**原TextVQA主评测仍只输入原图与问题，不依据评测答案或GT文字框选crop**。

### 单一变化因素

|固定项|对照方式|
|---|---|
|初始化、数值与生成策略|初始、控制、候选一致；若使用新FP32 head，先建立同策略新基线，不拼接旧指标|
|图像来源与顺序|优先同一批已有训练图；两组都保留相同原问答回放|
|参数与优化器|先固定语言LoRA范围和普通AdamW，不同时加投影、解冻视觉或换底座|
|候选变化|用一部分区域文字转写监督替代控制组的普通继续SFT曝光；研究新增监督粒度，不包装为原创loss|
|预算草案|每组200更新/3200QA曝光、45分钟/臂上限；分配可草拟为1600共同QA＋1600对应监督，实际可用数量须准备后确认|
|成本披露|新增人工来源标注、唯一QA数量、输出长度和实际视觉tokens分别报告；等QA不等于等信息/等计算|

旧约24.5分钟的SFT耗时不是新实验ETA，需正式首批实测。若现有图像/标注不足，不临时放宽划分或借用保留图。

### 量化验收与后续顺序

新OCR研究的指标建议：同口径TextVQA整体与rare都以至少+0.5pp为目标，整体相对初始及匹配继续SFT的图像簇95%区间需支持正收益，frequent和人工yes/no保留保护；具体样本与指标在数据准备完成后一次冻结。OCR词频与原VQA答案频次不能混为同一长尾切片。

这不是降低旧yes/no分支的QA+1pp、pair+2pp门槛，旧失败仍保留。筛选仍披露探索属性，冻结候选后按独立确认/种子复核流程验收；不能把含筛选题的full-dev当独立确认。

在这一步之后再选算法：

1. **固定可信数据后比较优化方法。** Adam位移约束可以作为一个候选，只看新增净纠错与保持；不能用4题loss替代任务分数。
2. **需要视觉偏好时，再做图像条件DPO。** 借鉴mDPO/P²-DPO的视觉干预与保持思想；chosen/rejected应由可见真值验证，同义、空答和错误方向不进偏好。先与相同chosen数据的强SFT比较，不能直接复制旧噪声RLHF-V实验。
3. **有有效奖励与探索空间时，再做GRPO/PAPO。** OCR可研究转写正确性/编辑距离，grounding可研究IoU；先确认候选与奖励分布。PPO暂不优先，它不会自动解决数据和视觉监督不足。
4. **若视觉编码确有瓶颈，再作受控视觉适配。** 固定新监督与目标，仅改变一个视觉更新分支；不能用旧merger-KL负结果否定它，也不能凭PIVOT论文直接保证它有效。

## 6. 检索、源码与证据记录

工件位于`outputs/vlm_frontier_bottleneck_review_20261019/`：`REFERENCES.json`记录6篇去重论文，`PAPER_SOURCES.json`记录公开HTML与SHA，`SOURCE_MANIFEST.json`记录26个文件，`SOURCE_READING_NOTES.json`记录精读函数与固定提交链接，`REPOSITORY_SNAPSHOT.json`记录star，`REWARD_FALLBACK_DEMONSTRATION.json`记录独立布尔表达式复算。

采用nature-academic-search多源检索流程：专用MCP未挂载，OpenAlex补充检索遇429；以arXiv、会议/作者页面与官方GitHub源码补足。两个大仓库的完整tree读取曾截断，但选定原始文件均已取得；新增P²-DPO仓库API遭403限流，因此本轮只核对其论文和官方网页，不称已取得它的固定提交训练源码。没有把Visual-RFT低star fork或同名masked-diffusion MDPO当作官方算法实现。

[交互学习页：前沿方法与本项目瓶颈](LOCAL_ARTIFACTS.md) · [上一轮实际Adam与精度证据](VLM_ACTUAL_ADAM_AND_NUMERICS_V1.md)
