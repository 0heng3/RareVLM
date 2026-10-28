# 物体适配与静态原模型KL保持 v1

2026-10-15，在本轮任何GPU forward、更新或新候选评测前登记。原`vlm_object_adapt_v2`保留完整负结果，不更改其门槛、源码、预测或标签：Only物体+1.7544pp且下界为正，但Text rare/frequent点保护失败；硬标签Replay物体+1.1696pp但单侧98.75%下界0，Text保持下界−.5021pp且切片保护失败。整体/切片点差与其区间不等于已证明普遍遗忘。

## 科学问题与原理

检验将原TextVQA回放的硬标签CE替换为固定初始模型的完整概率分布KL，能否保留物体收益并保持原短答/罕见答案能力。

唯一新候选`object_kl`：

`L = mean_QA(mean_assistant_token CE_object) + 1 * mean_QA(mean_assistant_token KL(p_initial || p_actor))`

KL每个token先对完整词表求和，然后每题对其实际assistant token求均值、最后QA均值；不能对词表再取平均。T=1，温度平方因子为1。teacher为同一个既有uniform SFT42初始checkpoint，非每步current actor.detach；两者同初始化时KL近0，而硬标签CE通常仍有非零梯度。它是保持行为的假设，不保证heldout非劣或收益。

参考[Hinton等知识蒸馏](https://arxiv.org/abs/1503.02531)与[TRL GKD固定源码](https://github.com/huggingface/trl/blob/0dcd18c48bf7e63b49a23da0cb030b0a93a2f39f/trl/experimental/gkd/gkd_trainer.py#L238)。已保存官方仓库/commit/API收据和原源码于本输出`upstream/`。其beta=0实现teacher到student的forward KL；本地保留旧每QA归一契约，上游默认是有效token全局汇总，不能称完全复现。TRL的lmbda是student-generated数据比例，本地replay权重1是另一概念。本研究使用原训练GT前缀、无新采样，属于监督式保持蒸馏，不称on-policy GKD、RLHF/GRPO或模型压缩（teacher/actor同为3B）。

## 固定训练与公平边界

- 沿用已通过61/64审查的数据v2完整1211图/2888物体QA和原完整32985 QA train_all回放池；新外部确认图不训练。训练数据协议、原质量准入与file/pixel/COCO ID边界逐字节绑定。
- 同原uniform SFT42起点、原BF16基座/视觉冻结、语言q/v rank8 LoRA1843200参数；初始LoRA先BF16舍入再升FP32 master，BF16 autocast、dropout0、非重入gradient checkpointing、tf32关闭、SDPA。
- 完整1epoch、181更新、物体每步16题（micro4×acc4）、最后8题按实际8归一；回放每步8题（micro4×2），共1448条既定SHA前缀，不声称32985条全曝光。物体/回放数据、顺序和真实输入与历史硬CE Replay seed42严格核对。
- AdamW lr5e-6、wd0、warmup .1后线性衰减、clip1；Final固定，无dev选epoch。T、KL权重均1，不同时加回放hardCE或调超参；hardCE只可detach作诊断。历史CE control仅seed42，算法差异对照标为单顺序seed探索；三seed主结论针对新方法与共同初始，不扩旧失败control。
- teacher软分布带来额外监督信息/前向/存储；同QA、参数和更新数不称等信息或等算力，不能把新增成本写成训练加速。

## 静态完整分布与工程预检

在正式optimizer之前，用同初始actor、同实际padded输入、同grad-enabled checkpoint前向路径，为全部预定回放微批计算FP32完整assistant log_softmax并detach持久化。无需额外teacher GPU模型。包含原结束token/换行，明确因果移位和label位置。完整cache绑定initial/base/processor/source/config/QA次序、像素/grid/IDs/mask/labels、GPU UUID；每批原子写入file/tensor SHA，缓存全量完成后manifest冻结，训练结束SHA不变。不得将序列sum或topK冒充全词表，不使用BF16压缩参考分布。

缓存前向后精确恢复LoRA、RNG、registered buffers及Qwen rope/cache属性，确保正式初始状态和历史控制相同；预计算成本另列。磁盘先估实际assistant数×完整词表×4 bytes并预留1GiB，缺完整cache批次可按同绑定恢复，不混源/精度/UUID。

独立与正式内置工程预检各做一个临时完整update，lr1e-4；初始KL绝对值<=1e-5、梯度有限、teacher无梯度、临时更新后KL响应>0，再恢复全部LoRA/RNG/runtime并丢弃Adam。工程步实际另计；任一allocated/reserved峰>65GiB或OOM/合同失败则工程拒绝，不静默降micro。正式每25更新原子保存FP32 master、Adam/RNG、游标、实际输入/teacher缓存/source/config/UUID锁；同role同UUID恢复，任何绑定变化拒绝。计算缓存不能改变物体次序或随机流。

## 数据与生成

旧物体val342QA/143图、均衡子集256QA/139图；原Textdev1617QA/1024图。历史初始Object来自v2完整真实结果，初始Text/POPE来自旧公开诊断完整结果；严格核验旧模型/data/processor/source/原始预测及实际input/pixel/grid/padding，复用不称新增生成，不在旧输出目录写新结果。

新增外部数据遵循`VLM_OBJECT_EXTERNAL_DATA_V1_PROTOCOL.md`：1791官方原JPEG、4372QA，实际逐类均衡1742图/3438QA（元数据3342只是预算）。与全部已知旧COCO ID、file/pixel精确重叠0；近重复与预训练重叠未知，不称盲测/COCO完整val或全80类。语义固定64和全fresh CPU必须通过，独立EVALUATION_DATA_ADMISSION绑定实际文件SHA后才允许prepare/任何新GPU。

所有生成保持BF16/1024上限、batch4、greedy、max_new_tokens32、SDPA、tf32关闭、不截断。保存逐题实际IDs/pixel/grid/padding与完整预测batch，target只训练/评分不用作生成输入。Text category missing/None/空串只在已校验原字节后的内存副本作无评分含义的规范化，非空Text类别或object/POPE类别差异必须拒绝；不得改变其他字段/标签。

## 阶段与验收：保持原目标，增加独立确认

四主端点为内部物体accuracy、Text非劣、外部物体accuracy和POPE accuracy，alpha=.05/4；配对图像簇bootstrap6000次、seed20261015，单侧98.75%下界，普通95%仅探索。均值先跨固定checkpoint/顺序seed再采整图，SD另列，不将seed×question视IID；图像CI条件于这些起点，不证明跨预训练模型稳定性，也不控制全历史研究FWER。

1. 独立KL工程预检通过后，pilot42完整训练+342物体+1617 Text。初始外部基线可在同队列另一空卡并行生成，不能用于选择训练参数、候选或epoch，candidate外部评测仍禁止提前。
2. pilot必须全部满足原规则：内部物体>=+1pp且98.75%下界>0，Text下界>−.5pp，物体recall点差>=0/specificity>=−.5pp，Text rare/frequent点差均>=0，内部逐类均衡accuracy点差>0。失败保存完整负结果，不下调阈值救结果。
3. 只为通过的唯一新候选追加order seeds0/200，同一个初始checkpoint/数据/方法；三seed均值过相同内部门槛，每seed物体accuracy点差>0、每seedText整体点差>=−.5pp。sample SD单列。
4. 内部复核通过后，三seed新模型全部评测完整4372外部QA，和真实生成的初始外部基线比较；全部结果齐后才判门槛：均值accuracy>=+.5pp且98.75%下界>0、每seedaccuracy点差>0、平均F1/recall不下降、specificity>=−.5pp、外部逐类均衡accuracy点差>0。各类描述性结果另列，不以外部结果回调训练或挑checkpoint。
5. 外部确认通过后，才对新模型三seed执行完整9000题POPE（不是18类子集）；共享500图整簇重采样，不当1500独立图。均值accuracy>=+.5pp且98.75%下界>0，三seed各自点差>0，平均F1/recall非降、specificity>=−.5pp，random/popular/adversarial各类均值accuracy非降；保留前述Text整体/每seed/rare/frequent保护。
6. 根代理复核完整预测/评分/模型与输入锁、真实权重更新、所有门槛、原图改进/退化病例及成本/学习材料后才可完成用户目标。任何单checkpoint、局部val、准备/缓存/脚本success均不能完成目标。POPE已在历史研究暴露，不能宣称完全盲测泛化。

共享research_runtime.queue继续零compute/空闲>=40GiB/util<10%的三次15秒快照与UUID文件锁，动态空卡准入，不终止或抢占现有任务。全部文件/缓存/临时/日志在private内。输出`outputs/vlm_object_kl_v1`，新代码独立；旧v2失败、原科学模块/模型/预测/正式CV/论文/Agent不改，无CI或自动push。若本轮仍失败，目标保持active，后续依据新证据另预注册，不追加同失败候选epoch/seed或偷偷改标准。
