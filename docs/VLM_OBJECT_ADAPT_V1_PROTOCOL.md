# 物体短答适配与TextVQA回放 v1

2026-10-15，在本轮任何GPU更新或新验证预测前登记。目标：在完整POPE上获得可复核的物体判断改善，并保持原TextVQA短答能力。数据质量与工程准入须先通过[数据协议](VLM_OBJECT_PRESENCE_DATA_V1_PROTOCOL.md)。本研究是监督SFT与回放对照，不称DPO、RLHF或GRPO成果；助手执行也不自动等于用户个人独立贡献。

## 假设与可归因范围

前轮公开长答DPO优化确有发生，但POPE误报少3、漏检多3，整体不变。这里直接提供物体存在性短答监督。两臂的物体QA、顺序、输入分组、初始化和更新次数相同；回放臂额外提供TextVQA训练监督，检验保持能力。与初始模型相比，变化包括监督任务、QA量与更新；不能把全部增益归因为单一格式。两臂并非等总QA或等算力；回放收益只能描述为有额外数据/计算的策略差异。

## 固定训练

- 初始：原Qwen2.5-VL-3B BF16基座，uniform SFT seed42 Final adapter。每一臂、每个训练顺序seed都从同一初始checkpoint出发，不将三个顺序seed当三个独立预训练模型。
- 原BF16权重与视觉编码器冻结；语言attention q/v、rank8、1843200 LoRA参数；先按历史评测路径BF16舍入，再升FP32 master训练，BF16 autocast、dropout0，gradient checkpointing开启。评测按历史BF16 LoRA加载路径，舍入差异不得混作新底座改进。
- 训练数据20120 QA，一遍完整曝光；完整image与QA hash顺序锁。`object_only`只做物体CE；`object_replay`在同一更新另加原TextVQA train_all回放CE。
- 每题assistant token先取平均CE，再按实际QA数平均。两臂物体逻辑批量16，实际microbatch4×累计4；末次只有8物体QA，分母用8。回放每次8QA、micro4×2，独立均值CE，lambda=1；两项相加，不因额外回放稀释物体CE。
- 1258次更新，AdamW lr5e-6、weight_decay0、warmup比例0.1后线性衰减、clip norm1；Final固定，不依dev挑epoch/checkpoint。
- 回放来自原32985 QA完整train_all池，独立带盐SHA顺序，耗尽才循环；本轮实际1258×8=10064 QA曝光，不声称使用完32985条。记录真实唯一QA/图像覆盖。回放分支不得改变物体输入顺序或全局随机状态。
- seeds先42做pilot，通过的臂再0/200复核；物体顺序跨臂严格匹配，额外回放的token、时间和显存分别报告。
- 视觉token上限1024、max序列2048，实际逐题记录，不截断。物体后缀换行`Answer yes or no.`；Text回放后缀换行`Answer the question using a single word or phrase.`。所有labels只在训练/计分使用，生成不读参考。
- 每25更新原子保存adapter、FP32 master、Adam、Python/NumPy/CPU/CUDA RNG、真实输入/顺序游标与源码/数据/config/UUID锁；恢复同role同UUID，输入不一致拒绝恢复。
- 独立与内置工程预检各做一个完整临时更新，lr1e-4后丢弃optimizer，并精确恢复全部LoRA、RNG、registered buffers和Qwen rope/cache属性；工程更新与正式1258次分列。allocated或reserved峰超过65GiB、OOM或合同失败则记录工程拒绝，不静默退回micro1。若需新runtime版本，先另行固定配置/协议，不能覆盖此运行。

## 内部筛选：先完整数据，后复核，最后只选一个候选

所有生成固定BF16/1024、batch4、SDPA、greedy、max_new_tokens32、tf32关闭，保存完整输入IDs/pixel/grid/padding和预测batch，断点按完整batch恢复。原Text内部dev1617 QA、1024图；新物体完整val2560 QA、423图，均衡子集1652 QA从同一预测筛出。initial新物体val真实生成；initial Text/POPE只严格核验并复用前轮全部预测，零重复GPU生成。候选与initial逐题实际输入/图像/批次匹配。

图像簇配对bootstrap6000次、种子20261015。四个主端点采用alpha=.05/4，单侧98.75%下界（分位.0125）；普通95%区间仅探索。先对checkpoint/顺序seed求均值，再整图重采样，不将seed×question视IID。种子均值±样本SD另列，单checkpoint的图像区间不代表跨训练seed不确定性。

pilot seed42必须全部满足：

1. 新物体完整val accuracy比initial至少+1pp，单侧98.75%下界>0。
2. 原Text整体差值单侧98.75%下界>−.5pp。
3. 新物体recall点差>=0、specificity点差>=−.5pp。
4. 原Text rare与frequent切片点差都>=0；切片区间单独报告，点值保护不称非劣证据。
5. 每类正负均衡子集accuracy点差>0，防止只利用问句类别先验。

只为pilot全部通过臂新增order seeds0/200，再按三seed均值检查以上门控，并要求每seed物体accuracy差>0、每seedText整体点差>=−.5pp。选择一个候选：三seed内部物体accuracy最高、再Text最高、再固定condition字符串排序。POPE结果不参与候选选择。

## 完整POPE确认及真正成功条件

只对内部三seed复核选出的一个候选运行三seed全部9000题POPE。三category共享500图，按500图簇重采样，不能当1500图或9000独立图。原POPE已被历史研究暴露，确认不等于盲测泛化。

- 三seed均值accuracy相对initial至少+.5pp，单侧98.75%下界>0。
- 三seed各自accuracy点差均>0；不额外要求每seed各自区间显著。
- 平均F1和recall不下降、specificity点差>=−.5pp；random/popular/adversarial每类均值accuracy均不下降。
- 同时保留内部Text整体差值的单侧98.75%下界>−.5pp、每seedText点差>=−.5pp及rare/frequent均值点差>=0。
- 复核全部预测计数/输入与模型来源、真实权重更新/冻结状态、生成不读target、统计实现和失败病例，展示原图/实际张量/代码与结果；报告成本和局限。

仅pilot或新物体val提升不构成用户“真实提升”目标完成。若没有候选通过，记录这一研究的负结果，保留原目标活跃，依据新证据设计独立研究；不自动追加epoch/改阈值/换seed救门控。原正式TextVQA5000、DocVQA、MMStar不用于此研究选择，不因POPE结果回调超参。

## 资源、成果与边界

共享GPU队列在零compute、空闲>=40GiB、util<10%的连续三次15秒快照及UUID文件锁后动态准入，按角色保持GPU亲和；既有任务不终止或抢占。全部环境/缓存/日志在private目录，输入/源码/配置/数据协议/审查与模型哈希在首个GPU前锁定。峰显存allocated/reserved、纯训练时间、生成时间、队列与完整壁钟分别报告，不能写成线上服务或跨硬件加速。

研究输出位于`outputs/vlm_object_adapt_v1`，数据位于`outputs/vlm_object_presence_v1`。中文项目入口与学习材料随真实状态更新；正式简历、原CV PDF、论文与Agent不改；不用CI、不自动push。原科学模块/config/结果均不覆盖。
