# 可见实体短答适配与TextVQA回放 v2

2026-10-15；新训练/验证预测前登记。建模v1因数据v1语义质量拒绝而从未进入GPU，不将其写成模型负结果。本v2只替换为另立协议的[18类可见实体数据v2](VLM_OBJECT_PRESENCE_DATA_V2_PROTOCOL.md)；不是同一总体重抽。未据任何新模型分数选择该变化。

本协议完整继承[建模v1的固定训练、生成、统计、阶段顺序、验收门槛、资源与成果边界](VLM_OBJECT_ADAPT_V1_PROTOCOL.md)，以下列出的数据版本/数量替代v1相应项，其余不变。v1原文件保留，不静默改写历史。两个协议、两版本数据来源、审查拒绝与新准入都进入本轮锁。

| 项目 | 本v2 |
| --- | --- |
| 数据 | outputs/vlm_object_presence_v2，动物10类+交通8类 |
| 训练 | 1211不同图，2888QA，1444yes/1444no，一遍完整曝光 |
| 物体内部val | 143图，342QA，171yes/171no |
| 固定更新 | ceil(2888/16)=181，最后8物体QA |
| 回放 | 每更新8QA，共1448 QA（原32985 train_all池的固定SHA前缀，不称全量曝光） |
| 初始模型 | 同原uniform SFT42，所有arms/order seeds共同初始 |
| 参数与数值 | 原BF16基座/视觉冻结，语言q/v rank8、1843200 FP32 master LoRA，BF16 autocast、dropout0 |
| runtime | object micro4×acc4=16，replay micro4×2=8；Final@1epoch；65GiB预检硬门控，无静默降批量 |
| 学习率等 | AdamW 5e-6、wd0、warmup .1后线性衰减、clip1；每25更新完整checkpoint |
| 工程预检 | 独立/内置临时完整update lr1e-4，精确恢复LoRA/RNG/buffers/rope，丢弃optimizer，工程步另计 |
| 成果输出 | outputs/vlm_object_adapt_v2；正式CV/论文/Agent不改，无CI/push |

两臂`object_only`与`object_replay`物体QA/顺序/微批分组/初始化/181更新严格相同。每题token均值CE再QA均值；回放臂额外加lambda1的8题独立均值CE，不能声称等总QA或等算力，也不将两项loss一起平均稀释物体监督。末8物体的分母用8。真实回放覆盖、token/显存/耗时分别报告。

完整val与逐类别正负均衡子集均报告；后者是冻结完整val的子集，按同一预测过滤，不新增/挑选预测。它可揭示类别问句先验，不能证明学到了全80类通用视觉知识。原Text内部dev仍1617QA/1024图；初始Text和POPE只按字节/模型/输入锁复用历史完整预测，初始新物体val要真实生成。

推进顺序与严格门槛不因训练数据变少而降低：

1. 两臂seed42完整训练+完整物体val+Textdev。物体accuracy至少+1pp且单侧98.75%下界>0；Text差值单侧98.75%下界>−.5pp；物体recall点差>=0/specificity>=−.5pp；Text rare/frequent点差均>=0；逐类正负均衡子集accuracy点差>0。
2. 只为pilot全部通过的臂追加order seeds0/200（同一个初始checkpoint），再按三seed均值过以上门槛，并要求每seed物体accuracy点差>0、每seedText点差>=−.5pp。
3. 只按内部物体accuracy、Text、固定名称选一个候选；只有它的三seed进入全部9000题POPE（包括本训练未覆盖的类别），不能用18类POPE子集当成功，也不按POPE挑模型。
4. 完整POPE均值accuracy至少+.5pp且单侧98.75%下界>0，三个seed各自accuracy点差>0；平均F1/recall不下降、specificity>=−.5pp，三个category均值accuracy都不下降；同时保留Text整体/每seed/rare/frequent保护。全部预测/模型来源/冻结状态/实际输入/统计/病例独立审查后才可认定真实提升。

所有CI按图像簇配对bootstrap6000次、seed20261015，alpha=.05/4，单侧98.75%下界；均值先跨checkpoint/order seed再采图，样本SD另列，不能将seed×QA当独立样本。POPE共500共享图不是1500独立图。原正式Text/Doc/MMStar不用于调参。POPE历史已暴露，不称盲测或预训练无重叠。较小内部val可能区间更宽，不能为此降低验收标准。

无候选通过则保存本轮科学负结果；数据/工程拒绝与算法负结果分开。任何局部门控通过、准备完成或脚本success都不自动满足用户目标。原目标保持活跃，后续以新证据另立协议，不擅改本轮阈值/数据/epoch或追加相同失败候选种子。
