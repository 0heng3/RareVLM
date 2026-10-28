# q/v＋视觉merger低秩适配与原始SFT未用图像迁移

2026-10-21事前登记。已有128错误/512回放的学习率与长度配比研究未达到总体提升；同网格换图探针显示训练关联增强，但验证G变化区间跨0。这里新建适配范围对照，不延长旧失败Final、不把诊断子集替代原512指标。

## 数据与评测范围修正

原uniform42初始化的TRAIN_RESULT绑定manifest fd469d…，完整train_all32985QA/20929图/1epoch/0skip。内部dev1617QA/1024图与train_all在image_id及解码RGB pixel SHA上交集均0。该dev已被历史实验观察，属于开发集，不是盲测；预训练重叠未知。新追加训练640题仍全部属于train_all，与dev无图片/像素交集。

原512仅未用于这轮追加训练，属于原SFT train白名单，单列`seen_validation`；新增加完整1617单列`transfer_dev`，不得混列或称原模型/预训练完全未见。两域均保留，不据结果重新切分。训练题仍原128已审不同图/125target各4次，以及原512不同回放各1次，共64更新/1024曝光，沿用plain128原微批顺序。训练128评测保留原468完整batch4背景、只128计分；保持64不变。

迁移1617原顺序/batch4、最后1题，单/多词按原目标空格词数；以原32985实际监督target的VQA规范化频次分unseen0=687/rare1–5=342/medium6–20=171/frequent>20=417。频次是答案文本曝光代理，不是实体类别/预训练频次。旧1617预测只提供已验证输入token/grid/图像/问题元数据，encoder源码、processor和提示绑定匹配后方可复用；另做固定SHA8题实际CPU编码。所有三个模型的1617分数均新生成，不复用旧BF16舍入路径分数；若元数据绑定不匹配则全部新编码并如实记录。

## 三个模型

1. `initial`：原uniform SFT42，FP32 q/v masters1843200、BF16冻结底座/视觉、冻结FP32头/dropout0。旧三cohort初始预测同帧只读复用，仅迁移1617新生成。
2. `qv_control`：旧`vlm_coverage_sft_v2/coverage128` fixed Final64（03c750…），相同起点/数据/顺序/CE/lr2e−4/64更新已完整完成；不重复训练。原三cohort同帧预测只读复用，迁移1617重新生成。
3. `candidate`唯一新候选：同初始144个FP32语言q/v因子逐项精确保留，新增`visual.merger.mlp.0`与`.2`的rank8/alpha16/dropout0 LoRA四因子139264参数，A按CPU seed42 Kaiming uniform(a=sqrt5)、B=0。因此新增分支初始输出0；全可训练参数1982464，比控制多7.56%，不宣称等参数/纯位置因果。语言q/v与merger均更新，视觉编码器、merger基底、语言基底、输出头冻结。不是旧“仅训练高rank merger/冻结语言/物体KL”研究复现。所有master保持FP32，无BF16中间舍入。

同128困难+512回放、assistant正文与实际EOS/去EOS后换行、每QA平均CE、有效batch16/4个micro4；lr2e−4/warmup8/线性衰减/AdamW wd0/全trainable clip1、固定Final64。新增参数用相同lr，数据和采样不另改。若控制历史科学源不等或初始不一致，不将其当匹配控制。

## 短预检与恢复

先CPU真实Qwen meta架构+真实PEFT：74目标/148因子、语言144原值exact、新B零、139264新增参数、保存读取roundtrip及通用FP32 restore通过后冻结。候选真实首步16题input/labels SHA相对旧qv控制必须exact，逐QA初始CE绝对差≤1e−5 nats（预先固定），新merger B梯度须有限非零；初始A梯度为0是B零初始化的预期。验证在第一optimizer前完成，失败0正式更新即退出；不重跑已通过的旧全局加载/损失审查。

逐步保存硬题/回放CE、总梯度、四个merger因子梯度、LR、耗时/显存和逐题trace；checkpoint保存FP32全148因子、Adam/RNG/日志offset和source/schedule/UUID lineage，同role同UUID恢复。最终generic PEFT加载使用实际148因子配置，不能用旧只允许144或旧rank96/120审查器。3任务共享队列，顺序initial/candidate/qv_control，两空卡独立准入；initial完成后自然接qv_control，不干预其他任务。

训练活动1800秒/allocated65GiB；每role评测活动2400秒/allocated65GiB，非有限/超预算退出并保存断点。candidate实际2661生成=468+512+64+1617，其中2321评分；initial/control各1617新生成、其余同条件复用，总5895新生成、64新训练更新，不把复用预测/背景题当新增独立样本。

## 固定验收

候选在`transfer_dev`与`seen_validation`两个域，分别对initial和qv_control整体差值均须≥+.5pp、6000次配对图像簇95%区间下界>0；dev1024簇/seen512簇，单seed固定checkpoint条件区间，未作全历史多重校正，不称种子稳定/盲测。两个域rare1–5对initial≥+.5pp，其余频次桶及单/多词对initial/control≥−.5pp，rare对control≥−.5；保持64对initial/control≥−.5，训练满分净增≥32。seen中原初始已满分多词243保持≥−.5一并保留。全门槛联合，不凭单域/单切片挑成功，不回选checkpoint或控制替补候选。

失败停止该配置，不扩步/seed/官方final5000/外部或CV；全过仅允许另登记种子0/200及原正式跨任务确认，不代表用户完整能力目标达成。结果、原图/模块示意/参数与真实预测写入学习页。

参考：[Qwen官方训练入口set_model](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py)区分视觉塔、merger、语言的训练范围；其当前代码/依赖不等于本地2.5-VL/Transformers4.51.1/PEFT0.15.2，本实验是独立低秩范围对照，不声称官方默认LoRA自动训练merger。实际执行只用已锁本地代码与checkpoint。
