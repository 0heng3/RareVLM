# 原生整图纠错：等参数语言/视觉骨干适配 V1

2026-10-26，任何GPU输出前冻结。上一轮KL实际降低Final回放分布KL23.2593%，但512条原生答案与SFT完全相同；完整训练梯度KL/CE11.1221%、cos−.112416不支持无反传或严重整体冲突。当前剩余假设之一是：仅语言侧适配改变输出习惯而未形成稳健图像条件纠错。此前merger适配和IIIT词图视觉LoRA均未证明整图问答收益；本研究不能重复包装其结果，而是在相同原生纠错QA上对照视觉骨干/语言侧适配组合。

## 唯一研究因素与共同起点

从稳定数值研究0109 OCR增强初始语言adapter保留144 FP32因子、1843200参数；与IIIT scope已验证初始适配器的60个视觉因子合并，全部30个视觉B精确0，A沿用其seed42初始值，不带任何视觉训练权重。CPU真实PEFT注入/形状/rank/scaling/保存往返核对，再冻结新的204因子复合起点。

两组同起点全部204因子；language仅更新语言q/v rank8/alpha16共1843200，视觉60因子冻结；visual仅更新视觉blocks2–31 attn.qkv rank12/alpha24共1843200，语言144因子冻结。等参数不是等FLOPs或纯位置因果；适配范围、rank与投影组合同时改变，按组合报告。底座/head全部冻结，视觉BF16、语言embedding/decoder/LoRA/head FP32，逐图视觉forward、TF32off/dropout0沿用修复策略。视觉候选实际visual.train启用非重入checkpoint；语言控制visual.eval。最终验证非本组adapter因子值保持，不能拿视觉初始A梯度0误判无学习。

## 数据、目标与预算

原384 QA/CPU chosen hash/32×4 micro顺序逐字节沿用native_stable_kl_v1：128看图支持纠错各2次+native回放128/OCR128各1次，每臂512 chosen曝光，micro4/effective16。共同标准逐QA有效assistant token均值CE，L=Σ16题CE/16，无KL/DPO、无负序列forward、无新标签/问题/评测选样。lr2e−4、warm4/linear32、AdamW wd0/clip1/seed42、1024视觉token上限、真实EOS及pad−100不变。两组各固定32更新，0追加步或seed。

## GPU准入与首步预检

两组各独立零optimizer预检：eval模式原hard前8题（原固定batch4）实际生成与冻结0109 baseline的completion IDs/prefix逐项一致，属于新复合图的8题spot check而非全域证明。随后同一首16QA forward/backward，实际chosen CPU hash相同，参数范围/count/FP32、有限梯度、冻结adapter grad None；visual必须30个B梯度非零、A梯度全0、actual visual checkpoint=True。模型hash保持，≤65GiB。两首16QA平均CE之差≤1e−5，未过则在0正式更新结束，不放宽或直接跑长训。

预检通过才生成全新共同baseline960（两队列各448/512），并各训练32/内部960。预检额外16生成；第一阶段合计2896生成/64正式更新。baseline使用真实复合起点，不直接把历史已测数字当新基线。两组首步同样检查selected/frozen梯度与真实scope，1/8/16/24/32完整Adam/RNG断点、Final非更新因子hash保持、实际512QA计数。每role训练2400秒/65GiB、内部评测1800秒；预检300秒、基线每片1800秒，shared空闲UUID三次15秒/≥40GiB/利用率<10%/零compute，不操作其他任务。

## 评价、条件推进与停止

报告hard满分/VQA、native512整体、single/multi/rare366、OCR256与ret64，按原图簇配对95%区间（OCR用filename-family）。已有内部反复使用、native图原SFT见过、OCR父研究已训练，非盲测/预训练未见。旧KL负结果不重评/回写，只作历史背景。

原学习门槛≥32hard满分且VQA+20pp、OCR及native整体/单/多/rare/ret各≥−.5pp保持。仅visual全部通过且native512对新初始≥+.5pp、对新language控制≥+.25pp时，才追加新初始、language、visual各1617（4851生成，0追加训练）。局部能力信号须对新初始/新language各≥+.5pp且正95%区间，同时保持原切片线。未过则停止，不扫rank/lr、不扩步、不过早上PPO/GRPO。原完整强基线/三seed/外部4372/全POPE目标保持未完成，不更新简历、科研/Agent/CI，不push。
