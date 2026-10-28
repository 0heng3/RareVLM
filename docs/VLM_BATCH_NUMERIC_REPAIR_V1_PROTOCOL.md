# 视觉分组数值修复验收 V1

2026-10-26，任何修复预测前冻结。定位V1锚点17716已发现：原始/实际pixels及patch_embed完全相同，visual.block00首先差异(max .0625)，逐层放大。未定位block内部某一算子，不能说注意力跨图泄漏。修复首选R1=imagewise：冻结视觉逐grid调用同一原HF forward，包括窗口/全注意力、merger与inverse permutation；不改权重、图像、位置ID、文本、LoRA或评分。R2=imagewise_fp32_language仅在R1未通过下面的稳定性验收时执行，将已加载BF16语言底座值上转FP32计算，LoRA/head仍FP32，visual仍BF16；这不恢复原始FP32底座尾数。选择只看训练病例稳定性，不看VQA好坏。

两策略都按旧6个事后训练病例的mining1/fixed1/mining2/fixed2（各4QA）与singleton（1QA）生成，共102条/策略，greedy32/相同输入SHA/同GPU。每次记录目标实际visual最终特征SHA；imagewise应使跨组visual输出完全一致。每策略另取原hard128按原顺序前8个不属于上述6的QA，比较单例8条与两组batch4共8条，16条额外生成；不按难易挑QA。每策略合计118条，最多236条，0optimizer。验收：6题组内重复、跨组及singleton输出IDs全部一致，视觉输出全相同；额外8题batch与singleton输出IDs一致。先R1，通过则不运行R2；未通过才R2，同门槛；均未过则不准入训练，不扩大数值网格。

生成输出/特征hash与原完整metadata保存；actual prefix/grid核对原DATA，LoRA hash不变。每策略900秒、65GiB；共享GPU三次15秒/零compute/40GiB/<10%门控，不抢占。仅已选训练病例诊断，不能称总体稳定、算法提分或未见泛化。新训练必须使用同一验收策略的新基线与候选，旧预测/原科学文件保留；不可用历史BF16分数作为新数值口径基线。用户已经授权修复后继续新实验，验收失败必须停在失败工件，不能绕过。
