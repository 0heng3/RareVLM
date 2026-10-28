# 同起点不同生成分组：零更新瓶颈检查V1

2026-10-25，任何新诊断输出前冻结。真实native纠错128中，有6题挖掘时≤.3但同0109起点在正式固定分组基线已满分。两处未填充prefix SHA及模型hash相同，当前训练收益已经用新6满分基线计算，未把这6题包装成训练提升。源码collate明确左填充、生成do_sample=False；这些事实不足以判断是BF16/批次边界还是其他状态问题。

本次仅复现这6个事后选定训练病例，不用新dev标签、不重训、不改历史分数。每题恢复原矿工4-QA同伴、正式128基线4-QA同伴，顺序mining1/fixed1/mining2/fixed2，全部同一已冻结0109模型、同一获准GPU。共6×4×4=96条新生成（24次目标题生成），0optimizer；不能当全验证集不稳定率或新能力收益。companion文本/图像、源码/包/原模型文件在PLAN先固定。

复用已验actor/generated函数，greedy32、相同FP32 head/LoRA与BF16 base、TF32off/dropout0。只挂只读decoder pre-hook与输出head hook，记录实际未填充位置编码hash和每步top10原logits及top1-top2间隔；不改任何tensor/模型方法，不添加forward。hook的CPU同步可改变耗时，不能据此声称逐bit复现历史无hook调用。

先判断同组重复是否稳定，再比较跨组输出、实际prefill位置ID和首分歧token间隔。prefill位置不同提示输入位置链路需继续定位；位置相同但top1在很小间隔下翻转提示数值/计算批次敏感，不能单凭此断言唯一根因；同组重复不稳定则先标生成状态/数值复现问题。单看prefix SHA不能证明实际位置编码与浮点logits完全相同。结果完整保存，不重评分旧缓存，不放松任何原native保护/三seed/外部门槛。

共享3×15秒空闲/≥40GiB/零compute/<10%利用率/UUID锁、不抢已有任务；单worker生成600秒/65GiB。原训练已完成且negative，检查不重启其队列、不扩大训练。原GPU8条零B移除spot check不重做，本次是新发现6例分组差异的独立诊断。无CV/科研/Agent/CI/push改动。
