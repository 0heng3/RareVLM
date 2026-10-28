# 批次敏感性首次偏差定位 V1

2026-10-25，输出前冻结。仅定位上一轮已复现的批次依赖，不训练、不选质量配置。锚点为训练题17716：上一诊断中两候选首token分差变化最大（2.378494）。恢复原DATA内mining/fixed两4-QA分组，使用同0109初始权重、原BF16底座/FP32 LoRA及独立head、TF32 off、dropout0。各做一次完整prefill，无KV缓存，无optimizer，共2次forward/8QA；只输出最后位置logits，不生成答案、不重新评分。

读取实际目标图像pixel_values、patch_embed、全部视觉block、merger、视觉最终输出、decoder实际inputs_embeds、全部decoder层最后token、最终head。视觉block内部按window顺序，每图占连续patch范围，按各图grid乘积划分；merger按4个patch合并，最终visual恢复原patch顺序。仅保存每层对应目标tensor的SHA、差值统计，第一组CPU副本只在内存暂存。实际pixel哈希、padding长度、网格和位置编码分别保存；不把不同token/图像片段误作同位置比较。先确认原像素一致，再定位首次非零差。

原HF计算和张量不改；head前切最后1位置只省去未使用输出投影，与旧路径最后位置同定义。结果不证明本轮新fix有效。单GPU共享准入3×15秒、40GiB、零compute、利用率<10%，不抢占；180秒forward/65GiB。源、协议、父DATA/PLAN/包/权重先冻结。若失败保留原版本，不静默改锁内代码。后续修复需另建版本与新同精度基线。
