# 温和频次加权 SFT：完整训练、单种子负结果

本实验已完成两次 seed42 全数据训练及 3,234 条内部 dev 预测。频次加权未改善预设罕见答案目标：整体 87.112→86.877（−0.235 pp），rare 87.398→87.281（−0.117 pp）；预设推进门控未通过。保留负结果，未追加种子或正式集评测。

## 问题、原理与实现

假设是提高低频监督答案的目标函数系数，可能改善其生成表现。每题先对 assistant 答案 token 的交叉熵取均值 $\ell_i$，再对真实 QA 取均值。加权目标是 $L=N^{-1}\sum_i w_i\ell_i$，不是对全部答案 token 无差别求均值。

```text
f_i = train_all 中实际 raw-modal 监督目标归一化文本的 QA 次数
r_i = clip((20 / f_i)^0.25, 1, 2)
Z   = mean_train_all(r_i) = 1.5860100540591475
w_i = r_i / Z
```

Z 在全训练集固定，不按 batch 重新归一化；监督目标原文本保持不变。权重范围 0.630513–1.261026、均值 1。rare 训练 QA 占 63.741%，加权目标系数占 75.730%；这是目标函数分配，未测量梯度贡献占比。

实现入口：[权重与标注审计](../src/vlm_weighted/data.py)、[weighted_objective 与 DDP 训练](../src/vlm_weighted/train.py)、[dev 评测](../src/vlm_weighted/evaluate.py)、[配对分析](../src/vlm_weighted/dev_analysis.py)、[配置](../configs/vlm_weighted_sft_v1.json)。DDP 用 `local weighted sum * world_size / actual_global_real_QA`，尾 batch 的 sentinel 权重为零，以抵消 DDP 的 rank 梯度平均并保留每真实题均值。

## 固定对照合同

底座 Qwen2.5-VL-3B-Instruct，revision `66285546d2b821cf421d4f5eb2576359d3770cd3`。两组各 32,985 QA / 20,929 图 / 2,062 更新、1 epoch；q/v LoRA rank8、alpha16、dropout0.05，共 1,843,200 可训练参数。训练使用语言 NF4、FP32 计算与视觉、visual cap256、sequence cap768、TF32 关闭。实际运行是 2×A100、每卡 microbatch8、accumulation1、有效 batch16；配置中的历史 4×4 布局被显式运行参数覆盖并记录。

两组新的 adapter 初始化张量、全局 QA 顺序、processor、输入 token 数及冻结参数均匹配。训练输入/答案 token 分别为 9,412,810 / 162,105；冻结参数前后哈希一致。checkpoint 固定最终 epoch，未按本次 dev 结果选择。

评测统一原权重 BF16、visual cap1024、batch4、SDPA、greedy max32，仅图像和问题，无 OCR 文本；共 1,617 QA / 1,024 图，每组相同原始输入。这里 1024 是上限，实际 visual token 为 333–1024、均值 938.856。评分为固定 EvalAI 归一化后的 TextVQA leave-one-annotator-out 共识分，差值单位为百分点。

## 结果与不确定性

频次按训练集实际监督 target 的文本次数分为 0、1–5、6–20、>20，是监督曝光代理，不是视觉实体长尾或预训练频次。

| 切片 | QA | 普通 CE（%） | 加权（%） | 差值 pp | 配对 95% CI（pp） |
|---|---:|---:|---:|---:|---|
| overall | 1617 | 87.112 | 86.877 | -0.235 | [-0.657, 0.205] |
| unseen_0 | 687 | 84.032 | 84.061 | +0.029 | [-0.747, 0.815] |
| rare_1_5 | 342 | 87.398 | 87.281 | -0.117 | [-0.357, 0.000] |
| medium_6_20 | 171 | 93.626 | 93.626 | +0.000 | [-1.221, 1.214] |
| frequent_gt20 | 417 | 89.281 | 88.417 | -0.863 | [-1.847, -0.167] |

每项 CI 对图像整簇做 2,000 次配对 bootstrap，条件于这一对训练 checkpoint；五项是点式探索区间，未多重校正，不表示种子稳定性或等价。推进门槛是 rare≥+0.5 pp、整体及 medium≥−0.5 pp，rare 不通过。正式 TextVQA/DocVQA/MMStar 未用于本候选回调。

## 工程实测与失败观察

| 条件 | 训练 active 秒 | 训练峰值 allocated / reserved GiB | 生成 QA/s | 预处理＋H2D＋生成 QA/s | 评测峰值 allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| 普通 CE | 8551.30 | 6.667 / 8.902 | 2.170 | 1.880 | 8.652 / 10.914 |
| 频次加权 | 8531.97 | 6.667 / 8.902 | 2.246 | 1.948 | 8.652 / 10.914 |

训练和评测控制器合计壁钟 18183.64 秒（约 5 小时 3 分）。评测计时剔除每 session 首 batch，保留 1,613 题，排除模型加载、预检、写盘和评分；没有线上吞吐或跨硬件加速结论。

两组保存回答中 39 题文本变化，21 题得分变化（8 升、13 降），18 题文本变化但分数不变。普通 CE 有 157 零分、166 部分分，共 323 未满分题。rare 的 −0.117 pp 来自一题措辞/共识分变化，不能推出普遍罕见答案退化。机制分类未人工审核；文字变长、数值变化等自动候选标签不等同视觉错误原因。[失败学习页生成器](../tools/build_vlm_weighted_failure_reader.py)复用原始图片及保存预测，未生成新结果；含图片 HTML 未收入 Git。

## 证据与边界

[机器可读汇总](../results/vlm_weighted_sft_v1_metrics.json)保留完整切片、CI、资源指标、adapter SHA、初始化/顺序合同及来源文件 SHA256。原始 `vlm_weighted_sft_v1/analysis/ANALYSIS.json` 的文件 SHA256 为 `5c566d1b7e0dae2308eafc5a3678345c8b9c5ff128d0207687640dd388c8e68b`。摘要中的 artifact 是相对实验输出目录的证据句柄；原始预测、训练 checkpoint、标注和图片留在本地输出，不是仓库可下载内容。公开摘要经过路径与设备身份清理，不冒充原文件的逐字节副本。

该内部 dev 曾被研究探索，公开数据的预训练重叠未知；只有一个 seed42 对照。当前证据支持“该固定温和加权候选未达到推进目标”，不支持“解决长尾”、BF16 训练加速或个人已独立掌握全部实现。下一步复用失败样本定位机制，当前未自动追加训练或正式评测。
