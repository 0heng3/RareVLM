# 来源机制与实际执行范围

TS-MOF 的多目标协调、AES 的状态感知监督、GALE 的 rare-class signal preservation 是本项目提出问题的来源。源码中的 Stage-II 瓶颈残差思路被重新实现为冻结 CLIP 特征上的 LN/GELU adapter；没有导入旧训练脚本或历史路由接口。

| 来源候选 | RareVLM 状态 |
| --- | --- |
| 瓶颈残差适配 | 已重写并用于受控 CE/LA 对照；不是原 BN/ReLU 模块的逐参数复制。 |
| GALE TailSpec 静态频率尺度/margin/权重 | 在 CIFAR pilot 比较；动态 boost 和 center penalty 未移植。 |
| AES ARS、E-PCG；TS-MOF 梯度投影/专家权重；GALE GES/类别融合 | 作为机制候选保留，未进入 VLM 主实验；不作性能排名。 |
| LA 与 prior correction | 已在固定框架测试，明确采用既有机制。 |

当前项目贡献是工程复现、受控干预与适用范围分析。相关工作包括 [Logit Adjustment](https://openreview.net/forum?id=37nvvqkCo5)、[LIFT](https://proceedings.mlr.press/v235/shi24g.html)和 [Prior2Posterior](https://openaccess.thecvf.com/content/WACV2025/html/Bhat_Prior2Posterior_Model_Prior_Correction_for_Long-Tailed_Learning_WACV_2025_paper.html)。固定强度 train-only P2P-style 不等同完整论文调参复现。

## Visualization workflow credit

The source/estimator/uncertainty and export review used the `scientific-visualization` skill from Scientific Agent Skills. This is a visualization workflow credit, not a RareVLM algorithm contribution. Current reference: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065), current record accessed 2026-10-01.
