# 分段SDPA的强基线正式确认

2026-10-14。旧单checkpoint内部dev在同batch16原始SDPA对照下通过：三轮阶段速度中位5.354×，allocated27.220→10.180GiB，共识86.970→87.130，单侧下界−.211pp高于−.25阈值；不显著提分，非盲测/无损。batch4虽快2.139×，质量下界−.436，未过门槛。上述冻结结果保留。

新`vlm_strong_confirm`改用原始dense4强配置，对比分段SDPA batch16联合配置。目的：验证相对已有更快/更省显存的基线实际吞吐—质量—显存取舍，并覆盖图文、文档及通用评测需求；不能把5.35×直接解释为相对最优batch4的收益。

固定普通CE weighted-v1 seed42最终adapter（与旧正式三checkpoint不同）、原权重BF16、原生整图、视觉上限1024、min_pixels50176、同原task图像/问题/后缀、无截断、greedy最多32token、SDPA、TF32关闭、lm_head最后位置切片；0新训练。不改旧`vlm_segmented`算子或基线库，变动为batch4→16和视觉执行联合包。输入参考只生成后评分。32题CPU输入预检；两条件各完整QA保存实际inputIDs/像素SHA/grid/token，并逐题校验候选与新基线相同，padding可不同。

TextVQA官方val5000QA/3166图，DocVQA官方val5349QA/1285图，MMStar1500QA/1430图。各两条件同GPU，新6格23698条model×QA，不当独立样本。使用旧冻结数据/评分：十参考LOO共识、ANLS、完整18子类宏均值，分别报告。已曝光的公共benchmark和未知预训练重叠不宣称盲测。完成全6格，不根据正式分数调整checkpoint/预算/提示/算子或阈值。

每task再取manifest前200QA、两条件、三轮固定顺序，共3600不评分计时观察；独立额外预热后计量全计划QA。记录CPU原图/校验/processor、H2D、生成、session wall、allocated/reserved、竞争轨迹；无网络/队列/TTFT或线上SLA。来源库/配置/模型谱系与完整batch锁支持一致绑定恢复；资源为每task同卡、3个15秒空闲快照和UUID锁，优先空闲卡不抢占。

配对图像整簇bootstrap6000次、seed20261014，MMStar每次保持18类宏定义；3个预设task非劣比较单侧alpha=.05/3。容许损失TextVQA .25pp、DocVQA .25pp、MMStar .5pp，下界须严格高于负容许值。每task速度三轮中位≥1.5且每轮≥1.1、allocated≤65GiB、资源观察无竞争/查询失败；各task独立门控，不合成跨指标总分。显存可能增加，按相对dense4实际值报告，不把同batch16的62.6%降显存移植到强基线。固定单adapter不能称三种子稳定/能力保持/无损；通过工程门控与个人贡献/掌握核对后再考虑简历条目。

算子解释沿用[视觉分段协议](VLM_SEGMENTED_SDPA_V1_PROTOCOL.md)和真实cu_seqlens图解；该复核补强跨任务与强对照证据，不替代新检索算法研究。
