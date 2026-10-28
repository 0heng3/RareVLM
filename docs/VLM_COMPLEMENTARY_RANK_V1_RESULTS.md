# 同问异图CE＋跨图排序 v1 结果

阶段：screen；联合通过：False。

| 模型 | QA准确率 | 两图全对率 | TextVQA |
| --- | ---: | ---: | ---: |
| initial | 95.312500 | 90.625000 | 86.074219 |
| ce | 95.507812 | 91.015625 | 86.464844 |
| ce_rank | 95.507812 | 91.015625 | 86.484375 |

single continuation seed42, official-train-derived internal image-disjoint data; not official benchmark or three-seed claim

完整区间、升降预测、实际训练输入与成本见outputs/vlm_complementary_rank_v1。
