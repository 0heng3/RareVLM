# 人工监督完整 SFT v1 结果

单 continuation seed42，完整两分支；不称三种子或正式测试提升。

| 模型 | 人工yes/no正确率 | TextVQA |
| --- | ---: | ---: |
| initial | 95.569620 | 86.753247 |
| control | 94.936709 | 87.353123 |
| human | 95.991561 | 87.371676 |

联合门槛通过：False

完整区间、切片、全部变化预测、实际训练输入与资源记录见 outputs/vlm_human_sft_repair_v1。

retain negative result; no LR/epoch grid or automatic seed/formal expansion
