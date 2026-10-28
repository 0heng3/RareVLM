# 数值路径隔离：已完成固定单例的8条件对照

2026-10-21北京时间07:26完成，8条件/0更新，worker活动59.052秒，controller655469自然结束、退出0、0竞争观察。相同图片/焦点前缀RoPE位置一致，LoRA哈希未变。固定事后单例the pointe（DPO32/q20675），比较e对EOS的实际head logit差：

|执行|full8|prefix8|prefix4|prefix4 generate1|
|---|---:|---:|---:|---:|
|原BF16底座|+.067114|+.067137|−.015055|−.015055|
|同底座数值上转FP32|+.095892|+.095892|+.095892|+.095892|

full8与prefix8在同一前缀的hidden完全相同；BF16 prefix8对prefix4的hidden最大绝对差.5/L2差1.71364，FP32对应约7.15e−5/2.64e−4。同一FP32点积读出也出现同方向翻转，不能仅归于输出头GEMM形状。相同prefix4的forward与generate1 hidden完全一致。

支持该单例的batch形状/组成与数值路径敏感，未支持以未来GT泄漏或该prefix4单token生成接口解释翻转。但generate1只比较prefill，不是对历史完整增量KV缓存路径的全面排除；本次prefix4保留原8批padding/所选子集，并非旧hard_eval完整4题生成批，所以−.015055不能替代旧完整greedy临界值−.143963。

FP32只是在进程内上转相同BF16存储值，也会改变可选SDPA内核；未恢复原FP32底座信息，未训练或导出模型。本事后单例不证明总体FP32提分、唯一训练病因或更换架构的必要性。

[固定协议](VLM_ERROR_NUMERIC_PATH_V1_PROTOCOL.md) · [当前队列](LOCAL_ARTIFACTS.md) · [已完成32题诊断](VLM_ERROR_DECISION_TRACE_V1_RESULTS.md)
