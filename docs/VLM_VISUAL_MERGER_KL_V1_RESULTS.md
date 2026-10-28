# 视觉merger KL：完整负结果

2026-10-16，单seed42、181 Final更新，完整342 Object和1617 Text。新4个FP32视觉LoRA因子1843200可训练元素；原语言144冻结，总adapter148/3686400。位置、rank、初始化共同变化，不是纯位置消融。

| 完整内部指标 | 初始SFT | merger KL | 变化pp |
|---|---:|---:|---:|
| Object accuracy |95.614035|95.614035|0.000000|
| Text overall |87.111936|87.087199|−0.024737|
| Text unseen |84.032023|84.366812|+0.334789|
| Text rare |87.397661|87.397661|0.000000|
| Text medium |93.625731|93.216374|−0.409357|
| Text frequent |89.280576|88.800959|−0.479616|

原八项门槛4/8通过。Object至少+1pp/严格区间下界>0、balanced点差>0及Text frequent点保护失败。Text整体单侧98.75%下界−0.391569pp，预设.5pp非劣通过；95%区间[−0.345258,+0.280501]pp。6000次按图像整簇抽样，单checkpoint、候选×四主要终点alpha=.05/4，不代表全部历史探索FWER；切片点值不证明显著性或切片NI。

Object 26条raw回答变化、0题得分变化；Text 20条raw变化，11条得分变化（5升6降）。独立1959逐题评分0差，0历史重评分/CI重抽。训练active6869.494秒，reference2442.022秒已含；session7102.454秒、1丢弃工程更新（actor80/reference64），1959正式+16额外评测QA，0新教师回答。工程权重与梯度成功不等于效果成功，候选不扩种子/外部。

原主控制器完成评测/分析后调用未定义汇总函数失败，原STATUS和冻结源码保持。新recovery_v1只汇总完整负结果，0重训/推理/评分；独立Final v4与prediction v3均passed。完整三种子→external4372→POPE9000→root目标不缩减，用户目标尚未完成。

证据：[独立预测验收](LOCAL_ARTIFACTS.md)、[负结果汇总](LOCAL_ARTIFACTS.md)、[原图/真实回答/原理学习页](LOCAL_ARTIFACTS.md)。

下一独立研究是同GT-prefix、同语言LoRA预算的fresh KL控制与KL+GTCE单gamma1候选；不按dev失分题修改训练，也不称DPO/RL。
