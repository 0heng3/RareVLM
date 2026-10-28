# 匹配错误覆盖SFT完成

|模型|训练128满分|新512 VQA|保持64 VQA|
|---|---:|---:|---:|
|initial|0|93.222656|93.750000|
|repeat32|8|93.417969|93.750000|
|coverage128|9|93.085938|93.750000|

候选对初始与新控制：{'initial': {'delta_pp': -0.13671875, 'ci95': [-0.41015624999999994, 0.0], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}, 'repeat32': {'delta_pp': -0.33203125, 'ci95': [-0.8593750000000001, 0.0], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}}

联合门槛通过=False；Stop this configuration;no steps/seeds/main/CV expansion.

两组各64更新/1024曝光，数据身份与难度/视觉token可不同，单seed42内部筛选，非等计算/盲测或三seed主benchmark。初始同条件预测复用，实际生成背景题不计128训练纠错。原分数与门槛不更改，不以语义审核改写成功。

[预登记](VLM_COVERAGE_SFT_V1_PROTOCOL.md) · [128原图及子集](LOCAL_ARTIFACTS.md)
<!-- POSTRUN_FIT_REVIEW -->

## 训练拟合瓶颈（只读已保存记录）

覆盖组同128题在线首次/末次CE均值1.062310→1.011773；末次CE<0.2为6/128、>1为55/128。重复组32题为1.353773→1.140840。首末在线出现于不同checkpoint和其他题padding上下文，不冒称固定模型的配对loss。训练128题满分0/8/9；候选未纠正119题，不再把扩大覆盖本身当作修复。

验证候选相对初始仅q11660由life box→life、原VQA 1→0.3（−0.7/512=−0.136719pp）；无其余计分变化。不能称普遍崩塌，也不能移除此题或重写原分数挽救门槛。候选对初始95%区间[−0.410156,0]，对重复控制[−0.859375,0]；未证明提分。

|审核错误类型|题数|初始满分|重复32 Final（共同128上）|覆盖128 Final|
|---|---:|---:|---:|---:|
|extra_context|4|0|1|1|
|missing_text|1|0|1|1|
|numeric_text|21|0|0|0|
|spelling|27|0|1|2|
|visual_scale|1|0|0|0|
|visual_time|20|0|0|0|
|wrong_target|42|0|4|4|
|wrong_text|12|0|1|1|

下一项是另行事前登记的[16题拟合正控制](VLM_FIT_POSITIVE_CONTROL_V1_PROTOCOL.md)：新鲜同16题/同4题batch/同初始的2e−5对2e−4，分离正文/EOS并记录目标token竞争。它不是加训失败Final、不是泛化实验，也不改变简历或旧门槛。

原图复核：q11660包装同时可见LIFE和BOX；这是保存答案漏掉词的具体病例，不作新专家标签、不改变原评分。
