# 区域关键决策监督与等QA权重CE：32步匹配对照

2026-10-20登记。前32步CE/DPO各8→11/28且无DPO优势；6QA只读诊断中TELL完整CE3.493→3.343而首决策CE3.424→3.461，PAR首决策margin−1.000→+.240且生成转对，BEFORE仍−6.254。假设完整平均CE在同图词/结束决策上分配不足；这不是已证唯一病因。仅检验关键token监督是否超过普通难题加权，不换框架或RL名称。

## 唯一变化与强对照

冻结前一研究初始生成：28区域QA中20错题，按正确文本与实际初始误答的完整assistant编码找共同prefix后首分歧token，可为EOS；8道初始正确区域QA与256回放不加权。不根据本轮结果重挖负例或挑题。原训练白名单、14图/双框、原完整图像/原生坐标及所有chosen/rejected编码、4条回放/8序列偏好micro、32步/512chosen曝光/256rejected前向、相同初始化/顺序、FP32 q/v1843200、BF16底座/FP32 head、视觉冻结、LR2e-5/warmup4、clip1/dropout0均不变。

普通SFT32复用已完成、绑定的对照；不冒充新训练。额外新训两臂，从同原SFT42起步：

- qa_weighted：20错题 L=CE_full+CE_full，其余L=CE_full。
- decision：20错题 L=CE_full+CE_at_first_divergence，其余L=CE_full。

系数均固定1，逐QA系数总和均2，监督分配不同；不声称梯度范数、实际更新幅度或token权重逐位置相同。保留全部正确答案/EOS/模板CE，不取消EOS、不改变标签。rejected仅保持旧批次形状和诊断DPO量；DPO权重为0，**本轮是加权SFT，不是DPO/RLHF/原创算法**。不扫系数或从负Final接训。

## 快反馈、证据与停止

新增20对/40条真实CPU编码与旧chosen token hash核对，明确共同prefix/正确token位置；新源/协议/配置/父data/reference/baseline/adapter冻结。复用既有trainer、保存/恢复、首正式步参考margin和FP32 master值核对；只新增决策loss和对应trace分量。Final固定32、每臂active900秒cap，1/8/16/24/32存档，共享GPU准入保留其他任务。

先完成同等预算的两臂，再各生成原28区域+64保持QA，不用主Text/Human dev挑点。候选短学习需比初始净>=4正确QA、双框全对增加>=2，保持总分损失<=1/64且新增降分题<=2；扩大研究还需比新qa_weighted至少多2QA和2pair、保持总分不低超过1。均为小样本训练学习准入，不是统计显著或独立泛化成功。

无自动200步/扩seed/主测试或简历更新；完整报告纠正/新增错、同答、保持与训练时间。原主VQA门槛不改。失败后不能由此否定所有token加权，也不能认定纯数据、容量或框架病因。
