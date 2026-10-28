# 训练失败分因诊断：数据、目标、框架与设置

2026-10-17，在既有精度排查后继续区分原因。复用已完成的权重、mask、reference、评分与数值路径证据，不重做上一轮检查，不修改旧实验。

## 新的可区分诊断：真实数据学习正控制

目的只检验当前模型和训练链路能否拟合可靠的真实训练QA，以及学习率/计算精度是否影响短程学习；**不是项目能力实验，不是简历提分或独立泛化证据**。

从现有TextVQA train_all取32QA：16条来自上一轮固定训练样本中NLL较高且人类答案至少8/10同意的题；16条按输入ID哈希选择的同等高共识题。排除空答案、unanswerable和标注反馈元回答。另取32条图像不重叠的训练域probe，不参与本诊断更新；它们曾参与原始SFT，不能叫未见测试。验证/正式集不用于筛选、监督或改参。

三臂共享同一个原SFT42 **raw FP32 LoRA**，BF16底座、language q/v rank8/alpha16、1,843,200可训练参数、整图最多1024视觉token、maxseq2048、不截断，micro4/effective16，dropout0、clip1、AdamW/wd0。固定32更新、每臂512次QA曝光，constant LR用于短程正控制，不是复刻原2062步warmup/decay配方。各臂同样的数据顺序与每QA assistant-token平均CE，CE统一FP32计算。

| 臂 | 全局BF16 autocast | 固定LR | 识别的问题 |
| --- | --- | --- | --- |
| amp_lr5e6 | 开 | 5e-6 | 小LR/BF16计算参考 |
| fp32_lora_lr5e6 | 关 | 5e-6 | 与上臂只变全局autocast策略；不能把全部差异仅归于LoRA matmul |
| fp32_lora_lr1e4 | 关 | 1e-4 | 与上一臂只变学习率20倍 |

所有初始/终点评分统一BF16底座＋FP32 LoRA/no autocast、同batch4、原短答提示、greedy32；防止将评测精度变化当训练收益。步骤8/16/32记录同一fit集合NLL，初始/Final保存fit与probe的生成及官方共识分；不按诊断分选checkpoint或追加训练。

学习正控制事前判据：fit CE至少降低50%，且fit共识点值不下降。即使通过，也只证明局部标签拟合链路有效，不证明视觉泛化、全TextVQA或Object/DPO训练都无问题。如果小LR弱而较大LR可学，只能说明这个局部设置的可优化性，不能推出全量最佳LR或直接以更大LR重训。若各臂均可学，优先数据覆盖/监督目标与生成决策问题，不换框架掩盖问题；若均不能学，才沿梯度、生成与教师强制差异继续定位。

三臂通过原共享GPU队列执行，遵守零compute、≥40GiB空闲、利用率<10%、3次间隔15秒与UUID锁。已有任务不终止、不抢占；资源忙时等待。保存每8步完整状态支持恢复，原科学设置不动。

实现：`tools/analyze_failure_data_v2.py`与`tools/run_training_factor_control_v2.py`。输入计划及数据分析在`outputs/vlm_failure_factor_review_v2/`；本文件须先于任何诊断更新登记。
