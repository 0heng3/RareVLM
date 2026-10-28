# 固定覆盖对照Final的完整内部验证 V1

2026-10-26，在本轮新生成前冻结。用户本轮要求检查结果并执行下一步；本次为独立、事后提出的零训练能力评测，不恢复旧控制器，不改变旧门槛/阴性判定或升级其通过状态。

原native_diversity_v1完整结束：64更新/1920生成、两worker exit0/0争用。diverse192的native512 93.554688→94.726563（+1.171875pp/95%[+.390625,+2.148438]）、rare366 93.005464→94.371585（+1.366120pp/[+.273224,+2.732240]），6题升0题降。对repeat96 +.585938pp但下界0。全部原点保护通过；旧128诊断33满分≥32，但VQA增益19.53125<20，因此原expand_transfer=false、screen_negative_no_expansion保留。旧128在两组训练中仅覆盖64/126条，不是两组完整训练准确率；不能继续把它的未过线等同真实任务未提升。

目的：直接检验固定Final在完整1617题上的泛化，不根据已有512再调模型、权重、训练步数、样本、阈值或生成。该cohort历史反复使用过，不是新外部盲测；事后跟进的选择偏差和单checkpoint条件区间明确保留。

两臂只读native_diversity_v1的repeat96/diverse192 Final32及原DATA评测cohort；原imagewise vision+FP32 language/head/adapter、1024整图、batch4、greedy32，原提示/图像/EvalAI VQA计分及actual prefix SHA不变。共同起点1617和前DPO128强参考1617预测原字节复用、0新起点生成。分别核对Final原FP32 LoRA hash，不重新做旧数值/加载/训练审查，不改变任何权重。

预算：两个模型各1617，共3234新自由生成，0 optimizer/reference/新训练数据。按已有空闲UUID共享队列，3×15秒/≥40GiB/<10%/0compute、不抢占；每角色评测helper原cap1500秒、allocated≤65GiB，逐完整batch保存预测可恢复。原源研究结果/源码/输出全部保持。

报告四模型的同协议得分、candidate对起点/repeat96/前DPO128的图像簇paired区间及完整rare342切片。使用原paired实现/2000 bootstrap，不改原VQA或历史区间。独立任务信号仍需candidate对**起点与repeat96均≥+.5pp且95%下界>0**、对前DPO128点值≥0、完整rare342对起点和repeat96均≥−.5pp，以及原OCR/原生问答点保护保持；不因为差值接近门槛降低标准。区间跨0不证明等价/无损。

信号即便通过，也只写“事后固定模型内部能力信号，待复现”；原研究筛选门槛仍failed，原强基线/三seed/外部保持总目标仍未完成，不自动扩训/seed/正式测试。失败则保留结果，不训练凑阈值。CV/科研/Agent/CI不改、不push，不终止其他任务。
