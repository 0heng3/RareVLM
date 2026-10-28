# 已有SFT深化：训练视觉预算对照

2026-10-16，用户要求优先完善已执行算法并获得真实量化提升。本实验是已有Qwen SFT的受控延续，不命名为原创算法，不以工程检查代替训练结果。

## 证据与假设

已有完整SFT使用语言NF4/FP32、最多256视觉token训练；统一原权重BF16/1024正式评测下TextVQA由79.560提升至81.138±.169。后续Object小数据及KL回放没有达到综合提升标准，不能据此判定完整TextVQA SFT已无潜力。

既有训练标注统计显示：raw-modal target的官方人类共识得分均值.986624，只有367/32985题与非空归一化众数不同（1.1126%）。多答案监督并非已证明无效，但这些统计不足以将其列为当前主要瓶颈，因此本轮不改变答案目标或权重。

待检验假设：原训练分辨率限制了学习清晰文字的机会；将完整训练的视觉信息提高到评测预算，可以在统一推理输入下进一步提升识字问答。现有推理预算研究仅说明输入干预效应，尚未隔离新训练的贡献；本实验直接检验权重学习。

## 唯一主比较

| 项目 | low256：匹配继续SFT控制 | high1024：候选 |
| --- | --- | --- |
| 初始化 | 同一个既有SFT seed42 adapter | 相同 |
| 训练图像预算 | 最多256视觉token | 最多1024视觉token |
| 训练数据 | 完整train_all 32,985 QA；原众数目标 | 相同 |
| 模型/精度 | 原权重BF16、FP32 LoRA master | 相同 |
| 可训练范围 | 36层语言q/v rank8、alpha16，共1,843,200参数 | 相同 |
| 更新 | 一轮、batch16、micro4、2062更新，最后9题 | 相同 |
| 优化 | AdamW，lr5e-6、warmup10%、线性衰减、wd0、clip1 | 相同 |
| 选择 | 固定Final，不按dev挑checkpoint | 相同 |
| 评测 | 全部内部dev1617QA/1024图；原权重BF16、最多1024视觉token、batch4、greedy、32新token | 相同 |

两组复用已验证的assistant-only/每QA平均CE、模型加载、断点和评分实现。输入仅图像、问题和原短答提示；评测参考答案只在生成后评分。训练min_pixels50176、max_sequence2048，无截断、OCR或裁剪。实际token数、训练耗时与显存保留；1024训练使用更多计算，不能称等算力收益。两组都转原权重BF16，只能用本轮两臂隔离视觉预算贡献，不能把相对历史NF4训练的全部差值归于分辨率。

## 预先固定的量化标准

主要内部筛选：候选相对fresh low256对照至少+0.5pp，按图像整簇配对95%区间下界>0；同时相对原始SFT起点至少+0.5pp。监督答案rare(训练频次1–5)及frequent(>20)点值相对起点均不得下降超过.5pp。原内部dev已多轮探索，筛选通过不能直接作为正式稳定提升。

通过后才开展seeds0/42/200的匹配复核，并在冻结配置下评估正式TextVQA及DocVQA/MMStar保持。若使用同一SFT初始化，仅能称共同起点的三个继续训练种子，不能伪称三套独立全流程。正式能力结论要求三种子收益方向一致，正式TextVQA均值相对匹配对照至少+.5pp且图像簇配对区间下界>0；能力保持按每任务独立报告，不能用Text得分替代保持证据。简历仅在正式结果齐后更新。原Object/RL实验门槛不变、不回写成功。

若内部未通过，本候选停止扩种，不扫学习率、epoch或预算找dev最好值。根据完整学习曲线和全部升降题判断训练拟合不足、过拟合、答案粒度变化或视觉错误；区分观察与因果，不仅挑成功图。

## 原理与代码

目标仍为每题assistant-token CE，再对QA平均：`L = mean_q mean_t[-log p(y_t|image,question,y_<t)]`。图像patch14、2×2 merger，每个视觉token对应28×28输入像素面积；四倍面积预算约对应两倍线性分辨率，但实际resize/图像长宽比由processor决定。

实现：[训练与分析](../src/vlm_sft_resolution/experiment.py)、[配置](../configs/vlm_sft_resolution_v1.json)。复用`vlm_deep.common.encode/answer_loss`、`vlm_object_train.train.load_actor/save_checkpoint/resume_checkpoint`、`vlm_probe.evaluate.evaluate_batch`。首个正式更新记录真实loss、梯度、输入token和显存，不重复原全量工程审查或另做丢弃训练。

GPU使用现有`research_runtime.queue`：共享UUID锁、40GiB空闲、零compute、利用率<10%、3次间隔15秒；不终止已有任务。结果保存至`outputs/vlm_sft_resolution_v1`，支持原UUID/同源断点恢复。此协议登记不代表训练或提升已完成。
