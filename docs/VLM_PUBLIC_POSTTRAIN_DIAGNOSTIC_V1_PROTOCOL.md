# 公开人类局部纠错偏好的噪声诊断：SFT 与 DPO，协议 v1

2026-10-15，全部正式优化器更新之前冻结。承接用户“执行下一步行动”，只研究 RareVLM，不改论文/Agent/正式简历，不使用CI或自动push。

## 问题、定位与不可替代的失败结论

原749与修复199规则候选的语义门控均失败，继续禁止训练这两批数据。新下载官方RLHF-V人类局部纠错数据也不是自动真值。其固定64图审查仍包含同义改写、主观判断、局部纠错后的共享/新增残余错误，未达到原>=90%明确且Wilson95%下界>=80%门槛。原门槛和失败结果保留，`DATA_ADMISSION.json`保持拒绝，不将官方human来源当成全回答无幻觉。

本实验是**独立的噪声偏好诊断**：在真实、不完美的公开人类局部纠错偏好上，标准DPO相比相同chosen的继续SFT，会改善还是破坏生成行为？它不以“数据已经可靠”为假设，不能晋级为可信偏好数据/完成无噪声对齐；不为救门控训练64小审计集、改原门槛或替换抽样。独立`DIAGNOSTIC_ADMISSION.json`只允许这一明确scope；质量gate仍false，自动晋级始终false，即使生成指标变好也需另行评价，不自动扩种子/正式集/简历。非明确比例不是随机标签翻转率，不用它拟合鲁棒DPO的flip epsilon。

## 固定数据

官方`openbmb/RLHF-V-Dataset` revision `1d8e9804b59e9da64ad7b1e17d505869ab9b2ad3`，parquet879427238字节，SHA256 `c72b171b2a3ef496df0e66e4e6ee6ccbdcea63902ac192cf5b526b1990554e5d`。5733原对，排除121（94既有benchmark来源、10 POPE同图、16相同回答、1空/非文本；理由多标签重复不累加）；5612对真实双completion编码均通过2048长度/1024视觉上限，0截断；一像素图一对再排1605，得4007独图。固定SHA/seed42约90/10：**训练3568、heldout439**。与旧train_all/dev/final、DocVQA/MMStar及POPE的文件/像素精确重叠0，来源名和COCO ID另校验；不宣称预训练未见或排除了近重复/裁剪重叠。

TRAIN SHA `7d1e08336b1b392f28a8cfdbd5f39af4996904960d5c4dcd2e611344bff32165`；HELDOUT SHA `9af2fd2ae00a65442f99a11e5f9f0cbb3b3f882fdc19a4496afa5bad514d41af`。固定审查64从整个4007候选按事先SHA抽样，未看模型分数，包含两split仅用于数据审查，故heldout不称人工完全盲审；heldout预测不调超参数/选epoch。原文question/chosen/rejected保留；训练自然问题`prompt_suffix=''`，不得套短词答案提示。中位chosen/rejected40/33词，不能称与旧短答SFT同信息/同token预算。

## 算法与精度

初始模型为同一已完成的普通CE SFT seed42：`outputs/vlm_weighted_sft_v1/runs/uniform/seed42/train/adapter`，原Qwen2.5-VL-3B revision `66285546d2b821cf421d4f5eb2576359d3770cd3`。载入原BF16基座与BF16舍入LoRA后将LoRA精确提升FP32，BF16 autocast；仅语言36层q/v rank8、alpha16、1843200参数更新，视觉/基座冻结，dropout均0。不与旧NF4/FP32训练混作单因素或训练加速。

SFT：`-sum assistant logp / assistant token_count`，每QA平均。DPO：`-logsigmoid(.1*((lp_c-lp_r)-(ref_c-ref_r)))`，assistant token sum含原mask定义的结束token，prompt/visual/pad忽略。reference在任何正式更新前由固定初始SFT按相同pair批2编码/前向缓存，后续只查值与输入SHA；不能每步current_actor.detach。DPO多出rejected和reference完整前向，其token、耗时、显存单列，等QA/更新不等算力。

固定pilot seed42（数据/顺序轴，不是新随机初始化底座），**1 epoch、3568 chosen曝光、223更新**；两组同torch.randperm(seed+epoch)，micro1 QA/pair×acc16，末不足16按实际len归一化，本数恰整除。AdamW lr5e-6、weight decay0、warmup10%、线性衰减、clip1，Final@1，不做HPO或按评测选checkpoint。当前不执行0/200；未来若另批准/协议化复核，配置预留{0,42,200}不代表已三种子。

真实预检各临时优化一步，随后完整恢复LoRA/RNG并丢弃optimizer，前后冻结参数哈希相同；临时工程更新和正式更新分别计数。工程LR固定1e-4使单步可观测越过BF16舍入阈值，正式LR仍5e-6；不把工程更新当正式训练轨迹。独立预检两组+正式runner重复预检两组，预期共4次丢弃工程step，不能笼统宣称0optimizer.step。静态reference与actor使用相同训练/gradient-checkpointing前向路径以排数值路径混杂。每25正式更新原子保存adapter、Adam、RNG、曝光/游标和谱系锁，恢复不得混卡/输入/源码/配置；各组实际QA/chosen输入与初始化哈希匹配。

## 三种评测，固定一次而非调参

1. **生成幻觉：完整POPE 9000题/500图**，random/popular/adversarial各3000，同500图跨category相关。官方源码commit `08d957b917e5a378a2f99d35b6293c536a66298b`；mirror revision `4db1276663dfa5eb8ad16a52d24c31a09e470896`只取JPEG，85MB单分片SHA匹配。mirror与官方57条imange/image差异归档，所有问题/答案用固定官方三JSON。自然问题+`\nAnswer yes or no.`，原BF16/1024、batch4、greedy32，无OCR/参考输入。官方首句去逗号后词级No/no/not否则yes，另报strict invalid率，不能用拒答/全no伪造高质量。逐category和pooled accuracy、F1、precision、recall、yes_rate；保留真实输入/像素/grid/生成IDs。
2. **能力保持：完整内部TextVQA dev1617QA/1024图**，旧短词提示、原BF16/1024、batch4、greedy32、十参考LOO共识。重新跑initial/SFT/DPO确保当前逐题IDs/pixel/grid一致，历史87.112仅历史强参考。train_all实际target频次0/1–5/6–20/>20切片分别带图像簇CI，不把答案频次当视觉类别/预训练频次。新的公开数据曝光另列，不改原切片定义。
3. **heldout偏好439对**，同自然问题、assistant sum-logp、原BF16/pair2。报告raw chosen>rejected率、margin、每token概率/长度与参考中心差值；它是teacher-forced相对概率诊断，不是实际生成正确率，不能单凭提高叫视觉幻觉降低。

initial只一个checkpoint；SFT/DPO各一个新最终checkpoint。训练/heldout自然长答与Text短答/POPE yes-no提示差异显式保留；这是任务适配与保持研究，不做提示调优。

## 数字判据与禁止自动晋级

POPE诊断主比较DPO−SFT及DPO−initial：pooled accuracy各至少+.5pp且各单侧97.5%图像簇bootstrap下界>0；6000次、按同500图同步抽三category（非1500独立图）。F1和recall各主比较点差>=−.5pp，点保护不称非劣。Text DPO−initial整体单侧97.5%下界>−.5pp为预设保持界限，切片区间另报；Text自己更严格的两个+.5pp提准gate独立保存，不要求为幻觉改善同时大幅提升Text。

主结果是固定模型/数据的探索性条件区间，不含种子不确定性；POPE已公开、预训练重叠未知，不称盲测。报告所有失败与分数，不据结果回调数据、beta、LR、prompt、epoch或checkpoint。**上述数字判据即便通过，也只描述这次噪声偏好诊断的生成行为；数据事实质量gate=false、promotion_allowed=false保留，不自动执行新seed/正式Text/Doc/MMStar或更新简历。**

## 资源、学习和来源

空闲>=40GiB、<10%利用率、零compute进程、3次15秒快照、共享UUID锁。每分支先预检再单卡训练/评测，利用空闲卡动态衔接；不可终止已有任务。两张A100相同算法比较不作跨卡纯训练加速声明。真实速度/总耗时由完成日志估算，不以预检代整轮吞吐。

学习页连接：原图/人类局部纠错、assistant mask、实际logp、静态reference、SFT与DPO梯度、完整评测成功/失败。图片展示不冒充processor或attention；助手执行不代表用户本人提出/独立实现/掌握。

来源：[RLHF-V官方数据卡](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset)、[RLHF-V上游](https://github.com/RLHF-V/RLHF-V)、[TRL DPO公式](https://huggingface.co/docs/trl/main/en/dpo_trainer)、[POPE官方](https://github.com/AoiDragon/POPE)。RLHF-V人类纠错与RLAIF-V AI反馈区别保留；研究用途与原许可随数据来源，仓库不打包分发原数据/模型。
