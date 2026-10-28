# 原生问答纠错 V2：当前行动

2026-10-25瓶颈检查：128训练题满分数6→SFT49/DPO48；SFT罕见366中4增益(+4.0 score)、8损失(−7.4)，净−3.4。已看原图：COLTS→calgary、Heidi→Heidie、蓝区CHUNK STYLE→相邻红区packing为3个明确退化；另4题涉及答案范围/实体歧义，合计3.4丢分只是匹配风险，不能改分宣布收益。DPO比SFT额外1.6丢分来自single→single malt与120→120-4cycle。训练hard101/128、native回放106/128已是罕见答案，不能归因于完全缺少稀有样本。

新增零更新诊断`vlm_native_batch_sensitivity_v1`已完成96生成/58.262秒/exit0/0争用：选定6个训练病例原矿工分组与正式基线分组均各自重复稳定6/6、跨组输出差异6/6，完整复现历史答案；未填充位置编码在首分歧及以前相同，模型hash不变。已证实分组敏感现象，未定位具体计算层；不能一律解释为极小舍入或称总体6/6不稳定率。原收益按正式6满分基线计算，没有把这6题计作学习收益。下一步优先在新协议中隔离batch padding/同伴与视觉编码/注意力计算路径，再研究问题条件下回答范围与保持；不重跑已完成32步，不自动启动PPO/GRPO。详见`outputs/vlm_native_bottleneck_check_v1/REPORT.json`与诊断`RESULTS.json`；原标签/分数不改。

完整结果已结束，状态为`screen_negative_no_expansion`：两组32步/各512 chosen曝光均正常完成；SFT/DPO native512为93.671875%/93.359375%，起点93.750000%。罕见366相对起点−.928962pp/−1.092896pp，DPO单词切片另−.625pp，均未过≥−.5点保护；相关区间跨0，不能称已证明显著退化。训练纠错128 VQA为6.5625%→40.15625%/39.375%，只证明拟合提高。两组内部都未过，不增加1617/官方test/步数/seed。

已新增[交互观察台](LOCAL_ARTIFACTS.md)和[最新总览](LOCAL_ARTIFACTS.md)，包含真实曲线、失败门槛、原图/前后回答、证据链接；每次检查会重新采集。事后展示说明只解释匹配风险和待核验实体变化，不修改原标签、预测或分数。下文运行中时间均为历史快照。

2026-10-25 20:22实测：参考32个实际forward批/64键缓存与128条同分组背景已完成exit0。SFT worker742164/GPU0完成11/32、DPO worker742167/GPU1完成12/32，约16秒/更新、峰12.839GiB、0争用。首步两CE同为.5171988567、两个偏好micro的actor/reference相对margin精确0、DPO=.6931471825，梯度3.37786/3.83834有限、保存FP32参数一致。只证明实现正常，不代表新能力提升；预计训练20:28、内部评测20:35–20:40，条件完整1617约20:55前后，以实际工件为准。此前准备和提交快照保留。

2026-10-25 20:14更新：两次固定筛选已完成，2048+1536=3584条真实问答、共251个候选全部助手看图审核，135条暂支持/116条不准入或不确定。首池78/146支持，补充57/105支持；这些是已筛选候选的审核结果，非全数据集错标率/专家真值。原标签与评分未修改。独立补充协议保留首池不动，两池每target联合cap2，合并前256审核上限不变。审核选前128条，归一化127target、单词66/多词62，训练/评测图像和像素交叉0。

新SFT/DPO对照CPU已准备、源/数据/配置冻结，controller741963/start504045006于20:12:28提交。起点和偏好参考均为OCR增强模型语言0109，144 FP32因子1843200参数；384QA/512 chosen曝光/每臂32步，不加训练步数或参数。两组新训练尚未取得Final指标，当前参考阶段在生成128条同分组拟合背景，之后64参考micro键/32实际批，再两A100并行SFT与DPO+SFT。原先“未准入”段落为早期快照。

实时读`outputs/vlm_native_correction_sft_dpo_v2/`：`reference_queue/STATUS.json`、`reference/PROGRESS.json`、`training_queue/STATUS.json`、`runs/{sft,dpo_sft}/train/{FIRST_UPDATE,PROGRESS,TRAIN_COMPLETE}.json`、`SCREEN_ANALYSIS.json`、条件`TRANSFER_ANALYSIS.json`、`COMPLETION.json`。先通过原学习与保持点门槛才额外评1617；没有新强信号不自动扩种子/步数。预计第一阶段训练+内部评测20:30–20:40，条件完整1617约20:50–21:00，依资源和实际生成，不是保证。

[实际图像、135/251审核与有效偏好讲解](LOCAL_ARTIFACTS.md) · [8固定训练图、SFT/DPO公式与代码](LOCAL_ARTIFACTS.md) · [独立补充登记](VLM_NATIVE_CORRECTION_V2_SUPPLEMENT.md)。后文为早期快照。

2026-10-25 19:24快照，实时信息以输出工件为准。

上一轮回放修复已完整负结果：native1617初始87.328386%、原OCR增强模型87.260359%、回放修复87.247990%。修复对初始−.080396pp/条件图像簇95%[−.835366,.650581]，对原回放−.012369pp/区间跨0。多词点保护恢复，但未提升整体问答；不扩该方案。收尾缺字段已独立恢复，0重训/重生成，完成入口是`outputs/vlm_iiit_replay_repair_v1/recovery_v1/COMPLETION.json`，原失败队列保留。不能把实现错误当作模型负结果原因。

新行动保留已完成OCR收益（94.9667%→96.9333%，原官方test3000），改为真实问题条件下的纠错监督。新增数据来自原train_all的高共识整图问答，原图、问题、答案保留，不使用dev错题训练，也不把单词ROI当整图QA。native图像曾进原SFT，当前评测历史使用过，不称外部盲测。

第一阶段已19:19提交controller740678，两A100共享准入，固定2048条（单/多各1024）。池内符合基础条件单词9350、多词2509；每目标≤2、每图/像素1条、原标注≥8票，排当前训练与held图像。CPU编码通过；从OCR增强模型移除30个精确B0视觉适配器，保留144语言因子，8固定nativeQA生成ID与输入SHA完全一致，属于结构等价加spot check，非完整生成核验。此阶段2056生成、0新训练。

第二阶段源码和协议已写定，但未准入：首256错误最多审核，至少128个问题-答案-画面一致的真实错误/96不同答案才训练。两个组同起点、同384 QA/512 chosen曝光，各32更新；SFT和DPO+SFT只差偏好损失权重，参考模型也是OCR增强起点，负例使用实际生成token IDs。先验证学习与保持，再按冻结门槛决定是否补完整native1617；禁止按official OCRtest调参。

- [筛选协议](VLM_NATIVE_CORRECTION_MINING_V2_PROTOCOL.md) · [筛选代码](../tools/run_native_correction_mining_v2.py)。
- [训练协议](VLM_NATIVE_CORRECTION_SFT_DPO_V2_PROTOCOL.md) · [训练代码](../tools/run_native_correction_sft_dpo_v2.py)。
- 筛选进度：`outputs/vlm_native_correction_mining_v2/{queue/STATUS,shard_0/PROGRESS,shard_1/PROGRESS,COMPLETION}.json`。
- 审核：同根`ERROR_CANDIDATES.json`、`review/board_*.png`、之后的`REVIEWED_ERRORS.json`；未审核候选不直接训练。
- 准入后训练：`outputs/vlm_native_correction_sft_dpo_v2/{reference_queue,training_queue,transfer_queue}`、`SCREEN_ANALYSIS.json`、`TRANSFER_ANALYSIS.json`、`COMPLETION.json`。

目前尚无V2候选训练或量化提升结论。需要用完整native结果验证新假设，不能凭更多实现工作称打破瓶颈。
