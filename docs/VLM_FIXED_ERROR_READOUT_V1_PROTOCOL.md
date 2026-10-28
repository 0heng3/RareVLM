# 固定checkpoint零更新错误分型 v1

2026-10-18。用户授权执行连续低收益的下一步诊断。只读原始SFT42、CE Final200、CE＋rank Final200，不训练、不选择其他checkpoint、不调分类阈值、不触及任一确认集。

## 固定样本与真实生成

源为已完成的`vlm_complementary_rank_v1`。原训练600对中45对实际生成未全对，全部作为固定错误锚点；按盐`fixed_error_readout_v1_correct_training_anchors`从原本全对的训练对固定hash抽45对作保持锚点。补齐原始完整四QA批次、按原batch顺序去重，得到340QA/170对：45错误锚点、45正确锚点、80批次邻居；这不是完整训练分布或可称独立验证的数据。

筛选继续用已使用的512QA/256对/222组件，不换数据、不称新盲测。初始训练340题的原生成按完整原batch复用为sidecar，保留原batch_start、不冒充新评测cache；三角色筛选生成也复用。仅两个Final模型各新生成训练340题。所有生成保持batch4、1024、greedy32、原提示与计分；新预测在独立目录，原文件不改。

## 新增读出

三角色各对340＋512题做一次prompt-only前向。输入由已验`inference_row`/`encode(...,False)`构造，无target、answers或teacher-forcing assistant答案。BF16冻结底座、FP32 LoRA、无全局AMP，原始图片与完整四题batch不变。读出使用完整lm_head、最后prompt位置logits；本地Qwen2.5-VL的generate也为dense prefill，避免新切片造成的BF16 GEMM舍入。仍以保存/新增的真实生成为主，读出不能替代生成。

canonical margin=`logit('yes')−logit('no')`；另对yes/Yes/YES及前导空格的单token别名做去重、不重叠集合的logsumexp差得到alias margin。多token别名仅记录跳过，不伪装单token概率。两种读出分开，保留全词表top1 token、解码与条件概率。GT只在读出后作分析。

对原标签yes图m+、no图m−，计算gap d=m+−m−和center c=(m++m−)/2。严格二元双正确为m+>0且m−<0，等价于d>0且abs(c)<d/2。分开报告：严格排序未正确（逆序/并列）、排序正确但零边界未双正确、读出双正确但真实生成失败。0为不变诊断边界，abs(m)<=0.125仅作近边界敏感性标记，不改评分。不能从每对各有合适阈值推断存在一个通用阈值，本轮不拟合阈值。

## 判断范围

- 固定训练45错对修复数、45已会对新错数；邻居单列，训练诊断子集不能包装完整train准确率。
- 筛选初始24错对修复、232已会对新错；描述训练域与筛选域的修复差异，不仅据两个比例宣称典型过拟合。
- canonical/alias与实际生成的agreement及差异单列，先排除读出不能代表生成的情况。
- 初始→Final同题margin/gap/center的变化用于判断相对区分和共同偏好变化，不把在线trace冒充固定Final，不把单个实例解释为全体因果。
- 固定训练错题仍错提示现配方没有充分纠正；训练错题纠正却筛选不改善提示迁移不足。架构、容量、视觉主干与LR作用仍需有针对性的后续对照，不能由本诊断直接定论。

新读出21项CPU检查通过。复用已有加载、生成、缓存与UUID共享GPU队列，不重跑旧加载/保存/优化器预检；每卡零compute、≥40GiB、<10%利用率、3次15秒空闲观察。只新增2556次QA的单前向读出和680QA生成；0 optimizer step，不终止其他任务。输入、checkpoint、代码锁在新推理前固定，批次级缓存可接续。

输出`outputs/vlm_fixed_error_readout_v1/{ANALYSIS,CHANGED_CASES}.json`及`PAIR_DIAGNOSES.jsonl`。结果是失败归因诊断，不是新训练提分、正式测试或简历成果，也不自动放行新训练。
