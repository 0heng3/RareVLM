# JD驱动的0/20/50候选预算 v1

先预注册再读取Top20校准结果。动机：上一协议实际pair仅减少21.881%，虽然R1非劣/1.277×阶段加速通过，未达到30%成本目标。此独立研究改变每个被重排query的候选数，不修改旧协议/旧τ。主要对应[1688 Embedding/Rerank](https://liamding.cc/hiring.html)与[国际站多模态表征及检索链路优化实习](https://career.nankai.edu.cn/correcruit/content/id/118629.html)；非所有JD能力或当前开放状态证明。

## 固定条件及变化因素

固定三个旧distill Final@10、一BLIP-COCO、原fullTop50、FP32 CLIP/224/77、BF16 BLIP384/64/[ENC]30523、class1−class0、两role同card/batch32、每role三轮完整计时。0新神经网络训练。固定上一轮cal-only得出的fast阈值；m>τ_fast仍直接CLIP。其余query：m<κ则Top50，否则Top20。只有κ按cal选择。m是top1−top2 rawcosine启发式，不是已校准正确率。

Top20只给原CLIP前20候选重新排序，尾部30保持原CLIP顺序，不伪造其ITM分数；完整返回仍为原Top50成员，因此R50保持不等于Top20分支可以将尾部正项提升到Top1。单列实际预算prefix正项覆盖，K0的有效Top1上限为原第一名是否命中，K20/K50为相应prefix是否含正项；该覆盖只在预测后评分，不进入运行时预算。

## 校准及准入

保持上一轮SHA-only cal500/confirm500，同图五caption同fold，检索gallery仍完整1000/5000。dev/confirm已历史曝光，不能称新盲测；原30%成本失败推动机制设计，不能声明校正所有既往探索偏差。旧confidence/fast策略完整按SHA绑定，不当新增GPU或独立重复。

对每方向，候选κ为该方向cal中仍需重排query的margin合并后q∈{0,.05,...,1}分位数，加−1（所有需重排query用20）/2.1（退回旧50）。使用已冻结完整ITM的原pool分数模拟20prefix并仅计算calR1/nDCG，与full50相比平均R1损失≤.1pp且nDCG≤.5pp。选择数值最小的可行κ；更大的κ需要50的query增多，pair集合单调扩张，所以该规则最小化当前grid中的合法成本。cal失败回退50，不能用confirm挑κ/K/τ或另找种子。

先报告cal里full已正确命中query的正项Top20保留率，此为参考诊断不是算法分数。若cal最小可行策略在完整gallery无标签规划的unique pair削减仍<30%，记录calibration_rejected_before_gpu，不浪费后续GPU/不宣称实际加速。否则锁定policy、所有0/20/50预算及pair key后执行fresh full/budgeted匹配。

## 实际测量与门槛

质量以confirm500图中的budgeted−fresh full双向R1为主；三固定checkpoint先平均，500图像整簇6000bootstrap，两端点单侧alpha=.025，NI损失上限.5pp，样本SD另列。成本以完整gallery共享三checkpoint实际unique pair减少≥30%、同role/card/batch32三轮parallel matcher＋route速度比中位≥1.2且每轮≥1为主；nDCG点值≥−.5pp、65GiB及资源/原full复现保护。该两端点校正不声称修复历史探索偏差。

两个配置的ITM分数分别重新forward；只公共vision可复用。复用上一轮已冻结matcher的native64pair预检、分块keys/logits/SHA恢复与三轮计时合同，新config/root绑定；旧cached完整分数不能用于声称成本收益。恢复跳过部分chunk的轮次不计时。logical requests/quality unique pair/额外重复forward/vision/IO/路由/排序/完整壁钟单列，不当线上或单模型SLA。

dev全部通过才在历史test固定一次，重新产生testmargin、规划预算、fresh full/budgeted同卡匹配和三轮计时；否则停止、不调κ/grid/预算/阈值/门槛，不新增test。预训练重叠未知，test已历史曝光。

## 学习、来源与完整性

展示真实cal预算分配、首20候选/真实图/实际margin、budget keys及实际Top1改进退化；图片明确为源图展示，非attention。讲解双塔与联合图文匹配、R50与有效prefix上限、质量/计算量取舍，参考[SentenceTransformers两阶段检索官方文档](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)，没有迁移其文本模型或借用Wikipedia演示作成果。

新模块/配置/输出独立，源码/输入/weights/policy按SHA冻结；新GPU任务遵守空闲准入与共享UUID锁，已有任务不终止；旧结果、正式简历与个人贡献范围保留，不启CI、不自动push。该研究不能替代数据合成、工业商品评测、文档检索、CPT/RL或在线部署经验。
