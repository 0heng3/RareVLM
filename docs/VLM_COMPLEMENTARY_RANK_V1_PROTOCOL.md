# 同问异图互补监督：CE与CE＋跨图排序的短预算对照 v1

2026-10-18。在新的官方训练来源内部数据上检验一种训练目标，不回写旧hard-SFT负结果，不把公式或loss下降称作能力提升。

## 证据与假设

旧hard-SFT的6次判定变化全no→yes，人工screen净0，对均匀控制净−1；不能据此证明失败原因就是校准，但未见额外视觉区分。现有训练折同问正反301QA中初始只错9题；题面较客观的106QA仅错2题，不重复这些少量案例冒充更丰富训练。

VQA官方训练互补列表200394对定义为同问、异图、异答案。join官方train2014人类十答案，只保留双方yes/no且各≥9票、原题文完全相同、不同图、无历史图像ID/hash重合的项。不是缺COCO实例框推导的no，不是模型生成伪标签。官方链接：https://visualqa.org/download.html 。公开模型预训练重叠未知。

新的假设：成组监督显式增加同问题下的图像条件差异，能否比完全相同数据上的CE更有效。令m(I,q)=首答案位置logit_yes−logit_no：

`L = mean_QA(assistant_token_mean_CE) + λ × mean_pair(softplus(-(m(I_yes,q)−m(I_no,q))))`

同一个问题的共同加性yes偏置在差值中相消。该项不能仅靠所有图统一增加yes偏置而降低；模型仍可能学习其他捷径，必须以独立图像的实际生成得分检验。该目标是CE＋成对logistic排序，不包装为标准DPO、PPO/GRPO或原创算法。首答案取预测位置的前一token，FP32逐QA CE保持原mask/shift/归一化；排序复用同一次前向，无额外reference。

## 固定数据与先决条件

最初完整官方图只有19个分量、最大82229图，不适合划分。正式划分采用双方高票yes/no和历史排除后的全eligible配对图，先计算完整连通分量，再排除>200张图分量；不是任意拆开同一合格分量。salt=`rarevlm-complementary-vqa-v1-eligible-component-20261018`，按排序组件image IDs的hash作为component_id，`sha256([salt,component_id])`前16hex取模100：0–69 train、70–84 screen、85–99 confirm。旧inventory的provisional salt/划分保留，不当正式划分。

允许存在、颜色、空间、可见状态题面tag，排除inference/context tag及固定主观/隐含属性词，完整规则与选择盐、次序、下载跳过记录固定在FINAL_SELECTION_PLAN。每折hash排序greedy选图像不复用的train600/screen256/confirm256对（1200/512/512QA），qid亦不复用；所有路径绝对化、原始COCO图片完整RGB解码，文件和像素hash排历史边界并核折间隔离。元数据/下载失败仅按预定次序补项并保留原因，不按模型预测挑题。数量不足不放宽。

在任何新模型预测前，固定hash抽train16/screen8对，A/B展示顺序随机，隐藏答案与模型输出，助手看原图给yes/no/unknown，再与不变的人类标签对照。至少14/16训练对和7/8筛选对，两边都清晰可判断且一致才准入；否则停止，不改screen标签/分母。仅为助手小样本图像依据核验，不称新专家标注或整体可靠率。confirm不展示、不预测。

先完整生成新train1200QA及screen512QA的原始SFT42基线，所有题固定batch4/1024/greedy32。只有train至少24个互补对未全对、screen至少12对未全对且screen至少32个连通分量时才训练；这是预定学习空间/测量门槛，不是成功率保证。门槛不足停止，不另挑更难验证集。

## 两组相同训练

ce：λ=0；ce_rank：λ=0.1。单一候选权重不扫参。0.1与每题三assistant token、每更新16QA/4pair的归一化下首答案CE梯度尺度约相当，避免给排序项任意十倍支配权；不声称最优。

每组200更新/3200次QA曝光：共同1600条既有高一致性TextVQA回放（原32985监督目标rare400/mid400/frequent800，非pilot重算切片）和800次互补对/1600次人工QA。600训练对按seed42/43完整随机排列循环，最多两次曝光；不是3200个独立案例。每更新8Text＋4pair(8人工)，4个microbatch各4题，配对两图同microbatch。匹配控制具有相同数据、顺序、梯度累积、数值与预算；旧hard模型不充新控制。

均从原始raw-FP32 SFT42开始，BF16冻结底座＋FP32语言q/v LoRA、rank8/alpha16/1843200参数，dropout0、无全局AMP、显式FP32 CE、TF32关、nonreentrant checkpoint。micro4/effective16、图像上限1024、序列2048无截断。AdamW peak2e−5、warmup20/total200后线性衰减、weight_decay0、clip1。Final@200唯一候选。首正式step保存并报告新loss/梯度，不重跑旧加载/保存预检；逐QA实际输入、source、视觉/assistant token和pair margin记录，checkpoint复用已验恢复。

每臂累计active训练上限45分钟，超时仅在本任务更新边界保存并停止，不能自动延长；保存、评测及排队壁钟单列。现有同形200步24.7分钟仅作25–35分钟训练预估，实际以新首步/10步为准。GPU沿用UUID共享锁、零compute、≥40GiB空闲、<10%利用率及3次15秒空闲观察，不操作其他任务。

## 筛选指标与停止规则

新screen256对/512QA；不是旧368/474的同一数值口径。实际greedy生成，原ASCII yes/no首词计分。成对全对率要求该对两个原题均答对，不能用margin/teacher-forcing替代。QA及成对CI按eligible component_id整簇，4000 bootstrap/seed20261018；新内部开发筛选，不声称历史全实验FWER。

候选对原始初始和新CE控制必须同时：

- QA准确率≥+1pp且95%簇区间下界>0，净纠正≥6QA。
- 成对全对率≥+2pp且95%簇区间下界>0，净纠正≥6对。
- 相对初始yes/no各类点值均≥−0.5pp。
- 共同固定512条Text screen（保持原完整四题batch）整体、parent rare/frequent切片点值均≥−0.5pp。初始从同raw-FP32完整基线缓存复用并保留来源SHA和分组；这不是新独立初始重复。

全部纠正、新错、格式变化、训练监督是否学会和成本分别报告。任一门槛失败/不确定：停止，不生成新confirm、不扩epoch、不换checkpoint、不扫λ/LR，也不写CV提分。

若screen全通过，保持权重，才增加新confirm256对的初始/CE/候选生成和候选/CE完整Text1617评测；QA与pair的增益、净数、正CI要求相同。完整Text对初始95%簇区间下界>−0.5pp且rare/frequent点保护。只增加评测，不增加训练。通过仅称单continuation seed42、相应预算和新内部确认集结果；三种子和正式跨任务尚须独立后续，不自动写简历。

旧confirm377始终不消耗。所有科学配置/源码/数据/图片来源在新预测前锁定；旧源码、权重、结果与协议不改。

## 前置数据修订2：尚无模型预测或训练

首版固定盲审训练15/16、筛选6/8，未达到筛选7/8门槛。原判断先于标签key保存，失败项为现代艺术的主观分类、体育场总容量未入镜、毛绒形象是否为人的歧义；首版DATA_REVIEW_V1_RESULT及原选择/图像/标签完整保留。此为数据准备失败，不改GT或旧实验结论，首版不进入模型实验。

第二版沿用同一eligible图分量、fold盐及固定候选次序，仅在全池额外排modern/art/artistic/stadium/plenty/enough/fruit/meat/vegetables/percent/%、Apple产品归属/brand/made-by，以及存在式人类身份问句；完整regex落第二版FINAL_SELECTION_PLAN。不是按单个qid删除错题，也不依据模型分数换筛选集。新manifest位于data_preparation/revision2。固定数量与14/16、7/8质量门槛不变，审核样本抽取salt改为rarevlm-complementary-vqa-v1-blind-review-r2-20261018，避开首版看过的任何图/qid，confirm仍不展示。

本轮最多这一轮数据修订；第二版若仍不够数量、质量或初始错误空间，停止，不再自动换数据/门槛。旧版本文件和此协议修订前快照见data_preparation/prelaunch_revision1；所有模型预测仍为0。新数据与代码会在首次新预测前统一冻结，后续不得据结果追改。
