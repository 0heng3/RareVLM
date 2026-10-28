# IIIT5K新增监督：数据覆盖与同参数视觉/语言适配

2026-10-25，新模型结果前登记。前完整词支线仍失败：128词反复训练最多70正确，五设置49词始终未纠正；新128验证含Tirsa遮挡片段和I/l歧义风险。旧分数、旧协议、旧冻结数据不回写。预启动revision1的协议/库存/源码已在prelaunch_revision1保留；仅将IIIT官方alphanumeric标注口径与提示/计分对齐，划分与样本完全不变。本研究换用任务原生公开监督并单独归档，不将新集合起点与旧集合相减当收益。

## 数据和任务修复

官方IIIT5K归档105,861,943字节，SHA256 `7872c9efbec457eb23f3368855e7738f72ce10927f52a382deb4966ca0ffa38e`，来自 https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz 。训练数据准备阶段仅读取官方train的2000词图及traindata/trainCharBound；三个Final64冻结前不打开test图像或标签，附带50/1000词表和大词典不用于输入、候选限制或训练。

任务为`Read the main text in this image. Return only English letters and digits, without spaces or punctuation.`，官方给定词图，不做整图定位、不要求补全遮挡字。保留官方GroundTruth作为评测参考，训练用NFKC/casefold/strip；官方词图指标采用无词典限制的ASCII36 exact：NFKC/casefold后仅保留a–z和0–9，原始输出及严格literal exact另存；这与TextOCR完整字面评分分开，native VQA完全不变。官方标签原本去掉了BT-Pusher的连字符、$2.00的符号和小数点；该口径由数据审核在任何新预测前发现并登记，不作事后改分。预筛ASCII字母数字长度2–20、字符标注chars与词级GT归一化一致、字符数和charBB数一致、框在图内（允许MATLAB边界1px）、长宽比<=30，按decoded RGB精确去重。这些是同一官方包的字段一致性，不称双人独立重标注。

文件stem去掉末尾下划线编号作为保守filename-family；它不等于已验证的原始网页照片ID。按`content_sha('iiit_scope_v1:family:'+family)`前8位模5为0的组持出，固定SHA取256不同canonical词作为内部验证。训练排整个持出family和全部持出词，按词去重。512训练词按长度<=5/>5、有无数字、原SFT精确整答案频次0/1–5/>5分层，配额为4的倍数；每层取其1/4组成128词稀疏子集。稀疏OCR槽位逐层循环替换密集槽位，每个词精确8次/2次，分层槽位和总曝光完全相同。

通用回放从已完成replay_diversity的正确训练回放池选256不同QA，排其hard128及本轮验证词出现在监督答案字词中的样本。所有条件用同一native顺序/标签。OCR训练/验证按family、canonical词与decoded pixel隔离；native训练与评测按已有图像隔离再核对。上述不能排除不同裁剪与原网页照片、原SFT或预训练的未知重叠；答案频次只指原train_all整答案监督曝光代理，不等同预训练未见或旧TextOCR词频口径。

256验证图及固定32训练图在新预测前展示。助手只看图像/官方文字，冻结supported/uncertain记录；官方256主指标完整保留，另报事先支持子集，至少192支持项才准备GPU。审核不是专家真值、不删难例救结果。最短边<28的词图仅BICUBIC等比放大到28，记录原生/实际输入尺寸，不称增加原生信息。

## 三组训练

|组|不同OCR词图|每词曝光|更新参数|
|---|---:|---:|---|
|sparse_language|128|8|原语言36层q/v rank8，1,843,200|
|dense_language|512|2|同上，1,843,200|
|dense_visual|同512|2|视觉blocks2–31的qkv rank12，1,843,200|

三组从同一个复合初始adapter fresh Adam/RNG开始，原uniform42语言FP32 LoRA逐值保留5d8；新增30视觉A独立seed42 Kaiming、B0，scale2，实际PEFT204因子/3,686,400存储参数/保存往返已通过。语言组冻结新视觉因子；视觉组冻结原语言因子，底座/merger/embedding/FP32 head均冻结。新视觉qkv与原语言q/v的rank/投影也不同，等参数不等于纯位置因果或等FLOPs；以前merger负结果不代替本次视觉encoder内部对照。

共同64更新，micro4/effective24，每步native4/OCR4/OCR4/native4/OCR4/OCR4，1024 OCR+512 native=1536QA。所有组普通CE；`L=.5 mean(OCR16逐QA token均值CE)+.5 mean(native8逐QA token均值CE)`，含实际EOS，prompt/padding不参与；每OCR micro求和/32，每native micro求和/16。lr2e-4、warm8、64步horizon线性衰减、wd0、clip1、seed42、BF16底座/FP32 LoRA与head、TF32off、dropout0、视觉token上限1024。无DPO/RL/新teacher/新loss搜索。

视觉训练时visual.train()以启用已有非重入gradient checkpointing；语言组visual.eval()。实际首步检查精确更新scope和参数量、CPU input/labels SHA、全部梯度有限，视觉30个新B需非零而首A应为零、另一分区梯度为空；检查后才optimizer.step。Final核冻结adapter分区hash不变。只验证新增结构，不重跑已过全局加载/mask审查；复用FP32 master/Adam/RNG/trace-offset/UUID断点。

## 评测和判定（预测前revision3：完整官方test作为主终点）

先用复合B0初始模型fresh生成：新256内部词图、稀疏128共同训练探针、原native512、保持64，共960，分两空闲卡交错完整4QA批。新模块改变forward图，因此同图共同基线独立记录，不把结构准备当收益。内部256只作诊断，不据其少量残余错误判断整体能力或挑checkpoint。为解决小样本检验力不足，全部三组Final64冻结后才首次读取官方test3000图与标签；对initial及三个Final统一完整评测，各3000，共12000新增生成。原官方train/test划分不改，test不按字符或质量剔除；附带词典仍不使用。测试不得反过来选择参数、数据、步数或模型，全部配置先冻结。

各Final64固定生成同四队列960/组，共2880；探针只作训练可学性，不算泛化。旧native512/保持提示、图像与计分不变；新验证官方256全量和冻结支持子集并报，原SFT精确答案频次0/1–5切片分列。新OCR按filename-family整簇、native按原image整簇，6000次paired bootstrap，固定单seed条件95%区间，非种子稳定或外部盲测。

完整官方test主指标为无词典ASCII36 exact，保留原始输出/严格字面exact作格式诊断。新域局部收益预设为对initial>=+.5pp且配对95%下界>0；内部官方256与213支持子集分别报告，不替代完整test。native点差>=-.5pp、单多词与rare1–5>=-.5pp、保持>=-.5pp；native95%下界未高于-.5时只能称点值保护。视觉配置优势还需对dense_language在完整test>=+.5pp/正下界，覆盖优势需dense_language对sparse_language同标准。原项目联合提升仍要求native>=+.5pp/正下界，保留强基线与{0,42,200}复核；单seed新域收益不自动算项目完成。三组固定64预算和全部官方test均在任何新预测前登记；无论内部dev结果如何都只完成这项固定研究，不按test扩步/扩种子或改标签。

官方test按filename-family整簇配对；该保守前缀不是已验证的原网页图像ID，官方train/test也不保证词互斥或预训练未见。报告与新增训练的精确target重叠切片，不据其改变训练数据。测试准备只在三个TRAIN_COMPLETE=64后运行，冻结各Final hash和TEST_DATA/图像哈希。完整研究新生成共15840（内部初始960+三Final各960+四test各3000）。

每组训练<=1800秒、内部评测<=1500秒、每模型官方test<=2400秒、allocated<=65GiB。共享零compute/>=40GiB/<10%/三次15秒/UUID锁，队列先dense_language/dense_visual，再sparse_language；不抢占已有任务。代码/数据/初始化/包版本在GPU前锁定；学习页给真实图像、scope/参数量、loss和保存回答。CV/论文/Agent/CI不改、不push。

来源：[IIIT5K官方](https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)、[PARSeq官方数据说明](https://github.com/baudm/parseq/blob/main/Datasets.md)。本研究使用原官方标注，未套用PARSeq修正版标注或模型，不能声称复现PARSeq。
