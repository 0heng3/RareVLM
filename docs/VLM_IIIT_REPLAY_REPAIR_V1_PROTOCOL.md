# IIIT训练后原生问答保持：回放分布修复 V1

2026-10-25，任何新候选输出前登记。目录`outputs/vlm_iiit_replay_repair_v1`。

## 问题与证据

上一轮稀疏语言LoRA使官方IIIT5K无词典ASCII36正确率从2849/3000升至2908/3000（+1.966667pp，条件family簇95%区间[1.380506,2.586825]）。原生512总体+.527344pp、区间跨0，多词答案−.507813pp，未通过原≥−.5pp点保护。不能称全失败，也不能称联合任务已解决。

发现回放筛选偏差：原正确1024池单词544、两词365、三词以上115；严格排OCR验证词后剩883，其中三词以上41、含the为0；最后256为150/91/15。排词规则误删常见功能词，引入分布偏移。这不是已证明的唯一遗忘原因，五个多词掉分病例只有一个是简单截短。禁止把这些评测错题补进训练。

## 唯一干预

从同一正确1024池选择256：单词64、两词152、三词以上40。桶内按固定`content_sha(['iiit_replay_repair_v1:native',qid])`排序。前40个native micro为[单、双、双、长]，后24个为[单、双、双、双]。原256在旧schedule首次出现顺序与新256一一映射，各曝光2次；OCR128原schedule完全不变、各8次。共64更新、1024 OCR +512 native=1536 QA。

排OCR内部验证答案中的内容词，固定豁免功能词：the a an of to in on and for at with by is be are you your it no not from as or we our。允许部分OCR dev功能词出现在native监督中，必须披露，不能称新增监督全面词隔离；OCR train/dev原词、像素与filename-family隔离不变。新增内容词验证子集在模型输出前按固定功能词表排除，原256全量、213支持子集与旧评分不改。原正确性筛选使用历史initial预测，不称全部prediction-free。native与512/64/1617评测图像及解码像素隔离。

这是“内容及答案长度分布”组合策略，非单纯长度因果；等QA、等更新、非等token/FLOPs。自然问题和答案保持，绝不改成所有回答都要变长。

## 不变训练与控制

复用冻结`vlm_iiit_scope_v1.train.train`，角色固定sparse_language，仅将模块ROOT绑定至独立unit。初始化40ccd9…83a1，原SFT语言q/v rank8 1,843,200参数更新，视觉rank12 B0冻结。FP32 LoRA/head、BF16 base、dropout0、TF32off、普通逐QA mean CE，源权重OCR/native各.5；AdamW lr2e−4、warm8、linear64 horizon、wd0、clip1、micro4/effective24、原6micro顺序、Final64，完整Adam/RNG/输入trace断点。没有新增DPO/KL/视觉训练/步数搜索。

order42控制复用已冻结sparse_language Final与其960预测，1617预测由已冻结native_transfer补测提供，0重复控制训练或生成。repair42从共同初始化fresh训练。所有新源/数据/包/图像绑定后冻结，首步实际CPU input/labels匹配、有限梯度与冻结分区检查复用。

## 预声明有限推进

每个unit先960内部生成（128训练OCR拟合、256 OCR dev、512 native、64保持），再1617整图问答；合计2577。1617是历史使用的内部集，与原SFT train_all图像ID分离，非外部盲测。主观察整图1617相对initial及匹配control。

repair42推进需同时满足：1617相对initial≥+.5pp且不低于control42；512总体、单词、多词、频次1–5与保持64相对initial均≥−.5pp；**内容词OCR dev**对initial≥+1pp且不低于control−.5pp。这是点值探索推进条件，不是统计提升结论。报告所有条件图像/family簇95%区间（6000次固定bootstrap）及原256/213口径，不以区间跨0称证实退化。正式native提升仍须≥+.5pp且区间下界>0，并满足原项目强基线、保护、独立三seed与后续外部要求。

若失败立即自然结束，不加步数、不调配额、不扫超参、不新增official test。若全部通过，才匹配复现order0/200的control和repair四次64步。因dropout0和固定初始模型，仅改torch seed不能代表随机复现：order0/200以`random.Random(seed).shuffle(range(64))`重排完整24-QA更新束，匹配控制/候选使用相同置换，保持每个micro构成与曝光。order42原序不变。明确是**固定SFT42起点的数据顺序复现**，不能当作原SFT独立三seed完成。

若两组新增order也全部过同一内部条件，才对五个新Final各3000条已曝光OCR官方test作固定协议复现；原control42与initial test复用。新test不参与推进、模型或配置选择；不再主张新盲测，不使用旧extra_train_exact_target_unseen字段作新池的切片。第一阶段64新更新/2577新生成；最大320新更新、12885内部+15000重复test=27885新生成。仍不自动宣称原项目完整目标完成。

## 资源与停止

共享队列空闲3×15秒、零compute进程、≥40GiB、利用率<10%、UUID锁，不终止当前native-transfer或其他任务。每unit训练1800秒、内部960评测1500秒、1617评测1500秒、重复OCRtest2400秒，65GiB；单臂失败不杀其他正在运行任务。阶段结果写独立ANALYSIS/COMPLETION，区分实现失败、负结果和研究目标完成。无CI/push，CV、科研、Agent不改。
