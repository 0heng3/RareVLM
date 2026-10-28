# MLP-only函数保持：软KL与加强硬CE

2026-10-22事前登记。MLP-only已18:46完整完成/退出0/无争用，训练71/128、seen92.832031/transfer87.408782/保持92.1875。对联合训练迁移+.970934pp/95%[.248714,1.727814]，但对原SFT仅+.080396/95%[-.676535,.821102]，联合失败。固定q/v抑制部分回退，但MLP仍改变隐藏态和输出，尚无证据超过初始能力。此前q/v-only KL也未胜过加强CE，不把旧负结果遗忘；本轮检验新MLP容量/固定qv条件下的函数保持，必须再与强硬标签目标对照。

## 固定结构、数据、目标

唯一KL候选与新hard控制均从同原始SFT+零B MLP初始化d1f重新训练，q/v144/1843200严格冻结5d8，只训练MLP216/2820096，结构/存储4663296。128困难各4次+512已会回放各1次，640QA/1024曝光、64更新/micro4×4、lr2e−4/warm8/linear/AdamW wd0/clip1/dropout0/seed42/Final64。复用原DATA/CPU/SCHEDULE与原FP32头/冻结BF16基底/视觉，不接训失败Final、不新增训练数据/编码。

Lbase=(Σ8hard CE+Σ8replay CE)/16。正文+真实EOS逐token平均后QA平均，EOS后换行不监督。

* soft_kl：Lbase+β mean_replay KL(p_initial||p_actor)，T1，完整151936词表，逐有效assistant位置先词表和再token与QA均值。
* hard_replay：Lbase+同β mean_replay CE_GT。

额外约束只在回放，不在128纠错题蒸馏原错误；同β不意味等梯度/Adam步长。KL初始近0、硬CE可非0。软目标额外计算单列，不称等FLOPs；非DPO/PPO/GRPO或原创损失。教师是GT前缀分布，不承诺约束所有自由生成前缀。

## 缓存复用与训练侧校准

直接只读旧reference_replay的128packet/512QA/1951位置/1185970688bytes完整初始教师logp，0新教师前向/生成/缓存写入。所有packet原SHA、source PLAN及当前padded输入/labels/grid/pixels与qid绑定，原criterion/数学验证复用。原q/v值相同、新MLP B0保证相同意图函数，仍需实际新GPU最初8条replay absKL≤1e−5才放行；不因CE看似相同就跳过完整词表检查。

新协议不强制旧producer UUID；记录原UUID与新消费者UUID，保持当前数值策略/相同A100硬件族，以上实际KL检查代替原协议的固定硬件亲和。此改变只适用于本新实验，旧源码/锁/结果不改。

系数只用已完成MLP-only checkpoint16/32/48、next block17/33/49，计算216参数下g_baseCE/g_replayCE/g_KL，β=clip(median(||g_baseCE||/||g_KL||),.1,100)，不用验证题/分数；记录cos和相对差。若三点cos≥.999且相对差≤.01则基本冗余，不训练两臂；非有限/梯度不充分则退出，不调阈值/补点。初始8QA前向+48QA CE梯度（其中24回放另有KL梯度），0optimizer。活动180秒/65GiB。校准结束精确恢复原初始，然后进程退出；两个新worker重新载入原初始化/fresh Adam/RNG，不继承校准模型状态。

首正式step前原16输入/labels exact、CE差≤1e−5，soft额外初始8KL≤1e−5；216可训练库存与原q/v值核验，Final q/v hash必须保持5d8。完整216 master/Adam/RNG/日志offset/UUID checkpoint，保存adapter仍360因子；旧加载/mask/评分/恢复和数学检查不重复重写。

## 评测、门槛与停止

两臂各2661生成（训练context468仅128评分+seen512+保持64+transfer1617），共5322，128新正式更新；全部原始/qvCE/qv加强CE/覆盖1024/联合MLP/MLP-only保存预测同条件复用，非新独立重复。原生成batch4/末批1、FP32读出/greedy32/VQA不变。原SFT train与transfer1024图像/像素隔离，但已反复内部观察，seen512曾用于原SFT；不称盲测/预训练未见。监督target频次仅作曝光代理。

KL候选在两个域对initial/qvCE/qvHard/unique1024/joint/MLP_CE/newMLP_hard各整体≥+.5pp且6000次配对图像簇95%下界>0；rare对initial≥+.5pp、其他频次/单多词对初始/各参考≥−.5pp、rare对参考≥−.5pp；保持64对各参考≥−.5pp，训练128满分净增32、seen原243满分多词保持≥−.5pp。原门槛保留，不仅靠较差joint或MLP-only对照宣称成功。

hard控制另预声明报告对initial/MLP_CE/qvHard的两个域配对区间，不能在KL失败后把控制改称原候选成功。完整错误分型和所有有效升降列出；病例列表仅去除≤1e−8浮点舍入差，官方原分数/均值/CI/门槛完全不变。

单seed探索区间不含训练随机性及全历史多重选择。固定Final64，不挑点/扫β/LR/epoch/seed，失败不扩步/正式集/简历，通过仍需另登记独立确认与0/200。无新提分先验。

先一个空闲GPU做校准，通过后两空卡并行两臂训练评测；仅一卡空闲则顺序接续。共享队列零compute/≥40GiB/<10%/3×15秒/UUID锁，不终止其他任务。校准180秒、训练累计1800秒/臂、评测2400秒/臂、65GiB，异常断点退出；ETA以实际阶段与步速，不将排队当训练。源码/协议/输入/cache SHA在GPU前冻结，所有文件持久保存。CV/科研/Agent/CI不改、不push。

学习页`learning/MLP输出分布保持_KL与硬回放.html`沿用8固定训练原图、真实回答、完整词表KL与系数来源、真实源码，训练拟合不当泛化。
