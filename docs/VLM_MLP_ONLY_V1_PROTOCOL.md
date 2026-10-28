# 固定原有q/v、仅训练新增语言MLP

2026-10-22事前登记。联合q/v+MLP已11:47完整退出0、0争用：128训练满分88（q/v控制56），数字7/21/读钟5/20/拼写26/27/目标36/42，但迁移86.437848对q/v87.192331为−.754484pp/95%[-1.442649,−.086245]，对初始−.890538pp，保持92.1875<93.75。相对初始迁移42升62降。支持拟合与迁移取舍，不证明唯一过拟合、数据或架构原因。

## 假设及唯一改变

检验：在同一原始初始化与同一图结构中，固定已学q/v、让新增MLP承担纠错，能否比联合更新更好地保留迁移。唯一变化是optimizer更新范围；不增加数据、epoch、rank或目标函数。

复用`vlm_language_mlp_v1/initial_adapter`，原144个q/v因子严格原SFT5d8值，新增MLP216因子/r2/alpha4/A独立seed42 Kaiming/B0。总适配存储4663296不变，只有2820096个MLP参数更新，1843200个q/v参数固定。全base、视觉、merger、embedding、FP32输出头原样冻结。候选不是从联合训练负Final继续；fresh原始起点、fresh Adam/RNG。

`requires_grad=False`用于已有q/v，并只把216个MLP参数交给optimizer。训练仍对所有必要激活求导，不冻结整个decoder或用no_grad包住注意力，避免切断MLP梯度。固定Wq/Wv不固定Q/V输出，因为MLP会改变隐藏输入H；仍可能遗忘，不能先验保证保持。

匹配直接控制是已完成联合q/v+MLP，同图结构、起点、数据、顺序、损失、学习率和64步；复用其保存预测，不当新独立重复。另报initial、q/v CE、加强回放、覆盖1024强参考。单变化仍改变优化参数量、梯度裁剪与计算量，不称严格等FLOPs或归因所有误差到q/v。

## 固定训练与复用

只读reference_replay DATA/CPU_INPUTS/SCHEDULE，128逐图支持困难题各4次+512不同回放各1次；640QA/1024曝光、64更新、micro4×4/effective16，每步8hard+8replay。L=(Σhard CE+Σreplay CE)/16；逐QA正文+真实EOS token均值，EOS后换行不训练。LR2e−4/warm8/linear/AdamW wd0/clip1/dropout0/seed42/Final64，不调冻结层或学习率。

复用已通过真实架构/PEFT360因子/180模块的初始化，不重复构建meta模型。CPU只核本次scope：原144 q/v与原SFT逐值相同、新MLP216/2820096、108新B全零。实际GPU加载后核216 trainable/144 frozen，q/v hash应为5d8；首原16输入/labels exact、CE差≤1e−5，108新B梯度有限非零、初始A0才执行首正式step。每步q/v梯度统计0、MLP A/B分别记录；Final q/v哈希必须与初始一致。

训练循环从冻结父train.py建立版本副本，只改scope引入、216因子数量、空q/v梯度统计、Final frozen-q/v核验；评测函数体原样复制并绑定新输出ROOT。已有加载、mask、评分、断点不重审。完整adapter仍360因子，checkpoint可训练参数与Adam为216，恢复先应用相同mask，再按已验证通用恢复；保存RNG/日志偏移/UUID与输入锁。

## 评测与验收

一个新候选64正式更新/2661生成：训练context468仅128评分+seen512+保持64+transfer1617。全部参考预测同原帧/原分组/FP32头/greedy32/VQA评分复用，末批1合法。seen512曾用于原SFT；transfer1617/1024图与原SFT train图像/像素隔离，但已多轮内部观察，不称盲测/预训练未见。原640训练与held像素隔离证明沿用，target频次仅监督曝光代理。

两个域分别对initial/plain/hard_replay/unique1024/joint整体≥+.5pp且6000次配对图像簇95%下界>0；rare1–5对initial≥+.5pp，其他频次与单多词对initial/参考≥−.5pp、rare对参考≥−.5pp；保持64对全部参考≥−.5pp、训练128满分净增32、seen原243满分多词保持≥−.5pp。原门槛保留且加入联合控制，不以较差joint控制独自宣告成功。完整训练错误分型与逐题升降记录；拟合/冻结成功不当能力提升。

单seed探索区间不含训练随机性或全历史多重选择。固定Final64，不挑中间点/扫mask/epoch/rank；失败不扩种子/正式集/简历，通过仍需独立确认及0/200另登记。

共享空闲队列零compute/≥40GiB/<10%/3×15秒/UUID锁，两张卡现有任务不动；等待任一卡自然释放后运行。训练活动1800秒、评测累计2400秒、allocated65GiB；资源等待不计训练ETA，无空卡时不得宣称已训练或承诺固定结束时间。所有新文件在持久目录，CV/科研/Agent/CI不改、不push。

学习页`learning/冻结注意力适配_只训练MLP.html`沿用事前固定8张训练原图，讲解冻结、梯度传播、Q=HWq与MLP输入变化，比较实际回答；不选新成功案例包装泛化。
