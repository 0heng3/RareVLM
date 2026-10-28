# 零更新图像条件探针：同网格、同问题、同答案路径

2026-10-21事前登记。`vlm_length_replay_v1`两臂已15:38前完成/退出0，初始93.222656%，weighted50为93.027344、expanded240为92.968750，训练满分55/56，保持均93.75，联合失败。expanded240恢复三个完整名称，但新拼写/数字/品牌错误抵消收益，长度错配不是已证唯一原因。所有旧分数、协议、参数、Final选择和失败门槛保留，不扩旧配置。

本轮0 optimizer/0参数更新，两个固定模型：原uniform SFT42与最新expanded240 Final64。问题是纠错有多少依赖正确图像条件，而非将拟合直接当新视觉能力。真实图依赖也可能是图像样本记忆，不能因此证明跨图泛化或视觉编码器已改善。

固定256个recipient：原128不同图困难题；另128个验证诊断题，单/多词各64，先纳入原矿池初始全部38个未满分且有同网格donor的题，再按固定SHA补满。509/512原验证满足同网格训练donor资格、38未满分全满足；资格不读Final。验证诊断是错误富集/内部已观察且属于原SFT train白名单，不代表原512准确率或底座/SFT未见盲测。

每recipient从原矿池training3072按图像网格分组，在不同图像中固定SHA取donor，不读任何答案、预测、分数作donor选择。无GT不匹配保证，donor可能恰有问题答案；报告同target巧合但不据此筛选，不将替换图自动标为负例训练。两条件问题/目标正文与真实EOS token IDs、attention mask、图像网格和pixel tensor形状必须相同，只有图像内容改变；生成输入不含GT。CPU复用原128训练编码，仅编码新增验证128和替换图256共384项。相同四题batch在两条件/两模型复用，不拿旧不同分组初始分数替代本轮新鲜对照。

每模型256recipient×2图像条件=512条真实greedy生成和512个QA teacher-forced前向；两模型总1024生成+1024 teacher QA，0训练。保存实际生成IDs/EOS和VQA共识，正文/EOS CE、逐token目标rank和最强竞争margin；canonical teacher前缀不是实际生成轨迹，两个视图分别标明。复用已验收的token统计/生成函数。`G=CE_content(donor)-CE_content(proper)`（nats/token），正值代表正确图像下原答案更易预测；比较Final−Initial的G变化和两边正文CE下降。每cohort按recipient图像簇6000次bootstrap95%条件区间、固定模型、不含种子/训练随机性、不作主性能gate。

预先固定解释规则，仅作机制线索：若Final训练诊断proper满分≥32、其中≥75%换图后不再满分，且训练G变化均值≥.1 nats/token、区间下界>0，标为`image_conditioned_training_corrections`。若proper满分≥32、其中≥50%换图仍满分，标为`substantial_image_invariant_component`。其他为mixed/inconclusive。任一标签都不能证明整个失败的唯一原因、模型只看/不看图、或新数据泛化；对donor也不能声称所有目标都不在图上。报告全部结果，不回选模型、阈值、recipient或donor。只有诊断证据允许另登记下一训练方案，本轮不自动新训练/seed/正式评测/CV。

共享GPU队列零compute/≥40GiB/<10%/三次15秒与UUID锁，不干预已有任务；每模型活动900秒、allocated65GiB。每四题完整对照批原子保存，包含模型/源码/数据/UUID绑定与进度，同role同UUID恢复；容器变化不需要重训。模型LoRA哈希前后不变，probe本身不可能是能力提升。学习页预先各cohort固定SHA8例，展示两张原图和真实答案/概率，不把donor图冒充正确输入、原图对照冒充注意力或专家标签。

源码`src/vlm_image_condition_probe_v1/study.py`，输出`outputs/vlm_image_condition_probe_v1`。旧完整总体门槛及用户真实能力提升目标未完成。
