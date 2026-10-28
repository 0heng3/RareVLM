# 整图ST-VQA替换裁词回放：同预算SFT/DPO实验 v1

当前无稳定native1617能力提升。新数据准备与独立headroom已结束：987图均匹配上游subtrain物理图路径/尺寸，排COCO、VG、IIIT和已知像素；固定128槽位/124答案，初始exact99/128、正文NLL.490631（事前≥16未匹配且均NLL≥.2通过）。256来源保持195/256，仅为内部镜像来源诊断。官方原始参考不是10人投票；未把来源保持或原任务评测用于选择新槽位。近重复与底座预训练重叠未知，非专家全量重标。

## 唯一数据策略变化

复用visual_preference_v1共同复合初始、视觉30层qkv r12/alpha24、60个FP32因子1843200参数，语言/基座/head冻结。原128看图支持纠错各2次、128原生回放各1次保持；只把原128 IIIT裁词正监督槽位，一对一换为预测前已固定的128 ST-VQA整图正监督。384不同QA、512chosen曝光、256负序列/角色，32更新、microchosen4/effective16、seed42/lr2e-4/warm4/linear32/AdamW wd0/clip1/dropout0/TF32off、BF16 imagewise视觉与FP32语言/head、1024预算全部保持。

新两臂`ce`和`ce_dpo`共同同数据/顺序/初始化，loss原逐QA token均值CE，DPO原sum-logP sigmoid β.1、λ0/1；无KL/RL或GT区域辅助。新场景来源与问题/答案构成一起变化，不称纯分辨率或单一词长因果。

直接复用冻结run_visual_preference_v1.train，仅绑定新ROOT；不改旧科学源。偏好128题、8轨迹micro和初始ref全部没变，旧reference CACHE/COMPLETE按字节复用并绑定SHA，0新reference GPU调用。新source chosen CPU输入由已完成headroom复用；原native/偏好实际CPU输入、batch绑定保持。首正式步原DPO margin≤.05、log2、60因子/初始A0/B有限梯度、FP32保存仍由原trainer检查；新整体CE因回放QA改变，不要求等旧值。

## 真实控制与评价

共同起点960+1617、旧裁词数据CE与全30层CE+DPO全部预测精确复用，并保留旧数值策略/输入SHA。新两臂都是实际新训，旧复用不是独立重复。两臂Final32后无论训练拟合如何，都完成原内部960及完整1617；不按部分评测加步或选checkpoint。原source hold256不用于训练，本阶段不新增其Final评分；源内exact基线不与native VQA混列。

主候选预先固定`ce_dpo`：保留原内部hard学习/OCR与native/单多词/rare/保持保护门槛；完整1617对共同起点、新同数据CE、历史强全视觉DPO均≥+.5pp且配对图像簇95%下界严格>0。原强CE也完整报告；全1617rare1–5对初始及强控制点保护≥−.5pp。新CE对旧CE/旧强DPO作为预定数据策略次要比较；不看结果改主候选。原强基线/三seed/外部完整目标不因单seed内部信号完成。

完整逐题升降及区间沿原paired实现，旧原分数不回写；内部集已有历史曝光，CI不含训练种子、全历史多重选择，不称盲测或稳定提升。

## 资源和停止

两臂经共享空卡队列可并行，零compute/≥40GiB/<10%/3×15s/UUID锁，不抢占其他任务。每臂训练≤2400秒，内部/1617评测各≤1800秒，allocated≤65GiB；原完整optimizer/RNG/1,8,16,24,32检查点保留。基于历史训练约20分钟、两臂全量评测共享两卡约25–35分钟，本轮总预计45–60分钟（排队另计）；为了保留全部匹配控制与完整评测，明确高于默认短pilot下沿，不裁题救门槛。失败停止本配置，不扫mix比例/LR/扩步/seed；通过也先单独确认，不自动长训。

本轮预计64正式更新、5154新生成，0新偏好/教师回答；数据预检384生成与384 GT前向另列，不混训练效果。CV/科研/Agent/CI不改、不push，不动其他GPU任务。旧教师/rank/投影关闭状态保留。
