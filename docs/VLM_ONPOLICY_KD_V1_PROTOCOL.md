# 学生生成前缀上的7B→3B蒸馏 v1

2026-10-29，在本轮学生生成和更新前登记。完整教师验证93.537415%对强学生88.027211%，+5.510204pp/95%[+4.232778,+6.834321]；rare+4.122807pp，保持+3.125pp。旧GT-prefix KD为86.846011%，恢复语言可挽回+1.063698pp但仍未超起点；本轮检验不同的监督状态来源，不重开旧scope/LR/T/λ扫描。

## 方法、来源与区别

参考GKD原论文 https://arxiv.org/abs/2306.13649 与官方TRL https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py 的`generate_on_policy_outputs`、`compute_loss`、`generalized_jsd_loss(beta=0)`。官方源码要求两head的vocab_size相同、当前collator路径主要处理文本；不直接安装/套Trainer。复用本项目已验证的相同151665实际token ID重归一化与图像输入，旧学生151936/教师152064的padding槽不进入KL。

旧KD在GT前缀计算T² KL(q7B||p3B)。新唯一候选`op_kd`在学生当前模型逐micro生成的前缀计算同方向KL；生成不反传，教师冻结，学生在同一原图/问题/原始token序列上反传KL。原GT CE（含EOS）仍覆盖全部512曝光。辅助mask仍固定103题/206曝光，不根据评测、当前错误、教师新置信或reward继续挑选。

本次是GKD思想的部署greedy变体：每次使用当前学生raw greedy32（与评测一致），不冒称论文的随机采样全分布复现或PPO/GRPO/RLHF。辅助仍只监督正文、排除第一终止符及之后token，保持旧KL的正文口径；空正文明确记0辅助，不伪造token，记录总数。直接拼接原始生成token到已验证推理prefix，不经decode再encode；mask、shift和共享词表均须正确。

CPU准入在新生成/训练前执行：复用强起点128保存轨迹、旧真实chosen labels和固定103准入索引，比较正文第j个token之前的原始前缀。至少16个不同QA存在偏离GT的受监督前缀，且总偏离位置≥32，才支持本小规模对比有足够不同状态；不足则关闭此固定greedy-prefix配置，不启动GPU或降低门槛。该检查是已见训练轨迹的机制准入，不是能力收益。

## 固定控制和预算

强ST-VQA SFT Final32为共同起点，旧384不同QA/512chosen/32更新/micro4/effective16/seed42、原排程不变；103辅助mask及缓存索引只用于固定准入，旧GT logits不用于新的学生前缀。联合现有204 FP32 LoRA因子/3686400参数，基座/head冻结；LR1e-4、warm4/linear32、clip1、dropout0、TF32off、T2/辅助系数1、1024视觉预算和数值政策均不变。

直接复用冻结旧GT-prefix KD、hard、uniform、历史强DPO、共同起点的完整结果，不重训关闭的候选。旧trainer核心不改；新worker在进程内替换额外目标、记录当前micro并加载冻结教师，其他优化、保存、恢复及同一GT CE原样复用。额外前向计算和正文token量不相等，必须实报，不称等算力实验或唯一因素因果。

新32正式更新，最多206训练生成/206教师正文前向/206学生额外正文前向。Final固定32，内部960+完整1617均完整生成，共2577次评测；不按训练拟合或部分分数挑点。GT CE首步应与旧0.6925567678481457相符；只核新原始token/教师-学生同框、有限梯度、两模型共卡内存与新目标保存，不重复旧loader/mask/教师准入审查。

## 验收与停止

主候选须在完整1617对强起点、GT-prefix KD、hard、uniform、历史强DPO全部≥+.5pp且图像簇95%下界>0。原OCR/native512/单词/多词/rare366/保持64对起点≥−.5pp；对旧GT-prefix/hard/uniform的OCR/native512/rare366/保持64≥−.5pp；完整rare342对起点及这三控制均≥−.5pp。固定6000次、seed20261021条件图像簇区间，非全历史/训练seed或外部确认。

主任务通过也仅内部单seed信号，三seed与外部仍须另登记。阴性关闭本固定on-policy配置，不扫T/权重/学习率/范围/rank、不扩步/seed或改名续训，不以教师93.537%充当学生提分。

## 资源、恢复与新运行检查

共享空卡队列零其他compute/40GiB/<10%/3×15秒/UUID锁，不终止其他任务。3B学生与冻结7B教师同卡，预计峰约50–60GiB、硬上限65GiB；旧trainer正式训练active≤1800秒原样保持，教师加载另报。每1/8/16/24/32保存原完整FP32 master/Adam/RNG/游标，学生生成使用greedy。额外rollout按update/micro保存原始IDs、prefix、KL、token量，恢复按旧checkpoint游标重新执行未提交update，不改旧轨迹。

训练worker结束后两项评测使用共享队列空卡并行，各沿原1500秒/65GiBcap，不同时留教师。预计新训练20–30分钟、评测15–25分钟，整轮约40–60分钟，依据首步实测修正，排队另计。CV/科研/Agent/CI不改，不自动push，旧实验科学源/协议/预测/权重保全。
