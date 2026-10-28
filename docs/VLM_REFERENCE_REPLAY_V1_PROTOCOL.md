# 初始模型软分布保留 vs 相同系数回放CE加强

2026-10-21事前登记。merger扩范围研究已完整退出0：训练满分64/128，但SFT未用图像dev1617由87.328386→86.914038%，qv控制86.951144。多词469题8升/23降，+4−13.9score，净−2.110874pp，条件95%[−3.677419,−.589473]；13个降分回答变短。不是全部错误都由截短或同一机制导致，切片未作全历史多重校正。新实验检验保留式微调，不扩旧失败步数、不使用merger、不换底座。

## 数据与两个训练臂

固定`vlm_length_replay_v1/expanded240`的128已审困难题/512不同初始满分回放（240多词/272单词），每困难4次/每回放1次，64更新/1024真实曝光，原顺序micro4×4。两臂原uniform SFT42精确FP32 q/v1843200、BF16冻结基底/视觉、冻结FP32头/dropout0；lr2e−4/warmup8/linear/AdamW wd0/clip1/Final64。训练从原SFT开始，不从历史负Final或校准checkpoint继续。

每题CE是assistant内容＋真实EOS的token均值，不监督EOS后换行。`L_base=(Σ8hard CE+Σ8replay CE)/16`。

* 唯一候选`soft_kl`：`L=L_base+β·mean_replay KL(p_initial || p_actor)`，T=1，完整151936词表，每题assistant有效位置均值后QA均值。
* 新强控制`hard_replay`：`L=L_base+β·mean_replay CE_GT`，相同β/QA/更新，检验软分布保留是否超过单纯强化硬标签回放。初始KL近0而硬CE可非0是目标语义，不强求两个总loss相同。
* `plain`：只读已完成expanded240 Final，不重训；原三cohort预测复用、dev1617首次新评测。候选须同时优于原初始、普通CE和加强CE，不能靠一个差控制挽救成功。

这属于有监督CE与冻结教师蒸馏约束，没有奖励函数/策略梯度，不称DPO/PPO/GRPO复现。概念参考[Learning without Forgetting](https://arxiv.org/abs/1606.09282)，不是原CNN任务的复现。

## 教师与训练数据校准

只在相同回放GT前缀/实际四题padded输入上缓存原初始全词表FP32 logp，128 micro/512QA，真实内容与EOS位置；每批input_ids/labels/attention/grid/pixel tensor的dtype/shape/bytes摘要须与消费者exact。teacher.eval/no-grad/权重哈希不变，soft消费者与cache producer同UUID。无解码后重新分词、无生成偏好对或自动负例。缓存本身0optimizer，约1GB级，实际字节/entropy/非GT质量记录。

校准仅读取历史expanded240控制checkpoint16/32/48及其next schedule block（更新17/33/49）：每点16QA，0新optimizer，计算完整q/v梯度`g_baseCE`、`g_KL_replay`、`g_CE_replay`。β固定为`clip(median(||g_baseCE||/||g_KL||),.1,100)`；不用任何验证分挑β。报告KL与回放CE梯度cos/相对差；若三点均cos≥.999且相对差≤.01，标为基本冗余，跳过两组新训练，保留普通CE迁移评测与诊断。KL梯度非有限/≤1e−8等不满足则退出，不盲目跑长训练。该校准仅平衡历史状态下原始梯度量级，不证明Adam步长相等或β最优。

校准后精确恢复原SFT FP32 LoRA并核验哈希，重新seed42/fresh Adam。β输出与cache manifest及PLAN绑定，训练lineage包含β文件SHA；完成校准不重复运行。首正式step在optimizer前核验16QA输入/labels与旧普通CEexact、baseCE差≤1e−5；soft初始每QA absKL≤1e−5，hard初始CE正值合法。新criterion有独立闭式KL方向/逐QA归一化/左mask/梯度oracle CPU验证。

## 评测与完成条件

全部复用已修正的merger数据协议：原SFT train白名单seen512与原SFT未用图像transfer1617/1024图分列，后者内部已观察而非盲测，预训练重叠未知。640新训练图片与两个域/保持均不相交；原SFT训练与dev ID/pixel隔离证明复用。迁移unseen0/rare/medium/frequent定义照旧。initial全部预测同条件复用0新调用；plain三旧cohort复用、迁移1617新生成；soft/hard各2661新生成（468训练背景仅128计分+512+64+1617），总6939新生成、128新正式更新，缓存512QA前向与校准48QA前后向另计，不算新增独立训练数据。

候选在seen512和transfer1617两个域，分别对initial/plain/hard_replay整体均≥+.5pp且6000次配对图像簇95%下界>0。两域rare对initial≥+.5，其他频次/单多词对initial/强控制≥−.5、rare对控制≥−.5；保持64≥−.5、训练128满分净增32、seen原初始满分多词243保持≥−.5。旧门槛保留，不以新域/子集替代失败；固定Final64不扫β/checkpoint/seed。单seed条件区间不含训练随机性，未校正全历史选择；通过才另登记独立确认/0,200与原正式跨任务，失败停止本配置，不扩步/正式/CV。

共享queue顺序plain/soft_kl/hard_replay：两空卡先并行plain迁移和soft缓存/校准/训练/评测；plain释放后自然接hard训练评测。保护已有任务，零compute/40GiB/<10%/3×15秒/UUID锁。cache活动每次600秒、校准180秒、每臂训练累计1800秒、每role评测累计2400秒、allocated65GiB；按实际缓存/校准/训练/评测分阶段进度。完整FP32/Adam/RNG及日志offset恢复；缓存与β输入错则拒绝混用，source冻结后不静默修改。

实现`src/vlm_reference_replay_v1/`，输出`outputs/vlm_reference_replay_v1`。原科学文件/分数/权重保留，用户真实能力提升目标仍未完成。
