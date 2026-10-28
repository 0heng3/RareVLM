# 回放约束Adam：训练侧机制准入与匹配CE对照

2026-10-22事前登记。上轮soft_KL迁移+0.148423pp/95%[-0.339737,+0.669177]，hard_replay点值+0.439085pp，训练满分普通CE56/KL38/加强回放26，联合失败。固定增大保留权重的收益与纠错取舍值得研究，但这些结果没有证明梯度冲突或β过大是唯一原因。

## 假设与唯一算法变化

假设：普通继续SFT中，部分Adam实际位移对回放CE有不利的一阶方向；条件性修正位移，比始终加强回放监督更可能兼顾纠错与保持。沿用现有经过验证的`vlm_adam_conflict_v1.projection.project_displacement`，不重复旧人工yes/no小cohort诊断，而在当前TextVQA训练池、真实历史Adam状态下检验。

`L_base=(Σ8hard CE+Σ8replay CE)/16`。两新臂均按原4微批顺序、每微批一次`autograd.grad`，同时累计总梯度与hard/replay两源梯度；无额外反传，不改变CE归一化。共同clip1并调用AdamW，得到真实位移`δAdam=θproposal−θbefore`。

* 新鲜`ce`：原样应用Adam位移，记录同一几何计算但不修正。
* 唯一候选`projected`：解`min_δ 1/2||δ−δAdam||²`，约束`g_hard·δ≤0, g_replay·δ≤0`，以修正位移更新参数。不加正margin，零向量始终可行；可保留正交分量。Adam一二阶矩仍来自共同总CE梯度，不将修正位移伪写入moments；恢复存实际参数及该状态。

一阶约束不保证有限步BF16 loss下降或泛化；FP32参数加法后的真实投影也记录，避免仅报理想投影。此为GEM/A-GEM思路在本地Adam位移上的适配，不是原创算法、完整A-GEM复现或强化学习。参考[A-GEM论文](https://arxiv.org/abs/1812.00420)及[官方TensorFlow代码](https://github.com/facebookresearch/agem)。以前局部Adam投影诊断4题生成0/4不作为当前能力收益。

## 先做训练侧机制检验

只读上轮已经预先指定的expanded240 checkpoint16/32/48及next block17/33/49；恢复各自FP32 q/v参数与真实Adam状态，按相同4微批算新梯度，执行3次丢弃的诊断optimizer步，仅作局部位移测量。每点记录源梯度夹角、两源对实际位移的cosine、QP修正比例；不读验证结果、不扫checkpoint/阈值。结束精确恢复原SFT LoRA，诊断模型销毁，正式训练fresh模型/fresh Adam/RNG。

**事前准入**：至少2/3点满足`cos(g_replay,δAdam)>.01`且`cos(g_hard,δAdam)<0`。否则完成诊断并跳过两组新训练，不靠改阈值/新增点放行。活动预算180秒、allocated65GiB；3次丢弃step另计，正式更新为0。通过只证明局部机制存在，不保证新训练成功。

## 固定数据、模型与预算

直接复用`vlm_reference_replay_v1/DATA.json,CPU_INPUTS.json,SCHEDULE.json`，不重编码：128逐图审核困难题各4次，512不同回放题各1次（240多词/272单词），1024曝光/640独立QA；micro4/effective16、每更新8hard+8replay。训练原始uniform SFT42起点精确LoRA哈希5d8e2e21cc98686a4adf755b23e46c8f3512c7ff27d196df1b53f07cd934dd7b，FP32语言q/v1843200、冻结BF16底座/视觉、冻结FP32头/dropout0。LR2e-4、warmup8、线性衰减、AdamW wd0、clip1、Final64/seed42；不是从负Final继续训练。

新鲜CE与候选同循环，仅应用投影的开关不同。首正式更新16题input/labels与旧控制exact、逐题CE差≤1e-5，复用已有加载/mask/断点功能；只对新增零margin约束做1项闭式CPU测试，不堆叠全量审查。源码/协议/输入在诊断前冻结，任一异常停止对应流程，已有其他任务不操作。

两新臂各64正式更新、各2661真实生成（训练context468仅128计分+seen512+保持64+迁移1617），共128更新/5322生成。无教师缓存/偏好生成；三个历史checkpoint诊断48QA前后向另列。现有initial、plain、hard_replay同输入/分组预测复用0新调用，披露相关性；freshCE是主匹配对照。图像、像素隔离和原始SFT train/dev证明复用，不把训练目标文本频次当实体频次。

## 评测、成功与停止

原SFT曾训练图像seen512与原SFT图像隔离transfer1617分列，后者1024图、已多次内部观察而非盲测；公开数据预训练重叠未知。训练640图与两域/保持图像像素隔离。原固定生成batch4/FP32 readout/原EOS/greedy32及VQA评分不改，末批1合法。

候选在**两个域分别**对initial、旧plain、新鲜ce、强hard_replay均≥+0.5pp且6000次配对图像簇95%下界严格>0。两域rare1–5对initial≥+0.5pp；对全部对照的其他频次/单多词保护≥−0.5pp，rare对控制≥−0.5pp；保持64对全部对照≥−0.5pp、训练128满分净增32、seen原243满分多词保持≥−0.5pp。保留上轮门槛并增加新鲜CE控制，不以某一对照、子集、训练loss或QP满足挽救联合失败。

所有升降逐题保存。单seed区间只反映固定模型图像抽样，不含训练随机性或全历史多重选择；不称稳定能力提升。固定Final64，不扫LR/系数/seed/checkpoint；失败不自动扩步/正式benchmark/简历，通过也须另登记独立确认与种子0/200。CV/PDF/科研/Agent/CI不改，不push。

共享队列先一张空卡做180秒机制诊断，通过后两卡各一臂训练与评测；任一卡忙则在其他空卡顺序接续，零compute/≥40GiB/<10%/3×15秒/UUID锁，不抢占。每臂训练累计1800秒、评测累计2400秒、allocated65GiB，checkpoint保存实际参数/Adam/RNG/日志偏移与UUID；进度分阶段，预估由实测更新。学习页`learning/回放约束Adam_纠错与保持.html`连接8张事前固定原图、公式、真实代码、诊断数值与后续保存回答。
