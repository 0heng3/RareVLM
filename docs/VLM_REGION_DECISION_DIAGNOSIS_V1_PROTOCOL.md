# 区域偏好失败：分歧token与训练信号只读诊断

2026-10-20登记。前32步CE与CE+DPO均仅8→11/28、pair0→1/14，未通过短学习门槛。本项检验序列均值是否掩盖关键生成决策，以及DPO与该决策是否存在明显局部方向冲突，不重复旧训练链路审查或扩长。

固定选择训练白名单中BEFORE/CLOSE、TELL/YOU、PAR/CHAMPAGNY三组A/B，共6QA；包含仍错A框、正确B框和已纠正PAR。不是随机样本或泛化评测。对初始错误使用实际初始误答文本作为contrast；初始已正确题用登记的另一框词。重编码实际回答文本，不冒称取回历史生成raw token流。

两序列包含完整assistant EOS/换行；在完整编码的最长共同prefix之后取首个不同token，并核验相同prefix。可以是EOS与继续文字的分歧。所有模型使用同固定2序列批次、原生全图、1024预算、相同BF16底座/FP32 q/v与FP32冻结head、dropout0/TF32off。不是历史batch4生成概率的精确复现，不重新生成或评估主dev。

初始、SFT32、DPO32各12条teacher-forced序列，总36序列/0更新。记录关键token正确/错词logit差、正确token rank、完整CE、关键token CE、完整序列logp。初始仅做局部q/v梯度：完整CE、关键token CE、DPO、关键token margin；记录范数和cosine，以及实际Final delta的一阶margin投影。一阶梯度不等Adam历史或BF16真实有限位移。读取视觉输出activation梯度只说明局部敏感性，不是注意力、merger参数梯度、容量充分或视觉训练必然有效。

参考为该固定诊断批次下完整初始SFT42。先运行初始再并行两个Final，以相同编码比对。复用共享GPU队列三次15秒空闲确认和UUID锁；不抢占其他任务。每worker300秒上限、无optimizer/新adapter；前后LoRA hash不变。新增脚本/协议/输入锁独立于旧冻结模块。

若full CE与关键token CE明显对齐且DPO也对齐，不再把局部梯度冲突或纯EOS稀释当已证主因；若观察局部冲突，只支持对相应样本另登记最小匹配对照。两种情况均不得用训练内读数宣布独立能力提升，也不自动启动200步/换PPO或GRPO。
