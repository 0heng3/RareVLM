# VLM Object Behavior Replay v1 实际结果（stage31）

本研究已完成，结论为 **`completed_negative`，主候选teacher_ce32通过6/8项pilot门槛，`eligible_candidates=[]`，`goal_complete=false`**。内部物体识别改善，Text整体0.5 pp非劣效保护通过，但rare和frequent两个切片点保护失败。控制gt_ce32仅作seed42探索，不可替补晋级。本轮未扩展seed0/200，未执行候选的4372题external或完整9000题POPE确认；项目真实提升目标仍active。

方法与预设门槛见[原研究协议](VLM_OBJECT_BEHAVIOR_REPLAY_V1_PROTOCOL.md)。两臂从同一SFT42初始化，各训练2888条object QA、5792条不同Text回放QA，181次正式更新；object每更新16QA、replay32QA、microbatch4、两项分别按实际QA及assistant token均值归一、λ=1。teacher_ce32使用固定initial贪心生成的真实token IDs监督，gt_ce32使用原GT目标；teacher回答是行为参考，不是新增事实真值。冻结BF16底座、1843200个FP32语言q/v LoRA master与1024视觉token上限保持原协议。两臂目标内容、长度及计算量可能不同，不作等信息或等算力声明。

**主候选相对共同initial的正式pilot结果：**

| 端点/点保护 | initial | teacher_ce32 | 变化（pp） | 官方98.75%单侧下界（pp） | 判定 |
|---|---:|---:|---:|---:|---|
| Object accuracy，342QA/143图 | 95.614% | 97.661% | +2.047 | +0.552 | ≥+1且下界>0，2项通过 |
| Text整体，1617QA/1024图 | 87.112% | 87.062% | −0.049 | −0.423 | 下界>−0.5，NI通过 |
| Object recall | 91.228% | 95.322% | +4.094 | 点保护不要求CI | 通过 |
| Object specificity | 100.000% | 100.000% | 0.000 | 点保护不要求CI | 通过 |
| 逐类yes/no均衡子集accuracy，256QA/139图 | 96.875% | 98.828% | +1.953 | 点保护不要求CI | 严格>0，通过 |
| Text rare，342QA/312图 | 87.398% | 87.281% | −0.117 | −0.465（切片诊断） | 点不下降，失败 |
| Text frequent，417QA/367图 | 89.281% | 89.041% | −0.240 | −0.955（切片诊断） | 点不下降，失败 |

Object保存计分有7题升、0题降；TP从156升至163，FN从15降至8，FP仍0、TN仍171。均衡子集是完整342题预测的固定逐类yes/no均衡子集，不要求18类QA数相同，也不替代完整验证或external/POPE。这里的物体收益限于本内部18类动物/交通验证范围，不能推广为全80类POPE或一般多模态能力提升。

全部区间直接复用官方分析：6000次配对图像整簇bootstrap、seed20261015、单侧置信度98.75%，一个候选四主端点α=.05/4；不是全历史实验的FWER保证。种子轴先均值再重采样，QA和seed×image不能当独立样本。本轮只有一个order seed，区间条件于这些checkpoint，不能证明三种子稳定性。切片/recall/specificity点保护不等于切片非劣效或等价检验。

**控制与历史点值分开报告：**

| 条件 | 来源与范围 | Object accuracy | Text整体 | Text相对initial | rare变化 | frequent变化 |
|---|---|---:|---:|---:|---:|---:|
| teacher_ce32 | 本轮唯一候选，seed42 | 97.661% | 87.062% | −0.049 pp | −0.117 pp | −0.240 pp |
| gt_ce32 | 本轮探索控制，seed42 | 97.368% | 87.025% | −0.087 pp | −0.088 pp | −0.576 pp |
| wide32 KL | stage30历史探索，仅点值对照 | 97.661% | 86.964% | −0.148 pp | −0.322 pp | −0.480 pp |

teacher相对本轮GT控制，Object多正确1题（+0.292 pp），Text仅+0.037 pp；官方探索性Text差值95%图像簇区间为[−0.292,+0.365] pp，不能宣称显著优于控制。相对旧wide32，Object点值相同、Text高约0.099 pp，但未为该跨研究比较新增配对区间，且损失、目标信息及teacher成本不同；不能将差异全部归因为生成前缀，也不能据点值相同声称预测或模型等价。旧结果见[stage30报告](VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md)。

**两个失败切片各仅一题得分下降，不能据此作人群或总体遗忘结论。** teacher的Text有17条原回答变化、10题得分变化（4升6降）。unseen八题4升4降，累计得3.4分、失2.8分，净+0.6；mid无得分变化。rare净失0.4分、frequent净失1分，整体净失0.8分，对应−0.049 pp。这里的分数为保存的0–1共识分转移，不是新增人工正确性标注；rare/frequent按原32985个实际SFT目标频次定义，是监督曝光代理。

| 失败切片/题号 | initial回答→teacher回答 | 保存共识分 | 已有图像观察与边界 |
|---|---|---:|---|
| rare / 29766 | `spring` → `spring water` | 1.0 → 0.6 | root已查看原图，Spring Water字样确可见；多数参考偏好`spring`。这是答案粒度/共识分变化，不能直接称视觉识别退化。 |
| frequent / 18020 | `0` → `channel 10` | 1.0 → 0.0 | root已查看原图，遥控器数字细小；两臂均输出`channel 10`。具体是定位、数字辨识还是生成偏移仍属假设，当前证据没有隔离因果。 |

这两题用于解释预设门槛为何失败，不得据此选取dev题修补训练、放宽点保护或重新挑checkpoint。少量回答扩写、缩写或停止位置变化会影响精确共识分；也有数字答案改变，不能把全部下降都解释为措辞。

**工程验收与性能结论分别成立。** [训练完成审查](LOCAL_ARTIFACTS.md)的8149项检查通过；8149是检查数，不是训练QA数。它核验两臂完整实际监督流、FP32 master与Final、Adam181、保存RNG、冻结底座摘要及来源。两臂各181次正式更新，共362次、两份新adapter；独立及内置训练预检共4次丢弃工程optimizer step，未混入正式更新。工程通过不能替代效果门槛。

| 实际成本项 | teacher_ce32 | gt_ce32 | 口径 |
|---|---:|---:|---|
| 正式object/replay训练曝光 | 2888 / 5792 QA | 2888 / 5792 QA | 教师强制训练曝光，不是生成评测样本 |
| 正式训练active时间 | 4476.74 s | 4742.72 s | 原训练器记录，不含全部工作流等待/CPU准备 |
| 正式训练session wall | 4575.62 s | 4840.09 s | 两卡并行会话，不可相加当整轮壁钟 |
| 正式训练峰值allocated/reserved | 8.694 / 10.818 GiB | 8.694 / 10.818 GiB | 训练器实测，不是评测显存或跨方法加速证明 |

共享teacher实际生成 **5808 model×QA = 5792正式回复 + 16额外工程预检**，只计一次；actor训练中新增teacher生成调用为0，没有新增全词表概率cache。数据/来源验收通过不意味着5792个teacher回答均为事实真值，详见[teacher完成凭据](LOCAL_ARTIFACTS.md)。

内部评测产生 **3918正式model×QA = 两臂×(342 Object+1617 Text)**，另外保留 **4×8=32 QA评测工程生成**；不是3918个独立样本，也不能称本轮0新增forward。initial Object/Text及external/POPE基线严格历史引用，没有重新生成。独立CPU验收重算3918条新预测，保存score与冻结scorer **0差异**；initial未重复评分，CI未独立重采样。

原controller在两份Final完成后的adapter guard处遇到历史序列化缺字段：旧wide32结果未保存`replay_microbatch_order_sha256`，但旧结果绑定的完整schedule与本轮181步内容、原始字节均相同。版本化恢复仅派生这个缺失字段的内存视图，其他Final/实际输入/模型验证全部保留；原89科学源码、配置、协议、原锁、训练和失败证据不改。没有重新训练、teacher生成或重复已过训练预检。原尚未执行的评测预检仍执行后才进入完整评测。详见[工程恢复协议](VLM_OBJECT_BEHAVIOR_REPLAY_RECOVERY_V1_PROTOCOL.md)及[恢复完成凭据](LOCAL_ARTIFACTS.md)。恢复controller会话1490.61秒是评测接续的会话时间，不是整轮研究或纯GPU时间；工程故障不是两个性能点保护失败的原因。

可复核证据与哈希：

| 工件 | SHA256 |
|---|---|
| [官方pilot分析](LOCAL_ARTIFACTS.md) | `90bc7a0ff8b10387da21db934a4f4d21b5daebc8ec7a1fad701bba1b8e36cf15` |
| [完整预测独立验收](LOCAL_ARTIFACTS.md) | `369940cee3edcd889b66b0f36c9f19ed46006d207e249fe9982855926d9b8e99` |
| [有效工程恢复输入/源码锁](LOCAL_ARTIFACTS.md) | `8aed87a50409fe463a45b44123e832246f41d15541e22980d1c2b42b90cac4f5` |
| [训练工程验收](LOCAL_ARTIFACTS.md) | `e0cf20c0c0bde47243aff542dc2fc763da726e67f9de8f8a38fa67cae6aa4935` |

[学习页：实际图像、token监督和代码接线](LOCAL_ARTIFACTS.md)。教学病例与root对两张原图的观察不是完整人类真值审查，也不代表用户已掌握全部实现。

本候选不继续扩种。原预设成功范围仍是三种子内部筛选→完整4372 external→全9000 POPE→root独立实际预测/病例验收；不能因pilot局部改善而缩减目标，公开数据预训练重叠亦未知。下一项“固定teacher生成前缀上的full-vocabulary soft KL”仅是待新协议登记的研究计划，用于检验硬目标锐化与行为保持的关系；尚未实现、训练或取得结果，不能写成已证明的失败机制或新算法收益。
