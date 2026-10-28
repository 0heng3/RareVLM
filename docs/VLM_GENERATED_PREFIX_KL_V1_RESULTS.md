# 固定生成前缀全分布KL v1 实际结果（stage32）

本研究已完成，结论为 **`completed_negative`：唯一候选teacher_kl32通过6/8项pilot门槛，rare/frequent点保护失败，`eligible_candidates=[]`、`goal_complete=false`**。Object改善和Text整体0.5 pp非劣效保护通过，但完整真实提升目标尚未达到。未扩展seed0/200，未执行候选的4372题external或全9000题POPE确认；原完整成功范围保持，不以pilot替代目标。

方法、变量与原成功门槛见[事前协议](VLM_GENERATED_PREFIX_KL_V1_PROTOCOL.md)。本轮逐题复用父轮seed42已验收的5792条teacher生成前缀、token IDs、角色及2888条object QA，不重新生成teacher回答或decode后重分词。actor从同一SFT42初始化，使用BF16冻结底座及144个语言q/v LoRA FP32 master，共1843200可训练参数；object16QA、replay32QA、microbatch4、181次更新、Final@1epoch及原学习率保持。

变化仅在回放目标：父teacher_ce32使用hard CE；本轮在相同固定生成前缀上使用独立固定initial reference的全词表forward KL（λ=1、T=1、词表求和、每题assistant token均值、再按QA均值）。reference使用BF16基座/FP32 LoRA、BF16 autocast、eval/no_grad，不共享actor参数storage，不参与optimizer。回放hard CE只作同次logits的detach诊断。父生成前缀来自nativeBF16解码，本轮soft reference为训练算术路径，两者角色及成本分列。本轮没有学生在线采样、奖励、偏好优化或RL。

**主端点与8项门槛：**

| 门槛 | 实际值 | 判定 |
|---|---|---|
| Object accuracy增益≥1 pp | 95.614→97.661%，+2.047 pp | 通过 |
| Object单侧98.75%下界>0 | +0.552 pp | 通过 |
| Text整体NI下界>−0.5 pp | 87.112→87.007%，差−0.105 pp，下界−0.463 pp | 通过 |
| Object recall点不下降 | 91.228→95.322%，+4.094 pp | 通过 |
| Object specificity点差≥−0.5 pp | 100→100%，0 pp | 通过 |
| 逐类yes/no均衡子集点差>0 | 96.875→98.828%，+1.953 pp | 通过 |
| Text rare点不下降 | −0.322 pp | 失败 |
| Text frequent点不下降 | −0.240 pp | 失败 |

Object共342QA/143图，TP163、FP0、TN171、FN8，相对initial有7题得分上升、0下降，FN由15降至8。逐类yes/no均衡子集为256QA/139图，是完整预测的固定子集，不代表每类QA数相同，也不替代完整验证集。Object收益限于当前内部18类验证范围，尚无本候选三种子或external/POPE确认。

**Text各切片的保存分数转移：**

| 切片 | QA数 | initial→KL | 差值（pp） | 官方98.75%单侧下界（pp） | 得分变化QA（升/降） | 净共识分 |
|---|---:|---:|---:|---:|---:|---:|
| 整体 | 1617 | 87.112→87.007% | −0.105 | −0.463 | 9（3/6） | −1.7 |
| unseen | 687 | 84.032→84.090% | +0.058 | −0.655 | 6（3/3） | +0.4 |
| rare | 342 | 87.398→87.076% | −0.322 | −0.950 | 2（0/2） | −1.1 |
| mid | 171 | 93.626→93.626% | 0.000 | 0.000 | 0（0/0） | 0.0 |
| frequent | 417 | 89.281→89.041% | −0.240 | −0.955 | 1（0/1） | −1.0 |

相对initial，Text原回答有17条变化，9题得分变化，累计得2.7分、失4.4分，净失1.7分。这里的分数是冻结TextVQA 0–1共识分；rare/frequent由原32985条实际SFT目标频次定义，是监督曝光代理，不是视觉类别或人群统计。

rare除父CE已有的29766外，新增15990扣分：

| 题号 | initial→KL回答 | 保存共识分 | 边界 |
|---|---|---:|---|
| 15990，rare | `$1.00` → `$1` | 1.0→0.3 | 金额字符串缩短，净失0.7分；本轮未新增原图语义审查，不能直接归因为视觉辨识退化。父CE该题保持1分。 |
| 29766，rare | `spring` → `spring water` | 1.0→0.6 | 与父CE相同；父轮root已看到原图Spring Water字样，参考答案粒度影响共识分。 |
| 18020，frequent | `0` → `channel 10` | 1.0→0.0 | 与父CE相同；细小遥控器数字的定位、辨识或生成机制仍未隔离。 |

少数题目即可触发事前登记的点保护，不能据此宣称群体遗忘或普遍视觉退化；也不能删除这些题、放宽门槛或针对dev病例构造训练补丁。整体NI通过不等于Text准确率提升，也不替代切片保护。

**相对父teacher CE，KL确实产生不同Text输出，但没有更好保持的证据。**

| 指标 | 父teacher_ce32 | 新teacher_kl32 | KL−CE |
|---|---:|---:|---:|
| Object accuracy | 97.661% | 97.661% | 0.000 pp |
| Text整体 | 87.062% | 87.007% | −0.056 pp |
| Text rare | 87.281% | 87.076% | −0.205 pp |
| Text frequent | 89.041% | 89.041% | 0.000 pp |

官方探索性KL−CE的Text整体95%图像簇区间为 **[−0.273,+0.138] pp**，98.75%单侧下界−0.304 pp；区间跨零，不能宣称普遍优于或劣于CE。父CE仅为不可变历史参照，不是新增独立重复，不能替补候选。父轮事实见[stage31结果](VLM_OBJECT_BEHAVIOR_REPLAY_V1_RESULTS.md)。

相对CE，Object逐题保存回答、生成IDs和分数均无差异；Text有14条原回答变化、7题得分变化（3升4降），净失0.9分。30条生成IDs变化包含padding差异，不能等同30道回答变化或错误。具体得分转移为：

| 题号 | CE→KL回答 | 保存共识分 | 变化 |
|---|---|---:|---:|
| 9811 | `cd hd` → `g4 hd` | 0.3→1.0 | +0.7 |
| 28200 | `8jc` → `8 jc` | 0.3→1.0 | +0.7 |
| 29122 | `browns edinburgh` → `browns` | 0.9→1.0 | +0.1 |
| 14958 | `incanto bliss` → `incanto` | 0.9→0.6 | −0.3 |
| 15990 | `$1.00` → `$1` | 1.0→0.3 | −0.7 |
| 16984 | `sept 6 2014` → `september 6 2014` | 1.0→0.6 | −0.4 |
| 34368 | `c5083` → `5083` | 1.0→0.0 | −1.0 |

因此，本轮不是复制CE输出或空跑KL；但改成全分布KL仍未消除两个原失败病例，并新增rare扣分，不支持“hard CE锐化是唯一充分失败解释”。同一seed上的输出变化、CI跨零及历史参照也不能证明完整因果机制、能力等价或KL在其他设置普遍无效。

**工程正确性与效果分别验收。** [Final训练工程验收](LOCAL_ARTIFACTS.md)通过6978项检查，核对实际输入/角色、181更新、FP32 master/Final/Adam/RNG、固定reference及来源；6978是检查数，不是QA数。GPU嵌入式预检实测初始KL为0、初始replay/object梯度比约1.847×10⁻⁷，临时更新后KL及恢复梯度为正，随后精确恢复；这些证据说明数值/训练接线通过，不等于性能门槛通过。只有一次丢弃工程optimizer step，没有独立重复训练预检。

| 实际成本项 | 数值与口径 |
|---|---|
| 新训练 | 1份Final adapter、181次正式更新 |
| actor正式曝光 | 2888 object +5792 replay =8680 QA；不是自由生成评测 |
| 独立固定reference正式前向 | 5792 QA；teacher-forced全词表分布，不是新增teacher贪心生成 |
| 正式训练active时间 | 6867.0028599 s，已包含reference前向2436.8988000 s；二者不能再次相加 |
| 正式训练峰值allocated/reserved | 15.72534 /16.42969 GiB，包含actor及reference路径，低于65 GiB上限 |
| 丢弃工程步骤 | 1次；另计80 actor F/B QA、64 reference前向QA |
| 正式内部评测 | 1959 model×QA =342 Object +1617 Text |
| 额外评测工程生成 | 16 QA =两格各8；与1959正式评分分列 |
| 新teacher贪心生成 | 0；5792条生成前缀严格复用父轮已验收token IDs |
| 全词表概率bank | 0；reference在线独立前向，不读取或新建旧GT-prefix概率bank |

参考前向时间属于训练active时间的组成部分，不是额外可相加的总时长；上述时间也不是完整研究壁钟或跨方法推理加速证据。与父CE相比新增reference计算、两模型存储及不同梯度路径，不能宣称等FLOPs或等时间。

[预测独立验收](LOCAL_ARTIFACTS.md)仅重算本轮1959条新预测，保存score与冻结scorer **0差异**；initial和父CE重评均为0，CI未独立重采样。全部区间直接复用官方6000次图像整簇配对bootstrap、seed20261015、单侧98.75%，一个候选四主端点α=.05/4。Object为143图簇，Text为1024图簇；切片按各自图簇，不把QA或seed×image当IID。单seed区间条件于这些checkpoint，不能证明三种子稳定性；切片点保护不等于切片NI或人口结论。

可复核工件与SHA256：

| 工件 | SHA256 |
|---|---|
| [官方pilot分析](LOCAL_ARTIFACTS.md) | `473db718ad2bec03b4e5b5f17afb841ed0a7993e8cf0327692fc90700c0a6698` |
| [1959条新预测独立验收](LOCAL_ARTIFACTS.md) | `cd0aec8cbf0b278b3e36d158c138533dd8279a90910d1bf73c004565b19802d2` |
| [6978项训练工程验收](LOCAL_ARTIFACTS.md) | `bc4bd35748901636e387508f70c5b2ab5a341e576fde92aaa31bfa754faf066c` |
| [本轮科学输入/源码锁](LOCAL_ARTIFACTS.md) | `5c70250fa42d7a51c9b54b69c95fb37ca9450c32f8a4fc53cc855ec9fd4193e6` |
| [本轮完成凭据](LOCAL_ARTIFACTS.md) | `4913a56c28c7f2f74724634985991034a54820b57b4666f902008ff63afce41c` |

[学习页：原图、实际token/角色、KL原理、代码及保存预测](LOCAL_ARTIFACTS.md)。教学演示、少量病例和工程验收不代表用户已掌握全部实现，也不是全量人类事实真值复审。

本负结果不扩种。原成功范围仍要求seed0/42/200内部筛选、完整4372 external及3438均衡子集、三seed全9000 POPE（500共享图像、三类别各3000）和root最终独立验收；各原收益、NI、切片及类别保护均保留。内部dev已反复探索、公开benchmark预训练重叠未知，均不作盲测或预训练未见声明。下一研究方向“固定更新收缩”仅为待定计划，尚未实施，亦不保证有效；须另有事前协议、代码/CPU检查和来源锁，不能针对这几道dev题修补结果。
