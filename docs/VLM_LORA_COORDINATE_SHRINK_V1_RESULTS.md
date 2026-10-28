# LoRA 坐标更新收缩 v1：完整内部负结果

2026-10-16，唯一候选 `lora_half32` 的 seed42 完整 Object342 + Text1617 已完成，独立1959题重评分零差异。原八项内部门槛 **4项通过、4项失败**，终态为 `completed_negative`、`goal_complete=false`。固定0.5坐标收缩没有恢复 rare/frequent 点值保护，并损失部分物体纠正；不追加该候选的0/200、external或POPE，不在失败后扫描α。用户真实提升目标仍active。

权威来源为[完整分析](LOCAL_ARTIFACTS.md)、[独立预测验收](LOCAL_ARTIFACTS.md)和[恢复终态](LOCAL_ARTIFACTS.md)。[学习页](LOCAL_ARTIFACTS.md)用于连接实际来源、代码和案例；展示不构成新增模型证据。

## 固定方法与模型身份

依据[原研究协议](VLM_LORA_COORDINATE_SHRINK_V1_PROTOCOL.md)，seed42仅引用[上一轮已验收的181步 generated-prefix KL Final](VLM_GENERATED_PREFIX_KL_V1_RESULTS.md)，没有新训练或伪造新 `TRAIN_RESULT`。在CPU FP32中，144个语言q/v LoRA A/B张量、1843200元素逐步执行：

```text
theta0 = raw_initial.BF16.FP32
delta = Final_FP32 - theta0
theta_half = theta0 + delta * 0.5
```

初始canonical master为 `6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7`，历史producer Final为 `d86e0ce25d30c57377e660a9c35cecdc03b3db2205338da81292a9239b13c485`。rank8、LoRA alpha16、底座与原生BF16评测路径、1024视觉上限、batch4、greedy32均不变。[导出manifest](LOCAL_ARTIFACTS.md)与[独立导出验收](LOCAL_ARTIFACTS.md)绑定实际来源、文件及数值关系。

这是A/B因子坐标上的固定变换，不是有效dense `BA` 更新的精确中点，也不保证KL或生成变化随α单调。单个固定α的结果不能推断所有收缩方法无效；本轮没有选择最优α。

## 整体指标与原八门槛

下表得分为百分数，差值单位为百分点。`teacher_kl32` 是同一历史producer的已验收控制，不能替补候选晋级，也不是本轮新独立训练。

| 指标 | 共同initial | 历史完整KL | 固定half |
| --- | ---: | ---: | ---: |
| Object准确率，342QA / 143图 | 95.614035 | 97.660819 | 96.491228 |
| Text共识分，1617QA / 1024图 | 87.111936 | 87.006803 | 86.982066 |
| unseen，687QA | 84.032023 | 84.090247 | 84.032023 |
| rare，342QA | 87.397661 | 87.076023 | 87.076023 |
| medium，171QA | 93.625731 | 93.625731 | 93.625731 |
| frequent，417QA | 89.280576 | 89.040767 | 89.040767 |

| 原预登记条件，half相对initial | 实测 | 判定 |
| --- | --- | --- |
| Object增益≥+1pp | +0.877193 | 失败 |
| Object单侧98.75%下界>0 | 0.000000 | 失败 |
| Text整体NI单侧98.75%下界>−0.5pp | −0.482168；点差−0.129870 | 通过 |
| Object recall点差≥0 | +1.754386 | 通过 |
| Object specificity点差≥−0.5pp | 0.000000 | 通过 |
| 逐类yes/no均衡子集准确率点差>0 | +1.171875，96.875000→98.046875 | 通过 |
| rare点差≥0 | −0.321637 | 失败 |
| frequent点差≥0 | −0.239808 | 失败 |

Object混淆矩阵为TP159、FP0、TN171、FN12；recall92.982456%、F1 96.363636%、specificity100%。逐类yes/no均衡子集256QA / 139图仅作原固定诊断，不替代完整342QA主端点，不表示18类QA数量相等或宏平均。

## 已保存区间及历史控制差异

所有区间直接引用官方完整分析；独立验收没有重新bootstrap。主端点家族为唯一候选×内部Object、TextNI、external Object、POPE准确率四项，α=.05/4。官方使用6000次配对图像簇bootstrap、seed20261015，Object143图簇、Text整体1024图簇；不能把QA或seed×image视为独立样本。当前仅seed42，无跨种子均值/样本SD证据。切片及控制对比区间是探索性点式区间，点保护不是切片NI或等价检验。

| half相对initial | 点差pp | 官方95%图像簇区间pp | 单侧98.75%下界pp |
| --- | ---: | --- | ---: |
| Object整体 | +0.877193 | [0.000000, +2.339181] | 0.000000 |
| Text整体 | −0.129870 | [−0.441449, +0.142510] | −0.482168 |
| unseen | 0.000000 | [−0.597451, +0.564650] | −0.695268 |
| rare | −0.321637 | [−0.855457, 0.000000] | −0.949555 |
| medium | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| frequent | −0.239808 | [−0.731752, 0.000000] | −0.954654 |

相对历史完整KL，half Object下降1.169591pp，官方探索95%区间[−2.424242, −0.282486]；Text下降0.024737pp，区间[−0.249688, +0.196560]，不能据此证明Text等价或更好。rare/medium/frequent逐题得分相对KL均不变；Text净差来自unseen的4题得分转移。

## 真实题目变化与可支持的解释

独立验收对新候选1959题应用原binary/TextVQA scorer，`new_score_mismatches=0`；历史initial和KL仅复用既有验收，不重评分。qid、图像、问题/提示摘要及保存的input IDs、mask、像素摘要、grid、padding、visual token均配对一致。这是保存输入metadata的核对，不是新的GPU tensor dump。

相对initial，Object有38条原回答、110条generated IDs变化，但只有3题得分上升、0题下降：`218985:category03`、`524338:category03`（car）及`524338:category08`（truck）由no变yes。相对完整KL，half有4题由正确yes退回no：

| Object固定qid后缀 | 类别 | KL→half | 得分 |
| --- | --- | --- | --- |
| `116694:category08` | truck | yes→no | 1→0 |
| `123842:category21` | cow | yes→no | 1→0 |
| `409030:category08` | truck | yes→no | 1→0 |
| `445327:category08` | truck | yes→no | 1→0 |

完整qid均带`object_presence_v2:000000`前缀。固定half损失了完整KL七个新增纠正中的四个；当前保留三个纠正，解释了Object增益由+2.046784降至+0.877193pp。这支持“该固定收缩削弱了部分已学物体纠正”的本轮观察，不证明整体物体能力或所有α的规律。

Text相对initial有17条原回答、26条generated IDs变化，9题得分变化（3升6降），净损失2.1个共识分。unseen六题得分变化净0；rare两题净−1.1，frequent一题净−1，medium无得分变化。切片频次沿用原32985条实际`train_all`监督目标的已验收划分，不等于视觉类别频率或预训练曝光。

| Text qid / 切片 | initial→half原回答 | 共识分变化 |
| --- | --- | --- |
| 15990 / rare | `$1.00`→`$1` | 1→0.3 |
| 29766 / rare | `spring`→`spring water` | 1→0.6 |
| 18020 / frequent | `0`→`channel 10` | 1→0 |

相对KL，Text有11条原回答、20条generated IDs变化；4题得分变化为4290（0→1）、20949（0.3→0.9）、5796（1→0）、20063（1→0），合计−0.4分。前两题改善没有转移到原rare/frequent扣分题，后两题新增损失。评分受答案覆盖、格式与自由生成共同影响；这些病例没有新增图像或logit证据，不能单独证明视觉遗忘、prefix mismatch或KL方向是因果。原两类切片保护仍失败，不能用整体NI通过替代。

## 实际成本与工程恢复

- **历史训练与本轮变换分开**：复用seed42真实181步producer及其原2888 Object / 5792 replay谱系；本轮新训练更新0、新teacher生成0、新reference forward0、新概率bank0。本轮只新增一份固定half导出，不将其称为新训练；恢复阶段未重复导出。
- **CPU导出**：manifest记录0.290709秒，范围是来源检查、FP32构造、诊断及序列化，不含后续独立`verify_export`。保存的更新L2为0.231521880899→0.115760940156，比例0.499999998731；FP32舍入下不声称范数数学精确减半，更不等同dense BA更新减半。
- **完整评测**：1959条正式model×QA，加16条额外评测预检。原Object342+8=350次已完成调用原字节复用；恢复仅新增Text1617+8=1625次。它们不是1975个独立样本。EVALUATION记录Object当前session wall53.908109秒、Text861.910496秒；恢复controller976.375390秒含该恢复会话的准备/排队/校验/分析等，不能替代整轮研究壁钟或纯GPU生成时间。
- **评测显存**：Object allocated/reserved为7.454186/7.728516GiB，Text为8.652318/9.835938GiB，均来自各自EVALUATION的`resources`。资源计时仅覆盖缓存VLM阶段，Object338题、Text1613题有计时，不将其外推为全流程或线上吞吐。
- **独立评分**：1959条新候选重评分零差，耗时4.127048秒；0新模型、生成、processor或图像编码，官方CI复用而非独立重采样。

原controller在Object完成后因整份`training_binding`把动态`export_verification.verify_CPU_wall_seconds`当身份比较而失败。该工程异常发生在完成件reader，不能解释为算法准确率失败。按[独立恢复协议](VLM_LORA_COORDINATE_SHRINK_RECOVERY_V1_PROTOCOL.md)，恢复仅对该精确计时叶子作copy后稳定身份比较，两侧必须是有限非负非bool实数；其它字段、SHA、导出数值、实际native BF16/升FP32、输入及完整缓存校验均保留。旧Object三工件和失败证据没有改写。

恢复首次prelaunch prepare另遇runtime相对source key与STUDY绝对key口径不一致，revision1源码、CPU与失败状态已归档。revision2只规范键表示并要求完整96项集合和值精确相等，没有删scope或放宽科学检查；正式统一87项CPU合同通过，0失败/跳过，runner wall5.291625秒。新11项恢复来源与原96项科学来源分域强绑定，新Text INPUT_LOCK/EVAL和analysis均强制携带恢复锁身份；原STUDY不伪称包含新代码。

## 验收边界与证据封条

本轮只检验一个固定α和一个历史producer checkpoint。原成功范围仍是新pilot全八项通过后，才允许0/200新producer同法导出、三种子内部筛选、各seed完整4372QA external（逐类yes/no均衡3438QA附加保护）、各seed完整9000QA POPE及最终独立验收；本候选未通过pilot，以上后续均未执行。历史控制不可替补，工程恢复成功不构成性能成功，不能针对dev病例修补或据负结果扫描α。

| 证据 | 文件SHA256 |
| --- | --- |
| [原STUDY](LOCAL_ARTIFACTS.md) | `283b63a86a8aa0f60892512e57dd0f9e46533a0796d7972e41b2a118e6d800af` |
| [恢复锁](LOCAL_ARTIFACTS.md) | `f3fc8305407653c98d624744b2c57381b3c27ef7605931c82112b21945fe7e84` |
| 完整分析 | `50cdf828d176cb296a43d657c485b9fc39424ca92c8cd5ddc1292bef6f4f2b89` |
| 独立预测验收 | `c5355ac3f742eca4d4db8c63a2c5e76ac7602a89f7d785e574d77797d33d4f86` |
| 恢复COMPLETION | `73fef127943e92956072c7573d339fc830b7f0bee164a9e72de93a15c10106b1` |
| 固定half导出manifest | `ad7821e1a348e0a165f9f11733b267c4e5b4769434bc69f5d3159fa9dd187f47` |

原Object预测文件SHA仍为`d653fae351b23e650714aca3af76abdc36859b9c8fbc39153f030f14746f797e`，INPUT_LOCK原文件SHA为`fdc3dd984ff96c6dfd950751d82fcae4b1b51701c00a7b08eb5a8a2c7fec3274`。manifest身份与真实producer TRAIN_RESULT身份独立保留，不混域。所有数值来自上述已完成工件；本结果文档没有重评分、重算区间或更改冻结源。
