# 物体监督适配v2：局部物体改善未通过整体晋级门控

官方pilot筛选及COMPLETION已完成，trial状态为`completed_negative`。两个候选均未达到预注册联合门槛，未追加seed0/200，未新增POPE；用户的真实提升目标仍active，脚本完成不等于目标成功。

原控制器在全部新预测完成后，因新汇总器把历史Text的空`category`与新记录缺省字段视为不同而停止。原[失败STATUS](LOCAL_ARTIFACTS.md)与controller.log原哈希保留。独立CPU恢复先由冻结reader完整验证，再只在内存Text副本中规范化空类别别名（移除1617个已有空key）；object/POPE、全部非category字段、分数、原assert／评分／bootstrap／门槛及科学文件字节均不变。详见[RECOVERY receipt](LOCAL_ARTIFACTS.md)；恢复新增更新、预测和model forward均为0。


## 固定实验与计数

同一完整TextVQA-SFT42起点、原BF16基座／FP32语言q/v LoRA（1,843,200可训练参数）、视觉上限1024、Final@1 epoch、lr5e−6、microbatch4×accumulation4=object16。两组各2888个object监督QA／181正式更新，Replay每更新另加8题Text，共1448额外QA，两类逐题均值独立归一；末尾object8题分母为8。

本轮新增2个adapter、362正式更新、4个已丢弃并精确恢复的工程optimizer step。正式新预测为3个object×342＋2个Text×1617＝**4260 model×QA**；原始initial Text1617题严格历史复用、0新增生成。评估工程预检另有**40 model×QA生成**，还有有限logit forward与训练前后向／checkpoint重算，不能将报告工具“0新forward”写成整个实验0 forward。上述QA量不是独立样本数。

object full val为342QA／143图，18类；平衡诊断256QA／139图取自同一完整预测，0额外生成。Text为1617QA／1024图。单固定checkpoint条件下按图像整簇6000次bootstrap，seed20261015，双臂×两个主端点采用单侧98.75%下界（alpha=.05/4）；95%区间与切片区间用于单独探索解释。

## 官方主结果

共同initial object为95.614035%，历史initial Text为87.111936%。差值均以pp计。

| 候选 | object accuracy% | object差值 | object95%CI | object单侧98.75%下界 | Text共识% | Text差值 | Text单侧98.75%下界 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| object_only | 97.368421 | +1.754386 | [0.304878, 3.488372] | +0.292398 | 87.149041 | +0.037106 | -0.367656 |
| object_replay | 96.783626 | +1.169591 | [0.000000, 2.710843] | +0.000000 | 87.019171 | -0.092764 | -0.502144 |

逐项失败门控（保留官方字段，不调阈值）：

```json
{
  "object_only": [
    "text_rare_1_5_point_nondecrease",
    "text_frequent_gt20_point_nondecrease"
  ],
  "object_replay": [
    "object_accuracy_one_sided_98_75_lower_gt0",
    "text_one_sided_98_75_lower_gt_minus_0_5pp",
    "text_rare_1_5_point_nondecrease",
    "text_frequent_gt20_point_nondecrease"
  ]
}
```

object_only在新物体任务上达到局部accuracy改善门槛，但整体Text保持与罕见／高频监督答案保护是不同问题：整体Text通过或点差为正，不能代替rare／frequent点值保护。object_replay的object单侧下界未严格大于0，点差达到+1pp仍不足以晋级；Text数值与门控一律以上述官方JSON为准。平衡子集只作为附加grounding保护，不替代完整val或全9000题POPE。

## Text切片与边界

| 候选 | 实际SFT答案曝光切片 | QA数 | 差值pp | 图像簇95%CI |
| --- | --- | ---: | ---: | --- |
| object_only | unseen_0 | 687 | +0.393013 | [-0.297619, 1.124129] |
| object_only | rare_1_5 | 342 | -0.321637 | [-0.855457, 0.000000] |
| object_only | medium_6_20 | 171 | +0.000000 | [0.000000, 0.000000] |
| object_only | frequent_gt20 | 417 | -0.239808 | [-0.731752, 0.000000] |
| object_replay | unseen_0 | 687 | +0.043668 | [-0.548986, 0.615028] |
| object_replay | rare_1_5 | 342 | -0.116959 | [-0.741840, 0.495627] |
| object_replay | medium_6_20 | 171 | +0.000000 | [0.000000, 0.000000] |
| object_replay | frequent_gt20 | 417 | -0.335731 | [-1.225565, 0.480769] |

rare/frequent是原32985QA实际SFT target文本曝光的代理，不是视觉语义类别或预训练频率。点保护失败不等于已证明切片总体显著退化；整体非劣也不能证明切片非劣。未通过联合筛选，因此0/200复制与POPE确认没有执行，不能宣称稳定泛化、通用保持或长尾改善。

## 真实成本与不等信息

| 条件 | object QA | 额外Text QA | 训练active秒 | 单分支session wall秒 | 训练allocated / reserved GiB | object / Text input token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| object_only | 2888 | 0 | 664.406 | 726.600 | 7.773 / 7.902 | 1121392 / 0 |
| object_replay | 2888 | 1448 | 1718.252 | 1802.278 | 8.695 / 10.887 | 1121392 / 1414049 |

原工作流从LAUNCH至原控制器失败约3021.481秒；独立CPU恢复46.536秒，两者分列，不用短恢复耗时替代整轮。原工作流时间包含排队、加载、输入／权重核验、预检、训练、评测和失败汇总，非纯GPU耗时；不以两个并行分支耗时相加冒充壁钟。训练active时间、分支session、评估缓存生成阶段和完整工作流壁钟分别保留。评估资源原值见便携JSON的`evaluation_resources`，首batch预热与加载等排除范围保留原字段。

Replay增加1448个真实Text监督、1,414,049个训练input token及额外前后向；同初始化、object顺序、更新数、参数量和视觉上限不代表等信息、等FLOPs或等算力。没有跨运行时加速／线上服务性能结论，也不把历史NF4到本轮BF16路径解释成纯dtype收益。

## 独立核验与真实病例

- [病例1：可见鹦鹉仍被漏检](LOCAL_ARTIFACTS.md)；[病例2：遮挡车辆类别存在歧义](LOCAL_ARTIFACTS.md)。保留原标签与官方得分；局部助手看图不是全量独立人类真值，也不证明小目标／遮挡的因果份额。
- [数学、图像簇与官方解析核验](LOCAL_ARTIFACTS.md)仅证明未发现阻断实现问题，不证明性能成功。
- [完整object_only Text重评分与频次核验](LOCAL_ARTIFACTS.md)区分整体NI与rare/frequent点保护；不替代官方整轮判决。
- [两臂真实首更新输入匹配核验](LOCAL_ARTIFACTS.md)核验初始object输入，而不是全历程／等算力或用户已独立掌握的证明。
- [真实图像、token、逐题CE与回放学习页](LOCAL_ARTIFACTS.md)将实际数据／GPU记录与教学假设分开。

v1质量审查57/64被拒、v2新总体61/64通过；类别、可见大小与排除图像不同，两比例不是配对质量提升，也不是全量噪声率。上一轮DPO相对偏好改善未转化为POPE收益及FN瓶颈，是本研究的机制动机，不自动证明新物体监督能跨任务成功。

## 来源、验收与下一边界

官方[完整pilot分析](LOCAL_ARTIFACTS.md)与[COMPLETION](LOCAL_ARTIFACTS.md)保持原字节；该独立导出工具未修改src/config/protocol/STUDY/模型／预测／正式CV，未重跑CPU全套测试或GPU。源码、数据、模型、初始化和CV在导出前后按研究锁逐文件验证；详细哈希和范围在[便携指标JSON](../results/vlm_object_adapt_v2_metrics.json)及导出receipt。

本trial保留负联合结果。新实验若继续，需要依据新的错误机制另立协议；本报告不改变原门槛、checkpoint、数据或目标状态，也不自动启动后续实验。
