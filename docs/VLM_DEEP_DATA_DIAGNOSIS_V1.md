# 数据与训练残错深入诊断 v1

本报告只连接冻结数据、实际训练追踪、固定Final读出与真实生成，不进行GPU计算、模型更新或确认评测，不改原标签、分母、科学源码或历史结论。新工具仅使用标准库CPU运算。

## 1. 原错曝光确实有限，但不能据此单因归责

全量训练600对/1200QA中，初始46QA错误、45对未双正确。实际人工1600QA曝光覆盖全部1200QA：800QA出现1次、400QA出现2次，没有原错完全未见或数据未送入模型的证据。

| 最终排序模型中的原错pair分组 | pair数 | pair曝光 | 仅1次/2次 | 原错QA曝光 | 平均曝光LR | 在线三token均值CE |
|---|---:|---:|---:|---:|---:|---:|
| 已修复 | 9 | 12 | 6/3 | 12 | 8.102e-6 | .1983 |
| 仍未双正确 | 36 | 48 | 24/12 | 49 | 1.028e-5 | .5157 |

两组都恰好2/3的pair仅出现1次。残错组曝光LR平均更高，不能把失败直接归因于只在衰减末期才见到样本。残错组初始判定更困难、CE更大，但曝光与难度不是随机因果干预；此处不证明增加epoch就有效。

36个未修复pair包含37个初始错QA，Final中36个QA错：35个原错仍错、1个原先正确的QA新增错误。原错pair与原错QA不是同一个分母。

## 2. 低曝光占比不等于只有同样比例的学习信号

原错QA只有61/1600个人工曝光（3.8125%），占全部3200QA曝光的1.90625%。但是排序臂中，这些曝光的CE值占人工CE总和45.865%、全部CE总和17.482%。普通CE臂相应为45.492%、17.405%。它们的高CE本身使错误样本并未按其数量比例被等比例忽略。

这些是**目标函数损失值总和的占比，不是参数梯度占比**，不能据此定量归因Adam优化、梯度裁剪和共享参数冲突。全部1600个人工曝光的监督都包含3token：yes/no、im_end和newline；保存的qa_ce是三token均值，不是答案首tokenNLL。所有3200QA曝光均未截断。

实际在线轨迹只给每个原错1–2个时间点。残错中12个出现两次的初始错QA，其后一次CE相对前一次平均下降.01471；不能据两个被其他样本更新间隔的点宣称“已收敛”或单调学习。

## 3. 已知语义不确定样本确实保留在训练池

事前固定盲审允许训练16对中14对通过即可准入，协议`VLM_COMPLEMENTARY_RANK_V1_PROTOCOL.md:23`已声明。实际代码`src/vlm_complementary_rank/study.py:54`只判断整体门槛；`:69`直接将全部train_all加入训练，`:91`对全部训练pair排期。实现遵守原协议，问题是该数据准入设计没有排除已知不确定样本，**不是代码暗中违反协议**。

| 原盲审ID | 原标记与原因 | 实际进入训练 | rank追踪位置 |
|---|---|---|---|
| T02 | pair_passed=false；“Is the door open?”一图门帘/门洞细节不足，助手给unknown | 是；step75、1次 | INPUT_TRACES.jsonl:1193、1194 |
| T08 | pair_passed=false；“Is that the right ball to be using in that situation?”依赖运动规则/场景推断，助手两图均unknown | 是；step18、1次 | INPUT_TRACES.jsonl:283、284 |

原审核文件位于`outputs/vlm_complementary_rank_v1/data_preparation/revision2/DATA_REVIEW.json`。T02从第44行、证据第51行、false第56行；T08从第134行、证据第141行、false第146行。T08属于最终9个修复pair之一，因此不能把9个修复全部称为已验证纯视觉能力增益。历史评分仍保留，不后验删题改分。

问题文本规则也无法保证问题可直接视觉验证。选择器快照`data_preparation/revision2/FINAL_SELECTION_BUILDER_SOURCE.py:264`明确tags仅为审核候选；`:274`把所有is there/are there打existence标签，`:278`把right打spatial标签，所以“predominance of silver”和“right ball”仍可能入选。未来新版本应对**语义可验证性**做准入，不能把词表命中等同可靠监督。

以上2对是已知样本级不确定性，不是标签一定错误，也不是全量噪声率估计；本轮不重标或追改原池。

## 4. 固定8原图残错核对

从36个未修复训练pair中的Final错QA，按固定salt SHA次序，选4个false-positive和4个false-negative。不是代表全池的随机噪声抽样，也不是专家新标注。图均以view_image读取原始文件，未裁剪或改图。下列margin来自冻结Final的canonical yes−no首token读出，非attention。

### 1. vqa2_train2014:509235000

Is his top button open?  原GT `no`（9/10票），initial/CE/rank：yes/yes/yes；rank margin=+1.125；训练曝光steps=[17]。

原图中领带与领口遮挡最上方纽扣；no标签未发现明确反证，但不能从此图可靠判断扣眼是否打开。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 2. vqa2_train2014:444344000

Is the candle striped?  原GT `no`（9/10票），initial/CE/rank：yes/yes/yes；rank margin=+0.375；训练曝光steps=[146]。

桌上红色蜡烛/烛杯可见纹理和亮点，未见成组条纹；原no标签有直接图像支持，rank却回答yes。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 3. vqa2_train2014:71754001

Is the container open?  原GT `no`（10/10票），initial/CE/rank：no/yes/yes；rank margin=+0.125；训练曝光steps=[82]。

暗光玻璃容器顶部可见疑似合上的玻璃盖及金属扣；原no合理，但反光与开合结构限制独立判断把握。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 4. vqa2_train2014:184218002

Is the woman trying to return a ball that went right to her?  原GT `no`（9/10票），initial/CE/rank：yes/yes/yes；rank margin=+1.375；训练曝光steps=[94]。

女子明显向侧方伸展接球；right to her可指直接到她面前，也易被误读为右侧，球路和意图不是单帧可完整验证的状态。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 5. vqa2_train2014:297559000

Is the packaging open for the controller?  原GT `yes`（10/10票），initial/CE/rank：no/no/no；rank margin=-2.375；训练曝光steps=[142, 190]。

Wii控制器与线缆已取出散放，支持包装已经拆开的yes语义；箱子本体开口不直接可见，不能把open一词一律当物理开口检测。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 6. vqa2_train2014:472676002

Is there more than one person?  原GT `yes`（10/10票），initial/CE/rank：no/no/no；rank margin=-1.375；训练曝光steps=[19]。

人物位于海滩底部且高度较小、身体与腿部重叠；可明确见一名主体，无法据全图可靠排除紧贴重叠的第二人，不认定原yes错标。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 7. vqa2_train2014:360575013

Are these buses the same color?  原GT `yes`（10/10票），initial/CE/rank：no/no/no；rank margin=-1.75；训练曝光steps=[147]。

前景和后方多辆巴士均为黄色主体配红色条带/车顶，原same color=yes有直接图像支持，rank仍回答no。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

### 8. vqa2_train2014:543166001

Is there a predominance of silver in this photo?  原GT `yes`（9/10票），initial/CE/rank：no/no/no；rank margin=-0.875；训练曝光steps=[63, 181]。

停车计时器/汽车/建筑混合灰银白色；predominance需要约定面积比例及silver与gray/white边界，无法按明确客观规则判定yes。

[本地图像说明：原始训练图](LOCAL_ARTIFACTS.md)

## 5. 数据侧下一步约束

- 不把全部失败归为数据噪声：巴士同色、蜡烛条纹等存在图像直接支持的原GT，却仍被模型答错。
- 不把600对含混数据继续反复增加epoch：先给未来新训练池保留可核对依据，对已知审核unknown建立显式排除，保留旧池和历史结果。
- 新方法验证需要同一可信数据池、相同曝光与保持任务的强CE对照。该数据诊断本身没有产生新的模型能力提升。
- 梯度方向、首答案NLL与目标失配由主诊断连接；本报告的均值CE不冒充首答案损失。

## 产物与复核

`outputs/vlm_deep_pipeline_diagnosis_v1/data/TRAIN_QA_JOIN.jsonl`保存340QA的三模型原读出、实际预测、两臂每次曝光step/LR/CE及原文件行号；错误富集诊断集不报整体训练准确率。`TRAIN_PAIR_JOIN.json`保存45个原错pair，`KNOWN_PRETRAIN_REVIEW_UNCERTAIN.json`保存事前不确定两对的全部证据，`VISUAL_REVIEW_COHORT.json`保存8图审查记录。原文件SHA记录在`DATA_DIAGNOSIS.json`。

工具`tools/diagnose_vlm_residual_data_v1.py`执行成功；对两臂共6400条训练trace逐条核对实际qid/source与冻结schedule、人工target/image SHA与训练视图一致。没有新GPU推理、训练、确认集预测或标签修改。
