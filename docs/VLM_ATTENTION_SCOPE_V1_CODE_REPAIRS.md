# Q/V与Q/K/V/O失败：代码、日志与定向修复

2026-10-19。响应“深入代码层面和实验日志挖掘错误原因并修复”。新增内容独立存放于`outputs/vlm_attention_scope_repair_v1`；原科学代码、配置、标签、预测、checkpoint和原失败门槛均保留。此次没有训练更新，也没有新的能力提升声明。

## 本轮结论

1. **训练监督仍有不确定性，已修正下一训练输入。** 原准入池1186题中1155题仅有官方高共识、未经逐图审核。此次针对全部10道历史no训练错题逐图核对，4题有可见支持，6题因遮挡、对象指代或范围不确定而不适合直接当成确定负例。不是认定6题全部错标，更不是全池60%噪声。新训练视图排除6个整对、12题：593对/1186题→587对/1174题，不改原答案。
2. **发现隐式解码配置未进入评测锁，已提供独立修复入口。** 底座带`repetition_penalty=1.05`；原`generate(do_sample=False)`仍继承它，原评测`INPUT_LOCK`没有记录完整生成策略。该策略能改变原始logits，不能将greedy理解为无logits处理。新入口显式声明生成策略并独立锁定，要求新同策略初始基线，不覆盖旧缓存。
3. **新增GPU实测排除了“该重复惩罚导致这批no失败”的解释。** 固定训练16题×3checkpoint×2策略，仅生成首token。48个checkpoint×QA的策略对照全部首答案一致，两个策略的10道困难no均0/10；处理器虽在48个观测中都改过部分logits，但没有改变argmax。worker18.309秒、0训练更新。
4. **数据修复尚不能解释或解决全部负提升。** 将已有固定训练读出按视觉审核拆分后，可信4题的正确no首词alias NLL仍由1.1348升至1.3095；更明确的蜡烛/镜像两题也由0.8637升至0.9370。联合训练后的模型对可信困难no也更不支持。尚未隔离全训练的具体梯度来源或唯一病因，不能宣称数据清理后模型已经修好。

## 逐层代码与日志对应

|环节|真实实现|本轮判断|
|---|---|---|
|图片、问题、标签|`vlm_deep.common.encode`分别编码prefix/full，`answer_labels`要求精确prefix一致；训练trace记录实际token、图像SHA、首答位置|10道no的目标均为`[2152,151645,198]`，对应no、结束符、换行；没有发现把yes写成no或首答漏mask的新增错误|
|因果预测位置|`vlm_complementary_rank.loss.loss_from_logits`使用前一位置logit预测首target，显式检查suffix alignment|没有发现本轮causal shift错位；不重复已通过的旧全量数学审查|
|QA归一与累积|`vlm_attention_scope_v1.train`四个4QA微批各`qa_ce.sum()/16`，最后统一clip/AdamW；两组lambda_rank=0|没有定位重复除累积次数、每微批错误step或旧排序损失仍参与优化|
|LoRA实际更新|新K/O第一步B梯度非零、A梯度0符合B零初始化；最终候选K/O SHA改变，控制不变|新增容量已参与训练；不能用“模块没更新”解释负结果|
|位置编码/缓存|本地Qwen2.5-VL在无past时重新计算RoPE；训练use_cache=False，生成prefill cache_position=0|未定位跨样本复用过期RoPE的代码问题；不是所有框架均无bug的证明|
|困难题轨迹|10道no在trace中均有曝光，多数一次；如单猫镜像update19、蜡烛update100，两者目标均no且当时margin仍偏yes|不是完全没有训练这批题；次数不能直接替代梯度份额或因果解释|
|生成配置|`vlm_probe.evaluate.evaluate_batch`传do_sample=False但未显式传repetition_penalty；底座generation_config为1.05|确认了隐式配置与锁缺失；真实对照表明它不是当前10道no失败的直接原因|

普通CE的实际目标是：

```text
L_update = (Σ8个Text的逐QA CE + Σ8个人工题的逐QA CE) / 16
人工题CE = (CE(no或yes) + CE(im_end) + CE(换行)) / 3
```

这是合法的训练目标，并不是数学实现bug。但总体平均CE下降并不保证每个语义子群改善。旧首答梯度诊断也没有支持“格式token主导”这个单一解释；本轮不通过直接放大首答或延长训练来碰运气。

## 新训练监督审核

|训练qid末尾|问题|此次判定及原图依据|
|---|---|---|
|238618005|Is the gate open?|不确定：牛头与栏杆近景，独立门体的开闭状态不清楚|
|483069000|Is the dog wearing a collar?|不确定：颈部有红色可见物且部分遮挡，不能据该图明确支持无项圈|
|373254015|Is the man wearing a white hat?|支持原no：可见成年男性为深色帽；前景儿童帽的上表面也为深色|
|310783011|Is she wearing a skirt?|不确定：下半身被前景大块粉色展示物遮挡|
|196971004|Are there two cats?|支持原no：一只猫及其镜像，不是两只独立猫|
|487941010|Is the man on top of a board?|支持原no：脚下为两块独立滑水板，而非单块board|
|89939000|Are there ads behind the skaters?|不确定：背景大图案/涂鸦是否构成广告缺乏可读依据|
|253166001|Is there an opponent?|不确定：画面内对手与比赛存在对手是不同范围，问题未限定|
|444344000|Is the candle striped?|支持原no：红色烛杯未见成组条纹|
|486361008|Is the cat in front of a window?|不确定：可见窗帘、电视和家具，窗户相对猫的位置缺乏明确依据|

审核人为助手，非独立专家重新标注。10题按历史训练错误选择，不能估计总体噪声率。两张蜡烛/镜像图已有前轮原图审查记录，其余八张此次实际打开原图。没有将筛选/确认错题回填训练；没有删改任何评测分母。

新视图仍有1140题仅为官方高共识、34题获已有或新增可见支持，**不是已经全量干净的训练集**。新审核证据与问题SHA、图片SHA、原target、pair绑定；未知覆盖旧支持时整对否决，保留数据来源。

## 可信负例分解：修数据仍不足够

正确no首词的alias NLL，越低越好。全部来自前轮已保存的固定prompt-only读出，0额外模型前向。

|训练子集|初始|Q/V|Q/K/V/O|
|---|---:|---:|---:|
|视觉依据支持的4题|1.1348|1.3010|1.3095|
|不确定的6题|1.5626|1.9238|1.9472|
|其中更明确的蜡烛/镜像2题|0.8637|0.8992|0.9370|

两题子集与4题子集重叠，不是独立重复；它们是事后训练诊断，不是泛化指标。可信负例也退化，因此“只是标签噪声”“只是推理解码”均不符合全部证据。仍需隔离真实Adam更新对可靠困难正负例与保持题的作用，才可选择下一算法变化；不能把局部原始SGD梯度直接当作历史Adam的全程因果。

## 已落地修复与使用入口

- 数据：`src/vlm_attention_scope_repair_v1/admission.py`、`tools/repair_attention_scope_training_v1.py --prepare`。新视图为`outputs/vlm_attention_scope_repair_v1/TRAINING_MANIFEST.json`，未来新训练采用它；旧运行不恢复、不改输入。
- 评测：`src/vlm_attention_scope_repair_v1/generation.py:evaluate`。复用已有编码与评分，新增完整生成策略锁，禁止给旧缓存追挂新策略。新研究初始与候选必须一同使用该入口；此次只诊断首答，**没有将全部TextVQA/长答案重评为新结果**。
- 记录：`VISUAL_TRAIN_REVIEW.json`、`PAIR_ADMISSION.json`、`RELIABLE_NEGATIVE_BREAKDOWN.json`、`DECODE_DIAGNOSTIC.json`和`COMPLETION.json`。
- 验证：7项针对本次修改的CPU检查全部通过（0.46秒）；GPU诊断自然完成、三组LoRA SHA前后相同、父实验source/package锁复核通过、已有任务未中止。首次测试临时目录父路径缺失的setup错误记录保留，补齐目录后全部通过；不是模型训练故障。

这次修复的是已确认的监督准入不足与解码可复现性缺口。**尚无新训练、没有新提分、没有理由恢复旧容量分支或立即启动长训练。** 后续立项应针对可信困难负例的实际更新机制，并以匹配强CE和独立生成的净纠错、yes/no及Text保持验收；不重复旧难例过采样、不盲加epoch、不以换DPO/GRPO名称代替机制。

[原图逐题学习页](LOCAL_ARTIFACTS.md) · [上一轮完整失败分析](VLM_ATTENTION_SCOPE_V1_FAILURE_ANALYSIS.md)
