> 已被用户2026-10-16最新要求替代为历史分析。默认主线改为直接改善共享模型的训练研究，见[模型能力后训练路线](MODEL_CAPABILITY_POSTTRAINING_NEXT.md)。此文的大规模零更新诊断及多adapter工程备选不再默认执行。

# 已有结果之后的下一步行动分析

状态：**建议方案，未登记为新实验、未实现或启动GPU**。2026-10-16。原完整三seed→external4372→POPE9000→独立验收目标保留，本轮分析不改原门槛、候选负结果、源码或简历。

## 决策

先完成一次固定Final的、零optimizer/零生成的保持机制诊断，再据证据决定是否注册新的共享adapter训练。现有GTC E+KL负候选不扩种；不继续扫gamma/lambda/alpha、不用dev失分题补训练，不新增DPO/GRPO来覆盖关键词。

短期求职继续使用已核验的CLIP/检索/SFT成果和代码讲述，不等待这条研究解决才投递。多adapter任务分支可作独立工程备选，不能代替共享adapter能力保持结论。

## 历史结果（同一内部集，探索性；不是完整单因素实验）

| 方法 | Object变化pp | Text整体pp | rare pp | frequent pp | 原门槛 |
|---|---:|---:|---:|---:|---:|
|固定教师答案CE回放|+2.047|-0.049|-0.117|-0.240|6/8|
|生成prefix完整KL|+2.047|-0.105|-0.322|-0.240|6/8|
|LoRA因子坐标收缩0.5|+0.877|-0.130|-0.322|-0.240|4/8|
|视觉merger LoRA+KL|+0.000|-0.025|+0.000|-0.480|4/8|
|GT-CE+KL|+1.462|-0.087|+0.175|-0.096|5/8|

固定教师CE回放是Object收益与Text整体保持较好的历史参照，但仍未过rare/frequent点保护；不能把6/8当成功。GTCE相对其fresh GT-KL控制Object少对1题、Text+.167pp的95%区间跨0，未建立新增GTCE的优势。视觉merger策略同时改变位置/rank/初始化，总adapter参数亦增加，不能据单次负结果否定全部视觉适配，也没有证据要求立即换底座。

## 当前能支持的判断

两Final和3918新评分已独立验收，0差异，88科学文件未变，已完成的全量审查直接复用。候选Object5题纠正在4图、Text5题升8题降。13个变分Text题保存生成ID最长13，均未达到max_new_tokens32；延长生成上限不是现有证据支持的修复。5个增分题中3个回答更短，8个失分题中也有2个回答更短，不能把统一缩短回答当答案。

罕见切片净收益涉及wednesday、portland等粒度变化；同时$1.00→$1、spring→spring water丢分。24111的blue→white有原图支持，但单例不证明通用视觉增强。评分表达偏差与具体识别错误共存，不能归为一个已经确证的机制。

训练日志mean KL=.000264589是不同更新/输入上的在线平均，不能当Final训练集KL，更不能代表heldout保持。教学反例：q=[.501,.499]、p=[.499,.501]，KL(q||p)≈0.000008000，贪心top1已翻转；这是数学例子，不是本轮模型测量。

## 优先行动1：固定Final的行为保持诊断

**问题**：保持约束是否只在回放输入成立？首答案/EOS位置是否比整体均值更敏感？输出变化是否集中在小top1/top2间隔？现有KL梯度诊断不能证明梯度冲突是失败主因，本轮不重复旧KL梯度实验。

模型固定为初始SFT42、stage35 fresh GT-KL Final、stage35 GT-CE+KL Final。它们已有同QA、同顺序、同初始的受控训练证据；不根据新诊断挑checkpoint。

数据固定为已使用5792回放训练题和完整1617内部dev，均使用**初始SFT已保存的greedy assistant prefix**。训练题从已验收轻量target index取ID，不读大型teacher bank；dev使用已保存初始生成ID。两cohort都按相同q0-prefix定义比较，不把GT-prefix训练loss与generated-prefix诊断混成一个数。此处训练cohort表示“被回放过的问题”，不等同新的原GT训练目标。

三模型在相同actual image/question/prefix上做teacher-forced forward，0新生成、0optimizer；预计7409×3=22227 model×QA forward。只保存逐QA/role的KL、熵、top1/top2及margin、EOS概率和实际first-divergence信息，不保存全词表概率银行。首token/共同prefix的结论与真实生成对应；初始prefix分歧以后的统计只作条件分布诊断，不能冒充actor实际rollout。dev答案不参与模型输入或采样/调权规则。

输出包括：回放cohort与dev的Final分布漂移；first-content/later/EOS/template分项；监督频次切片；按图像整簇的描述汇总；实际回答升降与低margin的关系。只证明关联，不据此宣称遗忘因果或新提分。使用当时空闲A100共享队列、不动已有任务；三BF16模型/卡的实际显存必须预检，不能静默降batch。预计GPU执行约1–2小时，CPU编码、实现/准入和排队另计，以首批实测修正。

执行前必须新独立协议/config/source/输入权重锁、CPU契约和真实预检；本分析文件不是完整实验预注册。

## 优先行动2：只依据诊断注册一个训练假设

如果漂移集中在首答案位置，且训练cohort存在可使用的同类信号，才评估固定位置权重或训练端低margin保持对照；权重规则只由训练数据/参考模型确定，不根据dev题逐题加权。若诊断不支持，就不执行该训练假设。小margin与失分相关本身不保证加权KL能改善泛化。

新候选仍保留原8门槛、固定Final、三seed及external/POPE完整范围。原失败候选不通过新增seed救统计显著性，不改95%替代98.75%、不改官方答案评分。

## 独立工程备选：任务明确的多adapter保持

可以复用现有权重，在用户显式任务模式下，物体存在性问答选stage31 teacher-CE Final，文字问答选原SFT；未知模式明确fallback。不根据benchmark名字、标签、参考答案、单题对错路由；不添加用dev训练的selector。

PEFT本地接口已核load_adapter/set_adapter可加载并选一个活动adapter。实现须显式冻结参数：set_adapter会把被选adapter设为trainable；load_adapter默认dtype提升也须核对，保持原native BF16评测合同。官方API依据：https://huggingface.co/docs/peft/v0.15.0/en/package_reference/peft_model 。

第一次验证可0新训练：原SFT单adapter、对象CE单adapter、显式任务模式双adapter系统三组独立真实执行完整342+1617，不拼接历史预测当新结果；每组都实际生成。检查A→B→A切换的权重/输出、batch内模式、缓存/RNG/精度、真实切换成本与显存。总adapter元素1843200→3686400，单请求只启用一份；不称等总参数/零成本。筛选生成约5877model×QA，纯评测预算约30–50分钟（按历史速度估计，另计准备/排队）。

这能验证**已知任务模式下的系统能力保持**，不能证明一个始终激活的共享adapter更强，也不能包装成DPO/RL、自由任务路由、或用它替代原共享训练的论点。若以后沿此路线推进，也须新协议、独立完整门槛与条件三seed/外部确认。

## 当前执行边界与学习产物

本轮只做已有结果比较和下一步设计，无新GPU/训练/预测，原正式CV保持。下一次执行优先行动1；工程备选按独立目标排期。学习产物沿用“公式—实际张量/代码—原图—所有正负结果”，诊断未执行的数据不写成结果。

[已完成能力核验](VLM_GT_ANCHORED_KL_V1_RESULTS.md) · [原图与已有真实预测](LOCAL_ARTIFACTS.md)
