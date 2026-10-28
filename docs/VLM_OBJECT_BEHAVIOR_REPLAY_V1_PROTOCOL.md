# 固定初始模型回答回放 v1：事前协议

日期：2026-10-16。状态：准备中；CPU合同、来源锁与共享GPU准入通过前不执行教师生成或训练。此协议延续“真实提升后才停止”的完整目标；准备、工程通过、内部单点均不算目标完成。

## 1. 问题与已有证据

上轮 [回放覆盖研究](VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md) 已 completed_negative：wide32物体准确率95.614035→97.660819%，差+2.046784pp、单侧98.75%下界+.552486；Text87.111936→86.963513%，差−.148423pp、保持下界−.519966，rare/frequent点差−.321637/−.479616。5/8内部条件通过，保持与两个切片保护失败；不扩该候选的种子/外部/POPE。

完整新候选1959题独立计分零差异，Text有21条原回答变化、11条分数变化（4升7降）。rare两下降题为15990（$1.00→$1，1→.3）与29766（spring→spring water，1→.6）；前者原图标价$1.00，金额等价但共识分敏感。frequent两题0→channel 10、2→5有真实目标内容错误。不能将全部差分称为视觉遗忘或全部称为评分问题；原评分不改。旧KL八病例中4首token、3结束位置、1后续内容分歧，也没有隔离前缀或覆盖的因果。

新假设：将固定初始模型在训练prompt上的实际greedy回答作为明确回放目标，可能比重新拟合原监督答案更好保留其回答内容、表达与结束位置。它可能锐化已有偏好或错误，不能保证保持，且不能仅凭此解决heldout首token错误。

## 2. 候选、控制与目标

命名空间：src/vlm_object_behavior_replay/；配置configs/vlm_object_behavior_replay_v1.json；输出为rarevlm_pilot/outputs/vlm_object_behavior_replay_v1。

唯一主候选 **teacher_ce32**：用固定初始SFT42的真实生成token作回放hard CE。探索控制 **gt_ce32**：同一训练QA流的原GT hard CE，仅pilot42、不可替补或晋级。旧wide32 GT-prefix KL作为已完成历史参照，不新增其训练、不冒充独立重复。

两臂都使用逐题平均CE，再对各来源独立平均：

\[
L=\frac{1}{N_o}\sum_i CE_i^{object}
  +\frac{1}{N_r}\sum_j CE_j^{replay},\qquad
CE_j^{teacher}=-\frac{1}{m_j}\sum_t\log p_\theta(\hat y_{j,t}\mid x_j,\hat y_{j,<t}).
\]

教师为共同初始模型q0，生成后固定；hard target为其greedy轨迹，不是q0全词表软分布。初始化hard CE和梯度通常非零；lambda1不等同此前KL的实际强度。两臂只改变回放答案来源，但随之改变内容、长度、teacher-forcing前缀及计算，不能归因于纯前缀或宣称等FLOPs。均不称RLHF、DPO、GRPO、当前student on-policy GKD、模型压缩或已实现无损。

方法依据为 [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139.pdf) 中对教师解码结果的CE训练；原论文使用翻译/beam，本轮为同规模初始模型的greedy行为回放，收益须另验证。[TRL GKD文档](https://huggingface.co/docs/trl/gkd_trainer)和已有固定源码只用于区分序列来源与散度目标；seq_kd=True在已有固定实现中仍走分布蒸馏，不能冒充本轮hard CE。

## 3. 固定数据、模型与预算

- 沿用已准入的object v2：1211训练图/2888QA、内部143图/342QA及256QA均衡子集；外部1791图/4372QA，完整POPE9000QA。旧准入与精确边界复用，不重审旧64图，不把教师回答当事实ground truth。
- Text回放只从原train_all32985QA抽取，沿用wide32的固定uniform SHA排序/每步32题；pilot42为同一5792题，不按teacher正确性、GT分歧或dev表现筛选/替换。0/200仅在pilot通过后准备各自对应流。
- 只向教师提供原训练图像、问题和同一短答suffix；原GT只服务GT控制和来源记录，不进教师生成输入。dev/外部/POPE答案不用于生成目标、训练或选checkpoint。
- 两臂同一初始uniform SFT42、原Qwen BF16基座与视觉冻结，BF16-round后升FP32的语言q/v rank8 LoRA共1843200参数；初始master SHA为6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7。
- 181正式更新/1epoch固定Final；每步object16（micro4×4，末尾8）及replay32（micro4×8），两源独立均值lambda1；lr5e−6、warmup.1、wd0、clip1、dropout0、非重入gradient checkpoint、SDPA、TF32关闭。
- 原生单整图、visual上限1024、seq上限2048、无输入截断，原prompt/OCR/生成计分规则保持。teacher rollout及正式评测batch4、greedy、max_new_tokens32。

本地原模型generation_config实含repetition_penalty=1.05，既有原始评测路径继承此值；本轮教师同样使用1.05并显式记录native文件SHA与native配置加实际overrides后的effective generation_config。greedy指该完整处理策略下的实际选择，不误写成未处理原始logit的argmax；非采样时温度/top_p等字段仍保留来源但不据其声称采样。此显式记录不改变原生成行为或评测策略。

执行前源码核对补充：原始评测的load_path(original_bf16)将所有浮点参数（包括LoRA）转BF16，并在inference_mode下关闭autocast。教师直接复用该评测加载路径，所有参数冻结，实际LoRA推理dtype为BF16；不使用升FP32后的train loader作无autocast教师。6a44…身份SHA按这些BF16 LoRA的FP32表示计算，不冒充教师native FP32；另记native dtype/hash，生成前后都核对。actor训练仍使用真正FP32 master及BF16 autocast，与原固定训练路径一致。loader实际还把无采样时的温度/top_p/top_k元数据设为1/1/50，因此effective配置须在loader之后记录，native文件保持原字节。这个执行前修正未产生任何GPU调用或改变历史结果。

## 4. 教师生成与真实token编码

每个获准seed先冻结完整5792题prompt流及microbatch，再分两个固定contiguous shard，各2896QA/724批。两role经共享队列独立准入；先来可用GPU先运行，无需等所有GPU释放。固定教师权重、包版本、实际producer UUID、prompt/padding/像素/grid与生成参数逐批绑定。不同seed的生成批次另锁，不按qid跨不同padding/批次静默复用回答。

每个teacher shard先做固定8题有限性/容量生成预检，另记工程生成调用，不静默计入或复用为完整5792输出。成功后从同一固定初始教师独立执行该shard全部原计划；预检与正式生成权重SHA均核对，按完整批次恢复。两shard预检计划合计16次额外QA生成，真实重试另计。它们没有optimizer，不与四次训练工程step混计。

从generate返回序列的共同padded prompt宽度之后截取实际continuation；按每题首个真实配置EOS结束，保存原始suffix（含返回padding）及切分依据。训练直接使用实际token IDs，禁止decode后再tokenize作为训练输入。

- 正常chat EOS：保留实际EOS，再显式附加由本地chat模板实测的post-EOS suffix（通常换行），另标为模板token，不伪称教师生成。
- 其他已配置终止token：保留实际终止token，无对应chat后缀则不补。
- capped且无EOS：保留实际最多32个token，不补EOS/模板后缀、不过滤/重采样。
- 首token即EOS的空内容回答保留并单列；无实际新token、非法词表ID、prompt不一致、截断/损坏或来源不符则拒绝，不填造回答。

所有teacher targets是行为参考，不是新人工标注。新generated-ID encoder独立实现：prompt与原图字段复用严格绑定的实际CPU编码，labels只覆盖真实生成内容/终止符及明确模板后缀；左padding为−100。GT控制继续调用冻结原编码，并核对与旧wide32同QA实际合同。全流逐题核验，实际训练输入仍在每次前向前核对。

## 5. 工程、恢复与诊断

先CPU合同/准备锁，再教师生成；两shard完整后才合并，生成序列顺序严格复原原计划，形成独立TEACHER_REPLIES_LOCK。随后构造全部新teacher-supervised合同并冻结TRAIN_INPUT_LOCK，才允许独立训练预检与正式训练。STUDY_INPUT_LOCK预先绑定科学源/协议/config/CPU receipt/原始数据、模型及历史证据，后续动态产物使用各自来源锁，不改旧锁。

预检验证CE/分源梯度有限、真实临时更新有效，精确恢复LoRA/RNG/registered buffers/rope等运行状态并丢弃工程optimizer。不得要求初始化CE或梯度为0。两独立与两正式内置预检计划4次丢弃工程step，真实历史（含重试/复制）权威计数；正式每臂181不混计工程步。batch或65GiB预检硬门槛不自动下调或放宽。

沿用每25步及Final原子保存master/adapter、Adam/RNG/游标和完整批次合同，按同role/UUID精确恢复。停止/换容器不因观察窗口到期重启模型。新研究不构建/读取全词表teacher概率bank，教师生成与actor训练/评测成本分别记录。

使用同一已有训练前向的token CE，分别累计首内容token、后续内容、真实EOS和模板后缀的位置统计；空内容首EOS只归EOS，位置类别互斥且完整。只作诊断，不改CE权重或据其调lambda，不增加模型调用。在线均值不能当收敛、遗忘或heldout保持证明。

## 6. 原完整成功条件不变

仅teacher_ce32可晋级，gt_ce32只作探索对照。先seed42完整object342/Text1617，按原8项筛选：物体差≥+1pp/单侧98.75%下界>0，Text .5pp非劣下界>−.5，物体recall点不降/specificity≥−.5pp、均衡子集点差>0、Text rare/frequent点差均≥0。任何失败则不追加此候选种子/外部/POPE，不改门槛。

pilot通过才运行主候选0/200；同一初始模型的三训练/回放顺序及对应固定教师流，不称三独立预训练模型。三seed均值仍过原内部条件，各seed物体差正、Text点差≥−.5pp。随后三seed全4372外部：均值accuracy≥+.5pp/下界>0、逐seed正、F1/recall点不降/specificity≥−.5pp、均衡点差>0。通过后才三seed完整POPE9000：均值accuracy≥+.5pp/下界>0、逐seed正、F1/recall点不降/specificity≥−.5pp、三category点不降，并保留全部Text保护。

四主端点alpha=.05/4、单侧98.75%、6000次按图像整簇paired bootstrap/seed20261015；先跨seed平均后重采样，样本SD另列；95%/切片/控制对照仅探索，不声称全历史FWER、切片总体退化、等价或公开集盲测。初始完整object/Text/external/POPE基线只读复用，预训练/近重复重叠未知。

最后必须有root独立完整输入/权重/评分/来源/病例/成本审查。内部通过、低loss、生成完成或新adapter均不完成用户目标。

## 7. 资源与成果边界

全部新文件/环境/缓存/log在private。共享research_runtime.queue按至少40GiB空闲、util<10%、零compute、3×15秒快照和UUID锁准入；不终止或抢占其他任务。正式teacher/GT两臂尽量并行，已有任务优先。只读观察/审查进程与GPU任务分开，前者超时不代表后者终止。

学习材料连结真实原图、原理、数值、源码、生成/监督token及完整结果，按事前固定首8训练QA展示teacher数据；失败病例固定排序展示并注明选择范围。科学结果和计划分开。正式RESUME/PDF、论文、Agent保持；无CI/账单/自动push，助手完成不等于用户已独立掌握或贡献。

## 8. 实现交接合同（准备期）

data.py提供load_rows(config)→(objects,replays)、build_plan(config,seed)→完整schedule/indices计划、prepare_generation_inputs(config,seed)→CPU prompt manifest、finalize_teacher_replies(config,seed)→完整teacher manifest、load_teacher_replies(config,seed)→qid映射；rollout.py提供共享队列worker使用的run_teacher_shard及CLI。具体数据schema与路径由实现显式记录，调用者先核manifest，不猜缺省字段。

encode_training_sample(processor,row,config,*,source,condition,teacher_replies=None)返回原标准tensor字段及answer_roles；collate_training(samples,pad_token_id)按原左padding处理并同步角色。角色0=prompt/pad、1=first_content、2=later_content、3=actual_eos、4=template_suffix，互斥覆盖所有labels≠−100。object/GT保持原encode的实际IDs/labels；teacher分支直接拼接实际token。训练forward显式去掉answer_roles后才交给模型，角色只供无新增forward的diagnostic。

统一路径为prepared/seed{seed}/PLAN.json与PROMPT_INPUTS.jsonl、teacher_replies/seed{seed}/shard{0,1}/各完整批次及完成收据、teacher_replies/seed{seed}/MANIFEST.json；正式训练仍为runs/{condition}/seed{seed}/train，评测子目录沿用object/text/external/pope。新model/data/teacher产物的动态SHA进入各阶段锁，不塞回已冻结的STUDY_INPUT_LOCK。实现合同如需修订仅在任何相应科学执行前明确更新并重新锁定，不能据结果改变方法或门槛。
