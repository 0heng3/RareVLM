# 保留q/v起点的语言MLP低秩适配

2026-10-22事前登记。本轮检验受限的可更新子空间是否妨碍当前纠错。上一覆盖回放候选seen93.613281/transfer87.612863%，对匹配控制迁移−.055659pp/95%[-.499080,+.377248]，训练满分24/128；联合失败。普通CE训练满分56，其中拼写16/27、目标选择29/42、数字3/21、读钟0/20。加强回放26，覆盖1024为24，数字0/21、读钟仍0/20。视觉merger旧实验数字5/21/总64，但读钟0/20且迁移退化；不据此断言唯一容量、数据或框架病因。

## 唯一候选与控制

原初始uniform SFT42全部144个FP32 q/v因子原值保留（1843200参数、原哈希5d8e2e21cc98686a4adf755b23e46c8f3512c7ff27d196df1b53f07cd934dd7b）。唯一候选在36层语言MLP的gate/up/down各加rank2/alpha4的LoRA，A按独立CPU seed42 Kaiming-uniform、B严格零；原q/v仍rank8/alpha16/scale2并继续训练。新108矩阵/216因子/2820096参数，总180矩阵/360因子/4663296参数。目标使用完整语言层路径，不误匹配视觉层；视觉、merger、embedding、lm_head与全部原基座冻结，读出保持独立FP32冻结头。

MLP(h)=Wdown[SiLU(Wgate h)⊙Wup h]，隐藏2048、中间11008。此为现有LoRA的适配范围/容量研究，增加参数及计算，不称等参数位置因果消融、原创结构或已经修复读钟。参考[PEFT v0.15配置文档](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora)的target_modules/rank_pattern/alpha_pattern/零B语义，本地版本0.15.2由真实注入验证；不采用跳转到Qwen3的新仓库默认配方替代本地Qwen2.5契约。

匹配强控制为既有`vlm_reference_replay_v1/runs/plain`（expanded240普通CE Final64），原始adapter、data/order/labels/精度/生成与本轮控制条件完全相同，直接复用预测，不当新独立重复。旧加强回放和最新unique1024是额外强参考，数据/目标变化已声明，不能用较弱q/v控制独自挽救成功。旧视觉merger数据的回放多词比例不同，只作背景，不据其与本轮之差归因纯适配位置。

## 数据、损失与预算固定

只读复用reference_replay的DATA/CPU_INPUTS/SCHEDULE与FIRST_UPDATE_REFERENCE：128逐图支持困难题各4次+512已会回放各1次、640独立QA、1024曝光、64更新、micro4×4/effective16，每更新8hard+8replay。**普通CE** `L=(Σ8hard CE+Σ8replay CE)/16`，无KL、β加强、偏好或额外生成监督。每QA正文+真实EOS token平均，EOS后换行不训练。LR2e−4/warmup8/linear/AdamW wd0/clip1/dropout0/seed42/Final64，不从上轮失败Final继续；0新QA/重编码/教师生成。

CPU仅核本次新增scope：真实Qwen meta架构+PEFT混合rank注入，360实际键/shape/FP32值与180目标/scale核对；原144逐值保留，真实PEFT保存及通用FP32恢复exact。已有加载/mask/评分/断点功能复用，不新建重复审核流程。新GPU首正式更新在optimizer前核原16input/labels exact、逐QA CE差≤1e−5，108个新增B梯度有限非零，初始A梯度全0为正常；未过即停止。通过后第一步就是正式更新，不做额外丢弃工程步。后续记录q/v、MLP A/B梯度范数与源CE/token/时间。

完整断点存360个实际FP32参数、Adam/RNG/输入锁/日志偏移与UUID，通用恢复不假定144/148个因子。新矩阵数与保存配置一并绑定源码和初始权重，不忽略rank_pattern导致的shape。源码/协议/数据在首个新GPU前冻结。

## 评测、成功与停止

仅一个新候选训练64正式步，评测2661生成=训练context468（仅128评分）+seen512+保持64+transfer1617；所有初始、普通q/v CE、加强回放、最新覆盖1024预测同原固定输入/分组/FP32头/greedy32复用，0新baseline调用。末批1合法。seen512曾用于原SFT；transfer1617/1024图与原SFT train图像/像素隔离，但已多轮内部观察，不称盲测或预训练未见。新640训练与held像素隔离证明原样复用。监督target频次只作曝光代理。

候选在seen与transfer两个域，对initial/plain/hard_replay/unique1024**分别**整体≥+.5pp且6000次配对图像簇95%下界>0；rare1–5对initial≥+.5pp，其他频次与单多词对初始/强参考≥−.5pp，rare对参考≥−.5pp；保持64对各参考≥−.5pp，训练128满分净增32，seen原243满分多词保持≥−.5pp。训练8错误分型完整报告，包括数字和读钟，但这些只说明训练纠错，不替代独立主指标。

固定Final64，不扫rank/LR/epoch或根据训练案例挑checkpoint。单seed内部探索，条件图像簇区间不含训练随机性或全历史多重选择；失败不自动扩步、种子、正式benchmark、简历。通过后仍需独立确认与种子0/200另登记，不提前承诺成功率。

共享空闲队列零compute/≥40GiB/<10%/3×15秒/UUID锁。GPU0当前有他人compute，不终止/不操作宿主PID；用空闲GPU完成唯一新训练与评测。训练累计1800秒/评测累计2400秒/allocated65GiB，异常保存完整步并退出，ETA依实测，不把任务存活当能力提升。CV/PDF/科研/Agent/CI不改、不push。

学习页`learning/语言MLP适配_容量与迁移.html`：8张在新训练前固定的真实训练图（读钟/数字/拼写/目标选择各2）、MLP与LoRA公式/矩阵形状、真实代码和模型回答。训练病例不当泛化证据。
