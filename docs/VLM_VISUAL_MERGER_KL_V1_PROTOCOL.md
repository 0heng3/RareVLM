# Visual Merger KL v1：同可训练预算的适配策略研究

## 证据与唯一候选

前两项完整结果：[生成前缀KL](VLM_GENERATED_PREFIX_KL_V1_RESULTS.md)通过6/8；[固定LoRA坐标收缩](VLM_LORA_COORDINATE_SHRINK_V1_RESULTS.md)通过4/8。前者Object +2.046784 pp/Text −0.105133 pp，后者+0.877193/−0.129870 pp，rare/frequent保护均未过。后者source/engineer问题已经版本化恢复，1959条新预测独立评分零差异，完整负结果由root验收；失败后不继续alpha网格。

新唯一候选`merger_kl32`保留共同SFT42语言适配器，冻结其144个语言q/v LoRA张量；只训练视觉merger两层线性的4个新LoRA因子，监督/回放仍为原object CE + generated-prefix forward KL。检验整套适配策略能否改善物体识别并保持Text。已有结果不能证明语言更新是退化原因，亦不能证明视觉接口更优；本候选不是基于验证题的补丁。

## 实际结构与预算

本地实际Transformers4.51.1、PEFT0.15.2；checkpoint配置中的旧库版本不替代运行版本。Qwen2.5-VL-3B视觉hidden=1280，spatial_merge=2，out_hidden=2048；merger为RMSNorm(1280)→四patch拼接5120→Linear(5120,5120)→GELU→Linear(5120,2048)。输入预算、processor及visual token规则不变。

| 新可训练位置 | A形状 | B形状 | rank | lora_alpha | 元素数 |
| --- | --- | --- | ---: | ---: | ---: |
| visual.merger.mlp.0 | 96×5120 | 5120×96 | 96 | 192 | 983040 |
| visual.merger.mlp.2 | 120×5120 | 2048×120 | 120 | 240 | 860160 |
| 合计 | | | | scaling均为2 | 1843200 |

方程10240r0+7168r2=1843200，选择两rank均为8倍数的96/120，不搜索rank或按dev择优。RMSNorm、base线性/bias、原视觉encoder、语言base及旧语言LoRA全部冻结。优化器只持有上述4个FP32 master。

总adapter为148张量/3686400元素，其中144张量/1843200元素是冻结的已有语言SFT。因此只匹配可训练元素预算，不称总存储/推理参数预算不变、等FLOP或纯适配位置因果消融；位置、rank分配和残差初始化同时变化。完整FP32 adapter约14.06MiB，比原方案增加约7.03MiB，实际推理/训练成本单列测量。

## 组合adapter与共同初始化

从相同原始BF16 base一次构建始终激活的同一`default`组合adapter，目标为原语言q/v及两个精确merger线性；语言保持r8/alpha16，两个merger由严格rank_pattern/alpha_pattern覆盖。先按实际模块名核对目标 inventory，拒绝误匹配/额外视觉attention模块。不得用`set_adapter('vision')`关闭已有语言SFT，也不得二次覆盖已有default。

原144语言LoRA逐项按真实raw initial→BF16 round→FP32 canonical加载，继续绑定`6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7`。复制只接受预期旧144与新4的完整键集，不能用不检查missing/unexpected的宽松加载。最后freeze全部参数，再只unfreeze新4；保存及实际加载均必须保持旧语言适配器启用且值不变。

新视觉A用CPU独立torch.Generator固定seed42，FP32 Kaiming-uniform（a=sqrt(5)，输入fan-in5120），B为FP32全零。prepare生成独立共同初始化safetensors+manifest，记录4形状/名字/原始和tensorSHA、初始化代码/版本。所有order/replay seeds共用这份A/B；训练A、B均可学，并非永久冻结A。构造模型时的随机分配不能替代该共同初始化，加载后精确核对。初始B=0时A首步梯度为零是链式求导结果，不据此断言空跑。

新增merger `.base_layer`包装会改变named parameter身份。必须显式一一映射原base名字并核对对应冻结值、语言144 canonical、视觉4 initial；只允许声明的两个merger路径重映射。分别记录actor原始named-hash、原值映射证据、frozen语言和视觉master；不能机械声称actor raw frozen hash一定等于旧4748。reference使用原SFT-only独立冻结模型，不与actor共用storage，不加入新视觉LoRA，避免两模型同构建错误却得到零KL。

## 数据、损失和训练数值

seed42复用已验收stage31的5792条真实teacher-generated IDs/前缀/roles与181计划、全部8680输入合同；0新teacher生成、0重新分词或完整输入重编码。旧条件名teacher_ce32只表示已保存目标布局，不路由新损失或伪造新模型身份。控制为已完成的language teacher_kl32，严格历史引用，不参与选择。旧初始Object/Text/external/POPE基线引用已验收来源，不重新生成。

共同初始化为原SFT42加上述零B视觉残差；order/replay axes为0/42/200，不是三个独立SFT/视觉初始。pilot只有新seed42；通过后才生成0/200实际prefix并各训练同一新策略，不调用旧负候选gate放行。

每seed固定Final@1epoch，181次更新、2888 object+5792 replay，object16/replay32/micro4，末object8按实际QA归一。训练lr5e-6、warmup0.1、wd0、clip1、dropout0；FP32新视觉master，BF16冻结base与autocast，非重入checkpoint；原模板、assistant/EOS/template mask、1024视觉token上限、2048序列限制不变、不截断。两项分别按QA及每QA有效assistant token均值，lambda1/T1。

回放为独立固定initial reference上的完整151936词表forward KL(initial||actor)，词表求和，不是均值，不用topk/probability bank；hardCE仅detach诊断，绝不反传。reference原语言LoRA同canonical FP32、冻结base/vision BF16、eval/no_grad/checkpoint disabled，全参数requires_grad=false且grad不存在，构造前后恢复actor RNG；resume时reference始终initial。5792次正式reference forward另计，已包含训练active秒数，不双算。

冻结decoder权重不能关闭其反传图：视觉merger仍需要经decoder反传。不得对整个语言前向加no_grad。记录每个新因子梯度、冻结语言/base无grad，确保真实梯度链。保存master/Adam4状态、边界/RNG/输入合同、完整148 adapter及冻结语言摘要，支持相同协议/UUID恢复。

## 必须实测的工程准入

在冻结源码、统一CPU及真实prepare通过后，只进行一个嵌入式可丢弃工程step：初始每QA max|KL|≤1e-5、回放参数grad norm≤object参数grad norm的1%，object有限正grad；actor/reference实际初始输入/IDs/roles一致。zero-B不代替实测初始parity。临时lr1e-4更新后KL及回放参数grad有限且为正，证明新视觉参数真实接入损失；不要求理论上应为零的初始A梯度非零。

之后精确恢复新4、旧语言144、冻结base值、RNG/buffers/rope，丢弃工程optimizer，才开始181正式更新。新actor和reference共同峰值allocated/reserved实测，65GiB cap不降低；不达标即工程拒绝，不换数值/目标或放宽阈值。物理执行版本、所有count/dtype/图形路径与理论预期分开报告。

eval沿原native BF16、SDPA、TF32关闭、batch4、贪心32token、同提示/processor/数据/scorer。新148 expected FP32/BF16/promoted摘要与GPU实际加载摘要分别记录并精确匹配；参考答案仅训练/生成后评分。baseline与candidate的参数dtype策略一致，inventory必须是声明的+4张量/+1843200元素，不能机械要求旧actual_parameter_dtypes整dict相等，也不能不验证增量。

身份只含稳定科学/来源字段；验证耗时等遥测单列，不把动态耗时嵌入模型身份整dict比较。推理重入必须验证稳定identity后复用首次raw INPUT_LOCK，保留其contentSHA与完整已完成prediction batch。来源relative/absolute表示只按显式规范化副本比较，原数据/锁不改。

## 完整成功范围

pilot完成342 Object+1617 Text全部1959正式model×QA，各8题额外评测预检分列。保留八门槛：Object accuracy≥+1 pp且98.75%单侧下界>0；Text整体98.75%下界>−0.5 pp；Object recall点不下降；specificity变化≥−0.5 pp；固定逐类yes/no均衡子集点增益>0；Text rare及frequent点值均不下降。

只有八项全过，才新0/200 teacher+输入+同181训练，三seed每QA先平均再图像簇bootstrap；平均过原八项，每seed Object正增益且Text变化≥−0.5 pp。随后每seed完整4372 external：平均accuracy≥+0.5 pp且下界>0、每seed正、F1/recall不降、specificity≥−0.5 pp、固定balanced点增益>0；balanced3438QA/1742图按QA汇总。再每seed完整9000 POPE（共享500图，三类别各3000），accuracy同上述门槛、每seed正、F1/recall不降、specificity保护、每类别点不下降并保留Text整体及rare/frequent保护。不以内部18类代替80类POPE。

四主端点为内部Object、Text整体NI、external、POPE accuracy，每端点alpha=.05/4=.0125；6000次配对图像整簇bootstrap、seed20261015、单侧98.75%。切片点保护不是非劣/等价；这些区间条件于固定checkpoint和历史已探索dev，不保证全历史FWER、独立随机SFT种子或预训练未见。

自动全通过也只能写`confirmation_passed_pending_independent_review`、`goal_complete=false`。root须独立检查全部初始化/实际梯度链/Final/完整输入/评分/病例/成本及全部原门槛后才可完成目标。pilot负则结束这个唯一候选，不把历史控制替补晋级，不按dev样例改数据/层/rank或checkpoint。

## 来源、资源与学习交付

新模块`src/vlm_visual_merger_kl/`、配置`configs/vlm_visual_merger_kl_v1.json`、输出`outputs/vlm_visual_merger_kl_v1/`独立；本协议先于实现与GPU。直接绑定stage31输入证明、stage32已验收language KL/initial控制、stage33完整负结果与root验收，不递归重审旧大teacher/概率bank/历史state；旧源码/配置/协议/结果不改。

完整0/200、external、POPE编排及来源/identity/预算/冻结/checkpoint/公式合同在首个GPU前实现并CPU验收。共享GPU按≥40GiB空闲、util<10%、无compute、3次15秒确认，role/UUID锁与恢复；不终止其它任务，不因观察超时重启。新方法成本与参数增长真实报告，四个矩阵不等于加速或省显存。

学习页连接真实图像、5120→2048形状、A/B数值/初始零梯度、冻结decoder的梯度链、真实源码和完整预测结果；教学图不冒充attention或模型输入。项目/README可同步真实结果，不改CV/PDF、CI或自动推送。

Primary参考：[Transformers4.51.1结构](https://github.com/huggingface/transformers/blob/v4.51.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)、[Qwen训练代码](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py)、[LLaVA projector训练](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py)、[PEFT0.15.2配置](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/tuners/lora/config.py)、[PEFT adapter切换](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/peft_model.py)。参考代码只支持方案可实现，不是本VLM收益证据。
