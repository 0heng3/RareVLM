# 固定教师KL回放覆盖预算对照 v1

2026-10-15，在本轮任何新GPU前向、训练或候选评测前登记。输出为独立`outputs/vlm_object_replay_coverage_v1`，源码`src/vlm_object_replay_coverage/`。旧KL结果和梯度诊断不改；用户“达到真实提升后再停止”的目标继续active，准备/工程/局部单点通过不算完成。

## 依据、假设与可解释范围

旧KL在内部物体342题上95.614035→97.953216%（+2.339181pp），但Text1617题87.111936→86.796537%，保持下界−.648809pp及rare/frequent点保护未过。三个预先固定训练点的KL/物体梯度范数比.030913/2.699215/.333181，cos−.014279/−.013333/−.068101；条件于实际Adam历史状态的三点KL一阶项均负。这不证明普遍冲突、KL始终太弱、增大lambda的必要性或覆盖不足因果。

本研究检验扩大实际unique回放是否改善未参与训练的Text保持，并保留物体收益。候选仅`wide32`；控制`repeat8x4`仅pilot42探索，不参与候选筛选、不能在wide32失败时替补晋级。

| 条件 | 每步32题的构造 | 181步实际回放曝光 | unique QA /32985 | 角色 |
| --- | --- | ---: | ---: | --- |
| wide32 | 原同seed SHA回放排序的前5792题，每步连续32题 | 5792 | 5792，17.559497% | 唯一候选 |
| repeat8x4 | 原KL8对应每步8题整块重复4次 | 5792 | 1448，4.389874% | pilot42控制 |

两臂回放每题先按assistant token均值，再除本步实际32题，lambda=1。重复4次不放大KL权重；无随机扰动的理想精确算术下其回放均值梯度等于原8题均值，实际FP32累加、优化轨迹仍实测，不声称逐位等价。wide32同时改变每批信息及梯度方差，不能把结果全部归因于全程unique覆盖。相同曝光、更新、参数不等于相同信息、教师计算或FLOPs。

## 固定训练与损失

`L = mean_QA mean_assistant CE(object) + mean_QA mean_assistant KL(p_initial || p_actor)`。

- 原完整qualified v2数据1211图/2888物体QA；回放池仍完整train_all32985题，仅按声明的前缀曝光，不使用内部dev/外部/POPE标签挑样。原61/64及外部63/64审查准入只读复用，保留原SHA与新study provenance，不重新宣称全量人类真值。
- 同一既有uniform SFT42初始化；原Qwen3B BF16基座/视觉冻结，语言q/v rank8 LoRA1843200参数，初始BF16舍入后FP32 master。BF16 autocast、dropout0、非重入gradient checkpointing、SDPA、关闭TF32。
- 物体micro4×acc4=16，最后一步实际8题；181更新/固定Final一epoch。每步另32回放题/micro4×8，独立按32均值；每臂正式2888+5792=8680 QA曝光。seed42物体顺序/实际输入与旧KL严格核对。wide32可追加顺序seeds0/200；控制只42。共同初始化不是三个独立预训练模型。
- AdamW lr5e−6、wd0、warmup.1后线性衰减、clip1；T=1、lambda=1、完整151936词表FP32 forward KL，先词表求和再每题token均值，不加hardCE反传。GT答案前缀与静态初始teacher，非on-policy、偏好优化或RL。hardCE只detach诊断。
- 原生整图视觉上限1024、长度上限2048、不截断；原prompt/processor/图像输入契约不改。checkpoint每25更新保存FP32 master/Adam/RNG/真实输入/teacher/source/UUID/游标，同role同UUID恢复，不按dev选epoch。

## 教师来源、成本与工程预检

wide32正式在optimizer之前为全部5792题建立自己的完整FP32词表teacher缓存；不得topK、量化或使用current_actor.detach。绑定模型/初始LoRA/源码/config/实际padded IDs/labels/masks/pixels/grid/producer UUID，manifest冻结后不变；磁盘估实际assistant数×151936×4并预留1GiB。

repeat8x4只读原`vlm_object_kl_v1/runs/object_kl/seed42/train/teacher_cache`完整1448题bank；每步8个micro映射到旧同一步micro0/1重复四次。逐批严格核对实际输入、目标、padding/assistant位置、像素/grid和原tensor/file SHA。旧producer GPU0绑定不改，新consumer UUID独立记录并做初始32题parity，absKL≤1e−5。新引用声明写新输出，不改旧bank/锁；正式新teacher forward=0，actor回放曝光5792/unique1448。不能称两臂教师成本相等，旧bank的历史生成成本另列。

两臂独立`--preflight`只用第一步32回放题作为`engineering_first_update_only`参考：候选fresh32，控制引用旧bank；此工程缓存不得冒充正式全量缓存。正式训练使用`formal_full_stream`完整来源，再各做内置预检。每次临时完整更新lr1e−4，检查初始KL≤1e−5、有限正梯度、更新后KL响应、teacher无梯度，精确恢复LoRA/RNG/registered buffers/rope等运行状态，丢弃工程Adam。两臂×独立/内置各1，共4次丢弃工程step，与每臂181正式更新分列。allocated或reserved超过65GiB或合同失败则拒绝，不降batch救结果。

预检/训练/缓存/生成的调用、QA/token、磁盘、allocated/reserved及纯调用/完整session成本分别记录。launcher在torch导入前固定OMP/MKL/OPENBLAS=2，避免旧CPU归约线程差异；不修改旧科学源码或放宽其精确检查。

## 评测、历史复用与晋级

Object内部342QA/143图及逐类yes/no均衡256QA/139图；Text内部1617QA/1024图。外部完整4372QA/1791原JPEG及均衡3438QA/1742图（各类数量不同，均衡accuracy为QA pooled非宏平均）。原POPE全9000QA/共享500图，非18类子集。已知file/pixel/COCO ID边界保留，近重复/预训练重叠未知；历史已暴露数据不称盲测。

初始Object/Text/POPE严格复用旧完整真实预测；初始外部严格引用旧KL完成目录`runs/initial/seed42/external`4372题，0新增baseline生成。核对原byte/model/processor/data/实际输入与评分后，在新输出写HISTORICAL_REFERENCE；不在旧目录写新结果。Text空category只在内存按旧无评分含义规范化，其余字段/标签不改。生成仍BF16/1024、batch4、greedy、max_new_tokens32、SDPA、TF32关闭、不截断，参考只训练/评分不入生成prompt。

四主端点仅针对wide32：内部物体accuracy、Text非劣、外部accuracy、POPE accuracy；alpha=.05/4。6000次配对整图bootstrap、seed20261015、单侧98.75%下界；95%/切片区间为点式探索，不控制全部历史研究FWER。三seed先求均值后按图簇重采样、样本SD另列，不把seed×QA当独立。pilot/control对比仅探索，不用作候选选择或成功证据替代。

1. 两臂独立工程通过后并行pilot42完整训练+Object342+Text1617，两臂均齐后分析。控制无后续seed或外部/POPE分支。候选与共同initial比较，不能在wide32失败后将控制替补或换checkpoint。
2. wide32必须全部满足原8项内部条件：物体accuracy≥+1pp且98.75%下界>0；Text下界>−.5pp；物体recall点差≥0、specificity≥−.5pp；Text rare/frequent点差各≥0；均衡内部accuracy点差>0。失败保存负结果，不扩该候选。
3. pilot通过才追加wide32顺序seeds0/200；三seed均值满足同内部条件、每seed物体accuracy点差>0且Text整体点差≥−.5pp。
4. 三seed内部通过后才全4372外部：均值accuracy≥+.5pp且98.75%下界>0，每seedaccuracy点差>0，平均F1/recall非降、specificity≥−.5pp、均衡accuracy点差>0。不得依据外部结果回调参数或选择checkpoint。
5. 外部通过才三seed全9000题POPE：均值accuracy≥+.5pp且98.75%下界>0，每seed点差>0，平均F1/recall非降、specificity≥−.5pp，random/popular/adversarial各类均值accuracy非降；保留Text全部保护。
6. root独立复核完整预测/计分/权重更新/输入与source锁/真实病例/成本及学习材料后才可完成真实提升目标；工程、诊断、内部单点不代替最终成功。

## 资源与交付

仅共享research_runtime.queue三次15秒快照准入：free≥40GiB、util<10%、零compute进程、UUID文件锁，动态使用空闲卡、不抢占或终止其他任务。两臂同role按原UUID恢复；观察超时不当失败或重启理由。所有文件/临时/缓存/日志在private。新CPU合同与来源锁通过后才GPU，科学文件冻结后不静默改设置。旧结果、正式CV/PDF、论文与Agent保持，不用CI/自动push。

学习页连接实际6张训练原图、逐题loss分母、代码/张量形状、真实成本/结果与失败病例。pending保持pending，不伪造学生概率、注意力或已发生的性能；助手实现/执行不自动写为用户独立贡献。若本候选失败，目标继续active，依新证据另预注册。
