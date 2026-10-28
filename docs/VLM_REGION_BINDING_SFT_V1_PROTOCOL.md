# 同图区域覆盖SFT v1：重复单框与覆盖双框

2026-10-20登记；原像素200步完整未过主门槛后停止，以下是新数据覆盖假设，从原SFT42重新起步，不接负Final，不改变旧门槛。

## 依据与边界

原无框Text：像素85.976563/初始85.859375/归一化85.371094，主联合失败。新256训练图诊断full casefold126→156（33修3破，22修原已有目标/11原未有），目标字符串出现161→161、范围错35→5，同图其他词43→45；crop179→180。固定同图双框64组初始/像素Final全对19→27、同答13→5，说明模型已有框依赖但精度不足，不称完全忽略坐标。
原1600训练白名单可构造1389组官方双区域，非模型错误选样；候选无专家全量干净标注声明。按已固定哈希顺序取前800组，包含原64组训练探针，绝不称heldout。

## 唯一因素：第二区域的监督覆盖

每图两条原TextVQA问题（已有>=8票的原控制清单）作共同回放，共1600唯一QA；每图两个官方区域A/B，非交叠、答案不同且互不为子串、原几何/可读性过滤不变，实际像素坐标/全图/模板一致。
repeat控制：第一辅助slot读取anchor，第二slot重复同一anchor。
dual候选：第一slot同anchor，第二slot读取另一官方区域。
anchor按图序交替，400张A、400张B，避免只重复原A造成难度方向不平衡；不根据模型对错选择。每组200更新/3200曝光，图像及微批顺序完全相同：每4图一更新，4条VQA1、4条anchor、4条VQA2、4条重复或另一框。
两组共享完整3200候选QA目录，但控制实际只曝光2400独立QA（1600VQA+800区域各重复两次），候选3200独立QA（1600VQA+1600区域各一次）。目录中未曝光条目不会送入模型；Final按实际INPUT_TRACES逐题核对计划顺序和独立曝光数。候选新增800个区域标签，等QA/图像不等于等监督信息/文本token；不能包装成新损失的纯算法收益。

## 固定训练与输入

普通完整assistant CE含结束符，不用取消EOS/排序/DPO/GRPO。直接复用冻结vlm_region_ocr_sft_v1.train.train：其3200目录约束满足，训练只按注册schedule索引；不复制或修改旧训练循环。
原SFT42 FP32语言q/v LoRA1843200；BF16底座/FP32冻结head和CE/视觉冻结/dropout0/TF32off；micro4/effective16/LR2e-5/warmup20/AdamW wd0/clip1，200更新Final，2700秒active cap/臂。共享GPU零compute/40GiB/3次15秒/UUID锁，不抢占他人任务。
1600共同和800原A的2400个实际CPU输入hash可复用（第二原VQA的source标签改为common，不改问题/target/token）；800新B全部实际编码、目标mask与实际像素frame核验，登记3200唯一目录输入。原图片与官方target不改，无GT答案输入生成。

## 评测与标准

两组各Text512/Human512（无GT框），同raw greedy/repetition1/max32/batch4/1024预算；复用完全同口径原initial预测。dual对initial与本轮fresh repeat均须Text overall及rare1–5至少+.5pp，整体6000图像簇95%CI下界>0；frequent>20与Human overall/yes/no点差>=−.5pp。频次仍为原32985监督目标，Human按既有组件聚类，保留全部升降，不改旧+1pp/pair+2pp分支门槛。
Final额外在同128QA/64双框训练探针评测casefold QA、两框全对与同答数，初始已知55/128与19/64，不能当独立能力验证，也不拿此替代主VQA门槛。
单seed42/已曝光内部screen的探索研究；失败不扩训/挑checkpoint/扫参/写CV，筛选全过仍需独立确认和复核，本轮不自动追加种子或训练。

## 学习入口

条件概率p(y|I,b)；同图两个真实框的完整逐QA CE检验区域覆盖是否能改善定位与迁移，不声称模型原先完全不用框。数据构造prepare_region_binding_sft_v1.py，旧实际训练vlm_region_ocr_sft_v1/train.py，编排vlm_region_binding_sft_v1/study.py。
