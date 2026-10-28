# 完整单词可靠监督：同数据SFT与DPO+SFT

2026-10-23，新增训练输出前登记；只有vlm_literal_roi_v1训练池里至少128助手视觉支持错误/96target才冻结本研究输入，不用256转写验证补题。128题均官方完整词给定区域，模型原始自然EOS回答错误，选择固定SHA顺序的前128个通过项；偏好不是原生品牌/姓名问题的共识分推断。助手复核不冒称专家重标注，不改变原官方转写或原分数。

两臂原uniform SFT42同起点5d8e2e...dd7b，BF16底座/FP32语言q/v144因子1843200与冻结FP32 head、dropout0/TF32off；无MLP/视觉新适配。128完整词各2次+固定256原回放各1次，共512 chosen曝光，32更新、micro4/effective16，lr2e-4、warmup4、AdamW wd0、clip1、Final32不挑点。256回放从既有512正确训练记忆按固定SHA取前256，两臂一致。每步common4/preference4/common4/preference4，chosen CE逐QA token均值后/16累积。

控制sft: L=CE；候选dpo_sft: L=CE+DPO，β=.1/权重1，只在128词题上加DPO，回放仍普通CE。DPO logp为含EOS序列SUM，ref是精确原SFT，相同实际图/问题/left-padding batch。rejected使用初始greedy原始completion_ids和真实EOS，attach_completion后直接teacher forcing，不把解码文字再分词当原动作。chosen保留官方词/既有assistant模板EOS切断，忽略EOS后模板换行；复用既有chosen_sample、DPO损失和32步循环。两臂都运行相同8序列chosen/rejected前向（控制额外DPO权重0），同布局/forward预算；不称比最省CE更低成本。首参考相对margin容差沿用旧.05、β后.005，不重做已过全框架数值审查。

唯一参考模型先缓存实际64 preference微批的初始chosen/rejected logp；完全相同重复batch可只forward一次并在64键中复用，记录实际唯一批/序列数。缓存前后权重hash保持。原初始模型额外生成selected128的新固定batch拟合背景（避免使用矿池旧padding上下文假定初始全错）；原seen512/保持64、同256转写验证在同起点/同batch/同生成设置下预测只读复用并记录来源。数据、输入、代码/权重/配置/参考缓存在训练前锁定，fresh Adam/RNG，不接训负Final。引用的新区域图实际factor28预适配与初始矿池完全相同。

固定第一阶段每臂128拟合+256词验证+512原生seen+64保持=960生成，0额外教师回答。完整字面exact与官方VQA分列。探索性放行须：256词验证相对初始>=+3pp且配对6000图像簇95%下界>0；seen512相对初始>=+.5pp且下界>0；seen单/多词和rare1–5点保护>=-.5pp；保持>=-.5pp；训练满分净增>=32。DPO额外优势还需相对新SFT在词验证与seen均>=+.5pp/严格正区间。切片区间单列，单seed内部已曝光研究，不称盲测/多种子稳定性。

若任一臂第一阶段放行，单独用同一Final给两臂补原transfer_dev1617（不修改权重），再检查原两个VQA域相对初始、新SFT及已有强q/v回放均>=+.5pp且严格正区间与原罕见/保持保护；第一阶段/转写收益不能代替主目标。未放行不运行1617、不扩步/扫β/加seed或正式测试。完整项目三seed与更广验证要求不因新任务缩减；用户目标只有相应完整证据齐后才完成。

参考缓存活动<=600秒、训练<=1800秒/臂、第一阶段生成<=1500秒/臂、allocated<=65GiB；原共享空闲GPU规则/UUID锁不抢占，完整optimizer/RNG/日志偏移断点。训练源码复制已验32步循环，只重绑定独立common/配置和结果scope，不修改旧源。CV/科研/Agent/CI不改、不push。
