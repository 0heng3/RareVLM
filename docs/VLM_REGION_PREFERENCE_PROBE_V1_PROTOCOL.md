# 区域偏好短学习正控 v1（SFT 与 SFT+DPO）

2026-10-20登记。旧同图覆盖200步完整负结果后停止；本项是32步训练内正控，不是新benchmark或简历成果，不自动扩长/看主dev挑点。

## 数据证据

从既有256训练图诊断的initial错误中，选择回答匹配同图其他官方词、且GT裁剪下initial能读对的例子。进一步用原几何规则过滤负词区域、非重叠和互非子串，得到16候选图；逐图查看32个A/B裁剪，1组低对比/可能缺重音不确定排整组，并检查第二区域不含其他word中心。当前准入14组，均官方转写与可见证据支持；这是助手裁剪审查而非专家全量标注，不报全池噪声率。
每组双向：框A chosen=yA/rejected=yB，框B chosen=yB/rejected=yA；原错误内容对应拒绝词，反向配对避免全局偏好某词或长度。只用完整原图＋原生像素框，裁剪仅用于数据证据，不输入训练。

## 训练与公平性

硬偏好共28QA，原白名单256条独立VQA回放；另外64条原SFT已见、但本轮未回放且图像分离的VQA保持题。所有数据来自训练白名单，不叫独立泛化。
32更新/effective16/micro4，每步8区域chosen QA（4个完整互反pair）＋8回放，共512 chosen曝光，其中偏好256/回放256。两组完全相同data/order/forward，偏好微批均chosen4+rejected4合批，SFT控制只对chosen算CE，候选额外加DPO；拒绝词forward成本和输入形状因此匹配。控制负序列不产生loss梯度，无batchnorm/跨样本注意力。实际数据/源码和包版本冻结。
原SFT42 FP32语言q/v LoRA1843200、BF16底座/FP32冻结head和loss、视觉冻结/dropout0/TF32off，AdamW wd0/LR2e-5/warmup4/clip1。Final@32，保存1/8/16/24/32，累计训练active每臂900秒cap。0丢弃预检更新，复用已有checkpoint保存恢复与FP32 master核验。共享GPU准入保留，不终止其他任务。

## 目标与参考

复用本地已核TRL标准公式的vlm_alignment_signal.loss，序列logp为完整assistant token和（含EOS/尾部），不是长度均值。
L = CE_chosen + lambda * [-log sigmoid(beta*((logp_pi(y+)-logp_pi(y-))-(logp_ref(y+)-logp_ref(y-))))]
beta=.1，SFT lambda=0，DPO+SFT lambda=1，回放lambda=0。CE仍按每QA assistant-token均值后QA均值，保持绝对正确答案学习，不跑DPO-only或扫权重。
reference是完整原SFT42，不是disable_adapter后的底座。按全部计划偏好微批缓存参考chosen/rejected logp，每批绑定完全相同padding后的input_ids/labels/grid和问题顺序；训练两组匹配缓存输入。初始差值/有限性记录，不将DPO loss或relative reward当实际纠错。
本地上游依据：TRL commit447be34c59a4ad7fb6212917382c62053bab147b，dpo_trainer.py 1330–1388，序列sum与sigmoid公式一致。未安装或替换新训练框架。

## 快速反馈与边界

先生成固定初始基线（偏好28QA＋64保持题）与参考缓存，再两卡训练32步并生成Final同题。即使新分组基线使少数旧错误变正确，也不重选样本。
每组仅以训练内正控判断能否学：偏好净>=4题纠正、pair全对>=2组增加；保持题总VQA score损失不超过1.0且新增降分题<=2。DPO扩大研究还需比同数据SFT至少多2个正确QA及2个pair，保持总score不比SFT低超过1.0。均是资源准入阈值，不是显著性/主能力成功声明。
结果报告casefold/raw、实际纠正/新错、pair全对、保持分与图像簇探索区间。不评估主Text512/Human512、正式benchmark或种子，不写CV。若CE同样学会或更好，不把DPO当必要；若两者都未学会，停止并查优化/可训练模块，不启动另一轮200步盲训。
旧主VQA整体/rare+.5pp/正区间及保护门槛全部保持；通过短正控后仍需另行登记完整数据和无框任务验证。

首个计划偏好微批在optimizer更新前核对actor/reference相对序列logp margin：绝对偏差需<=.05（beta=.1对应DPO logit<=.005），保存实测值。不是要求训练目标已正确，也不以此宣称泛化；输入padding/hash必须逐值一致。
