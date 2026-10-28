# SFT/KL阴性结果：完整训练分布的零更新梯度诊断 V1

2026-10-26，输出前冻结。父`vlm_native_stable_kl_v1`两臂32步/各512QA/各960内部预测完整、exit0；KL与SFT在native512字符串及分数全同，native+.117188pp、rare−.655738pp未过点保护，禁止扩原负配置。不能用累计KL占loss比例推断真实梯度强弱。本次只对两Final计算静态完整训练目标梯度，不重复初始/首步已完成检查，不优化、不使用验证题梯度、不改旧标签/预测/评分。

完全复用原384 QA、chosen CPU hashes与32×4个micro顺序、正确回放flag255、64批全词表FP32教师缓存、原实际EOS和pad mask，以及修复后的逐图视觉+FP32语言数值策略。按ordered 4-QA tuple合并重复batch并保留曝光重数；原hard128各2次、native/OCR各128一次，384次实际QA forward表示512曝光的同一固定Final目标。若分组绑定不同不能合并。预计96 unique forward batches/模型（32hard+32native+32OCR），无负序列、0optimizer。

对每Final以同teacher grad-enabled/dropout0/非重入checkpoint方式分别求5组FP32参数梯度：hard CE、native CE、OCR CE、native正确回放KL、OCR正确回放KL；均保留原512曝光归一与重数。common同一次forward计算CE与KL，分别autograd.grad，hard只CE。转CPU FP64累计，保存完整5个梯度向量和参数名/shape，以重现norm、cos与层级贡献；不把静态梯度当历史Adam更新或唯一因果。

报告gCE=gHard+gNativeCE+gOCRCE、gKL=gNativeKL+gOCRKL及合力；||gKL||/||gCE||、夹角cos、native/OCR分支与hard的cos、λ1合力对CE方向变化。norm0时cos记null；不以loss比代替梯度比，不根据诊断自动调λ或上PPO/GRPO。父新起点与候选原性能结果独立分列，原1617未放行仍未测。

科学源/父锁/两Final权重与TRAIN_COMPLETE/缓存索引/协议先绑定；缓存每实际使用batch hash与token counts复核，实际chosen与原CPU一致。两Final各独立worker，完整数据每模型900秒/65GiB，使用共享空闲GPU UUID三次15秒/零compute/40GiB/<10%，不抢占。仅新梯度诊断，不重跑旧完整验收/生成/训练。若未完成不宣称梯度根因；完成后提供同真值数据的可视化与下一算法建议。
