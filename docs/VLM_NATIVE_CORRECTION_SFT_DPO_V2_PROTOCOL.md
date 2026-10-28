# 真实原生问答纠错：SFT / DPO+SFT V2

训练输出前补充：首池146候选在看图后已确定不足128支持，独立[1536补充登记](VLM_NATIVE_CORRECTION_V2_SUPPLEMENT.md)保留原首池不动。此训练改读vlm_native_correction_pool_v2派生合并池，按首池146原序+补充候选固定序，合并仅审核前256；128/96准入、384训练QA、32步、所有损失和评测门槛均不改。首池最初first256表述按本显式补充执行，尚无训练STUDY锁/模型输出。

2026-10-25，候选训练输出前登记。必须先由native_correction_mining_v2取得至少128条助手看图支持、96不同target的真实错误；只能用固定首256候选及其审核顺序，不能补dev题或降低准入。审核不是专家重标注，历史标签/分数不改。若不足，本训练协议不准入。

同起点：OCR提升后的sparse_language Final64语言144 FP32因子（独立导出0109b884…520f9），视觉B0去除、语言值精确保留，固定8题GPU生成ID/前缀spot check须通过；不是原5d8参考模型。仍更新1,843,200个q/v rank8参数、BF16 base/FP32 head+LoRA、dropout0、TF32off。

同数据：128真实native纠错各2次；128旧正确native回放各1次；128原IIIT稀疏词图各1次，总384 QA、512 chosen曝光。native原问题/原图/原答案不变；OCR仅沿用canonical助手监督，原标签另存。两个组source/micro/order完全相同，每步OCR4/纠错4/native4/纠错4，effective16/micro4。旧正确native池并非保证当前OCR增强模型仍全正确；只是来源属性。所有训练与eval图像/解码像素隔离；原train_all图像曾进初始SFT，不称新图盲测。native监督可能出现OCR dev词，报告重叠，不再用排功能词造成额外长度偏移；不声称OCR新增监督全面词隔离。

复用冻结vlm_literal_preference_v1.train的32步数值循环，只绑定独立ROOT/CONFIG；make_batch/sample、sequence_logps、dpo_loss不重写。AdamW lr2e−4、warmup4、linear32、wd0、clip1，完整FP32 master/Adam/RNG/输入trace断点。角色sft权重0、dpo_sft权重1，β.1；每偏好QA CE为有效assistant token平均，DPO为正负序列log概率sum（含真实EOS），相同CE+.5×平均偏好DPO等价于候选的整步损失，CE的纠错/native/OCR权重为.5/.25/.25。负序列直接附原生成completion IDs到原prefix，不解码再分词。两个组均执行相同负序列forward，以同预算分离DPO目标贡献。

参考为新起点，eval/no_grad、完全相同的padded chosen+rejected batch，64 micro键/32唯一forward批可缓存；原训练首步relative margin≤.05和DPO数值检查保留。先以同新硬样本顺序重新生成起点128条拟合背景，避免把矿工不同batch分组全当模型变化；另baseline OCR256/native512/保持64/transfer1617引用同起点原记录，逐题输入SHA核对。保持64在新数据明确补冻结baseline实际SHA，不修改历史源或在收尾无条件取缺失字段。

第一阶段：每组Final32新native_hard128/OCR256/native512/保持64=960生成。推进需硬样本≥32满分且VQA较起点≥+20pp（只证明可学）、OCR256点差≥−.5pp、native512整体/单词/多词/频次1–5及保持64相对起点均≥−.5pp。这些是探索点门槛，不等于能力成功。两组都未过则结束，不延长训练、不调β/数据配额。任一过才补两组完整1617，公平报告对照，包括未过保护的一组，不用它替补候选成功。

真实native局部正信号需：原内部门槛通过、1617对OCR增强起点≥+.5pp且图像簇95%下界>0，并对原SFT初始也≥+.5pp且正下界。DPO特有优势还需相对同数据SFT≥+.5pp且正下界，不能仅用拟合或偏好margin称改进。bootstrap6000次固定源码，历史内部集、固定单seed，不能冒称独立种子或外部盲测。

只对满足native条件的组追加官方IIIT5K3000已曝光test作为读词保持复核，对原sparse96.933333%需点差≥−.5pp与区间下界>−.5才支持统计非劣；不据该test换配置。报告原始预测、ASCII36口径、未知预训练重叠，不使用旧extra_train_unseen标签。原强基线/独立三seed/外部/跨任务完整要求仍保留，本32步先导永不自动标记项目目标完成，也不自动扩seed/步数。

预算：总64新优化器更新，先128起点+1920内部=2048新生成；条件native3234、条件重复OCR最多6000，总最多11282新生成。reference600秒（另128拟合背景≤1500）、训练1800秒/组、内部1500秒/组、1617评测1500秒/组、OCRtest2400秒/组、65GiB。共享空闲3×15秒、≥40GiB/零compute/<10%利用率/UUID锁，不抢占已有任务。训练源码/数据/配置/环境审核后冻结。无CV/科研/Agent/CI/push改动。

原理与接口参考：[DPO原论文](https://arxiv.org/abs/2305.18290)、[TRL DPO官方说明](https://huggingface.co/docs/trl/dpo_trainer)，采用本地已验冻结数值路径，不切换在线最新库。配套学习页展示8张预固定真实训练图、正负答案、公式、源码和完整评测结果，训练示例不当泛化。
