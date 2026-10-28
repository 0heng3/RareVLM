# 已完成IIIT适配的整图VQA迁移诊断

2026-10-25，四组新输出前登记。前IIIT固定64步研究已02:18完整完成，官方OCR三组均有任务收益，通用native512保护尚未过；不按已曝光OCR测试调参。当前需检验同一四个冻结模型在更完整整图VQA上的迁移，0训练。

复用原`vlm_literal_case_sft_v1/DATA.json`的transfer_dev1617，问题、10参考、图像、1024视觉上限和生成策略不改。该cohort是历史内部dev，并非新外部盲测；在本轮IIIT模型上首次完整评价。CPU重新记录真实prefix SHA并核对旧长度/grid，只针对本次新cohort，不重复旧加载/mask审查。检查其source image ID与原SFT train_all及本轮训练图的交集，按实际结果报告，未知预训练重叠不作未见声明。

四模型initial/dense_language/dense_visual/sparse_language直接加载上轮已封闭初始或Final64，FP32 LoRA/冻结FP32 head/BF16底座、dropout0、TF32off、greedy32/repetition1、batch4（尾1合法）、原VQA计分。全四模型共同输入，0新optimizer、0模型或checkpoint选择。依共享空闲锁，先initial+sparse_language，再其余配置，已有任务不抢占。

1617整体、single/multiword、原监督整答案频次0及1–5分列，6000次image-cluster配对95%区间，单checkpoint探索、不涵盖seed随机性。不能将裁剪词图ASCII36收益或答案输出范围变化自动视为整图视觉推理提升；结果用于决定后续native回放分布研究的内部路线，不能回写原IIIT测试成功或旧保持门槛。

每模型评测<=1800秒/allocated<=65GiB，完整4QA批断点/源和模型输入锁。共6468新生成；未混入上轮15840。新模型初始为原语言5d8加B0视觉、复合40ccd9a5，Final身份取封闭TRAIN_COMPLETE。已有pipeline/源码/权重/评分不改；CV/Agent/论文/CI不改、不push。
