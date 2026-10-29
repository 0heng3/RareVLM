# 7B整图教师是否提供新增监督：固定训练池验证 v1

2026-10-28在任何新教师输出前登记。来源保持完整结果为新SFT197/256、起点195/256、新DPO196/256，四预定exact比较的98.75%区间均跨0；DPO正文NLL最低不能替代自由生成正确率。原固定ST-VQA训练配置关闭，不追加步数/seed或扫mix/LR。此前同3B标注辅助crop教师的数量与信息门槛失败也保持关闭。

本轮使用真正不同的冻结模型能力：官方`Qwen/Qwen2.5-VL-7B-Instruct`完整图像教师，HF revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`，权重经ModelScope传输、逐文件按HF LFS SHA256核验。对照为当前新整图SFT Final32的3B学生，不用较弱起点夸大教师优势。目标只判教师信号，0更新，不把教师训练池得分当学生能力提升。

## 固定数据、输入与计算

直接使用原`vlm_stvqa_replay_v1/data/MANIFEST.json`的全部128条已审核native_hard训练QA、原顺序和原图；不是从dev/保持/正式集挑样，不扩大错误池。它们已被学生训练，非未见数据；原GT与分数不改。学生128条自由生成按原SHA复用，教师新生成128；两模型各128 GT前缀前向，共256。全部权重冻结，不创建optimizer或新adapter。原小crop/teacher不足门槛不能借本协议改变历史结论。

两角色采用1024视觉预算、原问题/短答提示、greedy32/真实EOS、batch4、imagewise BF16视觉、BF16基座值上转的FP32语言及独立FP32 head、TF32off。学生使用原加载器、FP32 LoRA；7B不挂3B adapter。新增教师只用原FP32 head和imagewise安装函数，基座无参数更新。共享空卡队列2角色，各≤900秒/allocated65GiB，不占用其他任务的GPU。

CPU先核验两个tokenizer的完整实际token→ID映射、special IDs及128题prefix/grid/GT labels/pixels合同。3B和7B输出embedding尺寸为151936/152064，不直接截断后假称相同完整词表。软分布仅在完全一致的实际token ID集合上分别重新归一化，记录两个模型在T1/T2丢弃的填充输出概率质量；GT CE统计仍用各自完整输出。权重不齐或CPU映射不一致不进入GPU。

## 固定教师信息检查

自由生成采用既有NFKC/大小写/空白/外端标点literal规范化，保留冠词；不是新增官方VQA。正文/EOS NLL、GT argmax和终止状态分列。教师正确须literal匹配并自然EOS，学生输出直接复用当前Final。

软信息主温度固定T=2，T=1仅描述。逐GT正文token缓存共同实际词表上的FP32 logits，不含EOS。计算teacher KD输出logit梯度与同GT概率的均匀尾分布对照：

`g_real=T*(p_student(T)-p_teacher(T))`；`q_uniform[y]=p_teacher[y]`，其余有效token均分`1-p_teacher[y]`；`g_uniform=T*(p_student(T)-q_uniform)`。

同GT质量、同温度、同非GT总概率，只有非GT概率结构不同。逐QA合并正文梯度求cos和相对L2差；这是输出logit梯度，不是参数梯度或泛化因果证明。GT硬CE的梯度另列，不因KL或NLL非零就声称提供新知识。

事前教师信号门槛：

1. 教师至少48/128自然终止literal正确、至少40不同GT；
2. 学生错→教师正确至少16，纠正数减退化数至少8；
3. 这些新增纠正的平均正文NLL下降至少0.2；
4. 教师正确且正文NLL≤1、EOS NLL≤0.5的题中，至少16题的T2真实KD与同GT质量均匀尾对照cos<.995且相对L2差≥.1；
5. 两模型所有正文token的T2填充输出概率质量≤.01。

任何一项失败停止本固定教师配置，不扩池/降阈值/扫温度。通过也只允许另行设计学生的硬CE/同GT质量均匀尾/真实教师KD匹配对照，未授权自动学生训练。原完整1617对起点/匹配强控制≥+.5pp且正区间、OCR/长尾/保持、三seed/外部确认目标不变。

## 获取与停止

资产获取CPU上限1800秒/4流，断点与失败保留；完整HF校验后才允许教师加载。该时间是下载预算，不是训练时间。教师控制器可等待已提交获取任务；失败不启动GPU，也不改原数值策略救结果。源码、协议、CPU输入与模型身份冻结后执行；完成只归档教师准入结果。简历/科研/Agent/CI不改，不自动push，不终止其他GPU任务。
