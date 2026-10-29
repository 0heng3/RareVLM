# 完整7B教师：监督准入五项全部通过

固定128条已见训练题、两个不同冻结模型。7B教师自然literal正确105/128（104不同GT），当前较强3B SFT为36/128；新增纠正71、退化2，净69，纠正题平均正文NLL下降0.825937。73题通过T2真实KD与同GT概率均匀尾的结构信息检查，填充输出概率质量最大0.001973。全部预定gate通过。

这不是学生从36提升到105，不是benchmark、专家全量重标或纯参数量因果。两worker均exit0/0争用/0更新；学生128 GT前向active115.36秒/峰14.753GiB，教师128生成+128 GT前向active362.37秒/峰32.840GiB。资产15/15官方哈希过，获取1480.66秒另列。

CPU验证两个完整实际token字典及128题prefix/grid/pixels/GT标签一致，共151665实际token。输出head151936/152064的填充质量单列；软分布在共同实际token集合归一化，原CE使用各自完整head，不直接截断冒充同词表。

下一训练固定103题教师自然正确且正文NLL≤1/EOS NLL≤.5，其中70个学生残错被纠正；每题2次辅助曝光，不按dev或信息分数继续挑选。教师也会错，23个literal错误及其他置信不合格项均不进入辅助KD；全部原GT CE仍保留。

另行预注册hard/uniform/kd三臂，从较强SFT Final32共同起步，现有204 LoRA联合训练。GT硬监督、同GT概率均匀尾、真实教师KL仅辅助目标不同；学生能力须完整1617、匹配控制、长尾与保持验收，尚无新能力结论。

[教师固定协议](VLM_FULL_TEACHER_SIGNAL_V1_PROTOCOL.md) · [便携指标](../results/vlm_full_teacher_signal_v1_metrics.json) · [真实图像生成器](../tools/render_full_teacher_signal_v1.py) · [学生蒸馏协议](VLM_FULL_TEACHER_KD_V1_PROTOCOL.md) · [损失代码](../src/vlm_full_teacher_kd_v1/loss.py)
