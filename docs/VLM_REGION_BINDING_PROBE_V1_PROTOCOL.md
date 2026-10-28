# 同图双区域坐标切换诊断 v1

2026-10-20在像素SFT完整未过门槛及256图迁移诊断完成后登记。本轮仅0更新诊断，不追加训练/修改旧预测或门槛。

## 数据

完整原1600训练图中，使用原几何规则寻找官方第二文字区域，要求与原区域不相交，文字大小写无关不相同、相互非子串，候选不得包含其他word中心，按归一化中心距离最近/annotation_id破同分选择；未使用模型错误选样。得到1389组同图双区域候选，211图无合格第二区域。全池是未经专家全量审查的训练候选，不称新heldout或干净标注集。
按预定义SHA256('region-binding-v1:20261020:'+image_id)排序取前64组，每图两个问题/不同官方文字，共128QA。两个问题图像、模板、尺寸、suffix完全一致，只有像素坐标数字不同；CPU实际编码逐题检查grid与所声明输入维度。

## 推理

原SFT42 initial与既有像素Final@200，数值/LoRA/FP32 head/raw greedy/max32/batch4/1024预算同前。每模型128新生成，共256；0optimizer。先锁脚本/协议/全池/64组输入/模型及图片SHA，推理前后LoRA hash不变。
共享GPU准入保持零compute/40GiB/3次15秒/UUID锁；不终止其他任务。每worker active240秒上限，排队载入另计。

## 指标与下一研究

主要：同图两个官方词都casefold exact的pair exact；报告QA exact、A/B各自正确数、两个框回答相同的数量、A输出B目标/B输出A目标。回答不同不等于定位正确；图片属于训练，不写成独立泛化能力。
Final对initial的pair exact按image/pair聚类、6000bootstrap/seed20261020；区间探索性。不得把两个问题或两个模型当独立种子。
这项测试检验现有单区域SFT是否学会随框改变答案，为之后单独登记的同图多区域监督提供实测依据。后续最小候选是同图两个真实框各自完整CE＋原VQA回放，不在本轮引入DPO/GRPO或自动长训。
数据构造脚本prepare_region_binding_data_v1.py，实际推理probe_region_binding_v1.py；源输出vlm_region_binding_data_v1与vlm_region_binding_probe_v1。
