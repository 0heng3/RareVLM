# 统一简历项目描述

目标方向：**计算机视觉算法／多模态算法／多模态大模型算法**。只保留这份项目描述，使用已完成并可核验的成果。

**RareVLM｜视觉语言模型适配：长尾分析与多模态指令微调**  
独立算法研究与工程项目｜PyTorch · Transformers · PEFT · CLIP · Qwen-VL · QLoRA · DDP

- 构建冻结 CLIP 与 6.6 万参数残差 adapter 的长尾适配框架，在 ImageNet-LT、iNaturalist 2018 上完成三种子配对实验，覆盖完整 B/32/B/16 checkpoint 和在线训练视图复核。
- 发现 ImageNet-LT B/32 的 CE 适配使 Many 提升 **15.1 pp**、Few 下降 **21.5 pp**；固定 adapter 的验证集校正将 Few 从 **37.76% 恢复至 61.23%**、OA 从 **61.38% 提升至 65.36%**，结合逐图错误转换、有效先验和 margin 分析定位适配取舍。
- 新增 Qwen2.5-VL-3B-Instruct 多模态指令微调链路：冻结视觉编码器和语言底座，使用 NF4 QLoRA 训练 **184.32 万参数（约 0.049%）**；在 **4 张 V100** 上完成 TextVQA 图像互斥子集的 1,024 样本、一轮 SFT、64 次更新，采用助手回答专用 loss mask 与固定最终 checkpoint。
- 实现验证选点、模型/数据/checkpoint 哈希绑定、图像 ID/像素去重、配对在线增强与生成输入隔离；通过 **31 项 CPU 契约测试**。新模块提供相同设置下的生成式评测和本地图像问答推理入口，记录显存及延迟；原 CLIP 模块通过 CPU smoke 验证。

证据入口：[CLIP 主结果](RESULTS.md)、[生成式执行结果](VLM_SFT_RESULTS.md)、[生成式模块协议](VLM_SFT.md)、[岗位要求映射](JOB_ROUTE.md)。生成式模块属于单种子工程子集，不等同于完整 TextVQA benchmark、新微调算法或生产服务。全文为同一份项目简历描述，适用于上述三个相关岗位方向。
