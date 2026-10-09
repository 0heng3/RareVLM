# 岗位共性 → RareVLM 的统一技术路线

检索日期：2026-10-09。只依据官方 JD 提炼出现过的技术与工作链路，不把岗位样本当全行业统计。简历统一面向计算机视觉、多模态算法、多模态大模型算法。

| 官方岗位样本 | 出现的能力要求 | 本项目对应交付 |
| --- | --- | --- |
| [百度 AIDU 多模态算法](https://talent.baidu.com/jobs/detail/GRADUATE/d29404bd-55de-4cc1-929b-e67d304abc97) | CLIP/Qwen-VL、Python/PyTorch、数据管线、评测和模型训练部署 | 原 CLIP 受控研究与新生成式 VLM 模块形成连续链路。 |
| [腾讯微信内容理解](https://careers.tencent.com/jobdesc.html?postId=2099340394696458240) | SFT 工作，LoRA/P-Tuning 等实践优先，训练评测部署与数据问题定位 | assistant-only LoRA SFT、固定输入生成评测、可运行本地推理。 |
| [字节全球电商，新加坡](https://joinbytedance.com/search/7621874043991132421) | 数据构建、指令微调、PEFT、训练部署；混合精度/分布式知识加分 | 图像分组与内容去重、QLoRA/DDP、精度兼容排错、可训练参数及资源记录。 |
| [美团多模态文档理解实习](https://zhaopin.meituan.com/web/position/detail?highlightType=campus&jobUnionId=4635641129) | PyTorch、LLM/VLM 训练微调、OCR/文档理解、LLaVA/Qwen-VL | 包含场景文字的视觉问答任务，用 Qwen-VL 完成图像→文本答案链路。 |

所选路线：**视觉语言表征与诊断 → 生成式 VLM → 多模态指令数据 → LoRA SFT → 生成评测与错误分析 → 本地推理/资源测量**。它覆盖样本中反复出现的训练、数据、评测与工程实践；不为增加关键词而加入 RL、RAG、视频生成或新梯度 solver。

原 CLIP 长尾算法与结果保持各自冻结协议。2026-10-09 用户明确授权新增生成式模块，因此新模块有独立配置、数据、模型和结果，不与历史 OA/Few 表混列。它采用既有 Qwen 与 LoRA，贡献是实际实现与验证，不宣称新算法。

统一简历在 [RESUME.md](RESUME.md)。能力必须绑定实际完成的运行记录：代码可运行不等于已完成训练，少量工程子集不等于完整 benchmark 或商业部署。相关性能数字只在完成训练与评测后加入。
