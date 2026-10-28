# 新场景问答监督：ST-VQA训练来源核验 v1

当前完整native1617能力目标未达标；旧教师池、rank及语言/视觉回放投影分支关闭。新假设：已近零损失的IIIT裁词回放对整图识读新增监督较弱；ST-VQA完整场景问题可能提供不同的定位/语义监督。该推断尚未证明，不能直接开始长训。

本阶段仅下载与盘点新训练来源、检查隔离及准备有界headroom研究。只使用HuggingFaceM4/the_cauldron的st_vqa train，固定revision `847a98a779b1652d65111daf20c972dfcd333605`。发布者说明集合仅含各子集train；保留原镜像user/assistant/source，不把镜像格式化答案冒称原始全量参考答案或8/10票。源图是否在底座预训练出现未知。lmms-lab-encoder/ST-VQA当前只有test，不下载为训练。

两个parquet：`train-00000-of-00002-8f04db983951cc0f.parquet`，359226537字节/SHA256 a7a8dbbe2d4238ec3cb2419189b2ad49dd954d1c725e8acc6d30a05bd681957c；`train-00001-of-00002-9f9c1472593ff618.parquet`，361236353字节/SHA256 f85d7305b013e2ed0e407abba7f07eafcd2b3de0d8626fce2573ab32292b4c7f。总量720462890字节，≤1GiB，每分片下载≤300秒；失败保留partial，不改原训练研究。

盘点必须先确认真实解码分辨率、问题/答案模板、原始路径是否保留、每图不同问答数。排除现有训练/评测原图与像素重复；跨数据集COCO、VisualGenome及IIIT场景/词图家族可能交叉，必须明确能排除哪些身份，不能只凭新dataset名称声明独立。若来源缺少家族信息，保留限制并阻断不满足原保持/外部隔离要求的训练。

最多准备1536不同图的候选输入及256不同图的新来源内部保持（都来自官方训练镜像，非官方test）。QA选择和分组须在新模型预测前固定，后续单独冻结headroom协议/输入/模型。新来源只有一个或少量参考，不能复制10次冒充TextVQA投票；其exact或单参考ANLS仅为镜像池诊断，原native1617的VQA评分保持不变。

未来若数据准入与有效监督成立，唯一候选方向是在相同起点/视觉LoRA/32步/512chosen/原128已审纠错+DPO/128native保持下，用128整图ST-VQA正监督替代128已解IIIT裁词回放。原强CE/DPO对照须逐项复用或匹配，原OCR/native/长尾/保持门槛及native1617双强控制+.5pp/正CI、后续三seed/独立确认不变。新问题和答案类型随数据源变化，不能称单一分辨率或纯数据量因果。此训练尚未登记或运行。

本阶段0模型调用/0优化，不改简历/科研/Agent/CI、不push、不操作其他GPU任务。不能把下载成功或数据量写作能力提升。

来源：[The Cauldron数据卡](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)、[ST-VQA原论文](https://arxiv.org/abs/1905.13648)、[lmms-eval任务配置](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/stvqa/stvqa.yaml)。
