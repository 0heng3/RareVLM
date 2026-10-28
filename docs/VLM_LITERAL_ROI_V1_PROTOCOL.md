# 给定单词区域转写：可靠监督与偏好学习准入

2026-10-23事前登记。前固定网格+未曝光词方案只有90条，已不足停止，0GPU/训练；不改变其条件或完成记录。新任务为标准given-word-region transcription，参考[PARSeq官方TextOCR转换器](https://github.com/baudm/parseq/blob/main/tools/textocr_converter.py)按官方词框构建识别图像，不是按原VQA答案寻找裁剪，也不把given-ROI结果冒称整图定位。官方源码已保存在outputs/cocotext_source_v1/parseq_textocr_converter.py；这是数据处理参考，不宣称复现PARSeq模型。网络COCO-Text资源仅取得不完整文件，不使用其数据，不称新公开数据已接入。

**数据修复。** 原生brand/name问题与OCR目标可能语义错配；本任务固定问“Transcribe the complete word in the image. Return only that word.”，使用单个官方完整转写，不伪造10人共识。从原train_all排当前训练/context/seen/保持、原dev/final及前1269/104同图与decoded pixel；一原图一词、casefold词也唯一。官方polygon合法，水平<=10度，ASCII字母3–20、原生宽>=48/高>=16；包围框外加2像素构成给定区域，距原图边界>0，不能与任何其他OCR词框相交。模型只接收区域图和固定指令，不接收文字/坐标/原VQA答案。官方密集标注也可能不完整，仍需对实际训练错题图像复核，不称专家全量审核。

本轮用NFKC/casefold后正则[a-z]+统计原SFT完整答案中的词，不用会删除冠词的EvalAI归一化。限制词曝光0–5：prediction-free盘点4964图/词（2453曝光0、2511曝光1–5）。这是监督词曝光，不是预训练未见或视觉类别频次；所有原图都曾进入原SFT。源清单已保存，按固定SHA执行顺序取前2304，前2048训练候选、后256固定转写验证。两组原图/decoded pixel/词均隔离，验证不补训练、不挑参数；少于2304则停止，不放宽规则。

**初始评测。** 原uniform SFT42 q/v144/1843200 FP32 LoRA hash5d8e2e...dd7b、BF16底座与冻结FP32 head、dropout0、TF32off、1024视觉上限/min50176、batch4/greedy32/repetition1。真实CPU编码2304前缀，保存输入SHA/grid/长度、原图及actual crop raw/pixel SHA；复用已验加载、生成、EOS，不重审整个框架。完整4题批交替分两卡，900秒生成活动/片、allocated<=65GiB、共享空闲锁不抢占/同UUID恢复。

主指标为NFKC/casefold/strip后完整字面exact；保留内部空白/标点，不改GT，不是TextVQA共识分。报告训练候选2048和验证256全部固定分母，曝光0/1–5另列；初始得分不作为新训练能力提升。训练候选须初始错误、自然EOS、GT不在回答分词中且去掉预测空白仍不是GT；再按固定SHA逐图审核，至少128可靠不同原图/96target。数量不等于审核，控制器不自动优化；不足时不降门槛或回填验证。

GPU输出前的CPU输入适配修复：当前锁定Qwen处理器smart_resize会拒绝任一边<28的图像，首次实际编码在高22/宽211处停止，0GPU/模型调用。新区域在保存与编码前，仅当最短边<28时用BICUBIC等比例上采样至最短边28（另一边ceil）；记录original_crop_size与factor28_compatible_size，原标签与源框不变。所有角色和验证使用同一确定处理；插值不增加原生图像细节。此变更在新PLAN冻结和任何模型输出前完成，后续完整CPU编码验收。

**后续预期但尚未实施。** 审核通过后另立冻结训练协议：同128词错误+同256回放、32步普通SFT对DPO+SFT。DPO参考为精确原SFT，chosen为完整官方词，rejected保留实际自然EOS采样IDs；β=.1/权重1，序列SUM含EOS；SFT逐QA token均值。新增词转写与原生VQA、保持任务分别验收，不能用本任务提升替代原项目两个VQA域严格收益要求。训练拟合/loss/权重变化均不当能力证据。

学习页8例在推理前固定，图为实际given-ROI输入，链接到原图和源框；展示局部识字和整图问答的任务差别、SFT与DPO公式、真实回答。CV/论文/Agent/CI不改，不push。
