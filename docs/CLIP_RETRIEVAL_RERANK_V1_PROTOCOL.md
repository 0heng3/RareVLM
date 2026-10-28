# 固定Top50的BLIP图文匹配重排 v1

2026-10-14用户授权执行JD驱动的下一步实验。依照NEXT_ACTIONS候选，固定已有三个CLIP关系蒸馏Final@10 checkpoint；0新增训练、1个固定COCO匹配BLIP checkpoint。目标为增加Embedding后的图文重排证据，并实测质量与额外成本；不改正式简历、不启动Agent/RL/视频研究。

## 固定模型、输入和排序

召回保持CLIP-B/32冻结FP32/224中心裁剪/77文本上限、两侧132480参数的三个旧adapter，种子0/42/200作为既有模型的评测轴；完整gallery1000图/5000描述。全图库cosine与stable index排序生成Top50，先逐query核对旧Top10完全一致。新matcher只接收这些候选图文，不依据标注筛选；监督图ID仅用于生成后评分。

BLIP为Salesforce/blip-itm-base-coco，revision bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c；pytorch_model.bin完整SHA017fb3e7f4e125f13a8a4717f1402dbe0d0bb877474b4a203db13a4447b0227f，895139697字节。模型卡注明COCO ITM训练，未采用Flickr专门微调权重；预训练数据重叠仍未知。完整processor/词表/config/模型卡文件校验留档，不自动升级Transformers4.51.1及其余包。

固定processor的384输入，最大文本64token，按官方BLIP ITM源码将起始位置设为[ENC]30523；区别于HF处理器默认CLS101，不称默认HF输入等价。ITM正类1由官方训练ones正样本标签确认。前向BF16，TF32关闭，vision batch16、pair batch32；图像缓存[1000,577,768]，text input[32,64]、ITM logits[32,2]。排序分数为FP32(logit1−logit0)，不混合CLIP权重、不扫超参；同分保留原候选CLIP顺序，仅重排Top50。候选之外的尾部不影响R@1/5/10/nDCG10，R@50上界不变。

新BLIP增容量、视觉输入和计算，属于整个两阶段配置对照，不是等模型/等信息/等计算的纯算法归因，也不声称新原创、端到端训练或工业部署。

## 去重与恢复

每seed：T2I5000×50和I2T1000×50共300000个逻辑pair；三seed900000。以image_index*5000+caption_index为联合key，全两方向/三seed排序去重后仅计算一次，实际唯一pair数与逻辑请求数分开。两GPU role平分完整去重集合；各2048pair保存key、双类logits、margin、完整hash和区间，支持逐块恢复。

同role坚持原预检GPU UUID、输入和源码锁，vision缓存按role保存，不把换卡时间混作可比计时。图像编码复用一次/role；HF原生匹配器的image_atts factory使用作用域默认GPU设备以避免4.51.1创建CPU mask，未修改上游库。相同[ENC]输入分别运行原生全前向与缓存图像前向，有限数值证据不代表全输入/bitwise等价。

## 预检、资源与成本

从完整去重候选key均匀选64个pair，选择不使用答案或准确率。每role原生和缓存各3次重复：max logit abs≤.125、relative RMS≤.01，两role缓存logit最大差≤.125；这是BF16有限输入工程合同。allocated≤65GiB，按缓存64pair中位吞吐估算每role匹配≤3600秒才推进全量；不据完整dev/test调预算或阈值。

先声明输入/预检配置和准备源锁，真实预检通过后冻结完整STUDY_INPUT_LOCK及协议SHA，再做完整dev评分。使用既有共享UUID flock：A10080、无compute进程、至少40GiB空闲、GPU利用率<10%，连续三次15秒采样；不抢占或终止已有任务。仅本轮自己创建且失败的CPU模型下载器允许在所有权核验与审计后恢复传输，不涉及GPU任务。

分别报告CPU输入准备、三CLIP模型候选生成、vision编码、实际唯一pair匹配、排序、排队与完整壁钟；model加载/缓存IO与64pair工程预检单独识别。实际批量匹配吞吐不当线上p95/SLA；缓存历史baseline的0新增计算不当服务召回成本。所有耗时保留GPU与资源竞争记录，不称独占跨硬件加速。

## 完整评测与门槛

dev三个新rerank格，各1000image query和5000caption query，复用三个对应原排序格；三旧模型不是三次新BLIP训练。保存R@1/5/10/50、binary nDCG10、Top50及候选原位置匹配分数；独立检查旧Recall精确/nDCG在8×FP32epsilon一致性界、候选集合和R@50不变、全去重key完整且无重复缺失。NumPy数值只做JSON类型转换，不改分数。

两主端点：T2I R1至少+1pp且6000配对图像簇bootstrap单侧97.5%下界>0；I2T R1下界>−.5pp表示条件非劣。各alpha=.05/2；另至少2/3旧CLIP seed的T2I同向、两方向nDCG均值点值各≥−.5pp、各roleallocated≤65GiB。nDCG点值保护不表示显著非劣。同图5caption先平均，再对固定三个模型平均的逐图差采样；三seed样本SD另列，不能把seed×image当独立或将条件CI当所有训练方差。

全部dev门槛通过才以原配置追加完整历史test三rerank格，复用旧baseline三格。否则停止当前候选，不追加test/seed/训练/混合权重，不放宽门槛。历史dev/test已曝光、预训练重叠未知，不称盲测/无泄漏/SOTA/业务收益/视觉实体长尾已改善。

## 代码学习与来源

独立src/clip_retrieval_rerank：common缓存前向/稳定重排，prepare输入与旧文件绑定，candidates旧Top10复现与Top50联合key，matcher原生预检/逐块评分，analysis主门槛与质量成本，pipeline空闲准入及dev/test顺序，learning实际图文/分数/排序展示。学习页以真实原图、processor384展示、匹配logit差、Top50命中上限和实际Top5解释两阶段；展示图不冒充attention。

来源：Salesforce BLIP官方models/blip_retrieval.py与模型卡、固定Transformers4.51.1实际源码指纹，均保存在本轮来源记录。图表按scientific-visualization守则记录原始来源/变换，种子SD/条件CI、零基线与病例选择分别标注。模型、特征、原图和匹配数组留在本地，源码/协议/轻量指标可按用户后续授权发布。
