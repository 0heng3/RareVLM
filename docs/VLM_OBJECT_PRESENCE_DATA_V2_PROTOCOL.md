# 可见动物／交通工具短答数据 v2

2026-10-15。在新manifest冻结、新64例抽样/看图及任何新模型结果前登记。本v2是在v1固定64例57明确、7不确定、Wilson95%下界79.101%而拒绝后的不同机制研究，不重抽同一总体、不覆盖旧审查、不降低90%/80%门槛。旧数据工程通过与语义拒绝均保留。v2也不保证通过或提升。

## 为什么改变监督任务

v1的微缩食物图、媒介封套、杯型、暗轮廓和极小球暴露了闭集标签与可见语义的差距。v2只研究现实常见实体的动物/交通类别，并加入事前可见尺寸要求；用全部符合规则的图片/类别，不依据模型预测、验证/POPE结果挑图或挑标签。最终研究目标仍为完整9000题POPE和TextVQA保持，不以这个较窄内部验证代替成功。

固定18类：bird、cat、dog、horse、sheep、cow、elephant、bear、zebra、giraffe；bicycle、car、motorcycle、airplane、bus、train、truck、boat。交通类别仍可能有bus/train等口径边界，动物也可能是玩具/绘画；本规则只降低部分风险，不能证明标注真值。

## 来源与边界

继承v1已验证的官方COCO 2017 JSON、公开RLHF-V图像池、原固定SHA训练/留出划分与所有heldout边界。不是COCO全训练集，也不是COCO官方val划分，不使用公开偏好chosen/rejected文本作为标签。官方来源与SHA见[原数据协议](VLM_OBJECT_PRESENCE_DATA_V1_PROTOCOL.md)和`outputs/vlm_object_presence_v1/SOURCE_LOCK.json`；新的SOURCE_LOCK还绑定本v2协议、v1固定审查/完成收据、新源码、实际图像和processor，旧文件不改。

train/dev/final、DocVQA、MMStar及所有POPE500图的精确file/pixel SHA与COCO ID边界继承并核验。另从v2整个候选池统一排除v1已审查的全部64个不同pixel图像，不仅排除7个不明确图。不根据旧64逐例正确性作额外例外。近重复、预训练重叠、缩放/裁切风险未知，POPE历史暴露明确披露。

## 标签与固定采样

1. 在固定18类内，一类别至少有一个同一noncrowd实例同时满足：bbox按当前实际图像尺寸缩放后短边>=24 pixel，COCO segmentation_area / 官方图像面积>=0.005，才进入yes。每图保留全部符合条件的唯一正类。bbox只用于准入与展示，不裁剪模型输入；原整图不变。
2. no候选必须在该图全部annotation（含crowd、小实例）中完全缺席。未过尺寸的正类绝不能变成no。每图按新salt SHA选与yes相同数量、互不重复的absent类别。若无正类或negative不够则排整图，不截取部分正类救数量；事前统计不足负类图为0。
3. negative salt=`object_presence_v2_negative`，audit salt=`object_presence_v2_audit`，audit_order salt=`object_presence_v2_audit_order`，seed42；qid命名空间`object_presence_v2:COCO12:categoryXX`。实际实现的SHA串联与固定测试同锁，不在审查后改salt。
4. 问题仍`Is there a {COCO category} in the image?`，后缀换行`Answer yes or no.`，target lowercase yes/no。一图一类别只一题，不以重复QA增加数据量。

事前统计：训练1211图/2888QA（1444yes+1444no），内部验证143图/342QA（171+171）；18类两split均有正例。基于v1池，训练另有2064图/验证280图无合格正类而排除；训练旧64审查图统一排除。最终以冻结manifest核验这些统计，数量冲突则先停止查明，不静默改规则。逐类别正负均衡子集从完整val按固定SHA取min(yes,no)的两类；没有双类的类别不进入子集，不能据结果重新取样。

## 工程与质量准入

- 3230条全部fresh调用冻结Qwen processor/assistant mask，记录实际像素、input IDs、grid、视觉token与监督计数；0 model forward。不能把v1复用记录称为新编码；本v2直接全量新编码。视觉上限1024、总序列2048，不截断，全部完成无冲突。
- 新训练manifest先冻结，再选32yes+32no、64distinct pixel图像；与旧64无交集，按上述新salt SHA确定，不看模型分数或图像内容挑选。
- 助手逐图审查记录clear/ambiguous/incorrect及依据；全部64图看contact sheets，有疑问查看原尺寸。至少90% clear且双侧Wilson95%下界>=80%才通过；ambiguous不计clear。不得换样、把审查小集专门训练、下调阈值或用诊断旁路准入。
- 通过工程和语义两关后，建模研究独立DATA_ADMISSION绑定真实train文件SHA、本协议SHA、新固定64/审查/CPU完成收据，`quality_gate_passed=true`且`automatic_diagnostic_bypass_allowed=false`。准备完成仍不自动放行。

本研究的助手视觉审查不是独立人类全量真值；即使通过也不能声称零漏标、无偏、全80类覆盖或模型已提升。v1与v2总体不同，审查比例不能当配对质量提升。
