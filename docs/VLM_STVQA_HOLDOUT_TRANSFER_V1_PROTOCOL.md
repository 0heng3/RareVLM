# 四个固定Final的ST-VQA来源保持验证 v1

登记于2026-10-28，全部新模型预测前冻结。上一研究的预定DPO主候选未过门槛；次要SFT在原1617内部题上对起点+.828695pp，对历史强DPO仅+.395795pp且区间跨0。原停止决定保持，本轮不是同配置扩步、扩seed、调权重或改名续训，而是验证此前未测的来源保持泛化。

## 问题、数据与模型

固定`vlm_stvqa_headroom_v1/DATA.json`中全部256个`source_holdout` QA/物理场景，不另采样或挑错题。它们未进入128个ST-VQA训练槽位，与旧/新完整训练图的原始文件及解码像素均隔离。来源是固定TheCauldron第一分片与MMF subtrain匹配的原尺寸子集；不重新下载、不扩候选。此前已知共同起点195/256、正文NLL .471961，因此这是来源内部保持集，不是新盲测或官方ST-VQA测试集；近重复、预训练重叠未知。

四个模型在看到本轮结果前固定为`vlm_visual_preference_v1`的CE与CE+DPO Final32，以及`vlm_stvqa_replay_v1`的CE与CE+DPO Final32。共同复合起点的256条生成与GT前向结果按SHA精确复用，不当独立重复。所有角色的语言/视觉/基座权重全部冻结，0optimizer、0新adapter。原共同起点、视觉参数预算、seed42、32更新等训练事实不改；本轮不重新训练。

## 复用与资源

直接复用冻结`run_stvqa_headroom_v1.worker`和`run_stvqa_headroom_source_adapter_v1.source_generated`，仅绑定每角色新ROOT、原Final路径及256题子集。保留64个原4题batch、shard0/1分组；每角色两worker各128题，8个job经共享队列在两张空闲A100接续。原图/CPU编码SHA、逐题prefix/grid/chosen/labels、1024视觉预算、greedy32/真实EOS、imagewise BF16视觉/FP32语言与LoRA、TF32off保持。生成函数只读问题与图像；GT随后用于NLL和literal exact，不添加10份假参考，不调用TextVQA投票评分。

总计1024新生成+1024 GT前向、0更新；基线256+256精确复用。每worker沿原上限900秒/allocated65GiB；零compute/≥40GiB/<10%/3×15秒/UUID锁，不终止其他任务。按上次192题约230秒估计本轮共享两卡10–15分钟，实际以PROGRESS与队列为准，排队另计。

## 固定统计与解释

来源exact采用原NFKC/大小写/空白/外端标点规范化，保留冠词，不冒称官方ANLS或原生VQA。正文NLL与EOS NLL各逐QA平均，报告32token删失。

exact四项比较事前固定：新CE−起点、新CE−旧CE、新DPO−新CE、(新DPO−新CE)−(旧DPO−旧CE)。按物理图像整簇6000次bootstrap，固定seed20261021，同时报告条件95%和四比较Bonferroni对应的98.75%区间（双侧分位.00625/.99375）。它们只涵盖本次四个来源端点，不校正所有历史选择或涵盖种子方差。正文/EOS NLL仅描述性次要指标；单/多词切片描述，不混同原native rare342。

新CE对旧CE的exact正向且校正区间下界>0，支持本固定来源子集的数据策略泛化；NLL下降而exact不确定，仅称似然信号。新DPO对新CE的exact校正区间上界<0，支持来源保持损伤；若源内变好但原native变差，只支持任务取舍，不能称全面失效。无论结果如何，本研究不改变原DPO主候选/原门槛，不因次要结果挑checkpoint或启新训练。

固定学习例为256题顺序前4条，展示真实原图、问题、原参考及五个模型保存回答；不进行新人工真值重标，不将图像展示当注意力。

## 停止

八worker完整结束即汇总并停止。任何预算/身份/运行异常保留原失败，只修对应故障；不重训、重挑题或回调统计门槛。完成后仍需新的不同监督/新数据依据和独立协议才考虑未来训练；原完整强基线/三seed/外部目标保持未完成。简历/科研/Agent/CI不改、不push、不动其他GPU。
