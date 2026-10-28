# 高学习率覆盖研究完成

|单元|训练128满分|验证512 VQA%|保持64 VQA%|
|---|---:|---:|---:|
|initial|0|93.222656|93.750000|
|low_repeat32|8|93.417969|93.750000|
|low_coverage128|9|93.085938|93.750000|
|high_repeat32|45|92.226563|93.750000|
|high_coverage128|56|93.085938|93.750000|

联合门槛=False。

主配对：{'initial': {'delta_pp': -0.13671875, 'ci95': [-1.171875, 0.8989257812499928], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}, 'high_repeat32': {'delta_pp': 0.8593750000000001, 'ci95': [-0.546875, 2.34375], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}, 'low_coverage128': {'delta_pp': 0.0, 'ci95': [-0.9770507812500001, 0.9960937499999999], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}}

完整控制/切片/条件效应见ANALYSIS；单seed42内部探索、历史低lr单元复用、非盲测。训练拟合不替代验证能力，原门槛不改。

[协议](VLM_COVERAGE_SFT_V2_PROTOCOL.md) · [原图/四格答案](LOCAL_ARTIFACTS.md)

## 逐题迁移与保持

候选新512四题升分合计+3.7、五题降分合计−4.4，净−.7；同低lr总分不代表逐题相同。5下降中3为完整名称被缩短，原图Miller High Life/Sunshine City/Life Box支持完整文字；蓝W→U明确，ndaine/andaine左侧遮挡仅作评分下降、不认定确定语义错误。原分数不修改，区间跨0不称显著退化。

困难曝光484单词/28多词+回放462单词/50多词，总1024曝光946单词/78多词（92.3828%/7.6172%）；验证各256。该错配与缩短病例提示长度分布问题，但还没有因果结论。初始已满分474验证题合计损失4.4，另38题增加3.7；保持64持平不能替代整个验证域能力保持。

高lr重复32在参与监督32题中满分28，未参与该轮监督96题中17满分（19.5833% VQA）；覆盖128在两子集为14/42。不能把失败概括为完全没有迁移或所有收益只是背题；它们也不是原底座/SFT未见数据。

下一项固定同128困难数据/同高lr64步，比较扩大多词回放覆盖240与原50题等分层总权重重加权。来自原训练fold且初始满分，保持频次桶总权重，不用验证标签补训练。原完整门槛继续保留，并观察原初始已满分多词保持；先登记再启动，失败不扩训。

新预登记：[多词回放扩覆盖与重加权](VLM_LENGTH_REPLAY_V1_PROTOCOL.md)；[原图、权重与真实预测](LOCAL_ARTIFACTS.md)。旧64保持实际单/多各32，持平也未覆盖本轮验证的所有退化。
