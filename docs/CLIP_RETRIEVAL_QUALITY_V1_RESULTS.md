# 图文数据一致性筛选：完成结果

对应多模态数据管线、图文表示训练与科学评测；独立12次完整训练，原始数据提准与受控错配鲁棒性分别验收。

[冻结协议](CLIP_RETRIEVAL_QUALITY_V1_PROTOCOL.md) · [便携指标](../results/clip_retrieval_quality_v1_metrics.json)

## 公平比较与工程规模

完整28000图；每图固定保留4条，112000描述／1120000正描述曝光、10epoch／8750更新／132480参数；Final@10、meanCE+KL、三seed{0,42,200}。随机／分数两组在各分支的初始化、负样本序列、曝光和更新hash严格一致。原始与错配分支采样排除可不同，跨分支不作纯噪声单因素归因。

| 划分 | 条件 | 文搜图R1均值±SD | 图搜文R1均值±SD |
| --- | --- | ---: | ---: |
| dev | clean_random4 | 69.673 ± 0.412 | 84.567 ± 0.643 |
| dev | clean_score4 | 69.900 ± 0.312 | 84.967 ± 0.513 |
| dev | noisy_random4 | 69.073 ± 0.496 | 85.133 ± 0.945 |
| dev | noisy_score4 | 70.233 ± 0.110 | 84.600 ± 0.265 |
| dev | 历史clean-full5，监督量更多 | 69.907 ± 0.343 | 84.733 ± 0.404 |

## 主比较与结论边界

- **clean dev score4−random4**：T2I +0.227pp，单侧98.75%下界-0.300pp；I2T +0.400pp、下界-0.633pp；预设全门控False，T2I正向seed3/3。
- **clean 未通过dev全门控，没有本轮新test**。不回调保留数、噪声／分数／门槛或checkpoint。
- **noisy dev score4−random4**：T2I +1.160pp，单侧98.75%下界0.513pp；I2T -0.533pp、下界-1.934pp；预设全门控False，T2I正向seed3/3。
- **noisy 未通过dev全门控，没有本轮新test**。不回调保留数、噪声／分数／门槛或checkpoint。

区间按图像簇、固定三checkpoint均值计算，T2I每图五caption先聚合；四主端点alpha=.05/4。seed样本SD单列。历史dev/test已曝光、预训练重叠未知，不是盲测或全部历史多重比较校正。

## 筛选诊断与代码原理

训练集每图置换1/5描述，总28000条来源配对改变。random4保留22446条（20.041%）；score4保留440条（0.393%），排除98.429%已知置换。语义相似置换可能仍合理；这些是受控来源诊断，不能当原始真实错标率。

`selection.py:select_slots`只读实际train特征／图SHA／槽序；s=image·text，取最高4条。相同教师用于筛选与KL，分数同源偏差存在。原始数据上的低分描述可能包含真实细节，不能把低分自动叫错标。`loss.py:selected_terms`保持meanCE＋KL，I2T[32,384]、T2I[128,96]；未选描述不进入教师或监督。诊断flags／donor只在选择后检查，不用于修复训练标签。

## 工程与学习

12次训练active秒：总1047.48，单次范围84.84—93.63；筛选CPU1.04秒；首次启动至完成壁钟1098.21秒（含训练、两卡队列、评测和JSON修复；不含启动前代码准备）。恢复controller壁钟1.04秒单列。allocated峰值最高0.344GiB。不是完整CLIP预训练速度。
5项数学／边界合同、四真实训练batch梯度与manifest对齐通过；12份完整图库预测由独立Python标量Recall/nDCG复核，245个历史科学文件／正式简历哈希保持。0资源查询失败、0观察到的竞争，不抢占现有任务。
12份dev完成后原统计JSON保存遇到numpy.bool_错误；独立`tools/resume_quality_report.py`只在报告边界转换NumPy scalar，原冻结源码／计算公式／门槛／模型和预测不改、原失败日志保留，0重训练／0重复dev预测。复现或恢复本协议时用该wrapper调用冻结pipeline；helper与原来源锁另行归档。

真实图文／筛选／改善退化学习页：`/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/clip_retrieval_quality_v1/learning/学习页.html`。图片为来源原图缩略图，实际CLIP输入仍源协议224预处理；不是attention。个人掌握与语义案例审查范围另记，不自动当本人独立贡献。

## JD与项目使用

字节[Seed官方方向](https://seed.bytedance.com/en/seedearlycareer)明确多模态数据管线与科学评测；阿里[国际站多模态实习原文](https://career.nankai.edu.cn/correcruit/content/id/118629.html)强调图文表示／实验分析。腾讯既有社招数据质量要求仅作技术参考。本实验支持图文数据筛选的对照与诊断能力，不代表工业中文商品、生成式数据合成、CPT/RL或在线服务已完成。
参考[DataComp](https://github.com/mlfoundations/datacomp)固定架构／预算及CLIP score过滤思想，revision与参考文件SHA已归档；没有执行其预训练框架或复现其benchmark。正式CV保持；针对数据质量岗位可先准备该研究原理／负结果／压力测试与强基线边界。
