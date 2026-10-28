# CLIP Top50召回与BLIP匹配重排：完成结果

2026-10-14完成。依照[冻结协议](CLIP_RETRIEVAL_RERANK_V1_PROTOCOL.md)，固定三个旧CLIP蒸馏Final@10 checkpoint，新增一份COCO匹配BLIP（revision bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c）；0新训练。追加384图像、64文本上限、BF16模型容量和计算，按ITM logit1−logit0仅重排原Top50，不混合分数、不扫超参。

## 完整质量结果

每划分1000图/5000caption，三已有CLIP模型共三个新重排格，复用三个旧基线格。±为三个旧CLIP checkpoint的样本SD，共享同一固定BLIP；不是三次新BLIP训练。新dev/test各18000个model×query，不能视为18000独立样本。

| 划分/条件 | 文搜图R@1（%） | 图搜文R@1（%） | 文搜图nDCG@10（%） | 图搜文nDCG@10（%） |
| --- | ---: | ---: | ---: | ---: |
| dev原CLIP蒸馏 | 69.907 ± 0.343 | 84.733 ± 0.404 | 82.702 ± 0.184 | 79.671 ± 0.162 |
| dev加BLIP重排 | 86.687 ± 0.042 | 95.667 ± 0.058 | 92.880 ± 0.022 | 91.020 ± 0.088 |
| test原CLIP蒸馏 | 67.820 ± 0.122 | 79.833 ± 0.404 | 81.423 ± 0.070 | 76.705 ± 0.195 |
| test加BLIP重排 | **85.093 ± 0.023** | **95.067 ± 0.058** | **92.104 ± 0.028** | **89.662 ± 0.095** |

| 新配置减同checkpoint基线 | 差值pp | 条件图像簇95%区间/单侧97.5%下界 |
| --- | ---: | --- |
| dev文搜图 | +16.780 | [15.220,18.3135] / +15.220 |
| dev图搜文 | +10.933 | [8.833,12.933] / +8.833 |
| test文搜图 | **+17.273** | **[15.7265,18.880] / +15.7265** |
| test图搜文 | **+15.233** | **[13.033,17.467] / +13.033** |

两划分全部预设门槛通过：T2I≥+1pp且下界>0；I2T下界>−.5pp；至少2seed文搜图同向；两方向nDCG点值保护和资源上限。三旧模型test两方向均同向提升；未进行新重排训练。6000次图像簇bootstrap、两主端点alpha=.05/2；同图5caption先平均，再对固定三CLIP+一BLIP逐图差计算条件区间，训练重采样不确定性未包含，不当seed×image独立。

原Top50集合完全保留；test T2I/I2T的R@50上界分别约99.05/99.83%。因此增益来自本配置池内重排，未改召回覆盖；不进一步全部归因注意力机制，新增模型/训练来源/像素/数值路径共同变化。剩余错误应先复用实际排名核对池内排序及语义/标注歧义，不据test回调K/权重。

## 成本与工程结果

| 指标 | dev | test |
| --- | ---: | ---: |
| 三旧模型/双方向逻辑pair | 900000 | 900000 |
| 全局去重后的实际唯一pair | 312780 | 312116 |
| 两卡各匹配pair | 156390 | 156058 |
| role0/role1匹配阶段秒 | 75.731 / 76.119 | 75.555 / 75.771 |
| role0/role1图像编码秒 | 1.888 / 1.941 | 1.943 / 1.989 |
| 实际评分peak allocated GiB/卡 | 1.464 | 1.464 |
| 完整划分流程壁钟秒 | 252.717 | 247.797 |

两A100并行；逻辑请求与唯一计算分开，不能将900000当独立pair或推理吞吐样本。完整流程含CPU准备、候选再生、排队、64pair×3原生/缓存预检、模型/cache IO、匹配、排序和分析；首次权重传输、代码准备及最终教学/报告修复另计。全量评分缓存gallery视觉latent，不是实时新请求/冷启动成本；不宣称线上p95/SLA/百万图库性能。预检peak allocated2.851GiB，区别于实际缓存评分1.464GiB；两划分六个准入队列观察0资源查询失败/0竞争，不作跨硬件或绝对独占结论。

两role各64个实际pair与原生同输入前向最大logit差/relativeRMS为0；仅有限输入证据，不是全输入bitwise等价。旧三seed两划分Top10逐项精确重现；候选pool/R50不变、完整去重key无缺失重叠；六新重排格由独立Python逐row重算Recall@1/5/10/50精确一致、float64 nDCG误差<1e−12。数据、权重、source/processor/源码/GPU与每2048pair断点SHA归档。

## 运行恢复与结论边界

- 固定模型八文件完整核验；原HF超时/不完整流式下载以仅本轮自建CPU下载器恢复为4MiB HTTP区间、4线程，最终895139697字节整权重SHA通过；没有中止GPU外部任务。
- HF tokenizer缺官方DEC/ENC注册，GPU前词表合同拦截。原准备锁/日志归档revision1，按固定官方init_tokenizer补齐DEC30522/ENC30523，checkpoint原已有30524 embedding行，不resize/改权重；revision2在预检后冻结完整源锁。旧源通过反向精确补丁重建并核对原lockedSHA，恢复过程明确记录。
- 完整预测与统计完成后，学习页直接json.dumps的NumPy bool失败；独立[报告恢复工具](../tools/complete_rerank_report.py)只读归档的native JSON完成教学和指标导出，0重预测、0门槛变更，冻结科学源字节不改。原失败记录保留。
- BLIP模型卡注明COCO ITM训练，未采用Flickr专门微调权重；预训练重叠未知、公开dev/test历史已曝光，不称盲测/SOTA/业务收益/长尾泛化。当前为英文静态图文、1000图库；中文商品、文档检索、视频、线上并发尚未验证。

## 原理、代码与学习页

[common.py](../src/clip_retrieval_rerank/common.py)的cached_logits：文本通过encoder_hidden_states观察缓存图像patch，CLS/[ENC]表征送入2类ITM head；score=logit_match−logit_nonmatch，其sigmoid为匹配概率。双塔独立向量适合快速召回，联合匹配需要逐候选额外计算；候选池没有标注正项时重排不能救回。[候选生成](../src/clip_retrieval_rerank/candidates.py)、[实际匹配与恢复](../src/clip_retrieval_rerank/matcher.py)、[分析](../src/clip_retrieval_rerank/analysis.py)、[控制器](../src/clip_retrieval_rerank/pipeline.py)。

本地`rarevlm_pilot/outputs/clip_retrieval_rerank_v1/learning/学习页.html`包含5个真实查询原图、实际384输入记录的反归一化展示、两配置Top5/完整Top50、logit分数/实际位置、dev/test图和成本。固定前4dev行，加首个改进/退化病例（若存在且非重复），不用选择病例估计发生率；按源图ID标注，其他图描述不自动等于语义错误。3张图与5源图+1实际输入展示已检查，不冒充attention。学习第一个实际查询的原Top5候选ITM分数约4.38，升至Top1；原Top1约−1.70，降至Top6，完整实际值在TEACHING_DATA.json。

[便携指标](../results/clip_retrieval_rerank_v1_metrics.json)、[独立验证与展示工具](../tools/review_rerank_results.py)。图表工作流参考Kassis等（2026）[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。正式简历/PDF保持，完成代码与实验不自动代表用户已独立掌握或个人贡献确认；后续先学习原理/源码与失败病例，再按具体JD调整项目展示。
