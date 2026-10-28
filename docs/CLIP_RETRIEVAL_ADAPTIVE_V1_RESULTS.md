# 自适应图文重排序：完成结果

已完成0新神经网络训练、3旧CLIP checkpoint＋1固定BLIP、6新完整dev预测格和两GPU role各3轮实际计时。阈值只用SHA划分的500cal图选择；主质量在另外500confirm图/2500caption查询＋500图像查询评估，图库仍1000图/5000caption。全部6格有36000条model×query输出，确认统计用18000条重复模型输出，不是独立样本。

双向R@1条件非劣通过；真实去重pair减少21.88%，缓存matcher＋route中位加速1.277×。预设至少30%pair削减未达，联合gate=false，没有新增test；不得把该结果写成通过全部协议或全任务无损。

## 确认集质量

| 配置，500确认图，三旧checkpoint均值±SD | 文搜图R@1 | 图搜文R@1 |
| --- | ---: | ---: |
| full | 86.587 ± 0.023 | 94.400 ± 0.000 |
| adaptive | 86.533 ± 0.061 | 94.400 ± 0.000 |

adaptive−fresh full的T2I差−.053pp，95%条件图像簇区间[−.400,.320]，单侧97.5%下界−.400>−.5；I2T差0，当前所有对应R1样本一致，条件bootstrap区间[0,0]。两项预先定义.5pp非劣均通过，不能解释为未知总体严格等价或零损失。

T2I/I2T nDCG10点变化−.127/−.046pp，满足−.5pp点保护，未做nDCG非劣证明。Recall@5/10/50等完整指标单列于JSON，不合成总分。三旧CLIPcheckpoint的样本SD与固定模型平均后500图像簇/6000次bootstrap区间分开；双向NI单侧alpha=.05/2，不能把seed×image/五caption当独立样本。dev/test历史曝光，预训练重叠未知，SHA确认集不是新盲测。

## 实际计算与速度

| 成本口径，完整1000/5000图库、共享三checkpoint | full | adaptive |
| --- | ---: | ---: |
| 逻辑候选pair请求 | 900000 | 677000 |
| 真正独立forward的unique pair/质量预测 | 312780 | 244340 |
| 三轮parallel缓存matcher阶段中位秒；adaptive含route | 75.071 | 58.820 |

实际unique pair削减21.881%，配对三轮速度比1.277153/1.276753/1.275876，中位1.276753×；各轮同role同card、pair batch32。route＋union CPU .195秒已加入adaptive时间。两role各实际full质量forward156390、adaptive122170，并额外各完整重复2遍计时；无历史ITM分数查表冒充节省。计时不是新的独立质量样本。

三轮总质量/计时forward为3×(312780+244340)=1671360pair，另外原生/缓存64pair预检共256pair；这些重复不能当独立训练样本或QA。每seed单独去重pair削减25.465/24.816/25.486%，共享三checkpoint union的21.881%不是单模型线上成本。

vision由本轮每role新编码1000图，full/adaptive共享公共视觉缓存。模型加载/旧输入准备/vision/原生预检/排队/排序/研究壁钟与matcher阶段分开；本轮整体控制器519.18秒，约8分39秒，包含三轮full/adaptive，代码开发与此前CPU检查另计。没有测冷启动、vLLM/SGLang、线上p95、工业商品/中文文档/百万图库。allocated峰值每role2.851GiB，含原生预检，不能称adaptive独有显存收益。

## 路由及未达成本目标的原因

原理：margin=原FP32 CLIP top1 cosine−top2 cosine；margin≤τ才执行BLIP，否则原CLIP Top50直接返回。它是启发式margin，不是已校准正确概率。T2I τ=.07362494，I2T τ=.11425933，两方向共用跨三checkpoint阈值；参考答案/标注不进入route_mask或实际matcher。

固定cal grid中，T2I重排65%时calR1−.280pp，超过允许损失.1pp；70%时+.040pp才可行。I2T重排95%时calR1−.467pp、nDCG−.830pp；cal全部重排时才可行。实际dev约70%T2I和999/1000 I2T查询仍重排，因此节省有限。五正caption的I2T前两名可能都正确，top1−top2不是完整排名质量的充分证据。此解释由规则/扫描支撑，但未证明所有失败的唯一原因。

查询跳过比例不等于unique pair削减比例：图搜文仍需要匹配、多个查询/方向/checkpoint共享pair；full/adaptive全局去重后剩244340真实pair。达到当前质量保护所需的匹配覆盖与30%成本目标存在本固定配置下的取舍，不降低既定门槛。

## 真实代码、图片与复核

路由见[route.py](../src/clip_retrieval_adaptive/route.py)，校准见[calibrate.py](../src/clip_retrieval_adaptive/calibrate.py)，实际forward/分块恢复/同卡计时见[matcher.py](../src/clip_retrieval_adaptive/matcher.py)，主统计见[analysis.py](../src/clip_retrieval_adaptive/analysis.py)。

本地学习页：`rarevlm_pilot/outputs/clip_retrieval_adaptive_v1/learning/学习页.html`。有真实cal扫描表、实际margin/阈值/route、保存的改进/退化/保持样本及CLIP/full/adaptive实际Top1图片。照片仅展示缩略副本；文搜图输入是caption，图库照片不能冒充query输入、模型384/224预处理或attention。独立展示增强工具只读已保存排名，0新预测、0改源协议。

例：seed42 query3444描述黑衣与黑白帽男子，CLIP top1/top2=.299785/.196690，margin=.103095>τ，走fast路径；CLIP返回灰衣黑帽男子，完整BLIP返回黑衣浅帽男子的标注配对图。置信度大仍可漏掉细粒度服饰属性，图文实际对照比抽象解释更直接。其他图在部分模糊描述下也可能语义合理，标注相关性与所有语义相关性不等同。

10路由/重排合同（含修改confirm分数及margin不影响cal选择）通过；三checkpoint原完整Top50精确重现，64实际pair两卡原生/缓存预检logits差0。全部实际pair覆盖/unique预算/fast路径保持/Top50候选与R50不变、六格独立标量Recall/nDCG重评分均过；277历史科学/模型/输入/正式简历文件哈希保持。两GPU matching阶段0资源查询失败/0竞争观察，未终止其他任务。

[预注册协议](CLIP_RETRIEVAL_ADAPTIVE_V1_PROTOCOL.md)、[完整便携指标](../results/clip_retrieval_adaptive_v1_metrics.json)、[独立复核脚本](../tools/review_adaptive_results.py)、[图像展示增强脚本](../tools/enrich_adaptive_learning.py)。其他机器需要同版本旧资产及绑定路径，不能把原机命令当无资产可直接运行的示例。

## 后续候选（未执行）

若继续优化，优先在cal上检查每个重排query的候选预算，例如Top20/50分层；当前仍有约70%文搜图query各算50pair，这比继续增加模型训练更直接对应成本瓶颈。先核对cal中原full已正确命中的正项保留率，再独立预注册新路由/确认规则，不用本轮confirm/test调当前τ或把未执行方案写成收益。正式简历不变，不启CI、不自动push。
