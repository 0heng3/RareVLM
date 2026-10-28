# 0／20／50候选预算：完成结果

按阿里Embedding/Rerank及检索链路效率要求完善RareVLM。固定三个旧CLIP distill checkpoint与一BLIP，0新神经网络训练，只cal500图选择20／50 cutoff，旧fast阈值不变。dev确认500图、test1000图均通过全部预注册质量／成本／速度门槛。12新全gallery预测格共72000model×query不是独立样本；三个旧CLIP轴不是三次新BLIP训练。

## 固定历史test结果

| 配置，三旧checkpoint均值±样本SD | T2I R@1 | I2T R@1 |
| --- | ---: | ---: |
| 完整Top50 | 85.093 ± 0.023 | 95.067 ± 0.058 |
| 0／20／50预算 | 84.887 ± 0.012 | 95.067 ± 0.058 |

T2I−.207pp，条件图像簇95%区间[−.473,.060]；I2T差0、当前逐query对应R1相同。两方向单侧97.5%下界>−.5pp，预设条件非劣通过。T2I下界距离−.5仅.027pp，不能称严格无损、更严格.25pp质量保持或显著提准；bootstrap[0,0]也不证明未来总体差为0。T2I/I2T nDCG点差−.345/−.062pp满足点保护，未做nDCG非劣证明。

已历史曝光dev/test、未知预训练重叠；策略只cal选择，新test固定一次。三checkpoint先平均再按500/1000图像整簇6000bootstrap，双向NI各alpha=.025；seedSD另列，不把seed×image或五caption当独立样本，也不声明全历史探索FWER。

## 实际计算与同卡速度

| 数据／口径 | full | budgeted | 削减／速度比 |
| --- | ---: | ---: | ---: |
| dev实际unique pair | 312780 | 208535 | −33.33% |
| dev缓存matcher＋route三轮阶段中位秒 | 75.490 | 50.563 | 配对速度比中位1.493× |
| test实际unique pair | 312116 | 210397 | −32.59% |
| test缓存matcher＋route三轮阶段中位秒 | 76.031 | 51.498 | 配对速度比中位1.476× |

两个条件同role同GPU、batch32分别fresh forward；旧完整scores仅用于cal模拟，不查表伪造实际节省。test逻辑请求900000→590100，unique312116→210397；shared三checkpoint缓存不当单模型线上成本。全部三轮记录保留，计时重复forward不是新质量样本；公共vision每role约1.9秒、加载/旧输入准备/路由/排序/排队/完整壁钟分开，不称线上p95、冷启动或训练加速。allocated峰各role2.851GiB含原生预检，不作预算独有省显存。

整个研究控制器950.08秒，约15分50秒，含dev/test及每role三轮full/budgeted，代码开发／之前CPU检查另计。全部matching资源观察0查询失败/0竞争，未终止已有任务。

## 原理、候选覆盖与学习

固定fast τT2I .07362494、I2T .11425933；新cal-only κT2I .04284247、I2T .05803094。margin>τ直接原CLIP；否则margin<κ用Top50，剩余用Top20。Top20只排序原prefix，尾30保持原CLIP，不伪造ITM分数。

原返回Top50成员/R50保持，不等于全部50可竞争Top1：test T2I有效prefix上限99.047→97.907%，I2T99.833保持。K0有效上限是原Top1命中，K20/K50是对应prefix含正项；只预测后评分，不向路由传参考。cal里full已正确命中T2I的正项Top20保留率仅98.664/98.710/98.985%，所以不把所有query统一裁20。旧fast route单独节省21.88%未达30%，本轮在质量约束下缩减部分query的实际候选数。

源码：[预算分配与key](../src/clip_retrieval_budget/route.py)、[cal-only选点](../src/clip_retrieval_budget/calibrate.py)、[实际评分接口](../src/clip_retrieval_budget/matcher.py)、[质量／有效prefix／成本统计](../src/clip_retrieval_budget/analysis.py)。复用旧已冻结matcher，新增模块/配置/根目录，旧研究不改。

本地学习页：`rarevlm_pilot/outputs/clip_retrieval_budget_v1/learning/学习页.html`，真实margin／预算／cal保留率、原图和Top20改进退化／Top50病例。图片为源图缩略展示，不是224/384输入或attention；文搜图输入是caption。test汇总由独立展示补充，不改冻结科学源码／policy／预测。

## 验证与岗位表达

15数学／路由合同通过，含确认数据不影响calκ、prefix尾部保持、实际key单调预算；原fullTop50精确重现，两卡64实际pair原生/cached0差，实际pair覆盖与三个完整timing合同通过。12格保存排名经独立标量R@1/5/10/50与nDCG复算，288科学/模型/输入/正式简历保护文件保持。辅助复核脚本首次仍指向旧协议路径而失败，已只修正该入口、保留REVIEW_RECOVERY审计，0重预测/0门槛改变。

[固定协议](CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md)、[便携指标](../results/clip_retrieval_budget_v1_metrics.json)、[独立复核](../tools/review_budget_results.py)、[JD技术对齐表](JD_EVIDENCE_MAP.md)。核心可表述为质量约束下的候选预算优化，test实际pair−32.59%、缓存匹配＋route1.476×、双向.5pp条件非劣；不能称准确率提升、工业中文商品、文档检索、在线高并发、CPT/RL或新原创大模型架构。

正式简历不变；项目事实候选需按用户实际掌握/贡献使用。近期优先讲清对比学习/KL、双塔与ITM、预算/候选上限、统计非劣及真实失败图，不能因实验完成自动声明本人熟练。
