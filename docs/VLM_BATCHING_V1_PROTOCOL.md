# 保持质量的离线批量推理研究

2026-10-13，独立 `vlm_batching` / `vlm_batching_v1`；当前为预注册方案，完成状态以本地输出为准。此研究服务多模态项目的工程优化与评测能力，不新增独立项目、不替换底座、不修改简历、不启用 CI。

## 岗位与研究问题

[阿里AI导购研究实习JD](https://campus-talent.alibaba.com/campus/position/199909680005)涉及业务适配和实际模块开发，[大模型评测日常实习JD](https://campus-talent.alibaba.com/campus/position/199909720005)涉及评测数据、自动化和错误归因。2026-10-13再次取得官方搜索索引正文；不能据索引确认仍在招或个人完全符合资格，两者渠道与毕业窗口分别核对。批量推理是模型工程能力的补充，不替代业务、SFT或部署要求，也不声称多个合格JD都明确要求长度分桶。

本项目已有 SFT/正式指标，但历史 batch4 的高视觉预算吞吐约1.8QA/s。问题：同 checkpoint 与逐题实际输入不变，较大批量和长度分桶能否提高离线吞吐，并满足声明的质量保护？之前 OCR 两个方案为负结果，本轮不增加 OCR 信息、不追加训练。

## 算法与对照

批内长度为 `L_i`，补齐槽位 `B×max(L_i)`，有效槽位 `sum(L_i)`，输入 padding 比例为 `1−sum(L_i)/(B×max(L_i))`。这是形状代理，不能冒充完整 FLOPs、KV cache 浪费或注意力。按已有锁定 processor 的输入长度排序，再切成 batch；调度函数只接收 QA ID、长度与策略，不接收答案、得分或错误标签。先验长度元数据来自先前 CPU 输入合同，历史573.416秒准备成本单列，新准备成本另记。

| 同卡组 | 完整dev条件 | 可分离的比较 |
| --- | --- | --- |
| batch16 | FIFO4 / FIFO16 / bucket16 | 增大batch；同batch下长度分桶 |
| batch32 | FIFO4 / bucket32 | 较大batch和排序的联合效果；不能单独归因分桶 |

每格1617QA/1024图，共8085条新评分预测；两个组独立在两张空闲GPU上运行，各组必须重跑FIFO4基线，不直接比较不同卡的时间。时序保持预设，不据质量反馈选策略。

固定普通CE seed42最终adapter、Qwen2.5-VL-3B revision `66285546d2b821cf421d4f5eb2576359d3770cd3`、原权重BF16、SDPA、TF32关闭、原生整图、视觉预算上限1024、无截断、原问题与固定后缀、贪心最多32新token、最后位置lm_head切片。每次处理重新核对实际input IDs、像素SHA、image_grid、视觉token与历史baseline相同，仅padding及排列/批量改变。排序后的输出按question ID配对，不能按行号配对。

## GPU预检、计时和恢复

每策略的首个完整batch与最长输入完整batch各进行有限logits/生成/显存检查，不评分；超过65GiB allocated停止该候选，不自动缩batch或改协议。GPU准入为无compute进程、至少40GiB空闲、利用率低于10%、连续3次15秒快照、共享UUID锁，不抢占或终止已有任务。每批JSON持久化，恢复校验来源/权重/数据/策略/包/原GPU，只有未完成末尾JSON可修复。

每格先额外预热一批，然后计量完整计划QA。另取numeric QA ID最前256题，按三个预设轮次的策略顺序重复运行，共3840条不评分的计时观察；不是3840个独立样本。轮次轮换顺序，逐轮计算候选/同卡基线吞吐比，显示全部三次观察与中位数。纯生成、预处理+H2D+生成、完整wall、load/preflight/额外预热及allocated/reserved分别记录。没有网络、排队、TTFT、在线并发或服务SLA证据。

## 冻结门槛

三候选 `fifo16/bucket16/bucket32` 均相对本组FIFO4。TextVQA十参考LOO共识重计分；图像整簇配对bootstrap6000次，seed20261013；整体非劣界限0.25pp。三个预设比较使用单侧alpha=0.05/3，下界必须严格高于−0.25pp；同时保留探索双侧95%区间。此为固定checkpoint/既有dev上的条件非劣筛选，不是盲测泛化或等价/无损证明，不消除历史探索选择偏差。

推进还需：实际 `train_all` 监督target rare与frequent点值相对基线至少−0.25pp；三次配对阶段吞吐比中位数至少1.5、每次至少1.1；峰值allocated≤65GiB；资源快照无竞争或查询失败。切片点值保护不证明切片非劣或改善。未通过质量保护时即使加速也不作为成功优化写入简历；不自动追加正式集或三训练种子。

## 学习与开源依据

源码：`src/vlm_batching/schedule.py`、`evaluate.py`、`analysis.py`；复用 `src/vlm_deep/common.py` 的左padding与官方共识契约，以及冻结BF16加载器。已有版本为Transformers4.51.1；[该版本官方pipeline文档](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/pipelines)说明批量收益随序列长度与内存而变化，本项目实测不直接移植其示例结果。

完成后生成真实首批padding形状、五格成本图、可下载CSV、三轮计时、配对区间和8个预先按QA ID选取的原图/预测案例。padding图是输入形状重建，不是注意力；显示图片仅缩小副本，模型输入仍使用原图。掌握检查：解释mask与计算浪费；实现标签无关分桶；解释BF16批量改变为何可能改答案；区分非劣/等价与QA/s/单请求延迟。实验由助手完成不自动等于个人独立贡献或熟练掌握。

图表审查采用 scientific-visualization 技能，软件方法出处：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。已核对当前arXiv记录v2，引用不附版本号；该文仅为图表工作流依据，不是本实验性能证据。
