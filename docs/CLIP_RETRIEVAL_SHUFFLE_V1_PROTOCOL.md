# 教师负样本关系机制对照 v1

2026-10-14：用户授权按JD分析执行下一步实验。目的为解释已有检索蒸馏提升，继续同一RareVLM项目；不改正式简历、不推进Agent项目。

## 假设与唯一变化

已有教师关系蒸馏完整测试相对原CE的T2I/I2T R@1提升6.180/5.300pp，尚不能全部归于语义关系。本轮比较原distill与negative_shuffle：教师先在同候选池做一次log_softmax，然后每个query只在标注负项间随机置乱log概率。正项固定；置乱是双射，概率和、教师熵与监督/教师混合目标熵保持（浮点容差单列）。不改变候选池、监督loss、KL方向/系数/温度。

每方向 L=CE(P_gt,P_S)+KL(P_T||P_S)=2CE((P_gt+P_T)/2,P_S)-H(P_T)。负项的P_gt为0，因此负项置乱也保留混合目标熵。只破坏“哪一个负候选收到哪一个教师概率”的对应关系。若原教师优于对照，支持负关系信息有用；保留的正项置信度未被隔离，不证明教师全部语义或全部蒸馏收益的因果来源。此对照不是提分候选。

## 固定预算与历史对照

完整既存28k训练图、每图5描述、32锚图+64额外uniform负图；I2T[32,480]/T2I[160,96]。冻结FP32 CLIP-B/32缓存，双ResidualAdapter共132480参数；温度.07、AdamW lr.001/wd.0001、clip1、TF32关闭。仅新增三个negative_shuffle训练，seeds0/42/200、10epoch Final@10、每次8750更新/140万正pair暴露。

从clip_retrieval_distill_v1绑定六个历史checkpoint（mean与distill各三seed）以及两划分各七格排名（frozen+mean3+distill3）。配置/旧源码/权重/特征/训练采样流/排名SHA核验并只读复用；不充作新独立训练。完整数据划分与预处理保留上游锁，dev为源train内部固定1000图，非原Karpathyval。

负概率置乱用独立torch.Generator；种子为training_seed*1000003+update_step+104729。每步先I2T、后T2I；不会消费全局torch/NumPy采样RNG，step可恢复同设备的置乱。CPU/CUDA不同随机流明确区别。每250步保存model/optimizer/step，来源锁和GPU UUID一致才允许恢复。

## 统计与晋级

先完整dev：三个新增negative_shuffle格，复用七历史格，共10格、每格1000图与5000caption。两主端点分别为原distill减shuffle的I2T与T2I R@1；每项均要求均值至少+0.5pp、6000次配对图像簇bootstrap的单侧97.5%下界严格>0、至少2/3训练seed同向。Bonferroni alpha=.05/2，两端点必须都通过；不合成总分。T2I的同图5caption先聚合，再对固定三checkpoint平均按图像簇采样；种子样本SD单独给出。探索95%CI不能替代校正门槛或称seed×image独立。

dev通过才追加三个新checkpoint完整历史test一次，复用七旧test格。test做同样固定机制比较并明确历史曝光；dev失败停止候选，不追加test/seed/epoch、不调阈值。所有eight Recall/nDCG指标、Top10与训练契约留档，独立从Top10重评分。无总体或长尾新提升宣称：这是同一已知基线的机制研究，不新增长尾数据协议。

## 资源、学习与范围

共享UUID flock；空闲A10080、无compute进程、至少40GiB空闲、利用率<10%、连续三次15秒采样准入。动态使用空闲卡，保留已有任务，不抢占或终止外部进程。当前状态写入独立outputs，原源码/结果/正式简历哈希保护。

学习材料展示真实训练图/五描述、完整候选概率、置乱前后对应、P_S-P_T梯度及真实dev检索Top5；缩略图与诊断明确标注，不能冒充attention。结果图零基线、种子原始点与样本SD，条件CI另表，来源与变换记录；图表工作流参考Kassis等（2026）Scientific Agent Skills，https://doi.org/10.48550/arXiv.2609.00065（官方页面2026-10-14核验，当前v2）。

后续Rerank/文档检索另立协议，不作为本轮自动扩展；已有项目成果不代表用户独立掌握或个人贡献已确认。
