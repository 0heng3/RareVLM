# CLIP图文检索与困难负样本研究

2026-10-14。同一个RareVLM中新增`clip_retrieval`，不另凑独立项目。已有分类准确率/VQA共识不能替代检索Recall。新结果未完成不写简历，实验完成也不自动代表本人独立贡献或熟练。

## JD与缺口

[阿里1688团队负责人招聘页](https://liamding.cc/hiring.html)明确涉及Embedding/Rerank，研究实习面向在读硕博、杭州、3–6个月以上；[阿里国际站高校招聘原文](https://jdjywpt.jlu.edu.cn/mportal/recruit/details?id=2378ecb86a1b4a62a544060720416123)涉及跨模态及图文文档检索。资格/具体岗位开放状态仍单独核对，不把团队自述或索引当有效岗位数量。

[腾讯杭州多模态数据官方JD](https://careers.tencent.com/jobdesc.html?postId=2083008230946160640)的跨模态检索/数据质量要求作为技术参考，其2年以上经验不符合当前实习定位，不加入可投池。[字节Seed官方公告](https://seed.bytedance.com/zh/blog/bytedance-seed-2027-foundation-model-campus-recruitment-is-now-open-internships-included)为招聘方向参考，不能作为已确认杭州日常实习具体JD。[海康杭州算法实习](https://talent.hikvision.com/home/socity/position?postId=E916B963BC9E74350370DF43D2D2A8A3)涉及图文数据、预处理、算法复现与PyTorch，页面仍开放状态待核。这些异渠道样本不用于技术频率百分比。

## 开源数据与完整划分

使用[CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`的计分定义及[公开Flickr30k WebDataset](https://huggingface.co/datasets/clip-benchmark/wds_flickr30k) revision `f8af3ceee1b944bf761340682621dfafd23974d2`。全部10个归档4,288,051,200bytes按LFS SHA256验证、HTTP Range恢复，不用下载子集冒充完整训练。

提供的train/test分开，dev为源train中按`SHA256(20261014|image_sha)`选出的前1000个完整图像SHA组，非原Karpathy val。全源train实际行数/重复/派生train及dev人数在解析后记录；test须完整1000图、5000描述。源图字节SHA跨train/test重叠拒绝；train/dev按完整SHA组分割。上游是WebDataset图像表示，不称原始全分辨率Flickr；预训练重叠未知。图像/文本本体不随项目发布。

## 表征与算法

本地已缓存CLIP-B/32 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，冻结完整编码器、FP32/eager、224标准Resize/CenterCrop/Normalize、文本77token截断并报告实际比例。完整特征缓存记录来源/源码/权重/输入SHA、实际counts与图文顺序，按完整chunk恢复。

两侧512→64→512残差adapter（LayerNorm/GELU，上投影零初始化），总132480参数，初始归一几何保持。训练loss为对称多正样本CE：图像query对全部同图caption的log概率平均，caption query对对应图像，并按anchor image平均；确切同源SHA图共享训练正组。`P_ij=positive_ij/num_positive_i`，`L=-sum(P log softmax(cos/τ))`，τ=.07。

每更新32张anchor图、每图5描述，另64个负图及其描述。两方法固定初始化、参数、loss、epochs、anchor顺序与负图槽位：uniform为64随机；mixed为32困难+32随机。困难只在训练图库，以冻结CLIP的anchor-caption均值query与训练图相似度取top32，再补32随机；排除anchorID、同源SHA、已知相同NFC/空白/casefold caption关联图。这个排除仍不能解决全部语义误负例；锚点间重复描述等歧义单独看待。负例曝光分布不同，计数归档，不声称同梯度/同信息。

## 冻结实验与计分

seeds`{0,42,200}`、两方法各10epoch，共6次小参数训练，固定Final@10，不依据dev早停或挑参。AdamW lr=.001、weight_decay=.0001、clip_norm1.0。所有训练结束后，frozenCLIP及6个Final权重分别完整dev/test评测一次，不据test修改规则。R@1/5/10在T2I/I2T分别报告，使用上游“至少一个标注正项命中”定义，非检出全部五描述的比例；二值nDCG@10另列，不合成总分、不当VQA分数或工业召回指标。

主要门槛：mixed−uniform T2I R@1三seed均值至少+1pp，mixed−frozen至少+2pp；两主对各单侧alpha=.025、图像簇6000bootstrap下界>0；mixed−uniform I2T R@1点值至少−1pp，至少2/3训练seed T2I方向为正。seed均值±样本SD与固定三checkpoint平均后图像整簇CI分开，不把seed×image/caption当独立样本。dev/test分别归档，完整test不回调；门槛失败保留负结果，不包装成困难负采样成功。

## 执行与学习

数据下载→完整来源/分割→同卡冻结特征→两方法三seed训练→全部完整排序→配对统计/学习页；所有GPU阶段用已有任务不抢占的UUID锁、无compute进程、≥40GiB空闲、<10%利用率、3次15秒快照。checkpoint含optimizer/update/负例曝光，可在一致绑定下恢复。CPU数学小检查是工程契约，不是训练成果。

讲解用真实caption/原WebDataset图片、512维向量、正对矩阵、softmax/梯度及Top5：为什么一图五描述不该互作负例；`p−P`为何让hardnegative更有梯度；温度与排序的区别；如何计算hit-anyRecall/nDCG。图文相似度不是Transformer注意力。对实际贡献与独立讲述核对后才定向修改简历；这轮不宣称复现VLMEmbedding、CrossEncoderRerank、工业商品搜索或在线服务。

图表工作流引用：Kassis等(2026)，[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。指标依据为已锁定CLIP_benchmark，源码改造与标准方法实验不是算法原创声明。
