# 自适应图文重排序 v1

本协议在读取新margin及选择阈值前建立，状态以独立outputs为准。0新训练，复用旧三个CLIP distill Final@10和一固定BLIP-COCO权重。

## 问题与不变条件

完整Top50重排已有效，但每个划分的三checkpoint双向候选请求合计900000，去重仍约312000。研究问题：CLIP高置信度查询直接返回，低置信度查询才执行BLIP，能否保护质量并减少实际匹配？保持候选K50、CLIP FP32/224/77、BLIP BF16/384/64/[ENC]30523、正匹配类1、logit1−logit0、stable ties、pair batch32、两GPU role及固定gallery1000图/5000caption。

## 校准与确认

dev源图SHA必须1000唯一；按sha256('rarevlm-adaptive-v1:'+source_image_sha256)排序，前500图作为cal，后500图confirm。每图五caption与图像查询同属一fold，但检索图库仍完整1000/5000，不缩小负例空间。分组只用图SHA，不看答案或分数；dev历史已曝光，不称新盲测。

置信度m=cosine_top1−cosine_top2，由相同checkpoint完整原候选顺序的FP32分数计算。它是启发式margin，不是校准正确概率；I2T前两条caption可能都为正项，低margin不必然有错。

每方向共用一个跨checkpoint阈值τ，m≤τ重排，否则保留原CLIP Top50。阈值候选为三checkpoint的cal margin合并后q∈{0,.05,...,1}的分位数，加τ=−1（全部跳过）/2.1（全部重排）。按数值升序选择第一个同时满足cal平均R1损失≤.1pp、nDCG10损失≤.5pp的阈值；always保证可行回退。两方向选择可分解：较低阈值需要的pair集合是较高阈值集合的子集，独立最小可行τ也最小化共享union。预先固定此规则，不据confirm/test调整阈值、grid或容许损失。cal用已有已冻结CLIP/完整BLIP预测作事后校准，参考标注不进入运行时路由。

## 实际执行与耗时

重新产生CLIP分数并逐项核对旧Top50。执行路由后，先从所有被选查询提取pair key=image_index*5000+caption_index，再全局去重；仅这些keys进入adaptive matcher。对同GPU role分别新计算完整pair和adaptive pair，禁止把旧完整分数查表当节省。视觉特征由本轮固定模型重新编码一次/role，两配置共享这一公共前处理；全图视觉成本单列。

各role固定同card、pair batch32、相同BLIP/input；先对64实际pair作native/cached GPU预检，然后三轮full/adaptive交替顺序计时。第一轮完整质量评分按2048pair存keys/logits/margin/SHA；若中断则按输入/源码/原UUID锁接续，跳过已保存chunk的部分轮次不得用于计时，另外重跑完整计时轮。fresh第一轮可计入三轮，后两轮重新forward全部对应keys，不复用logits。计时为同步后的缓存matcher阶段，含gather/CPU回传、不含模型加载/vision/preflight/文件写入；CPU路由时间加入adaptive阶段。排序、输入缓存及完整研究壁钟另列，非线上SLA。重复forward次数与生成质量排名所需unique pair分开。

## 门槛与test规则

主质量比较为确认500图中adaptive−本轮fresh full的T2I/I2T R@1。先平均三固定checkpoint，再按图像整簇6000次bootstrap，单侧alpha=.025/方向，下界均>−.5pp。种子SD单列，五caption/seed×image不作独立样本。

成本在完整dev1000/5000共享三checkpoint union评估：实际adaptive unique pair比full至少减少30%；每轮parallel matcher phase取两role最大值，adaptive加入路由时间，三轮speedup中位数≥1.2、每轮≥1.0。两方向确认nDCG点值至少−.5pp、allocated≤65GiB、资源无竞争观察/查询失败。每seed独立候选union成本另外报告，不能把三checkpoint共享摊销写成单模型线上成本。

全部dev质量/成本/资源门槛通过才在历史test冻结τ进行一次完整正式研究；否则不新增test，不改margin阈值/门槛或追加新策略。test亦执行同卡fresh full/adaptive与三轮计时，但不当三个新的BLIP训练种子。test已历史曝光、预训练重叠未知，统计不能消除历史探索偏差。

## 教学与复核

保存校准扫描表、锁定阈值、逐query真实margin/路由、完整与选中pair列表、实际GPUforward/chunk记录、真实改进退化案例与图片。原图展示缩略图不冒充384/224输入或attention。CPU合同验证零/全路由、合法pair key、重复去重、缺失分数拒绝及稳定排名；从保存Top50独立评分，核对候选不变/原full预测、旧科学文件与正式简历锁。源码/配置独立，不动旧研究、不启CI、不自动push，已有GPU任务不终止。
