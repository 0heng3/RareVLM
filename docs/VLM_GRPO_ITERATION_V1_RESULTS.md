# 单轮GRPO对照完成

|模型|训练32 VQA|新512 VQA|保持64 VQA|
|---|---:|---:|---:|
|initial|4.062500|93.691406|93.750000|
|ce|16.562500|94.023438|93.750000|
|ce_grpo|19.687500|93.828125|93.750000|

候选对初始/新CE：{"initial": {"delta_pp": 0.13671875, "ci95": [-0.5859375, 0.8593750000000001], "image_clusters": 512, "scope": "Fixed single-checkpoint conditional image-cluster interval."}, "ce": {"delta_pp": -0.1953125, "ci95": [-0.5859375, 0.0], "image_clusters": 512, "scope": "Fixed single-checkpoint conditional image-cluster interval."}}

联合通过：False。Stop this configuration; no extra steps/seeds/main evaluation or CV claim.

本轮每臂32步，单seed42；初始新512实际生成、96历史预测同条件复用，两候选各608实际生成。候选额外136策略前后向，不能声称等计算。训练拟合/奖励不代替泛化，公开训练白名单及历史曝光披露保持。

[固定协议](VLM_GRPO_ITERATION_V1_PROTOCOL.md) · [原图/实际动作/优势学习页](LOCAL_ARTIFACTS.md)

事后分因补充：GRPO与CE在新验证511/512原始输出相同，唯一AGA/AGAI分差存在原图边缘截断转写歧义；California Style Absinthe完整短语也有原共识范围边界。原标准分数及失败不改，不把该单题直接断言为新增OCR错误。已存检查点显示SFT26题/GRPO25题在观察点从未满分，各1题回退；主要观察现象是困难题尚未学会，不据曲线挑checkpoint救Final。[原图与分因](VLM_GRPO_ITERATION_V1_FAILURE_REVIEW.md) · [完整纠错曲线](VLM_GRPO_LEARNING_CURVE_V1_RESULTS.md)
