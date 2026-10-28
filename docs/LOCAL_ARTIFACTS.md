# 本地实验产物与公开发布范围

公开仓库包含源码、配置、实验协议、聚合结果、统计边界与学习页生成器。数据集原图、完整逐题预测、模型权重、运行日志、私人简历及包含这些依赖的离线学习页保存在独立本地工作空间。

部分历史记录原来链接到本地outputs或离线学习页；公开副本将这些链接统一指向本说明，未用占位图冒充实验图像。对应生成器仍在tools/，数据来源与运行入口见各项协议。原本地协议与科学源码保留，公开文档链接整理和私人路径去标识记录于metadata/PUBLIC_EXPORT.json。

## 复现入口

- [项目与数据目录说明](ARCHITECTURE.md)与[运行说明](REPRODUCE.md)。
- [最新整图SFT/DPO完整结果](VLM_STVQA_REPLAY_V1_RESULTS.md)与[便携指标](../results/vlm_stvqa_replay_v1_metrics.json)。
- [整图监督协议](VLM_STVQA_REPLAY_V1_PROTOCOL.md)、[四Final来源保持验证](VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md)。
- [只读监控说明](monitor/README.md)：完成本地数据与输出配置后运行tools/rarevlm_monitor.py --snapshot --render。

历史研究脚本使用各协议登记的本地目录和输出布局。运行前须另行获取有权使用的数据与权重并设置相应配置；仓库目录内资源由src/rarevlm_paths.py解析，工作空间outputs位于仓库外。CPU smoke和发布检查无需训练数据或GPU；源码公开不等于已经打包全部GPU实验资产。

本轮仅同步项目研究，不修改个人简历、不启用GitHub Actions。已完成、失败、探索和待验证结果分开报告。
