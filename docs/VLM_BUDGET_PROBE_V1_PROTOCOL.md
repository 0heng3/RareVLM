# 固定权重的视觉输入预算诊断 v1

目录迁移说明：本文原机启动命令与持久环境路径是迁移前执行快照，历史结果/配置/来源锁保持原样。精确恢复旧归档须在独立 checkout 使用提交 `2049c0e`；当前 `src/` 版需先执行 `pip install -e . --no-deps`，使用新的输出目录与来源锁，不复用旧锁。详见[架构说明](ARCHITECTURE.md)。

冻结日期：2026-10-11。配置：[vlm_budget_probe_v1.json](../configs/vlm_budget_probe_v1.json)。本轮只做推理诊断，不更新模型权重。旧 A+B 源码、配置和结果继续冻结。

公开说明：本页保留冻结执行协议及原机命令。源码与配置可公开检查，完整状态、预测、数据和权重为本地归档；外部复现须准备对应依赖/数据/模型/adapter并重绑定配置与命令路径，不保证 clean clone 可直接启动 GPU 评测。完成结果见[当前研究路线](JOB_ROUTE.md)。

## 问题与原理

先检验：当前细粒度错误是否对视觉输入信息量敏感？如果输入在缩放时丢失了小字或数字，仅增加语言适配参数可能无法恢复这部分信息。增加输入预算能检验这个约束，但不能独立证明注意力机制、OCR 能力或模型容量的因果作用。

Qwen2.5-VL 的本地配置是 14×14 patch、2×2 空间 merger。每个合并后的视觉 token 对应 28×28 像素网格；预算 T 映射到 `max_pixels=T*28*28`。图像先按长宽比和离散网格缩放，因此 T 是上限，实际数量必须从 `image_grid_thw` 的乘积除以 4 计算，并与输入的 image token 计数核对。

```python
min_pixels = 64 * 28 * 28
max_pixels = visual_token_budget * 28 * 28
actual_visual_tokens = grid_t * grid_h * grid_w // 4
resized_h, resized_w = grid_h * 14, grid_w * 14
```

256→1024 把最大像素预算提高四倍，同长宽比下线性尺寸约翻倍。总耗时不必四倍或十六倍：视觉编码、全局/局部注意力、语言预填充、KV cache 与自回归生成的成本不同，需要实测。预算对照包含输入长度变化，不是等计算量对照。

## 固定矩阵与隔离

| 固定权重 | 256 上限 | 512 上限 | 1024 上限 |
|---|---|---|---|
| 原协议 NF4 底座，无 adapter | 完整 dev | 完整 dev | 完整 dev |
| stage2 qv_r8，seed42 最终 adapter | 完整 dev | 完整 dev | 完整 dev |

每单元 1,617 QA / 1,024 图，均为原训练池中预先固定的图像隔离 dev。qv_r8 来自冻结的 stage1 三种子 dev 第一名规则；seed42 在预算实验前固定，且在 A100 上训练。此处不按正式 TextVQA val、DocVQA 或 MMStar 分数挑选 checkpoint。

保持 NF4 语言量化、FP32 compute/vision、SDPA、TF32 关闭、原提示、无 OCR 文本、greedy、最多生成 32 token。每 GPU 固定 batch4，GPU0 跑底座，GPU1 跑 adapter，分别依次完成三档预算。base/adapted 的 GPU 分配固定，所以主要比较是每个模型在同卡上的预算变化。

训练时的 768 总 token 上限不适用于本轮高预算评测；推理不截断图像或问题，记录实际总输入长度。必须注意 adapter 训练预算是 256，本轮较大预算也属于推理输入分布变化。

## 工程预检和恢复

在最高 1024 预算先检查长问题及极端长宽比的内部 dev 输入：实际 token 与网格一致、有限 last-position logits、greedy 可生成、显存可容纳。压力样本只根据问题长度和图像尺寸选择，不按答案或是否失败选择，不进入主矩阵缓存。

新代码：[vlm_probe/evaluate.py](../src/vlm_probe/evaluate.py) 和 [pipeline.py](../src/vlm_probe/pipeline.py)。两种权重各常驻一张 GPU，无 DDP。每题保存预测及共识分、输入/图像 hash、原图/resize 尺寸、实际视觉和输入 token、batch 生成/预处理耗时、生成长度、峰值显存和 warmup 标签。恢复要求配置、源代码、processor、模型 provenance、adapter hash、数值路径和生成设置一致，不跨预算复用预测。

进度：STATUS.json（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_budget_probe_v1/STATUS.json`。运行目录：vlm_budget_probe_v1（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_budget_probe_v1`。GPU 预检失败时停止，不能静默改精度或 batch 后混列。

持久启动/恢复命令（先确认旧 controller 已停止；进程互斥锁也会拒绝重复启动）：

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
cd /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001
"$VLM_A100_PYTHON" -m vlm_probe.pipeline \
  --asset-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2 \
  --model-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1/model \
  --output-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_budget_probe_v1
```

当前后台启动信息保存于运行目录的 `LAUNCH.json`，控制器日志为 `pipeline.log`，两种模型的预检与评测日志分别在 `logs/`。向控制器 PID 发送 TERM 可中断两个 worker；中断期间不完整的最后 batch 会在恢复时重新生成，已经完整落盘的 batch 按锁复用。更换容器后仍使用上述持久环境与路径；如果包版本、GPU数值路径或源码锁发生变化，必须建立新输出目录，不能强行忽略锁。

## 分析和学习交付

先看每个模型的 512−256、1024−256 共识分差，再看差分中的差分 `(adapted1024-adapted256)-(base1024-base256)`。按图像整簇配对 bootstrap 2,000 次，保留同图全部问题；区间是固定权重的评测样本不确定性，不包含训练种子波动，也不做算法普适胜负结论。所有比较均为内部探索，多重比较未校正。

答案频次从全训练 `train_all` 重新计算。主要代理是实际 SFT 目标规范化文本的 QA 次数；辅助代理是每 QA 的规范化参考答案去重次数。dev 代表答案固定为规范化十条参考答案的众数，平票按字典序。分别报告 0、1–5、6–20、>20 四档，不能把旧 10k 频次直接用于全量训练，也不能把答案文本频次等同视觉实体频率或预训练未见。

同时展示质量、实际 token、稳态延迟、吞吐与峰值显存。按预先固定的三个内部 dev 教学样例（27482、10476、12257）呈现原图、真实预处理尺寸和各档实际预测；显示用放大图不输入模型，不制造注意力热图。个例仅解释结果，不替代总体指标。

高预算改善支持输入预算是当前约束之一；没有改善不等于底座必须更换。先结合分组和成本决定后续：自动预测区域＋等总 token/随机区域对照、等更新数据选择、能力保持；若仍有明确容量问题，再做同系列 7B 对照。BF16 工程桥接属于另一数值协议，不与本轮 FP32 表混列。
