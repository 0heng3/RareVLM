# BF16与高视觉预算联合诊断 v1

冻结：2026-10-11。配置：[vlm_joint_bf16_v1.json](../configs/vlm_joint_bf16_v1.json)。进度：STATUS.json（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_joint_bf16_v1/STATUS.json`。

公开说明：本页保留冻结执行协议及原机命令。源码与配置可公开检查，完整状态、预测、数据和权重为本地归档；外部复现须准备对应依赖/数据/模型/adapter并重绑定配置与命令路径，不保证 clean clone 可直接启动 GPU 评测。完成结果见[当前研究路线](JOB_ROUTE.md)。

## 问题与固定矩阵

此前1024预算的86.506分来自NF4/FP32，而2.44倍生成加速来自256预算的原权重BF16。两项收益不能直接叠加。本轮实际测量原权重BF16能否承接较高预算的质量与成本。

| 固定模型 | BF16、512上限 | BF16、1024上限 |
|---|---|---|
| 原始底座 | 完整dev | 完整dev |
| 原stage2 q/v rank8 seed42最终adapter | 完整dev | 完整dev |

共4个新单元、6,468 QA，每单元1,617题/1,024图。原BF16256两格为已完成历史参考，不冒充本轮新增评测。维持旧图像隔离dev；只推理，不重训或挑选新checkpoint。

原权重、vision、head和LoRA均为BF16；quantization=original，SDPA、TF32关闭、batch4、相同提示、无OCR文本、greedy最多32新token。14×14 patch与2×2 merger不变；min_pixels=50176，max_pixels=T×28²。实际视觉token由grid乘积除4计算，不能把预算上限当作实际token。

adapter原本NF4/FP32训练，当前是数值推理路径迁移，不是BF16训练成果。较高预算改变输入分布，不使用训练的768总token上限截断推理。

## 实现与资源

独立[worker](../vlm_joint/evaluate.py)、[controller](../vlm_joint/pipeline.py)、[CPU分析](../tools/analyze_vlm_joint_bf16.py)，复用冻结的加载/处理/生成/计分算法，不修改旧代码。

每role加载一次模型，先在1024压力样本检查有限logits、生成、网格/token一致与显存，再依次评测512/1024。压力样本只依据输入长度/长宽比和原有教学案例选择，不计入主矩阵缓存。

GPU按连续三次无计算进程、空闲≥40GiB、利用率<10%准入；role优先adapted→GPU1、base→GPU0，任一卡被占用时可在其他空闲卡顺序接续。每role两档预算固定同一UUID，并持有继承到worker的GPU锁直至退出。异常资源输出禁止放行，不向已有任务发送终止命令。持续记录同卡竞争事件，资源结果保留相应标签。

逐题预测、共识分、实际token/grid/尺寸、生成长度、去重批次计时、峰值显存和输入/权重/代码锁分别落盘。完整batch可恢复，中断末批重生成；不混用预算缓存。数据、数值路径、软件或硬件锁不一致时拒绝静默恢复。

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
cd /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001
"$VLM_A100_PYTHON" -m vlm_joint.pipeline \
  --asset-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2 \
  --model-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1/model \
  --output-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_joint_bf16_v1
```

当前后台启动PID与命令保存于LAUNCH.json，主日志pipeline.log、分role日志logs/。已有controller时互斥锁拒绝重复启动。

## 自动分析与结论边界

四格完整后与历史BF16256预测配对，报告每role的512−256、1024−256及1024增益交互，共五个预声明对比；按图像整簇bootstrap2,000次。条件于固定权重的探索区间，不包含重新训练的种子变化，未做多重比较校正。

同时生成六点质量—成本表、实际token/延迟/显存图、真实监督target频次切片与三个固定案例表。历史256的日期、UUID与竞争来源明确标记，不能把跨运行测速当作严格同卡、同主机负载的纯加速。NF4高预算作为另一数值路径的观察，不解释为纯dtype效应。

本轮不提供BF16/原FP32在高预算下的精度等价证明，因为没有同预算原FP32配对与预先定义等价界限。也不将dev分数或单adapter种子写成正式三种子结论。结果未齐时complete:false；分析源码就绪SHA握手后自动后处理。

完成本轮后可依据dev冻结工程设置，再单独声明base＋三个q/v seed、TextVQA/DocVQA/MMStar的12格47,396 QA正式复核协议。本controller不自动扩展到正式评测、重训、DPO或换底座。
