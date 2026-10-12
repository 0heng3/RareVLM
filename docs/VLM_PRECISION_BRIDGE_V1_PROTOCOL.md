# 原权重、NF4与BF16工程桥接 v1

日期：2026-10-11。本协议采用非抢占资源接续。配置：[vlm_precision_bridge_v1.json](../configs/vlm_precision_bridge_v1.json)。旧A+B和正在运行的视觉预算诊断不改设置、不终止。

公开说明：本页保留冻结执行协议及原机命令。源码与配置可公开检查，完整状态、预测、数据和权重为本地归档；外部复现须准备对应依赖/数据/模型/adapter并重绑定配置与命令路径，不保证 clean clone 可直接启动 GPU 评测。完成结果见[当前研究路线](JOB_ROUTE.md)。

## 研究问题与原理

已完成正式TextVQA数值对照中，原权重FP32底座70.328、NF4底座69.030；同一个早期pilot adapter在两路径为71.112与70.382。该观察支持先测量数值工程路径，而非直接把错误归因于模型容量。

FP32与BF16都是浮点计算路径。BF16与FP32有相同的8位指数宽度，但有效精度更低；A100可以通过BF16 Tensor Core提高矩阵计算效率。NF4则以4位表示语言权重，前向需要解量化，并有量化误差。这里不能按权重位数推算吞吐；需要同时看输出、质量、延迟与显存。

这不是训练实验。所有权重固定，保持同一模型revision、processor、256视觉token上限、原提示、无OCR文本、greedy最多32token、batch4、TF32关闭。BF16使用未量化权重，并记录vision、语言、head和LoRA实际dtype；SDPA在不同dtype下可能选择不同内核。因此它是独立数值工程路径，不宣称只改变compute dtype或量化权重bytes相同。

## 六个单元

| 固定模型 | NF4 / FP32 | 原权重 / FP32 | 原权重 / BF16 |
|---|---|---|---|
| Qwen2.5-VL-3B底座 | 完整dev | 完整dev | 完整dev |
| stage2 qv_r8 seed42 adapter | 完整dev | 完整dev | 完整dev |

每格原图像隔离内部dev的1,617 QA / 1,024图。seed42与适配策略沿用视觉诊断的预声明选择，不由本轮输出、正式val或视觉预算结果决定。adapter原本在NF4底座上训练，应用到未量化底座是明确的推理路径迁移，可能改善或恶化。

每个role在固定GPU上依次加载三条路径，释放上一模型后再加载下一模型；不将三模型同时驻卡。每条路径先做固定输入压力预检，检查有限logits、实际token/grid、greedy生成和显存；工程样本不计入主矩阵缓存。

## 资源接续与恢复

控制器：[vlm_precision/pipeline.py](../vlm_precision/pipeline.py)。worker：[evaluate.py](../vlm_precision/evaluate.py)。CPU分析：[analyze_vlm_precision_bridge.py](../tools/analyze_vlm_precision_bridge.py)。

GPU0接续base，GPU1接续adapted。两张卡分别等待，不要求同时释放；一张卡空闲时可先开始其role。启动条件为连续三个检查均无GPU计算进程、利用率低且空闲显存至少40GiB。检查其他任务只用于等待，不发送终止、暂停或修改命令。持久的每GPU互斥锁避免本控制器重复占卡；外部任务不保证遵循该协作锁，所以同时记录运行期间资源快照与竞争事件，质量数据保留，发生竞争时测速标记为受干扰。

启动先在CPU排队准备配置、源码锁和模型绑定；GPU释放后自动执行。当前队列进度：STATUS.json（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_precision_bridge_v1/STATUS.json`。运行根目录：vlm_precision_bridge_v1（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_precision_bridge_v1`。预测按完整batch落盘，中断后核对配置、源码、权重、adapter、数值路径和包版本恢复，不跨路径复用预测。

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
cd /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001
"$VLM_A100_PYTHON" -m vlm_precision.pipeline \
  --asset-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2 \
  --model-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1/model \
  --output-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_precision_bridge_v1
```

## 分析与判读

每role比较原权重FP32−NF4 FP32和原权重BF16−原权重FP32，共四个预声明对比。按图像整簇配对bootstrap2,000次，保留同图所有题；单种子固定权重的条件区间，不代表训练种子变异，未做多重比较修正。

同时报告原始/规范化预测一致率、正确→错误与错误→正确转换，并保留部分共识分的变化。高输出一致率、CI跨0或较小均值差均不构成精度等价证明。

资源按(session_id,batch_start)去重并排除每个session首个warmup batch。分别报告纯生成吞吐、预处理/H2D、批摊销延迟及峰值allocated/reserved显存。模型加载、预检和排队时间另列，不能合入稳态吞吐。共享主机的其他任务与竞争事件保留在结果中。

六格完成后自动生成JSON/CSV和真实结果PNG/SVG；未齐时明确complete:false，不推断未执行路径的质量或加速。原正式val三种子表、当前视觉预算dev表、新数值dev表保持分开。
