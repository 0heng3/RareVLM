# BF16/1024正式三任务复核 v1

目录迁移说明：本文原机启动命令与持久环境路径是迁移前执行快照，历史结果/配置/来源锁保持原样。精确恢复旧归档须在独立 checkout 使用提交 `2049c0e`；当前 `src/` 版需先执行 `pip install -e . --no-deps`，使用新的输出目录与来源锁，不复用旧锁。详见[架构说明](ARCHITECTURE.md)。

冻结日期：2026-10-11。配置：[vlm_formal_bf16_v1.json](../configs/vlm_formal_bf16_v1.json)。进度：STATUS.json（本地归档，未发布）：`rarevlm_pilot/outputs/vlm_formal_bf16_v1/STATUS.json`。旧训练、评测源码和结果不覆盖。

公开说明：本页保留冻结执行协议及原机命令。源码与配置可公开检查，完整状态、预测、数据和权重为本地归档；外部复现须准备对应依赖/数据/模型/adapter并重绑定配置与命令路径，不保证 clean clone 可直接启动 GPU 评测。完成结果见[当前研究路线](JOB_ROUTE.md)。

## 工程选择与结论范围

原权重BF16、1024视觉token上限，来自已完成内部dev诊断：固定seed42在256/512/1024的共识分75.343/82.678/86.926，最高预算峰值allocated8.652 GiB可容纳于当前A10080GB。选择在看过探索dev后、新协议正式评测前记录，不伪称探索前预注册。

这是当前最高tested-budget质量候选，尚无业务延迟SLA；成本继续测量。与历史NF4/FP32、同1024相比，未见/罕见监督切片点值低0.451/0.789 pp，无跨路径切片区间，不能宣布显著退化或省略该观察。本轮正式切片跟踪整体和低监督暴露答案，不把最高Overall当各子集最佳。

冻结后不依据本轮TextVQA/DocVQA/MMStar结果改预算、提示、数值路径或checkpoint。不论结果方向如何完成全部12格。原公开benchmark已有历史评测曝光，预训练重叠未知，不宣称新独立未见test证据。

## 正式矩阵

| 固定模型 | TextVQA official val | DocVQA val | MMStar |
|---|---|---|---|
| 同路径底座 | 5,000 QA / 3,166图 | 5,349 QA / 1,285图 | 1,500 QA / 1,430图 |
| stage2 q/v rank8 seed0最终adapter | 同上 | 同上 | 同上 |
| stage2 q/v rank8 seed42最终adapter | 同上 | 同上 | 同上 |
| stage2 q/v rank8 seed200最终adapter | 同上 | 同上 | 同上 |

12个新评测单元、47,396 QA，零新增训练。底座只评一次，不虚构三个训练种子。三个adapter均完整2062更新、train_all32,985 QA与固定最终epoch。seed0训练于4V100，seed42/200于2A100；均值±样本SD反映这三个实际checkpoint差异，不能纯归因于随机种子。

原权重、vision、head、LoRA为BF16；SDPA、TF32关闭、batch4、greedy最多32token，原固定task模板，不输入人类答案、训练target或参考OCR。min_pixels50176、max_pixels802816；逐题记录实际token与总输入长度，不套用旧训练768上限截断推理。没有原BF16重新训练。

## 计分与统计

- TextVQA主问题：每个seed在新路径下是否优于同路径底座；计分为冻结的十标注leave-one-out共识，完整5,000 QA。
- DocVQA为冻结ANLS；MMStar为冻结选择题提取器及**18子类别宏平均**，单独作为能力保持指标，不合并为统一总分，也不预设非劣效结论。
- 每task报告三checkpoint均值±样本SD与每seed−base的2,000次同图整簇配对区间，共9个预声明比较。区间条件于固定权重，不包含再训练不确定性、不将seed×image当独立样本，点区间未做多重比较校正。
- MMStar只在完整qid集合、18类别全集与各类题数匹配后发布正式宏平均。bootstrap若缺类别，拒绝该抽样并记录重抽，不能删除类别后改变分母。
- TextVQA正式val切片从train_all实际SFT目标归一化计数重建；代表答案按声明规则取规范化参考众数、tie字典序。未见/1–5/6–20/>20与辅助参考覆盖频次分别报告。参考答案用于评测后切片，不用于配置选择或模型输入。
- 新协议与旧NF4/FP32/256表分开。跨协议差异同时涉及预算、量化和精度，不证明纯dtype收益或精度等价。

## 执行与资源

新[worker](../src/vlm_formal/evaluate.py)、[controller](../src/vlm_formal/pipeline.py)、[CPU分析](../tools/analyze_vlm_formal_bf16.py)。复用冻结处理/加载/任务评分实现，每个worker常驻一个模型顺序完成三任务；两空闲GPU可同时执行两个model，再接续剩余model。

每task以输入长度/长宽比/最大图的固定工程样本预检有限logits、greedy生成、token网格和显存，不按答案或是否成功挑样本，不作为正式cache。manifest、图片与processor元数据、adapter与TRAIN_RESULT、源代码及包版本绑定。预测按完整batch落盘，task专用评分恢复校验，末批中断重生成。

连续三次无GPU计算进程、至少40GiB空闲、利用率<10%后准入；持UUID锁并继承至worker进程，动态空闲卡接续，保留所有已有任务。资源异常禁止放行；采样竞争单列，不能保证完全独占。

计时按task/session/batch去重，排除各session首批；分开报告纯生成、预处理/H2D/生成合计、峰值allocated/reserved及加载/预检/整体活动时间。吞吐不是线上p50/p95；ETA须按三任务各自20–50批的实际速率估计，不能统一套用TextVQA dev速度。

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
cd /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001
"$VLM_A100_PYTHON" -m vlm_formal.pipeline \
  --asset-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2 \
  --model-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1/model \
  --output-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_formal_bf16_v1
```

LAUNCH.json保存后台命令与PID，pipeline.log与logs/保存日志。完成后自动核验12格并生成JSON/CSV、逐seed差值、频次切片与真实结果图。未齐只报进度，不追加简历成果，不启动新训练/ROI/DPO或换底座分支。
