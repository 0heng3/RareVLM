# RareVLM 生成式深入实验 A+B：执行协议 v2

用户于2026-10-09授权执行上一轮深入实验分析。先执行完整评测与数值路径对照（A），以及同参数预算的三策略、三种子训练（B）。配置见 [vlm_deep_v2.json](../configs/vlm_deep_v2.json)。该版本单独保存数据、运行和报告。

## 固定主表

Qwen2.5-VL-3B-Instruct 固定 revision；所有新训练从同一预训练底座开始。NF4语言线性权重、FP32计算、FP32视觉主干、SDPA、固定图像resize预算64–256视觉token。只更新明确列出的LoRA参数，alpha/rank=2，dropout=.05。

| 策略 | 参数分配 | 预期可训练参数 |
| --- | --- | ---: |
| B1 qv_r8 | 36层语言q/v，rank8 | 1,843,200 |
| B2 qkvo_r4 | 36层语言q/k/v/o，rank4 | 1,843,200 |
| B3 qv_r4_merger | 语言q/v rank4；merger两层rank64/36 | 1,835,008 |

B1/B2比较相同预算下的参数分配，不将rank和覆盖范围解释为独立单因素。B3与参考预算差约0.44%。merger位于`visual.merger`；视觉trunk保持eval，merger LoRA参与训练。必须实际验证语言和merger梯度，不能以CLI开关代替验证。

## 数据、监督与选择

完整官方TextVQA训练来源，按图像ID的固定SHA顺序选1,024张内部dev图，排除历史pilot训练/内部验证图出现在新dev。保留图像的全部问题；stage1固定10,000条训练QA，所有策略和种子共享。训练目标采用人类回答的确定性众数代表，保留原文字面形式；标准归一化仅用于评分。无法构造非空训练目标的注释须记录并排除，最终5,000条官方val不据此过滤。

训练仅监督助手回答与结束token。损失为每问题有效答案token的平均CE，再对问题平均；梯度累计保持有效batch16。计划4卡、microbatch4、accumulation1；共同预检如不满足显存或数值契约，只根据这些约束降低所有策略的microbatch并相应提高accumulation，记录实现修复，不能根据评测分数调预算。

固定一轮AdamW、lr1e-4、weight_decay0、warmup占更新步数6.25%、线性衰减、clip_norm1；seeds={0,42,200}，三个策略共9次训练。使用最终epoch；内部dev生成共识分和token NLL用于选stage2策略。stage2取三种子dev均分最高的两个策略，平局依次由dev token NLL、方法名打破，扩展至dev之外全部可用官方train QA，仍三种子/固定一轮。正式val与迁移任务均不参与选择。

## 完整评测 A

固定FP32计算，比较original/NF4底座×无adapter/同一历史pilot adapter的推理2×2。该adapter在NF4下训练，矩阵不等同于训练量化因果分解。FP32单卡如超显存，使用记录清楚的跨卡推理布局；不把不同GPU预算的性能混作引擎加速收益。

TextVQA官方val全部5,000问题、3,166图像，图像和十个人类回答来源绑定，标准EvalAIAnswerProcessor与leave-one-annotator-out共识。固定短答案提示、greedy、max_new_tokens32；生成输入不含答案或OCR tokens。已有pilot部分val曝光如实披露，不称未见过的测试集。

DocVQA val（ANLS）和MMStar（选择题准确率）用于任务迁移/能力保持；其数据清单固定，标注在生成后用于评分，模型预训练曝光未知。原始图像须核对旋转/大小，记录实际下载传输、图像字节与像素哈希。跨split图像冲突仅从训练来源移除，不丢弃完整正式评测样本。

bootstrap按图像聚类，所有同图问题一起重采样；seed均值/SD另列，禁止把seed×question视为独立样本。答案频次切片由训练来源计算，并分离未见答案；它是下游标注频次，不等同于闭集类别先验。

## 执行与验收

执行前实现修复：Dataset Viewer标注服务出现429限流，改从固定revision的23个Parquet分片读取标注列，保留HTTP范围与范围字节哈希，逐字段核对已取得的Viewer标注。图片使用完整官方ZIP，整包SHA与所有成员CRC校验通过。范围读取不声称核验整份Parquet的LFS哈希。DocVQA固定到 `539088ef8a8ada01ac8e2e6d4e372586748a265e`、MMStar固定到 `bc98d668301da7b14f648724866e57302778ab27`，其完整Parquet文件均验证官方公布的LFS SHA。

共同GPU预检已通过：batch4、约757 token压力检查峰值低于7 GiB；回答尾部logits与完整logits的loss/梯度在FP32容差内一致；原始FP32推理路径约14.20 GiB且最后位置logits优化生成一致。NFS环境下CPU batch使用NumPy传输，再在训练进程转成相同dtype的tensor并pin，避免多进程tensor共享文件清理冲突。实现检查与正式研究结果分开保存。

每次训练保存数据/config/source/checkpoint哈希、实际参数清单、样本顺序、完整更新数、有限参数/梯度、冻结权重（含量化状态）前后哈希和资源记录。评测保存逐题生成、标准分数、dev/final边界以及各阶段状态。训练中断须从带optimizer/RNG/sampler绑定的完整checkpoint恢复；不能将部分epoch记成完成。

本阶段以完整A矩阵、9次stage1、dev规则选出的6次stage2、标准TextVQA/迁移评测及核验报告为交付。视觉分辨率、数据策略和DPO是后续有条件分支，不在A+B中同时引入。

## 2026-10-10 两张A100硬件迁移补充

用户更换容器后授权使用2张A100-SXM4-80GB继续剩余实验。第一阶段9次训练/36次评测均已完成；第二阶段qv_r8 seed0的全量训练、dev与TextVQA已完成。继续执行剩余5次训练和22项评测，不重做已完成结果。两策略仍由旧四卡第一阶段的冻结dev规则选择，为qv_r8与qv_r4_merger。

原`vlm_deep/`源码与`configs/vlm_deep_v2.json`保持冻结。新增`vlm_a100/`运行适配层：实际world2、microbatch8、accumulation1，有效batch仍为16；原数据/目标/样本全局顺序/2062次全量更新/AdamW与学习率/最终epoch/视觉与生成预算不变。继续NF4语言权重和FP32计算、FP32视觉，并显式关闭matmul/cudnn TF32。新布局、原科学配置与代码哈希分开记录。硬件、rank分配、dropout流与浮点归约可能不同，不宣称逐位等价或已排除硬件因素。

旧DocVQA四卡分片保存1748题，旧目录只读保留；两卡按question_id、问题/图像/adapter/模型/源码哈希和标准分数校验后复用，只生成缺失题。新评测目录为`eval_{task}_a100`。`runs/RUN_INDEX.json`为完成任务的唯一逻辑索引，每个训练或评测只计一次，最终要求15训练/67评测。汇总逐seed标注GPU/布局，旧缓存与新生成的资源计时分开，异构硬件下的均值不是纯算法因果结论。

实施检查在`outputs/vlm_deep_v2/a100_migration/preflight/`单独保存。两卡33真实QA检查验证3更新、最后1个真实样本/其余计算零loss权重、语言及merger实际梯度、冻结权重、完整optimizer/RNG恢复。FP32检查保留严格逐位/极小容差不一致记录，并报告实际数值差与采用的实用容差；检查产物不进入研究表。

新容器的PyTorch通讯库需匹配HPCX UCX/UCC路径，使用持久化环境入口启动：

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
cd /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001
"$VLM_A100_PYTHON" -u -m vlm_a100.pipeline \
  --asset-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2 \
  --pilot-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1 \
  --run-root /opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2/runs \
  --public-output results/vlm_deep_v2_metrics.json
```

运行记录见`a100_migration/MIGRATION_MANIFEST.json`、`PREFLIGHT_VERIFICATION.json`和`CALIBRATION.json`。实时进度与按实测速度更新的剩余时间见`runs/STATUS.json`；估算包含训练、dev与评测，不把短程检查当正式研究结果。
