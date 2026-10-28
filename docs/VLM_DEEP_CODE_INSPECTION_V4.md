# 深入代码排查：上一轮为什么还没有解决任务（2026-10-17）

上一轮解决了数值链路的已知问题，却没有完成可靠监督到目标任务训练的闭环。新的调查没有发现框架整体训练失效的证据。不能把“执行正确”“目标函数得到优化”“独立任务能力提升”写成同一个完成状态。

## 1. 直接的交付缺口：验证没有使用修正后的物体监督

[v3 验证配置](../configs/vlm_training_repairs_v3.json)第 4 行仍指向 `vlm_deep_v2/prepared/manifest.json`，第 8–9 行为 `validation_only=true/sample_questions=8`。所以此前 2 次更新只验证了旧 TextVQA 样本的数值实现。Object v3 的 9,576 正例候选和 9,576 未知项均未准入训练。

这不是一轮新的物体能力训练，更不能证明原 Object/POPE 问题已经修复。此前的 GPU 一致性证据仍有效，不需要再重复同样的 8 QA 检查。本轮新接入真实人工监督，而不是再增加数值检查器。

## 2. 源码中真正有影响的监督假设

[旧 Object v2 构造](../src/vlm_object_presence_v2/data.py)把范围固定为 18 类、尺寸门槛为短边 24px 和面积占比 .005，并由 `category_ids - annotated_any_including_crowd_category_ids` 产生负例。它只能支持所声明的闭集实例标注定义，不能自动推出开放视觉问答中的“该对象不存在”。原来 POPE 漏检中 79.4% 在训练范围外、7.2% 被尺寸/crowd 规则排除，是覆盖统计，不是可归因的性能损失比例。

v3 将不确定的负例转成 unknown，避免继续喂入可能冲突的监督，但没有补齐替代数据。未知项保持未知，不能凭需要训练就改回 no。

本轮从[官方 VQA v2](https://visualqa.org/download.html)训练集重新取得人类问答，接入 [human_row](../src/vlm_human_vqa/data.py) 和 [数据准备入口](../tools/prepare_human_vqa_supervision.py)：

```python
assert question['question_id'] == annotation['question_id']
assert question['image_id'] == annotation['image_id']
assert int(image['coco_image_id']) == annotation['image_id']
# 核对十个 answer_id，至少 9/10 对 yes 或 no 一致。
# 保留原问题，不把属性/关系的 no 改写成对象不存在。
```

预先固定的来源和 >=9/10 规则见 `outputs/vlm_deep_code_inspection_v4/HUMAN_SUPERVISION_PLAN.json`。只使用官方 `train2014`，在原允许的图像池内匹配；保留 v1/v2 已审查图像排除、train/val 分隔和 benchmark 排除。没有使用模型预测、loss 或 dev/test 答案挑选训练标签。

| 内部划分 | 图像 | QA | yes | no | 10/10 一致 | 9/10 一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,834 | 3,893 | 1,760 | 2,133 | 2,850 | 1,043 |
| val | 228 | 474 | 214 | 260 | 351 | 123 |

这里的任务是原始视觉 yes/no 问答，包含属性、关系、场景、存在性。存在性前缀的训练 QA 为 657，val 为 82；**不能将整个新数据集称作已覆盖 80 类的纯物体存在性监督**，不能与旧 Object val 或 POPE 得分拼成同一列。VQA 官方训练来源共 443,757 问题，数量不是本项目实际训练量。

新数据已通过现有 `vlm_deep.common.read_manifest(..., verify_images=True)`：全量匹配图像文件哈希核对通过；同图同问题文本（小写/空格归一）没有相反的高一致性标签。人工高一致性不保证所有语义绝无噪声。原 v3 候选未被追认准入，新标签只来自独立的人类问答。

实际数据：`outputs/vlm_human_vqa_v1/MANIFEST.json` 与 `SUMMARY.json`。`train_all` 可被现有 FP32 SFT 入口读取；兼容字段 `train_10k` 仅保存现有训练记录子集，不表示训练了 10,000 条。本轮没有启动该数据的新模型训练。

## 3. 两个真实原图例子

这些是训练来源的人工标签示例，不是模型预测，也不是代表性质量估计。原图按字节复制，未裁剪、未添加区域或注意力。

[本地图像说明：COCO 237344 原图](LOCAL_ARTIFACTS.md)

问题：**Is there a camera on the tripod?**；人工答案 **no，10/10**。它询问“相机在三脚架上”这一关系。画面中存在拿手机拍摄的人，不能把原问题改成“画面没有相机/拍摄设备”。来源：VQA train2014 question 237344010，COCO image 237344。

[本地图像说明：COCO 136027 原图](LOCAL_ARTIFACTS.md)

问题：**Is this a vegetarian meal?**；人工答案 **no，9/10**。画面包含鱼肉；这个 no 描述饮食属性，不能变成“没有盘子/西兰花”。来源：VQA train2014 question 136027000，COCO image 136027。

## 4. TextVQA 的目标函数问题要单独归因

[训练目标选择](../src/vlm_deep/common.py)取一条多数答案，[assistant CE](../src/vlm_fp32_adapter/loss.py)优化这条序列；生成后 [consensus](../src/vlm_deep/common.py)却按十人答案评分。两者的目标不是完全相同。

\[
L_i=-\frac{1}{T_i}\sum_t\log p_\theta(a^*_{it}\mid x_i,a^*_{i,<t})
\]

初始模型 dev1617 预测中，315 条与所选目标归一化后不同，其中 **82 条仍得至少 .9 分（占全部 5.07%）**。例如 question3502：生成 `tyburn`，所选目标 `course of the tyburn`，二者都得 1 分。单目标 CE 仍会鼓励后一条字符串。

这是需要验证的目标选择问题，不是 CE 符号写反；不能据此断言它导致全部负结果。训练目标本身的平均十人评分为 **98.6624%**，30,730/32,985 条目标得满分，不能把整套 TextVQA 说成脏数据。dev 案例只用于诊断，没有回填训练。

新的 human yes/no 数据解决的是 Object 分支的监督来源，不自动解决罕见 OCR 字符串泛化。多参考目标是后续候选，需要独立匹配对照，当前没有顺手更换 TextVQA loss、扫参数或降低门槛。

同理，[现有 DPO](../src/vlm_posttrain/loss.py)使用正确的相对偏好目标：提升 chosen/rejected 的参考校正概率比，不保证 chosen 的绝对概率或任务准确率上升。其 margin 改善不能替代能力评测；已有正控制表明链路能学习，不支持全面更换框架。

## 5. 排除与保留的疑点

| 检查 | 实际结果 | 能证明的范围 |
| --- | --- | --- |
| 公开图像池是否错配 COCO 原图 | 固定 hash 抽查训练图 16/16 与官方 2017 原图像素完全相同 | 这批样本未发现错配，不是全库证明 |
| VQA2014 与现有 COCO2017 ID 是否只在名字上匹配 | 4 张新人工问答图与官方 train2014 原图像素完全相同 | 抽查支持跨版本图像对应；不是仅比尺寸 |
| 新问答能否实际编码 | 4 条真实 CPU 编码通过；序列380–566，视觉345–529 token | 标签为 `yes/no<\|im_end\|>\n`、3个监督token；非推测长度 |
| 新标签构造回归 | 2 项通过 | 身份错配拒绝、8/10不足时不生成标签、原负问题语义保留 |
| 旧 FP32 数值修复是否再次失效 | 无新证据，复用上一轮已通过结果 | 本轮没有重复已通过的 GPU 检查 |

原始诊断：`outputs/vlm_deep_code_inspection_v4/IMAGE_ALIGNMENT.json`、`TARGET_OBJECTIVE_ALIGNMENT.json`、`HUMAN_SOURCE_ENCODING.json`、`HUMAN_EXAMPLES.json`。

## 6. 下一次训练应验证什么

首先使用同一个原始 FP32 LoRA 起点、固定数值策略，建立新 474 条人工 yes/no 的完整初始基线，同时保留 TextVQA 的匹配初始评测。然后才能按冻结协议训练完整可信监督，检验共享模型收益与 TextVQA 保持。新内部 val 图像此前用于 Object 研究，不能称为全项目未见的盲测；底座预训练重叠仍未知。

不能用旧 Object 基线与新人工数据候选跨题比较，不以 2 次更新或数据准备当提分，不从这 474 条反复挑参数。正式 POPE 和种子扩展仍由预注册能力门槛决定。当前状态是**监督数据与读取链路已补齐；新训练及独立提分尚未执行**。
