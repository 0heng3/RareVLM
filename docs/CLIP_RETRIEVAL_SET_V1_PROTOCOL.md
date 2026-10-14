# CLIP多正集合损失：受控后续研究

2026-10-14，用户授权“精读相关仓库，执行下一步行动”。本轮在RareVLM内补强图文对齐、对比学习和评测，不新增独立项目或修改正式简历。精读OpenCLIP、BLIP、lmms-eval固定源码，归档和提交在本地输出目录的`upstream/SOURCES.json`。

## 问题与假设

历史`clip_retrieval_v1`完整dev：冻结/随机/混合困难负例的文搜图R1为60.580/64.153/60.820%，图搜文R1为81.200/78.233/78.033%。随机策略存在双向取舍，困难负例比随机差；这些观察用于提出假设，不证明退化机制。历史test已曝光，预训练重叠未知。

现有图搜文目标平均所有标注正caption的负log概率；hit-any Recall要求前K命中任一标注正项。候选改为负log正集合总概率，检验这种目标差异能否改善图搜文，同时保护文搜图和多正项排序。它可能偏向已有高相似度caption，必须报告nDCG。这是已有多正学习目标的对照，不是原创算法或BLIP/SigLIP复现。

## 唯一改变与固定条件

令`s_j = cosine(z_image,z_caption_j)/0.07`、`P`为同源图SHA组的标注正项集合。

- `uniform_mean`：`L_I2T = logsumexp_all(s) - mean_positive(s)`，直接调用冻结v1完整loss函数。
- `uniform_set`：`L_I2T = logsumexp_all(s) - logsumexp_positive(s)`。
- 两者`L_T2I`保留v1多正平均CE，最终`L=(mean_I2T+mean_T2I)/2`。loss尺度不同，训练loss数值不能直接比较优劣。
- 同一完整28,000训练图/140,000caption、1,000图dev/5,000caption。WebDataset、预处理、CLIP-B/32 revision、FP32缓存与划分从v1只读复用；整文件SHA锁在`REUSED_DATA_LOCK.json`。
- 各seeds`{0,42,200}`、10epoch、Final@10、8,750更新、双侧残差adapter132,480参数；AdamW lr=.001、wd=.0001、clip_grad=1.0、FP32、TF32关闭。每次1,400,000标注正pair曝光。
- 每步32锚图及全部5caption，加64张uniform负图及全部5caption，典型logits为图搜文`[32,480]`、文搜图`[160,96]`。原锚点/同源SHA/相同规范化caption关联图排除规则不变；不声称消除了全部语义误负。
- 完全相同初始化、锚点顺序、每步负池、更新与正暴露；最终逐seed比较初始化SHA与完整负例计数SHA。不改变温度、encoder、head宽度或训练预算。
- 六训练全部结束后做冻结+六checkpoint共七个完整dev格；不按epoch/dev选点。完整图库、稳定索引破并列、hit-any Recall和二值nDCG沿用v1，不作ANN、Reranker或线上性能声明。

## 预设门槛与测试规则

三个固定checkpoint的逐图指标平均用于配对估计；图搜文每图一query，文搜图保留同图五caption整簇。6,000次图像bootstrap，seed20261014；三个主统计端点各单侧alpha=.05/3：

1. set对mean图搜文R1均值至少+1pp，单侧下界严格大于0。
2. set对mean文搜图R1容许损失.5pp，单侧下界严格高于−.5pp。
3. set对冻结CLIP图搜文R1容许损失.5pp，单侧下界严格高于−.5pp。

另要求图搜文nDCG10对mean点值至少−.5pp、文搜图R1对冻结点值至少+2pp、至少2/3seed图搜文正方向。点值门槛不等于对应统计非劣。

完整dev全部通过后，固定候选与设置，执行冻结+六checkpoint的完整历史test一次。dev失败不追加test、不放宽门槛、不追加epoch/种子。新协议不纠正历史探索偏差，不称首次盲测；test不回调loss、参数、种子或阈值。三seed均值±样本SD与固定checkpoint平均后的条件图像区间分开，不把seed×image当独立重复。

## 精读与零GPU证据

- OpenCLIP：一对一标签、对称CE、scale、gather梯度范围，与本项目多正矩阵对照。
- BLIP：同ID多正/动量软目标、ITM逐query负采样、跨模态交互、TopK重排；匹配分支困难负例不能等同共享对比负池。
- lmms-eval：十参考LOO、归一化、可选OCR/问题大小写、Qwen像素范围；执行固定上游评分函数/class节点，对两组已保存TextVQA各5000预测逐题复核。仅评分函数级对照，不称完整框架/input等价。
- 保存真实seed42第0步训练query的480 logits、正项、softmax与梯度，以及真实训练图和五caption；工程检查不计新增R@K成果。
- 复用v1六份负例计数，报告最大值、top1%份额与Gini；曝光集中不证明语义误负或退化原因。
- 学习页含原理、固定源码位置、真实图片/梯度、完整dev结果与固定前8个dev图搜文Top5。缩略展示与224中心裁剪区分，不作attention图。

## 执行与复现

代码`src/clip_retrieval_set/`，配置`configs/clip_retrieval_set_v1.json`；本地输出`rarevlm_pilot/outputs/clip_retrieval_set_v1/`，从工作区项目目录解析。

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
export PYTHONPATH=/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src
export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/opt/data/private/.venvs/rarevlm-vlm/bin/python -m clip_retrieval_set.pipeline \
  --config configs/clip_retrieval_set_v1.json
```

沿用共享UUID锁/三次15秒空闲确认，动态用空闲A100，不抢占或终止其他任务。250更新保存optimizer/model/step/完整负例计数；采样stateless，按原UUID及来源锁恢复。冻结v1源文件/结果保留，不使用CI，不写“本人已独立掌握/实现”。

可视化使用既有Matplotlib环境，保留源表与展示变换，标注SD与条件区间，人工检查输出；参考[Kassis等，2026，Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，无期刊目标。

实时状态以`STATUS.json`/`COMPLETION.json`、`analysis/dev/ANALYSIS.json`为准。
