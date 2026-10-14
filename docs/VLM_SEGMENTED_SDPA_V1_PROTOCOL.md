# 按cu_seqlens分段合批的视觉SDPA研究

2026-10-13。独立模块`src/vlm_segmented/`、配置`configs/vlm_segmented_sdpa_v1.json`、本地输出`outputs/vlm_segmented_sdpa_v1/`。新队列等待已有GPU任务结束后按三快照/UUID锁准入，不抢占。当前CPU数学合同3项通过，真实GPU特征与完整质量/性能尚待验证；不能提前写为简历成果。

## 真实动机与开源基础

之前的`vlm_batching_v1`压力预检：batch16最长输入峰值allocated约24.664GiB；batch32超过预设65GiB门槛，被拒绝进入完整评测。这是协议容量拒绝，不等于80GiB物理OOM或所有batch32都不可运行。原五格协议与失败记录保留，可行三格由补充调度器接续；空闲另一卡运行同策略硬件复核，不是训练种子。两个完整FIFO4基线已完成，各1617QA，内部共识分均87.111936；不能与官方TextVQA5000QA的81.138混为同一列。

读取锁定Transformers4.51.1的[`Qwen2_5_VLVisionSdpaAttention`](https://github.com/huggingface/transformers/blob/v4.51.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)，确认创建`[1,N,N]`布尔mask，按`cu_seqlens`把允许注意的块设为True。N是合批后的premerge视觉patch隐藏序列长度，不是语言input长度或合并后的image_pad数。单张mask存储按N²增长；实际allocated还包含权重、QKV、SDPA工作区、生成KV和其他张量，不能仅凭源码把全部显存/时延归因mask。

岗位关联：补充VLM模型源码定位、计算/显存优化及质量保护评测，保留已有QwenSFT与CLIP校准。该优化不是JD明文共同门槛、生产服务经验或新增独立项目；业务场景与个人贡献/掌握继续单独核对。

## 原理与实现

`Attention=softmax(QKᵀ/√d+M)V`。原mask允许范围是互不相交的块，块之间不通信。保持qkv投影、原始位置RoPE与输出投影，按cu_seqlens切块，把相同长度的块放入SDPA批量维度，不带mask计算，再用原索引写回。窗口层和全图层都使用各自实际cu_seqlens；不依据问题、答案、错误或得分调整。

代码[`attention.py`](../src/vlm_segmented/attention.py)：等长分组、index_select、`[segments,heads,length,head_dim]`、SDPA(dropout0/noncausal)、index_copy还原，32个视觉SDPA层；语言注意力/LoRA不变，冻结模型仅推理。共享layout索引缓存最多64项，切换条件清空，每格额外预热。该执行变化包括分段、合批、索引/缓存和SDPA后端选择；不能把全部收益归于单一因素或未实测的FlashAttention内核。

小张量数学合同检查：带非零RoPE的混合长度分段与原始稠密输出接近、不同块输入扰动不影响其他块、非法分段与训练模式被拒绝。它们是实现检查，非真实任务提分结果。

## 固定变量与完整对照

普通CE seed42最终adapter、原权重BF16、原生整图、视觉预算上限1024、min_pixels50176、原问题/后缀、无截断、SDPA、TF32关闭、贪心最多32新token、最后位置lm_head切片全部保持。逐题input IDs、像素SHA、grid和视觉token核对历史输入合同。参考答案只在生成后评分。

| 同卡组 | 固定batch与QA顺序 | 比较 |
| --- | --- | --- |
| batch4 | 4，原manifest顺序 | dense4 vs segmented4 |
| batch16 | 16，原manifest顺序 | dense16 vs segmented16 |

4格各1617QA/1024图，共6468条评分预测，0新训练。不能用dense4对segmented16单独归因注意力变化。另以同256QA、每组两条件、三个固定顺序轮次计时，共3072不评分观察，不能当独立样本/种子。

## 预检、指标与验收

资源释放后先在4条真实QA上，用原模型层输入和RoPE分别计算第0/7视觉层的原始/分段输出，保存实际hidden shape、cu_seqlens、dtype、max绝对误差与relative RMS；要求finite且relative RMS≤0.01。此检查只针对两个局部层的相同输入，不证明整个encoder或回答等价。每组各条件再做首批/最长输入全batch的finite logits、生成、编码与65GiB容量检查，通过后才全dev。

完整回答按QA ID重计十参考LOO共识，图像整簇配对bootstrap6000次。整体允许损失0.25pp，两项预设同batch主比较使用单侧alpha=.025，下界必须高于−.25pp；探索双侧95%区间同列。rare/frequent实际全量SFT target曝光切片点值至少−.25pp，不证明切片非劣。推进还需三次同卡阶段吞吐比中位数≥1.5、每次≥1.1、allocated≤65GiB、资源观察无竞争/查询失败。

记录纯生成、CPU原图/校验/processor＋H2D＋生成、完整cell session wall、load/preflight/额外预热、allocated/reserved。内部dev已多轮探索、固定一个checkpoint，结论仅是条件探索筛选，不证明无损、盲测、种子稳定性、能力保持或线上服务SLA。新数学执行关系不保证BF16逐bit一致；正式集不回调策略，不自动新增种子/正式评测/简历/CI。

## 边做边学与成果归档

解释链：实际batch32容量失败 → 读取[官方固定版本源码](https://github.com/huggingface/transformers/blob/v4.51.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) → 画cu_seqlens允许块 → 理解为什么块能成为batch维度 → 小合同 → 真实局部特征 → 完整QA质量与成本。mask图按真实cu记录重建并明确下采样，是允许范围，不是学得注意权重。原图显示副本只等比缩小，模型输入保持原图。完成后输出8预先按ID选择的图像/回答、四格成本图、CSV、重复计时和配对区间，个人理解仍需独立复现与讲述核对。

持久锁包括配置、基础输入计划、adapter/权重谱系、执行源码、Transformers实际文件SHA、包版本和GPU UUID；已保存完整批次可在同一绑定下恢复，拒绝静默混用。

图表审查工作流引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，arXiv当前记录v2已核对；这是图表方法来源，不是算法性能证据。
