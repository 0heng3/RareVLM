# 冻结CLIP教师关系蒸馏 v1

2026-10-14用户授权参考相关项目/实习执行下一步实验。本轮在RareVLM内补强多模态Embedding、对比学习、蒸馏与能力保持，不新增独立项目、不修改正式简历。参考[OpenCLIP固定DistillClipLoss源码](https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/loss.py)，对应[阿里1688团队招聘页](https://liamding.cc/hiring.html)的Embedding/训练优化技术方向；团队页非校园招聘统一入口，不据技术匹配自动认定申请资格。

## 假设与固定变量

原完整dev冻结CLIP T2I/I2T R1为60.580/81.200%，原多正均匀负例适配64.153/78.233%。多正集合目标未改善。后续检验固定教师分布约束是否缓解取舍；历史结果不证明表征漂移就是退化原因。

只改变训练目标：`L = 原多正平均CE + 1×(KL_I2T+KL_T2I)/2`，KL方向为teacher→student。teacher/student温度均.07、不额外乘T²、不扫描lambda。Teacher由同批次冻结缓存特征归一化/点积计算，no_grad且目标detach；不使用EMA、新模型或dev/test标签。OpenCLIP软目标CE减去固定教师熵得到KL，学生梯度相同；监督项仍保留多正矩阵，不套用其一对一标签。

源数据/CLIP revision/FP32缓存、28k图140kcaption、源SHA组正项及排除规则、32anchor图和64uniform负图、132480参数、初始化seeds0/42/200、10epoch Final@10、8750更新、AdamW .001/.0001、梯度clip1.0均保持一致。每次140万标注正pair曝光。典型logits[32,480]/[160,96]；KL增加计算，等数据/更新不等于等FLOPs。encoder不训练，不称完整CLIP或VLM蒸馏。

## 复用对照与数量

只新增3个distill训练。复用`clip_retrieval_set_v1`已完成且三seed逐query/Top10重现v1的uniform_mean checkpoints，保存配置、特征、权重、初始化、完整采样序列/负计数及原dev排名SHA。新三训练必须逐seed匹配这些合同。

Dev新增3格，复用冻结+三个mean共4格，每格1000图query与5000captionquery；不能称七格都是新预测或新独立对照。相同完整图库、hit-any Recall、二值nDCG、稳定索引破并列；旧源文件不改，symlink只读复用并绑定出处。

## 预设验收与停止

三个固定checkpoint平均的逐图指标用于配对bootstrap，caption按每图5条整簇。6000次、seed20261014，三个主统计端点各单侧alpha=.05/3：

1. I2T R1对mean均值≥+1pp且下界>0。
2. T2I R1对mean容许−.5pp，下界严格>−.5pp。
3. I2T R1对冻结容许−.5pp，下界严格>−.5pp。

另要求I2T nDCG10对mean点值≥−.5pp、T2I对冻结点值≥+2pp、至少2/3seed I2T正方向。点值保护不等于统计非劣。全部dev通过才固定配置执行新完整test七格（冻结、三个原baseline、三个distill）一次；失败不追加test/seed/epoch、调lambda或门槛。

历史dev/test已曝光、pretraining overlap未知，不称盲测；新端点校正不纠正全历史探索偏差。三seed样本SD与固定checkpoint平均后的条件图像区间分开，不把seed×image当独立样本；低KL/特征相似不证明质量保持。

## 验证、学习与执行

CPU验证教师停止梯度、KL解析梯度、lambda=0恢复原CE和全部参数梯度、identity初始KL=0。真实记录批次采用原mean seed42第一步输入，保存历史训练后baseline对教师的480项分布/梯度、源图五caption，核对上游CE−H=KL；这是教学/工程检查，不计新benchmark。

新模块`src/clip_retrieval_distill/`、配置`configs/clip_retrieval_distill_v1.json`、输出`rarevlm_pilot/outputs/clip_retrieval_distill_v1/`独立。每250步checkpoint，来源锁/原UUID恢复；使用持久VLM环境和共享GPU队列，三次15秒空闲准入，不抢占/终止已有任务，竞争观察保留，不将本轮时间称独占加速。正式简历/PDF、旧源码和结果保持，不启CI。

```bash
source /opt/data/private/.tools/vlm-a100-env.sh
export PYTHONPATH=/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src
export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/opt/data/private/.venvs/rarevlm-vlm/bin/python -m clip_retrieval_distill.pipeline \
  --config configs/clip_retrieval_distill_v1.json
```

学习页包含真实分布、梯度、源码、全dev与固定前8图真实Top5；图表保留数值/SD/缩略变换并人工检查，工作流参考[Kassis等，2026](https://doi.org/10.48550/arXiv.2609.00065)。无期刊目标、attention图、工业搜索/ANN/Reranker/线上SLA或本人独立掌握声明。实时看STATUS/COMPLETION及analysis/dev/ANALYSIS.json。
