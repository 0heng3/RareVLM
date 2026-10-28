# LoRA Coordinate Update Shrink v1：事前研究协议

## 问题与唯一候选

stage32完整负结果及独立验收已结束：generated-prefix full KL仍只通过6/8项内部门槛，Object +2.046784 pp，Text −0.105133 pp，rare/frequent点保护失败。该候选不扩种；新研究不改旧源码、权重、评分、门槛或结论。

唯一新候选为`lora_half32`：对共同SFT42 canonical初始化与stage32同法训练的FP32 LoRA Final，逐坐标执行固定更新收缩。研究检验较小的LoRA因子坐标位移能否在保留物体收益的同时减少Text行为扰动；不预设KL系数不足、梯度冲突或硬目标锐化为已证实原因。

## 固定变换与数学边界

系数事前固定为`alpha=0.5`，没有搜索、验证集择优、逐层系数或样例补丁。每个A/B张量在CPU上依次计算：

```python
theta0 = raw_initial.to(torch.bfloat16).to(torch.float32)
delta = final_fp32 - theta0
theta_half = theta0 + delta * 0.5
```

FP32逐步运算顺序固定，不采用FP64中间值、PEFT加权合并API、SVD、参数重排或重新分解。共同canonical初始张量SHA为`6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7`；seed42源Final为已验收stage32 FP32张量SHA `d86e0ce25d30c57377e660a9c35cecdc03b3db2205338da81292a9239b13c485`。

保持144个语言q/v LoRA张量、rank8、`lora_alpha=16`、1,843,200个参数元素。不得通过降低`lora_alpha`同时缩小原SFT更新。FP32导出文件与实际评测BF16转换的摘要/精度分开记录。语义是同一初始化和训练谱系中的**因子坐标**收缩；一般不等于dense矩阵`BA`的中点，也不能保证有效权重距离、KL或预测变化单调减小。两端A/B均训练时，交叉项按alpha平方缩放。

WiSE-FT提供固定0.5的参数插值参考，不是本方法与dense WiSE-FT等价或有效性的证据：[原论文](https://arxiv.org/html/2109.01903v3)、[官方代码](https://github.com/mlfoundations/wise-ft)。[PEFT官方合并实现](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/tuners/lora/model.py)中的`linear`不能替代上述明确计算。精确dense拼接会增秩/增元素，本协议不采用。若本候选失败，不沿同一内部dev追加alpha网格。

## 训练谱系与来源

seed42零新增训练：仅引用stage32实际181步、2888 object+5792 replay的已验收Final，并保存明确的历史producer引用及独立导出manifest。不得复制并改名伪造新`TRAIN_RESULT`，不得把导出说成新训练或新算法已提升。旧stage32完整负结论保持false，不伪造其passing pilot。

只有新候选完整pilot全过，才允许新seed0/200 producer：同一SFT42 canonical初始化、同一generated-prefix full-vocabulary forward KL训练数值协议，各181步，object16/replay32、micro4、lr5e-6、warmup0.1、wd0、clip1、lambda1/T1；QA/assistant分别均值、词表151936求和，FP32语言LoRA、冻结BF16底座/视觉、BF16 autocast。生成prefix沿原冻结初始模型，每seed5792QA，无GT筛选/重采样/重新分词；保存实际IDs/EOS/roles，参考模型独立冻结，不使用概率bank。复用数值源码须显式记录来源，新编排/准入/来源契约在独立模块中版本化，不修改旧globals。

每个新producer的原始Final与完整训练结果保留；收缩adapter及其manifest单独保存，只有后者是评测候选。每个seed应用同一alpha与初始化；seed是数据顺序/回放轴，非三个独立SFT初始模型。固定Final@1 epoch，不据dev选checkpoint。

## 完整成功范围（不得缩减）

先完成新候选内部342 Object+1617 Text共1959正式model×QA；独立初始/原KL控制只引用已验收结果，额外各8题评测预检分列。保留八项门槛：

1. Object accuracy相对initial至少+1 pp；
2. Object增益98.75%单侧下界严格>0；
3. Text整体增益98.75%单侧下界严格>−0.5 pp；
4. Object recall点值不下降；
5. specificity点值变化至少−0.5 pp；
6. 固定逐类yes/no均衡子集点增益严格>0；
7. Text rare点值不下降；
8. Text frequent点值不下降。

pilot全部通过后才生成/训练0、200并导出，组成0/42/200三seed。按每QA先跨seed平均，再图像簇bootstrap；三seed平均须过原八项，每seed Object正增益且Text整体变化至少−0.5 pp。只有其通过才进入每seed完整4372 external：整体accuracy平均至少+0.5 pp且下界>0、每seed正增益，F1/recall不下降、specificity变化至少−0.5 pp、固定balanced子集点增益>0。external原3438 balanced QA/1742图按QA汇总，不用宏平均替代。

随后每seed完整9000题POPE（共享500图、random/popular/adversarial各3000）：accuracy平均至少+0.5 pp且下界>0、每seed正增益，F1/recall不下降、specificity变化至少−0.5 pp，每类别点值不下降，保留Text整体及rare/frequent保护。不得用训练18类内部物体收益替代完整80类POPE确认。

四主端点为内部Object、Text整体NI、external Object、POPE accuracy；每端点alpha=.05/4=.0125。6000次配对图像簇bootstrap、随机种子20261015、98.75%单侧下界。切片点保护不等于切片非劣效/等价；不宣称历史所有试验的FWER或预训练未见。只有一个候选，历史KL/CE不参与替补选择。

最后必须由root独立检查完整来源、producer/导出模型、实际输入、预测评分、病例、成本及全部门槛。自动控制器即使全部通过，也只可写`confirmation_passed_pending_independent_review`和`goal_complete=false`；root完成全范围验收后才可完成用户目标。

## 工程、资源和学习

新实现放`src/vlm_lora_coordinate_shrink/`，新配置与输出分别为`configs/vlm_lora_coordinate_shrink_v1.json`和`outputs/vlm_lora_coordinate_shrink_v1/`。导出、历史producer、初始、源文件、协议、数据与评分来源均锁定摘要；不递归重审旧teacher流/大概率bank，不重训seed42。CPU合同及真实prepare通过且完整后续编排已实现后才准入GPU。

共享GPU准入沿原规则：至少40GiB空闲、利用率<10%、无compute进程，连续3次间隔15秒确认；role/UUID锁定与恢复，不操作其他任务。保存导出/训练检查点、完整预测batch与恢复状态；观察超时不代表任务终止，不据超时自动重启。

成本分别报告历史181训练谱系、零新增seed42训练、CPU导出、额外预检、正式生成和条件后续训练。保持参数元素数不等于输出相同，也不证明加速/省显存。沿原原生BF16/1024视觉token上限、batch4、贪心32token、相同提示和评分；参考答案仅评分，不能进入推理或路由。不引入benchmark身份路由或多adapter回退。

学习材料连接原图、真实坐标计算、因子与dense交叉项、导出精度、源码、原始回答和实际门槛。教学值与实测值分开；没有完成结果不写收益，不修改简历、CI或自动推送。
