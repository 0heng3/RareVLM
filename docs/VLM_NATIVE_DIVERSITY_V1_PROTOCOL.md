# 纠错监督覆盖与重复：同预算DPO V1

2026-10-26，在新reference/训练前冻结。问题：固定QA曝光下，增加独立且有图像支持的纠错监督，能否提高原生问答泛化？数据覆盖是待检验因素，不预先认定为唯一瓶颈。

依据：前visual_preference_v1完整1617起点87.198516/CE87.303649/DPO87.631416%，候选+.432900pp/95%跨0未达标。新矿工3072QA完整、195候选/194target；按固定顺序审核前152，70支持/82不准入或不确定，余43未审。支持中64 single/6 multi；新64 single为64不同normalized target，均不在旧135支持target中。助手非专家原图审核，非整个数据集错标率。语义等价、题义/品牌与赞助商歧义、图问不匹配、不可核实细字未准入；原GT/预测/分数不变。

**数据对照。** 旧135支持池按原顺序取每stratum前64个不同normalized target，组成旧128（64 single/64 multi）；加新审核选定64 single形成192（128 single/64 multi），每题一图且target唯一。按连续4QA成组，旧32组和新16组各取偶数组为控制：旧64+新32=96。共同锚点是控制的旧64。

| 角色 | 变量部分192次曝光 | 共同锚点 | 纠错总曝光 | 独立纠错数 |
|---|---|---|---:|---:|
| repeat96 | 96各2次 | 同一旧64各1次 | 256 | 96 |
| diverse192 | 192各1次 | 同一旧64各1次 | 256 | 192 |

因此控制旧64各3次/新32各2次；候选锚点旧64各2次、其他旧64与新64各1次。两组纠错曝光均旧192/新64、single160/multi96；不能漏报共同锚点或说所有样本均匀重复。每组仍为纠错256/native128/OCR128=512 chosen，另256 negative；相同32步/4chosen micro/effectivechosen16。沿原OCR4/hard4/native4/hard4的更新槽位；每两更新的四hard组为old/old/new/anchor，32个old变量槽和16个new变量槽使用上表覆盖/重复，16个anchor完全相同。普通回放顺序逐字节相同，训练/评测图像及像素隔离。独立QA总量控制352/候选448。等QA不等于等有效token、信息或FLOPs；样本难度/长度构成随覆盖一起变化，不能称纯重复次数因果。

**固定训练实现。** 直接复用已完成并冻结的`run_visual_preference_v1.train()`，只绑定各unit的ROOT/数据/config，不改原源码或训练循环。两组均CE+DPO，λ=1、β=.1、原sum-logP reference DPO及每QA有效assistant token均值CE、原EOS/assistant mask。共同native_scope复合初始adapter，语言144 FP32因子冻结；视觉blocks2–31 qkv rank12/alpha24的60 FP32因子/1843200参数训练；BF16 imagewise vision+FP32 language/head、TF32off/dropout0、视觉实际非重入checkpoint、1024视觉上限、greedy32/batch4评测。同seed42/lr2e−4/warm4/linear32/AdamW wd0/clip1，不扫参/扩步。

新reference在实际4chosen+4negative的同batch上冻结：控制24唯一batch、候选48唯一batch，每组64排程键，共576 teacher-forced序列、0更新；不要求复用旧不匹配的32唯一batch缓存。首个实际更新核对reference relative margin≤.05、DPO近log2、选中scope/梯度/FP32母参数保存；两组训练题不同，CE不要求相同。已过旧加载/数值/251图审查不重做。

每个GPU作业独立reference→32训练→960内部评测，控制reference提前完成可先训练，不等另一卡；共享队列3×15秒空闲、≥40GiB/<10%/0compute/UUID锁，不抢占。reference≤600秒、训练≤2400秒、内部评测≤1500秒、allocated≤65GiB。复用原完整optimizer/RNG断点。各unit的参考缓存和输入锁独立；main/runs为同一unit工件的路径别名，不计为重复训练。

**评测与推进。** 原共同起点960+1617预测逐字节复用、0新起点生成；原native_hard128只作旧纠错诊断，与本轮96/192训练集合并非完全一致，单独报告训练成员关系，不能当双方完整训练准确率。两Final各960生成；候选在该原128诊断需≥32满分且对起点VQA≥20pp，并守OCR/native512/单词/多词/rare366/ret64各≥−.5pp，对repeat96的OCR/native512/rare也≥−.5pp，才补各1617。保护/可学门槛不是能力成功。

完整1617主信号仍要求候选对共同起点及新匹配repeat96控制**均≥+.5pp且配对图像簇95%下界>0**；另报告前DPO128强参考87.631416及逐题比较，候选点值不得低于它；完整rare342相对起点及新控制均≥−.5pp。旧强参考与本轮数据构成不同，不将其差单归因覆盖。原paired实现2000次图像簇bootstrap、固定checkpoint条件区间，非三seed、外部盲测或全历史多重检验控制。失败保留原门槛，停止扩步/扫参/seed/正式集；局部通过仍待原强基线/三seed/外部保持目标，不自动执行。

新增预算：64优化更新，第一阶段1920生成，条件3234，最多5154生成；reference576序列另计。所有新训练数据原SFT train_all已曝光，不称底座未见。CV/科研/Agent/CI不改，不push、不终止其他任务。
