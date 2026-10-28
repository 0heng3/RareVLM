# 可靠训练难例的来源梯度诊断 v1

用户2026-10-18授权执行修复。数据准入和dropout修复使用独立新版本，不修改旧科学源。此实验仅测参数梯度，0 optimizer对象、0更新、不生成答案、不读确认结果、不训练新模型。

固定initial和普通CE Final200两个checkpoint，同一BF16底座/FP32 q/v LoRA1843200参数，无AMP/TF32，dropout0，非重入gradient checkpointing；训练processor1024、micro4、tail-head与旧训练一致。新metadata入口同步PEFT dropout配置0。只加载一次底座，随后精确恢复第二个原adapter，前后LoRA哈希必须不变。

样本：只使用新训练准入池。负例探针为蜡烛条纹no与单猫镜像no；正例探针为巴士同色yes与蓝色砧板yes。四图均由助手查看原图支持GT，属于真实训练难例，非专家重标；不保证能代表全部错误。它们在同一4QA微批中测两组首答案平均NLL梯度。普通人工来源选择旧排期中最先两组完整4QA、全部获新准入且不含上述探针的微批，共8QA；回放为原排期最先两个Text微批，共8QA。普通人工未经全量逐图复核，不能称所有均已验证干净。

每组梯度按本组QA平均，原始FP32可训练参数autograd.grad，CPU FP64复制/点积。普通人工额外测首答案梯度，两次反传复用同一forward；Text保留各题实际token均值。不给探针梯度加进合成更新。

比较两个明确的局部方向：

```text
G_CE = .5 g_TextQA + .5 g_普通人工QA
G_answer3 = G_CE + (1/3) g_普通人工首答案
```

第二个方向对应人工loss从`(L_answer+L_end+L_newline)/3`变为`(3L_answer+L_end+L_newline)/3`，保持格式项系数；不再次归一，不声称同梯度范数。合成系数来自16QA中的8Text/8human，普通human内容是固定局部诊断批，不冒称全部训练平均梯度或历史某一步实际Adam更新。探针均权重0，只比较其损失受该方向的一阶影响。

对探针损失L，假想未预条件SGD一步的一阶变化为`−eta * grad(L) dot G`。dot<0说明该局部方向一阶上增大L；pairwise负cos不自动意味着合成方向有害。两个固定点也不证明整段Adam轨迹因果，不保证有限步loss或能力提升。

来源、样本、规则、新源码和两checkpoint文件SHA在首个GPU forward前冻结。共享GPU队列要求零compute、>=40GiB、<10%利用率、三次15秒采样与UUID锁，不终止其他任务；只用一张空闲卡。worker累计上限10分钟，按微批边界停止；没有自动扫样本/权重/学习率、optimizer step或重试。仅新组件loss与梯度数学CPU测试，其余已通过加载/mask/save证据复用。

解释决策：若首答案加权方向对可靠no探针仍不利，不能凭热门配方立即启动加权SFT；若局部改善，也仅作为下一单变化短预算对照的立项线索，仍须独立生成得分、匹配CE和保持门槛。数据准入修复不等于模型已提分。
