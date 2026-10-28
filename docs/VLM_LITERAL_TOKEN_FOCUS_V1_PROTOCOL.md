# 完整词的最难词元损失：固定32步算法先导

2026-10-24，在新模型输出前登记。任务匹配回放128步已完整失败：纠错70/128，新128词117/128（初始121），native512=92.480469%（初始93.222656%），回放127/128只属训练保持；不追加该配置。在线训练词CE由首周期1.87560降到末周期.93024，但并非同一固定模型的逐题比较，不能据低平均loss直接认定训练链路异常。

## 先验证词元目标是否仍落后

固定上轮大小写canonical Final128（69/128正确）和其128个审核训练词；沿已验证的`token_statistics`，测每个GT-prefix词元的NLL、GT rank、GT减最佳其他词元的logit margin，正文与实际EOS分列。只改本次诊断的数据对象，不重审旧全局加载/mask，不用任何验证样本选择词元或loss参数。完整128均保存，不能只挑低loss例。

准入条件预先固定：在自由生成错误的训练词中，至少16词的固定模型全答案平均CE<=1、且至少一个正文token margin<0；这些词的最难token NLL/平均NLL中位数>=1.5；所有生成错误词的EOS CE中位数<.1。满足只说明存在可检验的词元目标问题，不能证明是唯一病因或保证新loss泛化。未通过则停止本分支、0新optimizer，不改阈值。

**学习率口径纠正。** 本轮逐值核对实际`vlm_object_train.train.learning_rate`，warmup后实现为`lr*(total-step)/(total-warm)`，即线性衰减；旧协议的“cosine”文字与实际函数不符。旧科学文件及负结果不回写，新对照沿实际日程，并检查复用step32的optimizer LR。这个记录错误不等于已确定失败主因。

## 唯一变化：纠错QA的loss聚合

普通目标：`CE_q = sum_t NLL(q,t)/T_q`。候选仅对hard词改为`H_q = max_t NLL(q,t)`；native回放仍用普通CE。最大值覆盖全部有效assistant token，包含实际EOS，不删EOS、不包含prompt/padding。每个hard答案把梯度集中到当前最差词元，避免容易词元平均后掩盖瓶颈；最差词元会随参数改变，硬选择可能引发振荡，是否有效看实测。该干预同时改变了hard的有效梯度强度及与长度的关系，不能把一切差异唯一归因于EOS稀释；不是声称新SOTA或RL成果。

只复用大小写实验的原128hard+256native、canonical hard标签、原CPU输入与schedule前32步。每步native4/hard4/native4/hard4，effective16/micro4；hard每题2次、native每题1次，各256曝光，共512QA。native标签和所有评测参考不变。两条件均为同一128步LR日程的前32步：lr2e-4、warm8、128步horizon的线性衰减；不当作独立32步收敛日程，也不与其他warm4/32cos协议混同。Final32固定，0按验证挑checkpoint，0自动追加128步/seed。

mean32控制直接复用旧canonical完整closed checkpoint_000032的原FP32 adapter，核对保存的step32探针模型hash，不重训已完成32步；初始和控制hard预测同输入复用。max32从原SFT5d8 fresh Adam/RNG，不从任何负Final接训。共同语言q/v144 FP32/1,843,200参数、BF16底座、冻结FP32 head/视觉、dropout0/TF32off/seed42/1024视觉上限、wd0/clip1。等QA、等prompt/labels、同样padding与forward shape，loss改变仍可影响运行时间，记录真实成本，不预设加速。

新loss仅增加一次局部数值验证：合成可变mask样本的均值与既有completion_logps一致，max>=mean，max梯度只到该QA最难有效位置，padding/无预测尾部无梯度；实际第一批从同logits核对mean和已验公式、逐样本CPU SHA、有限有效梯度。底座/加载器/旧数据/旧科学源码不修改。复用完整FP32 master/Adam/RNG/trace offset/UUID断点。

## 评测、推进与能力边界

原初始五队列预测只读复用。mean32只新增新词128+旧词256+native512+保持64=960次生成，hard128复用旧step32。max32生成完整hard128+新词128+旧词256+native512+保持64=1088。所有验证集不改、不加入训练；给定词区域不是整图定位，原图曾进原SFT train_all，内部dev反复使用，不称盲测。

先导推进信号：max32训练词比mean32至少多对16词；新词128及native512相对初始的点差均>=0、相对mean32也>=0；旧词/保持/单多词及rare1–5点保护>=-.5pp。该信号仅决定是否值得另登记长程对照，不是能力成功；本脚本即使通过也不自动长训。原正式提升要求保留：训练>=96/128只是可学性，新词相对初始>=+3pp/95%图像簇下界>0、native>=+.5pp/正下界，原保护条件及对强基线/三seed复核仍需完成。报告新loss对mean32的完整配对区间，不只对退步历史模型比较。

区间6000次配对图像簇bootstrap，单seed固定checkpoint条件探索，不涵盖seed稳定性；多端点不作全局显著性结论。固定预览8训练词，以实际词图、token片段、NLL/margin和保存自由生成回答讲解teacher forcing与greedy差异，不能把词元概率图称为视觉attention。

诊断300秒、max32训练900秒、每臂完整评测1500秒、allocated<=65GiB；正常共享零compute/>=40GiB/<10%/3×15秒/UUID锁准入，已有任务不抢占。所有新增内容在持久目录，CV/论文/Agent/CI不改、不push。
