# LoRA Coordinate Shrink：动态耗时身份比较的版本化恢复协议

本恢复只解决stage33完成Object后的工程读取错误。原`VLM_LORA_COORDINATE_SHRINK_V1_PROTOCOL.md`、96科学源码、配置、59项CPU凭据、STUDY锁、导出、GPU输入、Object预测/评分和失败证据全部保持原字节；原失败状态不改写为成功。用户完整真实提升目标保持active。

## 已定位的唯一不稳定身份字段

冻结`export.verify_export`每次产生工程计时`verify_CPU_wall_seconds`，冻结`adapter_binding`将整个返回值嵌入`training_binding.export_verification`。冻结`read_cell`重新验证后比较整份binding，因耗时变化拒绝已经完成的Object。原保存值为0.23652817774564028秒。两份只读定位均确认其余稳定字段一致，导出manifest SHA、原producer TRAIN_RESULT SHA及实际GPU原生BF16/再升FP32摘要分列正确；没有读取权重或重新评分来猜测原因。

新`src/vlm_lora_coordinate_shrink_recovery_v1/`提供显式版本化reader/analysis/编排及锁，不能修改旧globals。稳定身份视图只允许从copy中移除精确路径：

```text
training_binding.export_verification.verify_CPU_wall_seconds
```

两侧该值都必须存在、为有限非负实数且不是bool。其它字段、键集合、嵌套结构、路径、整数、QA/源/参数/SHA全部仍精确相等，原`verify_export`全部数值与来源检查照常执行。验证耗时单独作为工程遥测保留，不能修改旧INPUT_LOCK或其contentSHA，不能整体删除export_verification或放宽任何数学/质量门槛。

## 来源与实际执行

原STUDY SHA为`283b63a86a8aa0f60892512e57dd0f9e46533a0796d7972e41b2a118e6d800af`。失败归档索引位于原outputs下`independent_review/execution_failure_archive_20261016T023011Z/INDEX.json`，SHA `6fa1fcb884b9e984f974d8d80e886d59381f52185506d295114977f63672bfba`；54文件原件/副本一致，原102文件源码快照0差。

新RECOVERY_INPUT_LOCK须绑定原config/STUDY/协议、原96科学source、已通过CPU/prepare/独立导出验收、初始/真实producer/导出manifest链、已完成Object三工件及8QA预检、失败索引/原退出记录，并绑定全部新增恢复source/tool/protocol/tests与新CPU凭据。只直验必要小metadata/source；不重审父teacher流、模型或历史训练state。权重完整性继续由冻结evaluator在每次实际GPU模型加载前按原锁验证。

Object342正式QA+8额外评测工程QA已完成并原字节复用。当前只新增原配置Text1617正式QA+8额外评测工程QA，直接调用冻结evaluator数学/生成路径。新worker、queue、admission、runtime receipt和STATUS放独立恢复命名空间，绑定有效恢复锁、原STUDY及原角色GPU UUID，不能覆盖旧失败worker或重新生成Object。原actor训练、teacher生成和CPU导出均不重复。

新analysis显式记录恢复锁/有效恢复source及冻结的原数学依赖；不能把恢复代码遗漏为未经声明的wrapper。若原后续函数需要原`source_sha256`数学依赖字段，新增mandatory恢复来源字段单独完整绑定，并由恢复控制器/独立reader严格验证。旧STUDY原字节不变，不伪造旧锁包含新source。

## 完整门槛与后续编排

恢复仍完成全部1959正式内部model×QA，独立评分1959条新增候选预测、历史initial/KL控制不重评。原八门槛、6000次图像簇bootstrap/seed20261015/单侧98.75%、Text整体NI及rare/frequent点保护不变。

pilot失败只结束此固定alpha候选，不扩种、不继续alpha网格、不按dev题修补。只有新候选真实全过才能接续原0/200 fresh teacher→181步unshrunk producer→同一0.5导出→完整内部三seed筛选→各seed完整4372 external→各seed完整9000 POPE及Text保护→root独立终审。恢复编排须在GPU接续前实现并CPU覆盖原完整条件分支及恢复，不以仅pilot的stub替代。原stage32负结果仍为false，不能借其伪准入。

所有质量条件以原研究协议为准，不因工程恢复重定义成功。自动全过状态仍只能为`confirmation_passed_pending_independent_review`、`goal_complete=false`。工程恢复通过不等于算法提升；不能修改简历、CI或自动推送。

## 准入与验证

新CPU检查必须覆盖：实际消费者可接受同稳定字段/不同计时；缺失、bool、负数、NaN/Inf计时拒绝；任一其它字段变化拒绝；copy不修改旧binding；原Object缓存原字节复用及完整Text接续；负pilot/三seed/external的禁止扩展；全成功完整链仍pendingroot；精确GPU UUID/源/完整完成件优先于自然controller退出。

新增源码冻结、统一CPU和真实prepare-only通过、原controller/worker已自然退出后，才按原40GiB/利用率<10%/无compute、3次间隔15秒共享准入接续。只读观察超时不是任务终止，不据此重启。学习页最终结果须来自完整analysis及独立评分凭据；当前尚无本候选完整质量结论。
