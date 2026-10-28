# 视觉merger迁移研究：执行入口v2

继承`VLM_MERGER_TRANSFER_V1_PROTOCOL.md`全部科学设计、数据、模型、64步和验收门槛。v1已完成CPU准备，尚无GPU/optimizer；准备冻结与可选元数据修复发生在结束边界，原验证拒绝变化。已将v1 study恢复为PLAN原SHA5522db…，原冻结文件保持。

独立`evaluation_v2.py`仅将保持64题缺失的`frequency_slice`改为可选读取，值为None；保持集不作频次切片。对新旧evaluate函数作AST比较，仅这1处读取不同，其余包括本次新增的1617末批1题恢复、累计评测预算及生成逻辑完全一致。该修复属于新入口，不是此前负提升的已证原因。

`execution_v2.py`明确调用新评测函数，复用原冻结train、prepare产物和analyze，不修改旧函数全局、不重复1617编码或初始化检查。`EXECUTION_V2_LOCK.json`绑定原PLAN SHA、新入口/评测/工具/本说明及差异/输入字段检查的SHA；工作流只使用`queue_v2`，运行三组initial/candidate/qv_control。旧v1入口不用于本次执行。

GPU前原PLAN及新执行锁均须通过；相同共享空卡准入和模型/UUID绑定，原始Adam/RNG/checkpoint恢复保持。结果仍写入同研究ROOT，完整评测前无性能结论。CPU源版本恢复和元数据差异证据保存，0新增模型/GPU/optimizer成本。
