# 执行v2：明确任务名与模型目录名

继承`VLM_REFERENCE_REPLAY_V1_PROTOCOL.md`全部科学设计。准备锁定后核对发现任务键`plain_transfer`与逻辑固定模型名`plain`在v1调度/汇总中混用；尚未启动GPU。v2显式映射plain_transfer→plain，另外soft_kl/hard_replay同名。普通CE固定模型仅评测，不进入新训练。

原v1科学源、PLAN、数据、已过闭式损失测试保留；`analysis_v2.py`的analyze/render函数体逐字复用，只在新模块明确模型角色列表为plain/soft_kl/hard_replay。训练、teacher cache、β校准与评测函数未改。新调度入口、汇总模块、工具和本说明单独绑定到EXECUTION_V2_LOCK，工作流使用queue_v2。

CPU调度收据验证三任务分别对应plain/soft_kl/hard_replay目录及marker，只有后两者训练；0重复编码/新模型/optimizer调用。后续使用`tools/run_reference_replay_v2.py`，旧v1入口不用于执行。原全部成功门槛、预算、seed、比较对象及科学计数不变。
