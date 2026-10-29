# 学生前缀蒸馏：训练完成，完整评测中

2026-10-29 11:40:21最新快照：训练于11:27:03自然结束，32/32更新、512 chosen/206辅助曝光，worker exit0、0争用，active1600.95秒，峰allocated50.039GiB。206次当前学生生成和206次教师正文前向已完成，正文710token，空正文0、长度删失0。首宏步GT CE0.6925567678481457与旧控制一致，语言/视觉梯度有限正、204 FP32因子保存exact；不重复首步或重训。

内部960题已全部完成、worker exit0；完整1617题当前1016/1617，合计1976/2577评测生成，控制器和完整评测worker仍正常运行、0争用。ANALYSIS/DECISION尚未产生，当前无新学生最终提分结论。下一步由原队列自动完成剩余601题并统一验收；按完整评测实测约0.75秒/题，估11:48–11:52出齐，资源/IO变化另计。完成前不启动其他新训练，不依据partial分数选点/调参，原所有强控制及保护门槛保持。

以下为保留的历史排队记录，不代表当前仍未启动训练。

2026-10-29 10:44:15提交controller793427/start535195691，原4项解析CPU检查通过；已有强起点103训练题轨迹中47题/127位置偏离GT前缀，超过事前16题/32位置准入。全部科学输入已冻结。

10:46观察：training_queue等待GPU，0/32新更新；两张A100有其他任务、各约72GiB占用，不抢占。空卡后自动训练，首个新宏步检查新目标/实际raw-token同框/有限梯度/FP32保存及教师共卡内存；不是重审旧loader或首步。按当前设计训练20–30分钟、完整评测15–25分钟，排队另计、首步后修正。

新32正式更新、固定512GT CE/103辅助mask共206曝光；每micro当前学生生成、教师反馈同一原始prefix/正文token，部署greedy GKD变体，T2/λ1/LR/范围不变，0reward model。原旧trainer在worker进程内绑定新目标，源码及原实验锁不改；旧控制直接复用。Final32必须完整960+1617；训练拟合或7B教师93.537%均不能替代新学生提升。

后续首读本ROOT training_queue/STATUS、runs/op_kd/train/{LIVE_TEACHER,FIRST_UPDATE,PROGRESS,CURVE,ROLLOUT_SUMMARY,TRAIN_COMPLETE}与rollouts逐micro文件，再读evaluation_queue及内部/完整完成和ANALYSIS/COMPLETION/DECISION。不要重启旧队列/教师评测，不扩超参/步数/seed，不动其他任务，不改CV/科研/Agent/CI或自动push。

[协议](VLM_ONPOLICY_KD_V1_PROTOCOL.md) · [新目标源码](../src/vlm_onpolicy_kd_v1/objective.py) · [训练适配与验收](../tools/run_onpolicy_teacher_kd_v1.py) · [教师完整结果](VLM_FULL_TEACHER_TRANSFER_V1_RESULTS.md)
