# 区域偏好失败：关键token只读诊断结果

2026-10-20 13:34北京时间完成：三个checkpoint各12序列，共36条teacher-forced序列、0optimizer更新、LoRA前后hash全部相同。以下是6道选定训练题，固定新2序列批次，不能称独立泛化或精确重放历史batch4生成概率。

|目标／对比|分歧token|初始logit差|SFT差|DPO差|正确token初始→SFT名次|
|---|---|---:|---:|---:|---:|
|BEFORE / CLOSE|BE / CLOSE|-6.912064|-6.254271|-6.394120|19→15|
|CLOSE / BEFORE|CLOSE / BE|7.748350|7.184410|7.147915|1→1|
|TELL / YOU|TE / YOU|-2.611746|-2.806335|-2.578373|6→6|
|YOU / TELL|YOU / TE|5.158573|5.623472|5.729977|1→1|
|PAR / CHAMPAGNY|PAR / CH|-1.000103|0.239695|0.252888|2→1|
|CHAMPAGNY / CHAMPAGNY PAR|<|im_end|> /  PAR|-3.937014|-2.217472|-2.353504|2→2|

## 能支持的机制

1. BEFORE正确首token BE初始排19，SFT后排15，margin仍−6.254；PAR初始排2、margin−1.000，SFT后排1、margin+.240。此前真实生成仅PAR纠正，与不同难度/改善幅度一致，但6题不能归因全池。
2. TELL完整CE3.493293→3.343126，首个正确token TE的CE3.423954→3.460842，margin−2.611746→−2.806335。完整目标改善并不保证关键生成决策改善；DPO首token仅到−2.578373，仍排6。
3. CHAMPAGNY目标文字已输出，分歧是EOS对继续的“ PAR”，index4。SFT将差−3.937014改善到−2.217472，仍不足停止。选词和范围控制不是同一个失效点，不能统一删除EOS。
4. BEFORE/TELL的full CE与关键token CE初始q/v梯度cosine约.953/.726，DPO与关键token约.905/.817，均非完全无信号或局部反向。局部不等Adam全轨迹，也不证明容量充分。视觉OUTPUT activation梯度非零只说明局部敏感性，未测merger参数训练收益。
5. 原FP32 cosine两条近1数值出现<=.000212超界，保留原始产物，仅作约同向读数，不用作精确角度或主因证据。主要发现取直接logit/CE，不由这两条数值决定；这不是新增训练故障。

## 已执行的下一步

新`vlm_region_decision_sft_v1`已冻结并提交，两组从原SFT42新训32步，数据和顺序/模型/学习率/前向形状不变：在20道固定初始错题上分别加完整CE（强QA加权控制）或首分歧token CE（候选），系数均1。其余题不额外加权，保留完整答案监督。普通SFT32另复用，不冒充新控制。目标是区分只增加难题权重与改变token监督分配。
本轮是常规加权SFT方法研究，不是新DPO/GRPO、原创算法或成功修复声明；新候选待真实28难例+64保持生成评测。学习准入及对加权CE的优势门槛已预先登记，不自动扩长/扫参/写简历。

[协议](VLM_REGION_DECISION_DIAGNOSIS_V1_PROTOCOL.md) · [实际只读代码](../tools/diagnose_region_decision_v1.py) · [下一实验协议](VLM_REGION_DECISION_SFT_V1_PROTOCOL.md) · [原图与公式](LOCAL_ARTIFACTS.md) · [全部读数](LOCAL_ARTIFACTS.md)
