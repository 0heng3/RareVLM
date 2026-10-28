# 实际像素坐标SFT：完整结果与迁移瓶颈

2026-10-20北京时间01:04完整结束。两组各200次更新/3200QA；normalized训练2132.278秒、pixel2157.765秒，无报错/超时/争用。联合门槛未通过，不扩训该候选。

|模型|Text512|Human512|同像素提示的64训练题casefold exact|
|---|---:|---:|---:|
|initial|85.859375|95.117188|41/64|
|normalized|85.371094|94.921875|38/64|
|pixel|85.976562|95.117188|45/64|

## 可以证明什么

坐标适配对照有正向点差：pixel相对新normalized Text+.605469pp，95%图像簇CI[−.135142,+1.453840]跨零；相对初始只+.117188pp，CI[−.718450,+1.002099]。不能称稳定训练收益。

rare对初始+.608696pp，CI[−.762712,+2.631579]跨零；对normalized+1.217391pp。rare只有maps得分+1.0、all seasons→all得分−.3，净+.7score，不视为已证新字符识别能力。

frequent对初始−.901639pp（122题中2题净−1.1score），保护门槛失败。两题为yes→unanswerable（−.4）和mahou negra→mahou（−.7）。后一题原标注mahou negra和heineken各4票，按监督target=heineken（频次23）归入高频，当前输出的mahou negra并不是该频次标签。频次是既定监督代理，不是视觉实体类别；保留原口径/分母/门槛，只补充分歧解释。

Human整体/yes/no点差保持0：相对初始仅yes1题改善、1题降低，no全部相同，不能凭均值0证明等价或无损。

## 训练内改进的具体类型

同原生像素提示，casefold41→45（5修1破），raw33→44。5个casefold纠正为Northwest Grown...→Grown...、App Tools→App、Changing Place→Changing、Northern Constabulary→Constabulary、Aupollo American IPA→Aupollo；目标文字原来已出现，主要是范围收紧。新错WHO→DO。不能将+4严格匹配当新读出4种字符。

同一训练脚本、200更新、数据标签/图像/微批顺序保持，真实梯度/保存正常；当前未定位新的框架或优化器故障。可确认原坐标输入不贴合底座的约定，并且单词转写收益尚未有效迁移到原问答。新数据监督需要检验是否要求模型正确使用框，而不仅是输出单词。

## 已执行下一行动

新零更新诊断vlm_pixel_transfer_diagnosis_v1：排除旧64探针，按预先哈希从已训练区域取256独立图；初始/Final分别对完整图+像素框与官方word裁剪各生成256条，共1024条。全部仍是训练图，不叫heldout验证。主要casefold exact，raw及目标整词出现仅分型；配对区间和同图其他官方word匹配用于区分范围、内容选择与识字。

准备阶段在CPU发现官方smart_resize拒绝19px高的原始ROI；保留preparation_attempt1，按短边至少28px统一保比例LANCZOS放大并记录native/saved尺寸，随后512实际前缀全部通过。旧训练完整图不受影响，不据此追改历史失败归因。

只读controller628554已启动，复用两个冻结adapter、无optimizer更新、共享GPU准入不抢占。GT裁剪同时改变干扰和有效分辨率，不能作部署提升或纯定位因果。不得仅据错误示例改原标签/评测分母，也不在新结果前继续延长训练。

诊断协议：VLM_PIXEL_TRANSFER_DIAGNOSIS_V1_PROTOCOL.md。诊断完整结果以outputs/vlm_pixel_transfer_diagnosis_v1/{RESULTS,COMPLETION}.json为准。
