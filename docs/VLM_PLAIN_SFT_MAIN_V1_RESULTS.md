# 普通SFT32主VQA筛选：区域迁移未转化为主任务收益

2026-10-20 15:04北京时间完整完成，新增Text512/Human512预测，0新训练；同raw greedy/1024原图/FP32 qv与head、复用同策略初始。内部dev已历史曝光，单seed42，不是正式外部benchmark。

|指标|初始|普通SFT32|变化|
|---|---:|---:|---:|
|TextVQA整体|85.859375%|85.488281%|−.371094pp|
|Human整体|95.117188%|94.921875%|−.195313pp|
|Text rare1–5|同原频次/115QA|同一批题|−.260870pp|
|Text frequent>20|同原频次/122QA|同一批题|−.983607pp|
|Human yes|同256QA|同一批题|−.390625pp|
|Human no|同256QA|同一批题|0pp|

Text整体95%图像簇区间[−1.063013,+.270283]跨0，不能断言稳健退化，但没有达到预设+.5pp与正区间；rare亦未过+.5pp，frequent低于−.5pp保护。Human整体/yes/no点值保护通过。联合screen=false，不扩seed/确认/长训/正式简历。

## 失败机制被缩小到哪里

Text512中2题评分上升、6题下降，净−1.9个VQA score point。6个原答案前缀截短变化净−1.3分，其中4个有损截短合计−2.3分：murphy's red ale→murphy's（−.3）、all seasons→all（−.3）、online video→online（−1）、mahou negra→mahou（−.7）。另yale law library→yale增加1分，doktor多词→doktor仍0分。数字、空格和Human一题yes→no是其他变化，不能全部归于长度。

例如原问题“what does the ov in ovguide stand for?”，初始online video得1分，新模型online得0分。缩短区域转写输出与回答完整缩写含义是不同要求。另品牌题本身含mahou negra/heineken各4票的歧义；保留原VQA计分和分母，不据此声称唯一真实标签或回填评测样本训练。

同一普通SFT在本轮图像分离区域题上122→133/256，11修0破、QA+4.296875pp/探索95%[1.953125,7.03125]，pair40→45/128。其6修为收缩已有目标上下文、5修更换错词。这证明更新并非完全无效，却不能替代无GT框主VQA能力。更符合任务/监督粒度迁移取舍的解释；尚未做因果训练对照，不能认定数据、框架或模型容量的唯一病因。

## 已完成针对性数据修正准备

原32985训练QA按空白词数统计多词29.4983%，当前实际512曝光仅13.0859%（回放67/256多词，区域256/256单词）。原源中499条固定响应“answering does not require reading text in the image”不属于普通多词答案；仅排这项后多词28.4153%，偏移仍存在。未修改历史训练或评测标签。

训练白名单已整理279组同图真实短/长VQA问答，原答案各>=8/10一致，无人为截断/补词、无评测图回填。从中固定256图，准备两组同图回放视图：empirical_mix为67多词、scope_balanced为145多词，连同相同256区域曝光后分别13.0859%/28.3203%。两组78个槽位更换真实问题与完整答案，不是纯长度因果操作；相同图像与QA次数不等于相同assistant token计算。

334条唯一回放实际CPU编码全部通过原2048长度和有效assistant标签检查。状态明确为prepared_data_only_not_trained，0更新/0新adapter，没有放行新训练或效果结论。下一轮仅考虑该粒度混合的匹配普通CE对照，另登记图像分离的短/多词保持集和固定Final；不复用重叠题作新holdout，不继续残差/额外token权重或换RL名称。

[完整指标与门槛](LOCAL_ARTIFACTS.md) · [所有变化](LOCAL_ARTIFACTS.md) · [同图修正视图状态](LOCAL_ARTIFACTS.md) · [后续匹配设计](VLM_SCOPE_BALANCED_REPLAY_V1_DESIGN.md) · [原图与互动解释](LOCAL_ARTIFACTS.md)
