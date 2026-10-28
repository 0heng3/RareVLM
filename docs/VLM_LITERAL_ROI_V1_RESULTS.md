# 给定单词区域完整转写：初始结果

{'training_pool': {'literal_exact_percent': 81.201171875, 'qa': 2048, 'full': 1663, 'frequency': {'zero_target_word_exposure': {'qa': 1008, 'literal_exact_percent': 70.63492063492063}, 'rare_1_5': {'qa': 1040, 'literal_exact_percent': 91.4423076923077}}}, 'validation': {'literal_exact_percent': 82.421875, 'qa': 256, 'full': 211, 'frequency': {'zero_target_word_exposure': {'qa': 129, 'literal_exact_percent': 69.76744186046511}, 'rare_1_5': {'qa': 127, 'literal_exact_percent': 95.2755905511811}}}}

训练候选错误381，数量门槛=True，仍需实际区域图视觉复核与独立训练协议。0训练/新adapter，不是整图定位或原SFT未见图证据。

[协议](VLM_LITERAL_ROI_V1_PROTOCOL.md) · [实际区域图、原图与回答](LOCAL_ARTIFACTS.md)

后续训练准入：全部381困难候选经助手审核图查看，130视觉支持/251排除或不确定，固定SHA首128用于新训练；不是专家重标注或全数据错标率。独立REVIEWED_ERRORS.json为后续准入凭据，原COMPLETION的visual_review_passed=false是初始评测时快照，不改写。实际训练与迁移结果另见[VLM_LITERAL_PREFERENCE_V1](VLM_LITERAL_PREFERENCE_V1_RESULTS.md)。
