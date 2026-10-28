# 训练／对齐／强化学习：JD统计与下一步选择

2026-10-15。定向便利样本，沿用2026-10-14清单并新增国际站岗位原文；4具体岗位原文与2方向公告分别统计。样本阿里5、海康1，不能推断阿里／字节／腾讯市场频率、HC或用户可投资格。腾讯4社招不入实习分母，字节Seed方向公告不作为杭州具体JD。

| 来源层级 | 明确微调／后训练 | 明确RL／RLHF实践 | 检索 |
| --- | ---: | ---: | ---: |
| 具体岗位JD，n=4 | 1/4（1项仅加分） | 0/4 | 2/4 |
| 多模态方向公告，n=2 | 2/2 | 1/2 | 2/2 |

这里统计明确提及，关键词未出现不代表该工作无需训练。图文表征语义对齐与生成偏好对齐分别编码；公告方向和具体JD可能指同一业务团队，不合成6个独立岗位。两份旧阿里官方JS页面本轮未返回正文，保留2026-10-14审核快照，不能称全样本fresh核验。

## 来源与方向

- [智多星-大模型评测（通用方向）](https://campus-talent.alibaba.com/campus/position/199909720005)：official_jd；inherited20261014 snapshot;official pages may require JS。
- [AI算法实习生-AI导购](https://campus-talent.alibaba.com/campus/position/199909680005)：official_jd；inherited20261014 snapshot;official pages may require JS。
- [Accio研究型实习招聘：方向三多模态学习](https://jdjywpt.jlu.edu.cn/mportal/recruit/details?id=2378ecb86a1b4a62a544060720416123)：university_employer_announcement；fresh source body。
- [1688研究型实习：工业级VLM方向](https://liamding.cc/hiring.html)：team_leader_announcement；fresh source body。
- [研究院-算法实习生-多模态-杭州](https://talent.hikvision.com/home/socity/position?postId=E916B963BC9E74350370DF43D2D2A8A3)：official_jd；inherited20261014 snapshot;official pages may require JS。
- [算法研究员-多模态](https://career.nankai.edu.cn/correcruit/content/id/118629.html)：university_original_job_reprint；fresh original job reprint。

[Agentic模型后训练原文](https://career.nankai.edu.cn/correcruit/content/id/118628.html)明确CPT/SFT/RL，是后训练需求存在的直接证据；该职责包含coding/computer-use/online RL，与本CV项目任务不同，不把一项VLM DPO算成Agentic RL全覆盖。DPO/GRPO未在这份原文点名，不用标题推定具体算法。

## 本轮决策与已执行准备

现有SFT已经覆盖训练能力。为增加可验证的偏好对齐深度，先执行独立`vlm_alignment_signal_v1`：固定SFT seed42、512不同train图、greedy1+sample4=2560实际生成；供给比例>=10%为数据筛选门槛，非模型质量门槛。规则共识chosen>=.9、rejected=0且非空／规范化不同；GRPO统计组内奖励差异。参考只生成后评分，现有dev/final不生成、不用于训练选择。
真实DPO assistant-logp/梯度预检最多8对、0optimizer更新，不保存新adapter；不能写成已完成DPO／GRPO训练或收益。信号足够并检查真实语义后，再独立预注册完整10k偏好构建及继续SFT强对照；完整训练采用必要种子与能力保持保护，未通过不回调门槛。
8 CPU数学／mask／reference边界／报告序列化合同通过；来源、数据、历史权重和正式简历保护，使用原共享GPU队列，不抢占已有任务。

[冻结协议](VLM_ALIGNMENT_SIGNAL_V1_PROTOCOL.md) · [实际生成代码](../src/vlm_alignment_signal/generate.py) · [DPO序列logp与loss](../src/vlm_alignment_signal/loss.py)

公开商品／文档分支仍有价值，训练和RL实践不是所有多模态岗位共同硬要求；后续根据匹配岗位和面试反馈选择，不同时补所有算法。本人掌握／贡献待核对，正式简历／论文／Agent保持，不启CI、不自动push。
