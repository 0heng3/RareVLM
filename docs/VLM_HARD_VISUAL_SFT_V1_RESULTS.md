# 困难样本SFT 200步筛选结果

| 模型 | 人工screen368 | Text screen512 |
| --- | ---: | ---: |
| initial | 94.293478 | 86.074219 |
| uniform | 94.565217 | 86.464844 |
| hard | 94.293478 | 86.269531 |

筛选通过：False。

stop: no confirmation generation or longer training; retain complete negative result

这是新划分上的单种子探索结果；独立题数、曝光、原图复核和全部升降预测均保存在输出目录。

人工主指标：候选对初始0pp，95%图像簇区间[−1.282134,+1.298701]pp；对控制−0.271739pp，区间[−1.215898,+0.563380]pp。候选3纠正/3新错，6次均no→yes；yes +1.744186pp，no −1.530612pp，未过类别保护。此观察支持检查回答倾向，不能单独证明失败原因。

两组训练各约24.7分钟，权重保存重载通过。Text评测因迁移清单的相对图片目录缺失中断，已恢复data/images至原始目录，357张图文件SHA匹配；原始失败日志存于recovery_v1/failed_attempt。未修改科学设置、权重或评分标签，未重新训练。完整训练完成文件哈希前后相同。

本轮停止：不评confirm、不追加训练、不改简历；学习原图与全部升降入口见learning/困难样本SFT_200步.html。
